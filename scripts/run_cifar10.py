#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# CIFAR-10 对比实验：Adam + L2 / AdamW / Adam(no decay)
#
# 优化器直接调用 PyTorch 官方 torch.optim.AdamW（未自行实现更新公式），
# 权重衰减通过 optimizer 的 weight_decay 参数传入，而不是重写更新逻辑。
#   许可证：BSD-3-Clause
#   文档  ：https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
#   项目  ：https://pytorch.org
#
# 用法示例（默认读取脚本上一级目录的 configs/cifar10_config.json）：
#   python run_cifar10.py
#   python run_cifar10.py --config ../configs/cifar10_config.json
#   python run_cifar10.py --seeds 0,1 --lambdas 0.01 --alphas 1e-3 \
#        --schedules cosine --models small_resnet --epochs 5 --device cpu
# =============================================================================

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms


# -----------------------------------------------------------------------------
# 常量
# -----------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CSV_HEADER = [
    "timestamp", "method", "model", "lambda", "alpha", "schedule",
    "seed", "epoch", "lr", "train_loss", "test_loss", "test_error",
    "weight_l2_norm", "epoch_seconds",
]

METHOD_ALIASES = {
    "adam_l2": "adam_l2",
    "adamw": "adamw",
    "adam": "adamw",
    "adam_no_decay": "adam_no_decay",
    "adam_nodecay": "adam_no_decay",
}


def normalize_method(name):
    key = str(name).strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError("未知方法: {}".format(name))
    return METHOD_ALIASES[key]


# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------
class Logger(object):
    def __init__(self, path):
        self.path = Path(path)
        self.fh = open(self.path, "w", encoding="utf-8")

    def __call__(self, msg):
        text = str(msg)
        print(text, flush=True)
        self.fh.write(text + "\n")
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# 模型：小型 ResNet（BasicBlock 堆叠）
# -----------------------------------------------------------------------------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SmallResNet(nn.Module):
    """通道数按单机可承受规模缩小的小型 ResNet。"""

    def __init__(self, num_classes=10, widths=(16, 32, 64), blocks=(2, 2, 2)):
        super(SmallResNet, self).__init__()
        self.stem = nn.Sequential(
            conv3x3(3, widths[0]),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.in_ch = widths[0]
        self.layer1 = self._make_layer(widths[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(widths[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(widths[2], blocks[2], stride=2)
        self.fc = nn.Linear(widths[2], num_classes)

    def _make_layer(self, out_ch, n_blocks, stride):
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(int(n_blocks) - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# 模型：小型 VGG（三层卷积 + 两层全连接）
# -----------------------------------------------------------------------------
class SmallVGG(nn.Module):
    """通道数按单机可承受规模缩小的小型 VGG。"""

    def __init__(self, num_classes=10, widths=(32, 64, 128), hidden=256, dropout=0.1):
        super(SmallVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(widths[0], widths[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(widths[1], widths[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(widths[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        flat_dim = widths[2] * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(name, num_classes=10):
    key = str(name).strip().lower()
    if key == "small_resnet":
        return SmallResNet(num_classes=num_classes)
    if key == "small_vgg":
        return SmallVGG(num_classes=num_classes)
    raise ValueError("未知模型: {}".format(name))


# -----------------------------------------------------------------------------
# 参数分组：权重矩阵一组；偏置 / BatchNorm 权重与偏置一组（weight_decay=0）
# -----------------------------------------------------------------------------
def build_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = (param.ndim <= 1) or name.endswith(".bias")
        is_norm = ("bn" in name.lower()) or ("norm" in name.lower())
        if is_bias or is_norm:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# -----------------------------------------------------------------------------
# 学习率日程：cosine / step
# -----------------------------------------------------------------------------
def compute_lr(schedule, base_lr, epoch, total_epochs, milestones, gamma):
    schedule = str(schedule).strip().lower()
    total = max(int(total_epochs), 1)
    if schedule == "cosine":
        return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * float(epoch) / float(total)))
    if schedule == "step":
        lr = float(base_lr)
        for m in milestones:
            boundary = int(round(float(m) * total))
            if int(epoch) >= boundary:
                lr *= float(gamma)
        return lr
    raise ValueError("未知学习率日程: {}".format(schedule))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


# -----------------------------------------------------------------------------
# 度量
# -----------------------------------------------------------------------------
def squared_param_sum(model):
    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        s = param.pow(2).sum()
        total = s if total is None else total + s
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total


def weight_l2_norm(model):
    total = 0.0
    for param in model.parameters():
        total += float(param.detach().pow(2).sum().item())
    return math.sqrt(total)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += float(loss.item()) * inputs.size(0)
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        total += int(inputs.size(0))
    total = max(total, 1)
    return total_loss / total, 1.0 - float(correct) / float(total)


# -----------------------------------------------------------------------------
# 数据
# -----------------------------------------------------------------------------
def build_loaders(cfg, device):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    root = str(cfg["data_root"])
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf)

    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# 单次运行
# -----------------------------------------------------------------------------
def run_one(cfg, method, model_name, lam, alpha, schedule, seed,
            device, writer, logger, run_index, run_total):
    num_classes = int(cfg.get("num_classes", 10))
    epochs = int(cfg.get("epochs", 20))
    milestones = cfg.get("step_milestones", [0.5, 0.75])
    gamma = float(cfg.get("step_gamma", 0.1))

    torch.manual_seed(int(seed))
    random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    train_loader, test_loader = build_loaders(cfg, device)

    model = build_model(model_name, num_classes=num_classes).to(device)

    weight_decay = float(lam) if method == "adamw" else 0.0
    param_groups = build_param_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=float(alpha), betas=(0.9, 0.999), eps=1e-8)

    criterion = nn.CrossEntropyLoss()

    tag = ("[{:>3d}/{:>3d}] model={} method={} lambda={:g} alpha={:g} "
           "schedule={} seed={}").format(
        run_index, run_total, model_name, method, float(lam),
        float(alpha), schedule, int(seed))
    logger(tag)

    for epoch in range(epochs):
        epoch_start = time.time()
        lr = compute_lr(schedule, alpha, epoch, epochs, milestones, gamma)
        set_optimizer_lr(optimizer, lr)

        model.train()
        train_loss_sum = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            data_loss = criterion(outputs, targets)
            loss = data_loss
            if method == "adam_l2" and float(lam) > 0.0:
                loss = loss + 0.5 * float(lam) * squared_param_sum(model)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(data_loss.item()) * inputs.size(0)
            seen += int(inputs.size(0))

        train_loss = train_loss_sum / max(seen, 1)
        test_loss, test_error = evaluate(model, test_loader, device, criterion)
        l2_norm = weight_l2_norm(model)
        epoch_seconds = time.time() - epoch_start

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            method, model_name, float(lam), float(alpha), schedule,
            int(seed), int(epoch), float(lr), float(train_loss),
            float(test_loss), float(test_error), float(l2_norm),
            float(epoch_seconds),
        ])
        writer_fh.flush()

        logger("    epoch {:>3d}/{:>3d} lr={:.6e} train_loss={:.4f} "
               "test_loss={:.4f} test_err={:.4f} w_l2={:.4f} ({:.1f}s)".format(
                   epoch + 1, epochs, lr, train_loss, test_loss,
                   test_error, l2_norm, epoch_seconds))

    return test_loss, test_error, l2_norm


# -----------------------------------------------------------------------------
# 参数解析与配置
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 上 Adam+L2 / AdamW / Adam(no decay) 对比实验")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径，默认取脚本上一级目录的 configs/cifar10_config.json")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备，如 cpu / cuda / cuda:0；不给则自动判断")
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子")
    parser.add_argument("--lambdas", type=str, default=None, help="逗号分隔的权重衰减系数")
    parser.add_argument("--alphas", type=str, default=None, help="逗号分隔的学习率")
    parser.add_argument("--schedules", type=str, default=None, help="逗号分隔的学习率日程")
    parser.add_argument("--models", type=str, default=None, help="逗号分隔的模型名")
    parser.add_argument("--epochs", type=str, default=None, help="训练轮数（整数）")
    return parser.parse_args()


def load_config(path):
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_csv(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def apply_overrides(cfg, args):
    if args.seeds:
        cfg["seeds"] = [int(x) for x in split_csv(args.seeds)]
    if args.lambdas:
        cfg["lambdas"] = [float(x) for x in split_csv(args.lambdas)]
    if args.alphas:
        cfg["alphas"] = [float(x) for x in split_csv(args.alphas)]
    if args.schedules:
        cfg["schedules"] = split_csv(args.schedules)
    if args.models:
        cfg["models"] = split_csv(args.models)
    if args.epochs:
        cfg["epochs"] = int(args.epochs)
    return cfg


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = (script_dir.parent / "configs" / "cifar10_config.json").resolve()

    if not config_path.exists():
        print("配置文件不存在: {}".format(config_path), file=sys.stderr)
        return 1

    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, args)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_cfg = cfg.get("output", {}) or {}
    data_dir = Path(output_cfg.get("data", str(script_dir.parent / "data")))
    log_dir = Path(output_cfg.get("logs", str(script_dir.parent / "logs")))
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / "cifar10_run_{}.log".format(timestamp)
    csv_path = data_dir / "cifar10_results.csv"

    logger = Logger(log_path)

    methods = [normalize_method(m) for m in cfg.get("methods", ["adam_l2", "adamw", "adam_no_decay"])]
    models = list(cfg.get("models", ["small_resnet", "small_vgg"]))
    seeds = list(cfg.get("seeds", [0]))
    lambdas = list(cfg.get("lambdas", [0.01]))
    alphas = list(cfg.get("alphas", [0.001]))
    schedules = list(cfg.get("schedules", ["cosine"]))
    epochs = int(cfg.get("epochs", 20))
    cfg["epochs"] = epochs

    total_runs = (len(models) * len(methods) * len(lambdas) *
                  len(alphas) * len(schedules) * len(seeds))

    logger("配置路径: {}".format(config_path))
    logger("运行设备: {} (cuda_available={})".format(device, torch.cuda.is_available()))
    logger("数据根目录: {}".format(cfg.get("data_root")))
    logger("日志目录: {}".format(log_dir))
    logger("数据目录: {}".format(data_dir))
    logger("结果 CSV: {}".format(csv_path))
    logger("方法: {}".format(methods))
    logger("模型: {}".format(models))
    logger("种子: {}".format(seeds))
    logger("lambda: {}".format(lambdas))
    logger("alpha: {}".format(alphas))
    logger("日程: {}".format(schedules))
    logger("轮数: {}".format(epochs))
    logger("step_milestones: {}, step_gamma: {}".format(
        cfg.get("step_milestones", [0.5, 0.75]), cfg.get("step_gamma", 0.1)))
    logger("总运行次数: {}".format(total_runs))
    logger("优化器: torch.optim.AdamW（PyTorch 官方实现，BSD-3-Clause）")

    csv_exists = csv_path.exists()
    global writer_fh
    writer_fh = open(str(csv_path), "a", newline="", encoding="utf-8")
    writer = csv.writer(writer_fh)
    if not csv_exists:
        writer.writerow(CSV_HEADER)
        writer_fh.flush()

    run_index = 0
    started = time.time()
    try:
        for model_name in models:
            for method in methods:
                for lam in lambdas:
                    for alpha in alphas:
                        for schedule in schedules:
                            for seed in seeds:
                                run_index += 1
                                run_one(cfg, method, model_name, lam, alpha,
                                        schedule, seed, device, writer,
                                        logger, run_index, total_runs)
    finally:
        writer_fh.flush()
        writer_fh.close()
        logger("全部运行结束，用时 {:.1f}s，共 {} 次".format(
            time.time() - started, run_index))
        logger.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())