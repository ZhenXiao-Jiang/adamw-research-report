# run_synthetic.py — synthetic diagonal quadratic experiment
# -*- coding: utf-8 -*-
"""
run_synthetic.py

对角二次合成目标上的对比实验脚本，比较三种方法：
  - adam_l2       : 损失中加入 0.5 * lambda * ||theta||^2，优化器 weight_decay=0
  - adamw         : 优化器 weight_decay=lambda（解耦权重衰减）
  - adam_no_decay : 优化器 weight_decay=0（无衰减基线）

优化器声明：
  PyTorch 方法一律直接实例化 torch.optim.AdamW，不自行编写参数更新公式。
  优化器来源：PyTorch（https://pytorch.org/）BSD-3-Clause
  https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  Keras 3 对照为可选分支，使用 keras.optimizers.AdamW，仅在环境具备时运行。

输出：
  1) 每个配置一个逐 step CSV，写入 data/func/，列含 step、lr、loss、loss_total、
     grad_norm、weight_l2，可直接供绘图脚本画曲线；
  2) 汇总 CSV data/synthetic_summary.csv，保留原有字段并新增 budget 列。
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Windows 控制台代码页可能无法编码部分字符，进程退出时刷新标准流会报错并返回非零退出码。
# 此处仅加固输出编码，不影响任何实验逻辑。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import torch

METHODS = ("adam_l2", "adamw", "adam_no_decay")
METHOD_ALIASES = {"adam_nodecay": "adam_no_decay"}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "func_config.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="对角二次目标上的 Adam+L2 / AdamW / Adam(no decay) 对比实验"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="配置文件路径，默认为脚本上一级目录的 configs/func_config.json")
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子列表")
    parser.add_argument("--lambdas", type=str, default=None, help="逗号分隔的权重衰减 lambda 列表")
    parser.add_argument("--alphas", type=str, default=None, help="逗号分隔的学习率 alpha 列表")
    parser.add_argument("--schedules", type=str, default=None, help="逗号分隔的日程列表（cosine,step）")
    parser.add_argument("--budgets", type=str, default=None, help="逗号分隔的训练步数档位列表")
    parser.add_argument("--keras", action="store_true", default=False,
                        help="同时运行 Keras 3 AdamW 对照（需已安装 keras 与后端）")
    parser.add_argument("--no-keras", action="store_true", default=False,
                        help="强制跳过 Keras 对照")
    return parser.parse_args()


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip() != ""]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip() != ""]


def parse_str_list(text):
    return [item.strip() for item in text.split(",") if item.strip() != ""]


def load_config(args):
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.seeds is not None:
        config["seeds"] = parse_int_list(args.seeds)
    if args.lambdas is not None:
        config["lambdas"] = parse_float_list(args.lambdas)
    if args.alphas is not None:
        config["alphas"] = parse_float_list(args.alphas)
    if args.schedules is not None:
        config["schedules"] = parse_str_list(args.schedules)
    if args.budgets is not None:
        config["budgets"] = parse_int_list(args.budgets)
    return config, config_path


def resolve_methods(config):
    raw_methods = config.get("methods", list(METHODS))
    resolved = []
    for name in raw_methods:
        key = METHOD_ALIASES.get(name, name)
        if key in METHODS and key not in resolved:
            resolved.append(key)
    for key in METHODS:
        if key not in resolved:
            resolved.append(key)
    return resolved


def resolve_output_dirs(config, config_path):
    output = config.get("output", {})
    base = config_path.parent.parent

    def pick(key, default_name=None):
        value = output.get(key)
        if value:
            return Path(value)
        return base / (default_name or key)

    return pick("logs"), pick("data"), pick("func_data", "data/func"), pick("figures")


def build_curvature(dim, curvature_min, curvature_max, device):
    if dim <= 0:
        raise ValueError("dim 必须为正整数")
    if curvature_min <= 0 or curvature_max <= 0:
        raise ValueError("curvature_min 与 curvature_max 必须为正数")
    if curvature_max < curvature_min:
        raise ValueError("curvature_max 不能小于 curvature_min")
    log_min = math.log(curvature_min)
    log_max = math.log(curvature_max)
    if dim == 1:
        exponents = torch.zeros(1, device=device, dtype=torch.float32)
    else:
        exponents = torch.linspace(0.0, 1.0, dim, device=device, dtype=torch.float32)
    return torch.exp(log_min + exponents * (log_max - log_min))


def build_scheduler(schedule, optimizer, steps):
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    if schedule == "step":
        milestones = sorted({int(steps * 0.5), int(steps * 0.75)})
        milestones = [m for m in milestones if 0 < m < steps]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    raise ValueError("不支持的学习率日程: {0}（可选 cosine, step）".format(schedule))


def run_tag(method, lam, alpha, schedule, seed, budget):
    return "func_{0}_lam{1:g}_alpha{2:g}_{3}_seed{4}_budget{5}".format(
        method, lam, alpha, schedule, seed, budget)


def curve_filename(method, lam, alpha, schedule, seed, budget, framework="pytorch"):
    return "{0}_{1}_lam{2:g}_alpha{3:g}_{4}_seed{5}_budget{6}.csv".format(
        framework, method, lam, alpha, schedule, seed, budget)


CURVE_FIELDS = ["step", "lr", "loss", "loss_total", "grad_norm", "weight_l2"]


def run_single(method, lam, alpha, schedule, seed, budget, config, device, log_dir, func_dir):
    dim = int(config["dim"])
    log_interval = max(1, int(config["log_interval"]))
    record_interval = max(1, int(config.get("record_interval", 1)))
    curvature_min = float(config["curvature_min"])
    curvature_max = float(config["curvature_max"])
    steps = int(budget)

    tor_curv = build_curvature(dim, curvature_min, curvature_max, device)

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    theta = torch.randn(dim, device=device, generator=generator, dtype=torch.float32)
    theta.requires_grad_(True)

    weight_decay = lam if method == "adamw" else 0.0
    optimizer = torch.optim.AdamW([theta], lr=alpha, weight_decay=weight_decay)
    scheduler = build_scheduler(schedule, optimizer, steps)

    tag = run_tag(method, lam, alpha, schedule, seed, budget)
    log_path = log_dir / ("synthetic_{0}.log".format(tag))
    curve_path = func_dir / curve_filename(method, lam, alpha, schedule, seed, budget)

    rows = []
    header = ("[{0}] dim={1} steps={2} curvature_min={3:g} curvature_max={4:g} "
              "lr={5:g} weight_decay={6:g} schedule={7} seed={8} budget={9}").format(
        tag, dim, steps, curvature_min, curvature_max, alpha, weight_decay, schedule, seed, budget)
    print(header)

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(header + "\n")
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            quad_loss = 0.5 * torch.sum(tor_curv * theta * theta)
            if method == "adam_l2":
                total_loss = quad_loss + 0.5 * lam * torch.sum(theta * theta)
            else:
                total_loss = quad_loss
            total_loss.backward()

            grad_norm = float(torch.linalg.vector_norm(theta.grad.detach()).item())
            weight_l2 = float(torch.linalg.vector_norm(theta.detach()).item())

            optimizer.step()
            lr_now = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            step_index = step + 1
            if step_index % record_interval == 0 or step_index == steps:
                rows.append({
                    "step": step_index,
                    "lr": lr_now,
                    "loss": float(quad_loss.item()),
                    "loss_total": float(total_loss.item()),
                    "grad_norm": grad_norm,
                    "weight_l2": weight_l2,
                })
            if step_index % log_interval == 0 or step_index == steps:
                line = ("[{0}] step={1:6d} lr={2:.6e} loss={3:.6e} loss_total={4:.6e} "
                        "grad_norm={5:.6e} weight_l2={6:.6e}").format(
                    tag, step_index, lr_now, float(quad_loss.item()),
                    float(total_loss.item()), grad_norm, weight_l2)
                print(line)
                log_handle.write(line + "\n")

    with curve_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    last = rows[-1] if rows else {k: float("nan") for k in CURVE_FIELDS}
    return {
        "framework": "pytorch",
        "method": method,
        "lambda": lam,
        "alpha": alpha,
        "schedule": schedule,
        "seed": seed,
        "budget": steps,
        "weight_decay": weight_decay,
        "steps": steps,
        "final_step": last.get("step", 0),
        "final_lr": last.get("lr", float("nan")),
        "final_loss": last.get("loss", float("nan")),
        "final_loss_total": last.get("loss_total", float("nan")),
        "final_grad_norm": last.get("grad_norm", float("nan")),
        "final_weight_l2": last.get("weight_l2", float("nan")),
        "min_loss": min((r["loss"] for r in rows), default=float("nan")),
        "curve_file": str(curve_path.name),
    }


def run_keras_control(keras_cfg, func_dir, device):
    """Keras 3 AdamW 对照：与 PyTorch 同配置跑一组，用于框架一致性核对。"""
    try:
        import keras
        from keras import ops
    except Exception as exc:  # noqa: BLE001
        print("[提示] 未检测到可用的 Keras 3，已跳过 Keras 对照：{0}".format(exc))
        return []

    dim = int(keras_cfg["dim"])
    curvature_min = float(keras_cfg["curvature_min"])
    curvature_max = float(keras_cfg["curvature_max"])
    record_interval = max(1, int(keras_cfg.get("record_interval", 1)))

    curv = build_curvature(dim, curvature_min, curvature_max, device).cpu().numpy()
    results = []
    for lam in keras_cfg["lambdas"]:
        for alpha in keras_cfg["alphas"]:
            for schedule in keras_cfg["schedules"]:
                for seed in keras_cfg["seeds"]:
                    for budget in keras_cfg["budgets"]:
                        import numpy as np
                        rng = np.random.RandomState(seed)
                        theta = rng.randn(dim).astype("float32")
                        theta = keras.Variable(theta, name="theta")
                        opt = keras.optimizers.AdamW(learning_rate=alpha, weight_decay=lam)
                        rows = []
                        for step in range(int(budget)):
                            def loss_fn():
                                return 0.5 * ops.sum(ops.convert_to_tensor(curv) * theta * theta)
                            grads = keras.gradients(loss_fn, [theta])
                            opt.apply_gradients([(grads[0], theta)])
                            step_index = step + 1
                            if step_index % record_interval == 0 or step_index == int(budget):
                                value = float(keras.ops.convert_to_numpy(loss_fn()))
                                wl2 = float(np.linalg.norm(keras.ops.convert_to_numpy(theta)))
                                rows.append({"step": step_index, "lr": float(opt.learning_rate),
                                             "loss": value, "loss_total": value,
                                             "grad_norm": float("nan"), "weight_l2": wl2})
                        path = func_dir / curve_filename(
                            "adamw", lam, alpha, schedule, seed, budget, framework="keras")
                        with path.open("w", encoding="utf-8", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
                            writer.writeheader()
                            for row in rows:
                                writer.writerow(row)
                        last = rows[-1] if rows else {k: float("nan") for k in CURVE_FIELDS}
                        results.append({
                            "framework": "keras", "method": "adamw", "lambda": lam,
                            "alpha": alpha, "schedule": schedule, "seed": seed,
                            "budget": int(budget), "weight_decay": lam, "steps": int(budget),
                            "final_step": last.get("step", 0), "final_lr": last.get("lr", float("nan")),
                            "final_loss": last.get("loss", float("nan")),
                            "final_loss_total": last.get("loss_total", float("nan")),
                            "final_grad_norm": float("nan"),
                            "final_weight_l2": last.get("weight_l2", float("nan")),
                            "min_loss": min((r["loss"] for r in rows), default=float("nan")),
                            "curve_file": str(path.name),
                        })
                        print("[keras] 完成 {0}".format(path.name))
    return results


SUMMARY_FIELDS = [
    "framework", "method", "lambda", "alpha", "schedule", "seed", "budget",
    "weight_decay", "steps", "final_step", "final_lr", "final_loss",
    "final_loss_total", "final_grad_norm", "final_weight_l2", "min_loss", "curve_file",
]


def write_summary(summary_rows, data_dir):
    summary_path = data_dir / "synthetic_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_path


def main():
    args = parse_args()
    config, config_path = load_config(args)

    log_dir, data_dir, func_dir, figures_dir = resolve_output_dirs(config, config_path)
    for path in (log_dir, data_dir, func_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch version : {0}".format(torch.__version__))
    print("cuda available: {0}".format(torch.cuda.is_available()))
    print("device        : {0}".format(device))
    print("config file   : {0}".format(config_path))

    methods = resolve_methods(config)
    seeds = [int(s) for s in config["seeds"]]
    lambdas = [float(x) for x in config["lambdas"]]
    alphas = [float(x) for x in config["alphas"]]
    schedules = [str(s) for s in config["schedules"]]
    budgets = [int(b) for b in config.get("budgets", [config.get("steps", 2000)])]

    for schedule in schedules:
        if schedule not in ("cosine", "step"):
            raise ValueError("不支持的学习率日程: {0}（可选 cosine, step）".format(schedule))

    total = len(seeds) * len(methods) * len(lambdas) * len(alphas) * len(schedules) * len(budgets)
    keras_cfg = dict(config.get("keras", {}) or {})
    keras_enabled = bool(keras_cfg.get("enabled", False))
    if args.keras:
        keras_enabled = True
    if args.no_keras:
        keras_enabled = False
    keras_total = 0
    if keras_enabled:
        keras_total = (len(keras_cfg.get("lambdas", [])) * len(keras_cfg.get("alphas", []))
                       * len(keras_cfg.get("schedules", [])) * len(keras_cfg.get("seeds", []))
                       * len(keras_cfg.get("budgets", [])))

    print("methods   : {0}".format(methods))
    print("seeds     : {0}".format(seeds))
    print("lambdas   : {0}".format(lambdas))
    print("alphas    : {0}".format(alphas))
    print("schedules : {0}".format(schedules))
    print("budgets   : {0}".format(budgets))
    print("PyTorch 组数: {0}；Keras 对照组数: {1}".format(total, keras_total))

    summary_rows = []
    done = 0
    for budget in budgets:
        for seed in seeds:
            for method in methods:
                for lam in lambdas:
                    for alpha in alphas:
                        for schedule in schedules:
                            summary_rows.append(run_single(
                                method=method, lam=lam, alpha=alpha, schedule=schedule,
                                seed=seed, budget=budget, config=config, device=device,
                                log_dir=log_dir, func_dir=func_dir))
                            done += 1
                            print("[进度] PyTorch {0}/{1} 已完成".format(done, total))

    if keras_enabled:
        for key, default in (("dim", config["dim"]),
                             ("curvature_min", config["curvature_min"]),
                             ("curvature_max", config["curvature_max"]),
                             ("record_interval", config.get("record_interval", 1))):
            keras_cfg.setdefault(key, default)
        summary_rows.extend(run_keras_control(keras_cfg, func_dir, device))
    else:
        print("[提示] Keras 3 对照未启用；如需生成表 2，请加 --keras 或把 config 中 keras.enabled 置为 true。")

    summary_path = write_summary(summary_rows, data_dir)
    print("完成：共 {0} 组运行（含 Keras 对照 {1} 组）。".format(len(summary_rows), keras_total))
    print("日志目录: {0}".format(log_dir))
    print("数据目录: {0}".format(data_dir))
    print("逐 step 目录: {0}".format(func_dir))
    print("汇总文件: {0}".format(summary_path))


if __name__ == "__main__":
    main()
