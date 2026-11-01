# plot_results.py — figures and summary tables from experiment data
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_results.py

从实验数据目录中的 CSV 文件生成图 1 至图 5，并输出表 1、表 2 的汇总 CSV。

默认路径（相对于本脚本所在目录）：
    数据目录   ../data
    输出目录   ../figures

用法：
    python plot_results.py [--data-dir DIR] [--fig-dir DIR]

文件名约定（能识别多少用多少，缺省项不影响其余流程）：
    任务名（func / cifar10）、方法名（adam_l2 / adamw / adam）、
    lambda 或 lam 或 weight_decay、alpha 或 lr、seed、budget（或 steps / epochs）、
    framework（pytorch / keras）。
若文件名中不包含某些字段，则尝试从 CSV 的列名与列值中读取。
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = [
    'DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
    'WenQuanYi Zen Hei', 'Arial Unicode MS',
]
rcParams['axes.unicode_minus'] = False

NUM_PAT = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

METHOD_ORDER = ['adam_l2', 'adamw', 'adam']

STEP_EXACT = ('step', 'steps', 'iteration', 'iterations', 'iter', 'it',
              'update', 'updates', 'epoch', 'epochs', 'nstep', 'niter')
STEP_CONTAINS = ('step', 'iter', 'epoch', 'update')

LOSS_EXACT = ('loss', 'trainloss', 'trainloss', 'objective')
LOSS_CONTAINS = ('loss', 'objective')

L2_EXACT = ('l2', 'weightl2', 'weightl2norm', 'l2norm', 'weightnorm', 'wnorm')
L2_CONTAINS = ('weightl2', 'l2norm', 'weightnorm', 'wnorm', 'l2')

ERR_EXACT = ('testerror', 'testerr', 'testerrorrate', 'testerrorrate',
             'valerror', 'valerr', 'error')
ERR_CONTAINS = ('testerror', 'testerr', 'valerror', 'valerr', 'error')


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------

def default_dirs():
    """返回默认的数据目录与图片输出目录（脚本上一级目录下的 data / figures）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    return os.path.join(parent, 'data'), os.path.join(parent, 'figures')


def normalize_name(text):
    """把列名或文件名片段归一化成小写字母数字串，便于宽松匹配。"""
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def find_col(cols, exact=(), contains=()):
    """在列名列表中按精确匹配优先、包含匹配其次的方式定位一列。"""
    norm_map = {c: normalize_name(c) for c in cols}
    for c in cols:
        if norm_map[c] in exact:
            return c
    for c in cols:
        n = norm_map[c]
        for key in contains:
            if key and key in n:
                return c
    return None


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def to_int(value):
    f = to_float(value)
    if f is None:
        return None
    try:
        return int(round(f))
    except (TypeError, ValueError):
        return None


def first_scalar(df, col):
    """取某列的第一个非空值。"""
    if col is None or col not in df.columns:
        return None
    vals = df[col].dropna()
    if vals.empty:
        return None
    return vals.iloc[0]


# ----------------------------------------------------------------------------
# 文件名解析
# ----------------------------------------------------------------------------

def parse_filename(path):
    """从文件名中提取任务名、方法名、lambda、alpha、种子、预算与框架。"""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    info = {}

    m = re.search(r'(?:lambda|lam|weight_?decay|wd)[_\-=]?(' + NUM_PAT + ')', stem)
    if m:
        v = to_float(m.group(1))
        if v is not None:
            info['lambda'] = v

    m = re.search(r'(?:alpha|alp|learning_?rate|lr)[_\-=]?(' + NUM_PAT + ')', stem)
    if m:
        v = to_float(m.group(1))
        if v is not None:
            info['alpha'] = v

    m = re.search(r'(?:seed|sd)[_\-=]?(\d+)', stem)
    if m:
        v = to_int(m.group(1))
        if v is not None:
            info['seed'] = v

    m = re.search(r'(?:budget|bud|nsteps|niter|steps|epochs)[_\-=]?(\d+)', stem)
    if m:
        v = to_int(m.group(1))
        if v is not None:
            info['budget'] = v

    if 'cifar' in stem:
        info['task'] = 'cifar10'
    elif 'func' in stem or 'quad' in stem or 'synth' in stem:
        info['task'] = 'func'

    if 'adamw' in stem:
        info['method'] = 'adamw'
    elif 'adam' in stem:
        if re.search(r'(?<![a-z0-9])l2(?![a-z0-9])', stem):
            info['method'] = 'adam_l2'
        else:
            info['method'] = 'adam'

    if 'keras' in stem:
        info['framework'] = 'keras'
    elif 'torch' in stem or 'pytorch' in stem:
        info['framework'] = 'pytorch'

    return info


# ----------------------------------------------------------------------------
# 读入与解析 CSV
# ----------------------------------------------------------------------------

def analyse_file(path, df):
    """把一个 CSV 解析成带元信息的记录，供后续绘图与汇总使用。"""
    rec = {'path': path, 'df': df}
    rec.update(parse_filename(path))

    cols = list(df.columns)

    rec['xcol'] = find_col(cols, STEP_EXACT, STEP_CONTAINS)
    rec['loss_col'] = find_col(cols, LOSS_EXACT, LOSS_CONTAINS)
    rec['l2_col'] = find_col(cols, L2_EXACT, L2_CONTAINS)
    rec['err_col'] = find_col(cols, ERR_EXACT, ERR_CONTAINS)

    lam_col = find_col(cols, ('lambda', 'lam', 'weightdecay', 'wd'),
                       ('lambda', 'weightdecay'))
    alpha_col = find_col(cols, ('alpha', 'lr', 'learningrate'),
                         ('alpha', 'learningrate'))
    seed_col = find_col(cols, ('seed', 'randomseed'), ('seed',))
    budget_col = find_col(cols, ('budget', 'nsteps', 'totalsteps', 'steps', 'epochs'),
                          ('budget',))
    method_col = find_col(cols, ('method', 'optimizer', 'opt', 'algo', 'algorithm'),
                          ('method', 'optimizer'))
    task_col = find_col(cols, ('task', 'dataset', 'problem'), ('task', 'dataset'))
    fw_col = find_col(cols, ('framework', 'backend', 'fw'), ('framework', 'backend'))

    scalar_targets = (
        ('lambda', lam_col, 'float'),
        ('alpha', alpha_col, 'float'),
        ('seed', seed_col, 'int'),
        ('budget', budget_col, 'int'),
        ('method', method_col, 'str'),
        ('task', task_col, 'str'),
        ('framework', fw_col, 'str'),
    )
    for key, col, kind in scalar_targets:
        if rec.get(key) is not None:
            continue
        val = first_scalar(df, col)
        if val is None:
            continue
        if kind == 'float':
            val = to_float(val)
        elif kind == 'int':
            val = to_int(val)
        else:
            val = str(val)
        if val is not None:
            rec[key] = val

    if rec.get('task') is None:
        xname = normalize_name(rec['xcol']) if rec['xcol'] else ''
        if rec['err_col'] is not None and ('epoch' in xname or rec['loss_col'] is None):
            rec['task'] = 'cifar10'
        elif rec['loss_col'] is not None:
            rec['task'] = 'func'
        elif rec['err_col'] is not None:
            rec['task'] = 'cifar10'
        else:
            rec['task'] = 'unknown'

    return rec


def load_records(data_dir):
    """扫描数据目录下的全部 CSV 并解析。"""
    paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    records = []
    for path in paths:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print('[警告] 读取 %s 失败，已跳过：%s' % (name, exc))
            continue
        if df.empty:
            print('[提示] %s 没有数据行，已跳过。' % name)
            continue
        rec = analyse_file(path, df)
        rec['name'] = name
        records.append(rec)
    return records


# ----------------------------------------------------------------------------
# 指标提取与曲线聚合
# ----------------------------------------------------------------------------

def final_metric(rec):
    """取一个配置的最终指标：优先测试误差，其次损失。"""
    df = rec['df']
    for key in ('err_col', 'loss_col'):
        col = rec.get(key)
        if col is None or col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if not vals.empty:
            return float(vals.iloc[-1])
    return None


def aggregate_over_seeds(subset, xkey, ykey):
    """把同一方法下多个种子的曲线按 x 轴对齐，返回以 x 为索引、种子为列的 DataFrame。"""
    series = []
    for idx, rec in enumerate(subset):
        xcol = rec.get(xkey)
        ycol = rec.get(ykey)
        df = rec['df']
        if xcol is None or ycol is None:
            continue
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = pd.to_numeric(df[xcol], errors='coerce')
        y = pd.to_numeric(df[ycol], errors='coerce')
        mask = x.notna() & y.notna()
        if not mask.any():
            continue
        s = pd.Series(y[mask].values, index=x[mask].values.astype(float))
        s = s.groupby(level=0).mean()
        seed = rec.get('seed')
        s.name = ('seed%s' % seed) if seed is not None else ('run%d' % idx)
        series.append(s)
    if not series:
        return None
    out = pd.concat(series, axis=1, sort=False).sort_index()
    return out


def plot_method_curves(records, xkey, ykey, xlabel, ylabel, title, out_path,
                       method_order=None):
    """按方法分组，绘制种子均值曲线与 ±1 标准差带。"""
    subsets = {}
    for rec in records:
        if rec.get(xkey) is None or rec.get(ykey) is None:
            continue
        subsets.setdefault(rec.get('method') or 'unknown', []).append(rec)
    if not subsets:
        return False

    methods = sorted(subsets)
    if method_order:
        methods = [m for m in method_order if m in subsets] + \
                  [m for m in methods if m not in method_order]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plotted = 0
    for method in methods:
        agg = aggregate_over_seeds(subsets[method], xkey, ykey)
        if agg is None or agg.empty:
            continue
        mean = agg.mean(axis=1)
        if agg.shape[1] > 1:
            std = agg.std(axis=1, ddof=1).fillna(0.0)
        else:
            std = pd.Series(0.0, index=agg.index)
        ax.plot(mean.index.values, mean.values,
                label='%s (n=%d)' % (method, agg.shape[1]))
        ax.fill_between(mean.index.values,
                        (mean - std).values,
                        (mean + std).values,
                        alpha=0.22)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# 图 4：λ–α 网格热力图
# ----------------------------------------------------------------------------

def plot_lambda_alpha_grid(records, out_path):
    """按任务分别绘制 λ–α 网格上最终指标的均值热力图，并标注各自最优点。"""
    groups = {}
    for rec in records:
        lam = rec.get('lambda')
        alpha = rec.get('alpha')
        if lam is None or alpha is None:
            continue
        value = final_metric(rec)
        if value is None:
            continue
        groups.setdefault(rec.get('task') or 'all', []).append((lam, alpha, value))

    if not groups:
        return False

    tasks = sorted(groups)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.0 * len(tasks), 5.0),
                             squeeze=False)
    drawn = 0
    for ax, task in zip(axes[0], tasks):
        data = pd.DataFrame(groups[task], columns=['lambda', 'alpha', 'metric'])
        pivot = data.groupby(['lambda', 'alpha'])['metric'].mean().unstack('alpha')
        if pivot.empty:
            continue
        pivot = pivot.sort_index().sort_index(axis=1)
        matrix = np.ma.masked_invalid(pivot.values.astype(float))

        im = ax.imshow(matrix, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(['%g' % v for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(['%g' % v for v in pivot.index])
        ax.set_xlabel('alpha')
        ax.set_ylabel('lambda')
        ax.set_title('λ–α 网格（%s）' % task)

        if not np.isnan(pivot.values.astype(float)).all():
            flat = np.nanargmin(pivot.values.astype(float))
            i, j = np.unravel_index(flat, pivot.shape)
            ax.scatter([j], [i], marker='*', s=320, c='red',
                       edgecolors='white', linewidths=1.0, zorder=5)
            ax.annotate('%.4g' % pivot.values[i, j], (j, i),
                        textcoords='offset points', xytext=(10, 10),
                        color='white', fontsize=9, zorder=6)
        fig.colorbar(im, ax=ax, label='最终指标均值')
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return False

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# 图 5：不同训练预算下最优 λ 的移动趋势
# ----------------------------------------------------------------------------

def plot_budget_trend(records, out_path):
    """在有多个训练预算档位时，绘制最优 λ 随预算的变化。"""
    rows = []
    for rec in records:
        lam = rec.get('lambda')
        budget = rec.get('budget')
        if lam is None or budget is None:
            continue
        value = final_metric(rec)
        if value is None:
            continue
        rows.append((rec.get('task') or 'all',
                     rec.get('method') or 'unknown',
                     budget, lam, value))

    if not rows:
        return False

    data = pd.DataFrame(rows, columns=['task', 'method', 'budget', 'lambda', 'metric'])
    if data['budget'].nunique() < 2:
        return False

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    drawn = 0
    for (task, method), grp in data.groupby(['task', 'method']):
        agg = grp.groupby(['budget', 'lambda'])['metric'].mean().reset_index()
        xs, ys = [], []
        for budget, sub in agg.groupby('budget'):
            sub = sub.sort_values('metric')
            if sub.empty:
                continue
            xs.append(budget)
            ys.append(sub.iloc[0]['lambda'])
        if len(xs) < 2:
            continue
        order = np.argsort(xs)
        xs = np.array(xs, dtype=float)[order]
        ys = np.array(ys, dtype=float)[order]
        ax.plot(xs, ys, marker='o', label='%s / %s' % (task, method))
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return False

    ax.set_xlabel('训练预算')
    ax.set_ylabel('最优 λ')
    ax.set_title('图 5 不同训练预算下最优 λ 的移动趋势')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# 表 1 与表 2
# ----------------------------------------------------------------------------

def build_table1(records, out_path):
    """各配置最终指标（测试误差/损失）的均值与标准差汇总表。"""
    rows = []
    for rec in records:
        value = final_metric(rec)
        if value is None:
            continue
        rows.append({
            'task': rec.get('task') or 'all',
            'method': rec.get('method') or 'unknown',
            'lambda': rec.get('lambda'),
            'alpha': rec.get('alpha'),
            'seed': rec.get('seed'),
            'final_metric': value,
            'source_file': rec.get('name', ''),
        })

    if not rows:
        return False

    data = pd.DataFrame(rows)
    grouped = data.groupby(['task', 'method', 'lambda', 'alpha'], dropna=False)['final_metric']
    table = grouped.agg(['count', 'mean', 'std']).reset_index()
    table = table.rename(columns={
        'count': 'n_seeds',
        'mean': 'final_metric_mean',
        'std': 'final_metric_std',
    })
    table['final_metric_std'] = table['final_metric_std'].fillna(0.0)
    table = table.sort_values(['task', 'method', 'lambda', 'alpha'],
                              na_position='last')
    table.to_csv(out_path, index=False, encoding='utf-8-sig')
    return True


def build_table2(records, out_path):
    """跨框架（PyTorch 与 Keras 3）一致性对照汇总表。"""
    rows = []
    for rec in records:
        framework = rec.get('framework')
        if framework is None:
            continue
        value = final_metric(rec)
        if value is None:
            continue
        rows.append({
            'task': rec.get('task') or 'all',
            'method': rec.get('method') or 'unknown',
            'lambda': rec.get('lambda'),
            'alpha': rec.get('alpha'),
            'framework': framework,
            'final_metric': value,
        })

    if not rows:
        return False

    data = pd.DataFrame(rows)
    frameworks = sorted(data['framework'].dropna().unique().tolist())
    if len(frameworks) < 2:
        return False

    agg = data.groupby(['task', 'method', 'lambda', 'alpha', 'framework'],
                       dropna=False)['final_metric'].mean().reset_index()
    pivot = agg.pivot_table(index=['task', 'method', 'lambda', 'alpha'],
                            columns='framework', values='final_metric').reset_index()

    cols = [c for c in pivot.columns if c not in ('task', 'method', 'lambda', 'alpha')]
    if len(cols) < 2:
        return False

    left, right = cols[0], cols[1]
    pivot['abs_diff'] = (pivot[left] - pivot[right]).abs()
    denom = pivot[[left, right]].abs().max(axis=1).replace(0, np.nan)
    pivot['rel_diff'] = pivot['abs_diff'] / denom
    pivot = pivot.rename(columns={left: '%s_mean' % left,
                                  right: '%s_mean' % right})
    pivot.to_csv(out_path, index=False, encoding='utf-8-sig')
    return True


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main(argv=None):
    data_default, fig_default = default_dirs()

    parser = argparse.ArgumentParser(description='由实验数据 CSV 生成图表')
    parser.add_argument('--data-dir', default=data_default,
                        help='数据目录（默认：脚本上一级目录的 data）')
    parser.add_argument('--fig-dir', default=fig_default,
                        help='图片输出目录（默认：脚本上一级目录的 figures）')
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    fig_dir = os.path.abspath(os.path.expanduser(args.fig_dir))

    print('数据目录：%s' % data_dir)
    print('输出目录：%s' % fig_dir)

    try:
        os.makedirs(fig_dir, exist_ok=True)
    except OSError as exc:
        print('[提示] 无法创建输出目录 %s：%s' % (fig_dir, exc))
        return 0

    if not os.path.isdir(data_dir):
        print('[提示] 数据目录不存在，未生成任何图表。')
        return 0

    records = load_records(data_dir)
    if not records:
        print('[提示] 数据目录中没有可用的 CSV 文件，未生成任何图表。')
        return 0
    print('共读入 %d 个 CSV 文件。' % len(records))

    # ---------------- 图 1 与图 2：函数级 ----------------
    func_records = [r for r in records if r.get('task') == 'func']
    if not func_records:
        print('[提示] 未找到函数级数据，跳过图 1（函数级损失曲线）。')
        print('[提示] 未找到函数级数据，跳过图 2（函数级权重 ℓ2 范数曲线）。')
    else:
        path1 = os.path.join(fig_dir, 'fig1_func_loss.png')
        if plot_method_curves(func_records, 'xcol', 'loss_col',
                              '迭代次数', '损失',
                              '图 1 函数级三种方法的损失曲线',
                              path1, METHOD_ORDER):
            print('[已生成] 图 1：%s' % path1)
        else:
            print('[提示] 函数级数据缺少损失列，跳过图 1。')

        path2 = os.path.join(fig_dir, 'fig2_func_weight_l2.png')
        if plot_method_curves(func_records, 'xcol', 'l2_col',
                              '迭代次数', '权重 ℓ2 范数',
                              '图 2 函数级权重 ℓ2 范数随迭代变化',
                              path2, METHOD_ORDER):
            print('[已生成] 图 2：%s' % path2)
        else:
            print('[提示] 函数级数据缺少权重 ℓ2 范数列，跳过图 2。')

    # ---------------- 图 3：CIFAR-10 ----------------
    cifar_records = [r for r in records if r.get('task') == 'cifar10']
    cifar_selected = [r for r in cifar_records
                      if r.get('method') in ('adam_l2', 'adamw')]
    if not cifar_records:
        print('[提示] 未找到 CIFAR-10 数据，跳过图 3。')
    elif not cifar_selected:
        print('[提示] CIFAR-10 数据中没有 adam_l2 / adamw 记录，跳过图 3。')
    else:
        path3 = os.path.join(fig_dir, 'fig3_cifar_test_error.png')
        if plot_method_curves(cifar_selected, 'xcol', 'err_col',
                              'Epoch', '测试误差',
                              '图 3 CIFAR-10 测试误差随 epoch 曲线',
                              path3, ['adam_l2', 'adamw']):
            print('[已生成] 图 3：%s' % path3)
        else:
            print('[提示] CIFAR-10 数据缺少测试误差列或 epoch 列，跳过图 3。')

    # ---------------- 图 4：λ–α 网格 ----------------
    grid_records = [r for r in records
                    if r.get('lambda') is not None and r.get('alpha') is not None]
    if not grid_records:
        print('[提示] 数据中缺少 λ 或 α 标记，跳过图 4。')
    else:
        path4 = os.path.join(fig_dir, 'fig4_lambda_alpha_heatmap.png')
        if plot_lambda_alpha_grid(grid_records, path4):
            print('[已生成] 图 4：%s' % path4)
        else:
            print('[提示] 无法从数据中计算最终指标，跳过图 4。')

    # ---------------- 图 5：最优 λ 随预算移动 ----------------
    budget_records = [r for r in records
                      if r.get('budget') is not None and r.get('lambda') is not None]
    if not budget_records:
        print('[提示] 数据中没有多训练预算标记，跳过图 5。')
    else:
        path5 = os.path.join(fig_dir, 'fig5_best_lambda_vs_budget.png')
        if plot_budget_trend(budget_records, path5):
            print('[已生成] 图 5：%s' % path5)
        else:
            print('[提示] 可用训练预算档位不足两档，跳过图 5。')

    # ---------------- 表 1 ----------------
    table1_path = os.path.join(fig_dir, 'table1_final_metric_summary.csv')
    if build_table1(records, table1_path):
        print('[已生成] 表 1：%s' % table1_path)
    else:
        print('[提示] 没有可汇总的最终指标，跳过表 1。')

    # ---------------- 表 2 ----------------
    table2_path = os.path.join(fig_dir, 'table2_framework_consistency.csv')
    if build_table2(records, table2_path):
        print('[已生成] 表 2：%s' % table2_path)
    else:
        print('[提示] 数据中没有跨框架（PyTorch / Keras 3）对照记录，跳过表 2。')

    print('完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
# end of plot_results.py