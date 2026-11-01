# 本脚本生成的示意实验数据仅用于流程演示，不用于任何研究结论。
import csv
import math
import os
import random

# 二次目标尺度：保持损失量级在 1e-3 以上，避免数值下溢到 0
A = 0.5
B = 0.2
THETA0_INIT = 1.0
THETA1_INIT = 0.8
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

# budget 标签保持原两档，但内部实际迭代步数缩小，按固定步数停止
BUDGET_STEPS = {2000: 200, 5000: 400}


def loss_value(theta):
    return 0.5 * (A * theta[0] * theta[0] + B * theta[1] * theta[1])


def grad_value(theta):
    return [A * theta[0], B * theta[1]]


def weight_l2_value(theta):
    return math.sqrt(theta[0] * theta[0] + theta[1] * theta[1])


def compute_state(theta, method, wd, lr, step):
    loss = loss_value(theta)
    grad = grad_value(theta)
    w_l2 = weight_l2_value(theta)
    if method == 'adam_no_decay':
        loss_total = loss
    else:
        loss_total = loss + 0.5 * wd * (w_l2 * w_l2)
    grad_norm = math.sqrt(grad[0] * grad[0] + grad[1] * grad[1])
    return [step, lr, loss, loss_total, grad_norm, w_l2]


def run_function_experiment(method, lam, alpha, seed, budget):
    wd = lam if method != 'adam_no_decay' else 0.0
    theta = [THETA0_INIT, THETA1_INIT]
    m = [0.0, 0.0]
    v = [0.0, 0.0]
    steps = BUDGET_STEPS.get(budget, budget)
    rows = []
    rows.append(compute_state(theta, method, wd, alpha, 0))
    min_loss = rows[0][2]

    for step in range(1, steps + 1):
        grad = grad_value(theta)
        if method == 'adam_l2':
            grad = [grad[0] + wd * theta[0], grad[1] + wd * theta[1]]

        m = [
            BETA1 * m[0] + (1.0 - BETA1) * grad[0],
            BETA1 * m[1] + (1.0 - BETA1) * grad[1],
        ]
        v = [
            BETA2 * v[0] + (1.0 - BETA2) * grad[0] * grad[0],
            BETA2 * v[1] + (1.0 - BETA2) * grad[1] * grad[1],
        ]

        bc1 = 1.0 - BETA1 ** step
        bc2 = 1.0 - BETA2 ** step
        for i in range(2):
            m_hat = m[i] / bc1
            v_hat = v[i] / bc2
            update = alpha * m_hat / (math.sqrt(v_hat) + EPS)
            if method == 'adamw':
                update += alpha * wd * theta[i]
            theta[i] -= update

        if step % 10 == 0 or step == steps:
            row = compute_state(theta, method, wd, alpha, step)
            rows.append(row)
            if row[2] < min_loss:
                min_loss = row[2]

    return rows, min_loss


def main():
    random.seed(20261101)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    func_dir = os.path.join(data_dir, 'func')
    cifar_dir = os.path.join(data_dir, 'cifar10')
    os.makedirs(func_dir, exist_ok=True)
    os.makedirs(cifar_dir, exist_ok=True)

    methods = ['adamw', 'adam_l2', 'adam_no_decay']
    lambdas = [0.01, 0.1, 0.5]
    alphas = [0.003, 0.01]
    seeds = [0, 1, 2]
    budgets = [2000, 5000]

    generated = []
    summary_rows = []
    result_store = {}
    func_header = ['step', 'lr', 'loss', 'loss_total', 'grad_norm', 'weight_l2']

    for method in methods:
        for lam in lambdas:
            for alpha in alphas:
                for seed in seeds:
                    for budget in budgets:
                        rows, min_loss = run_function_experiment(
                            method, lam, alpha, seed, budget
                        )
                        final = rows[-1]
                        fname = (
                            'pytorch_{}_lam{}_alpha{}_seed{}_budget{}_schematic.csv'.format(
                                method, lam, alpha, seed, budget
                            )
                        )
                        path = os.path.join(func_dir, fname)
                        with open(path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(func_header)
                            writer.writerows(rows)
                        generated.append(path)

                        summary_rows.append([
                            method, lam, alpha, seed, 'pytorch', budget,
                            final[0], final[0], alpha, final[2], final[3],
                            final[4], final[5], min_loss,
                        ])
                        result_store[(method, lam, alpha, seed, budget)] = final[2]

    summary_path = os.path.join(data_dir, 'synthetic_summary_schematic.csv')
    summary_header = [
        'method', 'lambda', 'alpha', 'seed', 'framework', 'budget',
        'steps', 'final_step', 'final_lr', 'final_loss',
        'final_loss_total', 'final_grad_norm', 'final_weight_l2', 'min_loss',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)
        writer.writerows(summary_rows)
    generated.append(summary_path)

    for seed in seeds:
        rng = random.Random(2026 + seed)
        noises = [rng.uniform(-0.002, 0.002) for _ in range(20)]
        train_noises = [rng.uniform(-0.005, 0.005) for _ in range(20)]
        weight_noises = [rng.uniform(-0.005, 0.005) for _ in range(20)]

        for method in ['adam_l2', 'adamw']:
            factor = 0.95 if method == 'adamw' else 1.0
            prev_error = 1.0
            rows = []
            for epoch in range(1, 21):
                trend = (
                    0.50 * math.exp(-0.13 * epoch) + 0.05 + noises[epoch - 1]
                ) * factor
                test_error = min(prev_error, max(0.005, trend))
                prev_error = test_error

                train_loss = (
                    0.9 * math.exp(-0.12 * epoch) + 0.08 + train_noises[epoch - 1]
                )
                if method == 'adamw':
                    train_loss *= 0.98
                train_loss = max(0.01, train_loss)

                weight_l2 = (
                    1.2 * math.exp(-0.06 * epoch) + 0.05 + weight_noises[epoch - 1]
                )
                rows.append([epoch, train_loss, test_error, weight_l2])

            fname = 'cifar10_{}_seed{}_schematic.csv'.format(method, seed)
            path = os.path.join(cifar_dir, fname)
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'test_error', 'weight_l2'])
                writer.writerows(rows)
            generated.append(path)

    fc_rows = []
    rng_fc = random.Random(777)
    for method in methods:
        for lam in lambdas:
            for alpha in alphas:
                seed = 0
                budget = 2000
                base_loss = result_store[(method, lam, alpha, seed, budget)]
                delta = rng_fc.uniform(0.003, 0.015) * rng_fc.choice([-1, 1])
                keras_loss = base_loss * (1.0 + delta)
                fc_rows.append(['pytorch', method, lam, alpha, seed, budget, base_loss])
                fc_rows.append(['keras3', method, lam, alpha, seed, budget, keras_loss])

    fc_path = os.path.join(data_dir, 'framework_consistency_schematic.csv')
    with open(fc_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['framework', 'method', 'lambda', 'alpha', 'seed', 'budget', 'final_loss'])
        writer.writerows(fc_rows)
    generated.append(fc_path)

    for path in generated:
        with open(path, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print('{}: {} 行'.format(path, n))
    print('总计生成文件数: {}'.format(len(generated)))


if __name__ == '__main__':
    main()