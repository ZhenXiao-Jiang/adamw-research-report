# 本脚本使用示意数据绘制 SVG 图并生成汇总 CSV，仅用于流程演示，不用于研究结论。
import csv
import glob
import math
import os
import re
import xml.etree.ElementTree as ET

WIDTH = 800
HEIGHT = 500
SVG_NS = "http://www.w3.org/2000/svg"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")

COLORS = {
    "adamw": "#1f77b4",
    "adam_l2": "#d62728",
    "adam_no_decay": "#2ca02c",
    "sgd": "#9467bd",
}


def _norm_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _fmt_num(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.6g}"
    return str(value)


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in ("nan", "none", "null", "na", "n/a", "inf", "-inf", "+inf"):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def get_field(row, candidates):
    lookup = {}
    for key, value in row.items():
        if key is None:
            continue
        lookup[_norm_name(key)] = value
    for candidate in candidates:
        key = _norm_name(candidate)
        if key in lookup:
            value = lookup[key]
            if value is not None and str(value).strip() != "":
                return value
    return None


def read_csv_rows(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
        except OSError:
            return []
    return []


def _from_filename(path, names):
    base = os.path.basename(path).lower()
    for name in names:
        pattern = r"(?:^|[^a-z0-9])" + re.escape(name) + r"[^a-z0-9]*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)"
        match = re.search(pattern, base)
        if match:
            return parse_float(match.group(1))
    return None


def _method_from_text(text):
    value = _norm_name(text)
    if "adamw" in value:
        return "adamw"
    if "no_decay" in value or "nodecay" in value:
        return "adam_no_decay"
    if "adam_l2" in value or ("l2" in value and "adam" in value):
        return "adam_l2"
    if "adam" in value:
        return "adam"
    if "sgd" in value:
        return "sgd"
    return ""


def parse_method(row, path):
    value = get_field(row, ["method", "optimizer", "method_name", "algorithm"])
    if value is not None and str(value).strip() != "":
        method = _method_from_text(value)
        return method if method else _norm_name(value)
    method = _method_from_text(os.path.basename(path))
    return method if method else "unknown"


def parse_task(row, path):
    value = get_field(row, ["task", "dataset", "data", "task_name"])
    if value is not None and str(value).strip() != "":
        return _norm_name(value)
    lower = path.lower()
    base = os.path.basename(lower)
    if "cifar" in lower or "cifar" in base:
        return "cifar10"
    if "func" in lower or "function" in base:
        return "func"
    if "synthetic" in base:
        return "synthetic"
    return "unknown"


def parse_lambda(row, path):
    value = parse_float(get_field(row, ["lambda", "lam", "lambda_value", "l2_lambda"]))
    if value is None:
        value = _from_filename(path, ["lambda", "lam"])
    return value


def parse_alpha(row, path):
    value = parse_float(get_field(row, ["alpha", "alpha_value"]))
    if value is None:
        value = _from_filename(path, ["alpha"])
    return value


def parse_budget(row, path):
    value = parse_float(get_field(row, ["budget", "compute_budget"]))
    if value is None:
        value = _from_filename(path, ["budget"])
    return value


def _add_text(parent, x, y, text, anchor="start", size=12, fill="#111111", transform=None):
    attrs = {
        "x": f"{float(x):.2f}",
        "y": f"{float(y):.2f}",
        "text-anchor": anchor,
        "font-family": "Arial, sans-serif",
        "font-size": str(size),
        "fill": fill,
    }
    if transform is not None:
        attrs["transform"] = transform
    node = ET.SubElement(parent, "text", attrs)
    node.text = str(text)


def _add_line(parent, x1, y1, x2, y2, stroke="#333333", width=1.0):
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{float(x1):.2f}",
            "y1": f"{float(y1):.2f}",
            "x2": f"{float(x2):.2f}",
            "y2": f"{float(y2):.2f}",
            "stroke": stroke,
            "stroke-width": str(width),
        },
    )


def _new_svg(title):
    root = ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "width": str(WIDTH),
            "height": str(HEIGHT),
            "viewBox": f"0 0 {WIDTH} {HEIGHT}",
        },
    )
    ET.SubElement(
        root,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": str(WIDTH),
            "height": str(HEIGHT),
            "fill": "#ffffff",
        },
    )
    _add_text(root, WIDTH / 2.0, 30, title, anchor="middle", size=18)
    return root


def _save_svg(root, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _scale(value, data_min, data_max, pixel_min, pixel_max):
    if data_max == data_min:
        return (pixel_min + pixel_max) / 2.0
    return pixel_min + (value - data_min) * (pixel_max - pixel_min) / (data_max - data_min)


def _pad_range(vmin, vmax):
    if vmin == vmax:
        if vmin == 0:
            return -1.0, 1.0
        pad = abs(vmin) * 0.1
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.05
    return vmin - pad, vmax + pad


def _fmt_tick(value):
    if abs(value) < 1e-12:
        value = 0.0
    absolute = abs(value)
    if absolute >= 1000 or (0 < absolute < 0.001):
        return f"{value:.3g}"
    if absolute >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if absolute >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def write_line_chart_svg(path, title, series, x_label, y_label, annotate_points=False):
    usable = []
    all_points = []
    for name, color, points in series:
        clean_points = []
        for x_value, y_value in points:
            if x_value is None or y_value is None:
                continue
            if not (math.isfinite(x_value) and math.isfinite(y_value)):
                continue
            clean_points.append((float(x_value), float(y_value)))
        if clean_points:
            clean_points.sort(key=lambda item: item[0])
            usable.append((name, color, clean_points))
            all_points.extend(clean_points)
    if not all_points:
        return False

    xmin = min(point[0] for point in all_points)
    xmax = max(point[0] for point in all_points)
    ymin = min(point[1] for point in all_points)
    ymax = max(point[1] for point in all_points)
    xmin, xmax = _pad_range(xmin, xmax)
    ymin, ymax = _pad_range(ymin, ymax)

    left = 80.0
    right = 760.0
    top = 70.0
    bottom = 440.0

    root = _new_svg(title)
    _add_line(root, left, bottom, right, bottom, "#333333", 1.2)
    _add_line(root, left, top, left, bottom, "#333333", 1.2)

    for index in range(6):
        value = xmin + (xmax - xmin) * index / 5.0
        x_pixel = _scale(value, xmin, xmax, left, right)
        _add_line(root, x_pixel, bottom, x_pixel, bottom + 5, "#333333", 1.0)
        _add_text(root, x_pixel, bottom + 20, _fmt_tick(value), anchor="middle", size=11)

    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5.0
        y_pixel = _scale(value, ymin, ymax, bottom, top)
        _add_line(root, left - 5, y_pixel, left, y_pixel, "#333333", 1.0)
        _add_text(root, left - 10, y_pixel + 4, _fmt_tick(value), anchor="end", size=11)

    _add_text(root, (left + right) / 2.0, 485, x_label, anchor="middle", size=14)
    _add_text(
        root,
        25,
        (top + bottom) / 2.0,
        y_label,
        anchor="middle",
        size=14,
        transform=f"rotate(-90 25 {(top + bottom) / 2.0:.2f})",
    )

    for index, (name, color, _) in enumerate(usable):
        legend_y = 85 + index * 18
        _add_line(root, 600, legend_y - 5, 630, legend_y - 5, color, 2.0)
        _add_text(root, 636, legend_y, name, size=11)

    for name, color, points in usable:
        mapped = []
        for x_value, y_value in points:
            x_pixel = _scale(x_value, xmin, xmax, left, right)
            y_pixel = _scale(y_value, ymin, ymax, bottom, top)
            mapped.append((x_pixel, y_pixel))
        point_text = " ".join(f"{x_pixel:.2f},{y_pixel:.2f}" for x_pixel, y_pixel in mapped)
        ET.SubElement(
            root,
            "polyline",
            {
                "points": point_text,
                "fill": "none",
                "stroke": color,
                "stroke-width": "2",
                "stroke-linejoin": "round",
                "stroke-linecap": "round",
            },
        )
        if annotate_points:
            for (_, y_value), (x_pixel, y_pixel) in zip(points, mapped):
                _add_text(root, x_pixel + 6, y_pixel - 6, f"{y_value:.4f}", size=11)

    _save_svg(root, path)
    return True


def _heat_color(value, vmin, vmax):
    if value is None:
        return "#e6e6e6"
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = (value - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    red = int(round(30 + 225 * ratio))
    green = int(round(80 * (1.0 - ratio)))
    blue = int(round(200 * (1.0 - ratio)))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _heat_text_color(value, vmin, vmax):
    if value is None:
        return "#333333"
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = (value - vmin) / (vmax - vmin)
    if ratio < 0.25 or ratio > 0.85:
        return "#ffffff"
    return "#111111"


def write_heatmap_svg(path, title, row_values, col_values, cell_values, row_label, col_label):
    if not row_values or not col_values:
        return False
    finite_values = []
    for value in cell_values.values():
        if value is not None and math.isfinite(value):
            finite_values.append(value)
    if not finite_values:
        return False

    vmin = min(finite_values)
    vmax = max(finite_values)

    left = 140.0
    right = 700.0
    top = 100.0
    bottom = 400.0
    row_count = len(row_values)
    col_count = len(col_values)
    cell_width = (right - left) / col_count
    cell_height = (bottom - top) / row_count

    root = _new_svg(title)
    _add_line(root, left - 20, top, left - 20, bottom, "#333333", 1.2)
    _add_line(root, left, bottom, right, bottom, "#333333", 1.2)

    for row_index, row_value in enumerate(row_values):
        y_pixel = top + row_index * cell_height
        for col_index, col_value in enumerate(col_values):
            x_pixel = left + col_index * cell_width
            value = cell_values.get((row_index, col_index))
            fill = _heat_color(value, vmin, vmax)
            ET.SubElement(
                root,
                "rect",
                {
                    "x": f"{x_pixel:.2f}",
                    "y": f"{y_pixel:.2f}",
                    "width": f"{cell_width:.2f}",
                    "height": f"{cell_height:.2f}",
                    "fill": fill,
                    "stroke": "#ffffff",
                    "stroke-width": "1",
                },
            )
            label = "NA" if value is None else f"{value:.4f}"
            _add_text(
                root,
                x_pixel + cell_width / 2.0,
                y_pixel + cell_height / 2.0 + 5,
                label,
                anchor="middle",
                size=12,
                fill=_heat_text_color(value, vmin, vmax),
            )
        _add_text(
            root,
            left - 12,
            y_pixel + cell_height / 2.0 + 5,
            f"{row_label}={_fmt_num(row_value)}",
            anchor="end",
            size=12,
        )

    for col_index, col_value in enumerate(col_values):
        x_pixel = left + col_index * cell_width
        _add_text(
            root,
            x_pixel + cell_width / 2.0,
            top - 18,
            f"{col_label}={_fmt_num(col_value)}",
            anchor="middle",
            size=12,
        )

    _add_text(
        root,
        25,
        (top + bottom) / 2.0,
        row_label,
        anchor="middle",
        size=14,
        transform=f"rotate(-90 25 {(top + bottom) / 2.0:.2f})",
    )
    _add_text(root, (left + right) / 2.0, bottom + 45, col_label, anchor="middle", size=14)
    _save_svg(root, path)
    return True


def aggregate_curve(directory, allowed_methods, x_fields, y_fields):
    sums = {}
    for path in glob.glob(os.path.join(directory, "*.csv")):
        rows = read_csv_rows(path)
        for row in rows:
            method = parse_method(row, path)
            if method not in allowed_methods:
                continue
            x_value = parse_float(get_field(row, x_fields))
            y_value = parse_float(get_field(row, y_fields))
            if x_value is None or y_value is None:
                continue
            method_data = sums.setdefault(method, {})
            if x_value not in method_data:
                method_data[x_value] = [0.0, 0]
            method_data[x_value][0] += y_value
            method_data[x_value][1] += 1

    series = []
    for method in allowed_methods:
        method_data = sums.get(method)
        if not method_data:
            continue
        points = []
        for x_value in sorted(method_data):
            total, count = method_data[x_value]
            if count > 0:
                points.append((x_value, total / count))
        if points:
            series.append((method, COLORS.get(method, "#333333"), points))
    return series


def _std_sample(values):
    count = len(values)
    if count <= 1:
        return 0.0
    mean_value = sum(values) / count
    variance = sum((value - mean_value) ** 2 for value in values) / (count - 1)
    return math.sqrt(variance)


def write_table_csv(path, header, rows):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _record_from_row(row, path, final_loss):
    return {
        "task": parse_task(row, path),
        "method": parse_method(row, path),
        "lambda": parse_lambda(row, path),
        "alpha": parse_alpha(row, path),
        "final_loss": final_loss,
    }


def extract_final_records(path):
    rows = read_csv_rows(path)
    if not rows:
        return []

    records = []
    has_explicit_final = False
    for row in rows:
        value = parse_float(get_field(row, ["final_loss", "final_metric"]))
        if value is not None:
            has_explicit_final = True
            break

    if has_explicit_final:
        for row in rows:
            final_loss = parse_float(get_field(row, ["final_loss", "final_metric"]))
            if final_loss is None:
                continue
            records.append(_record_from_row(row, path, final_loss))
        return records

    groups = {}
    for index, row in enumerate(rows):
        loss = parse_float(get_field(row, ["loss", "train_loss", "test_error", "testerror", "test_loss"]))
        if loss is None:
            continue
        task = parse_task(row, path)
        method = parse_method(row, path)
        lambda_value = parse_lambda(row, path)
        alpha_value = parse_alpha(row, path)
        x_value = parse_float(get_field(row, ["step", "epoch", "iteration", "iter"]))
        if x_value is None:
            x_value = float(index)
        key = (task, method, lambda_value, alpha_value)
        if key not in groups or x_value >= groups[key]["x"]:
            groups[key] = {
                "x": x_value,
                "task": task,
                "method": method,
                "lambda": lambda_value,
                "alpha": alpha_value,
                "final_loss": loss,
            }

    for group in groups.values():
        records.append(
            {
                "task": group["task"],
                "method": group["method"],
                "lambda": group["lambda"],
                "alpha": group["alpha"],
                "final_loss": group["final_loss"],
            }
        )
    return records


def _append_generated(path, generated):
    print(path)
    generated.append(path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    generated = []

    summary_path = os.path.join(DATA_DIR, "synthetic_summary_schematic.csv")

    func_dir = os.path.join(DATA_DIR, "func")
    if not os.path.isdir(func_dir):
        print(f"[跳过] 未找到目录: {func_dir}")
    else:
        series = aggregate_curve(
            func_dir,
            ["adamw", "adam_l2", "adam_no_decay"],
            ["step", "iteration", "iter"],
            ["loss", "train_loss"],
        )
        if not series:
            print("[跳过] fig1: data/func 中没有可用的 step/loss 数据")
        else:
            output = os.path.join(FIG_DIR, "fig1_func_loss_schematic.svg")
            if write_line_chart_svg(
                output,
                "图1 函数训练损失（示意数据）",
                series,
                "step",
                "loss",
            ):
                _append_generated(output, generated)
            else:
                print("[跳过] fig1: SVG 生成失败")

        series = aggregate_curve(
            func_dir,
            ["adamw", "adam_l2", "adam_no_decay"],
            ["step", "iteration", "iter"],
            ["weight_l2", "weight_l2_norm", "weight_norm", "l2_weight"],
        )
        if not series:
            print("[跳过] fig2: data/func 中没有可用的 step/weight_l2 数据")
        else:
            output = os.path.join(FIG_DIR, "fig2_func_weight_l2_schematic.svg")
            if write_line_chart_svg(
                output,
                "图2 函数训练权重 L2（示意数据）",
                series,
                "step",
                "weight_l2",
            ):
                _append_generated(output, generated)
            else:
                print("[跳过] fig2: SVG 生成失败")

    cifar_dir = os.path.join(DATA_DIR, "cifar10")
    if not os.path.isdir(cifar_dir):
        print(f"[跳过] 未找到目录: {cifar_dir}")
    else:
        series = aggregate_curve(
            cifar_dir,
            ["adamw", "adam_l2"],
            ["epoch", "step"],
            ["test_error", "testerror", "test_err", "error", "test_loss"],
        )
        if not series:
            print("[跳过] fig3: data/cifar10 中没有可用的 epoch/测试误差数据")
        else:
            output = os.path.join(FIG_DIR, "fig3_cifar_test_error_schematic.svg")
            if write_line_chart_svg(
                output,
                "图3 CIFAR-10 测试误差（示意数据）",
                series,
                "epoch",
                "test error",
            ):
                _append_generated(output, generated)
            else:
                print("[跳过] fig3: SVG 生成失败")

    if not os.path.isfile(summary_path):
        print(f"[跳过] 未找到文件: {summary_path}")
        summary_rows = []
    else:
        summary_rows = read_csv_rows(summary_path)

    if not summary_rows:
        print("[跳过] fig4: 汇总 CSV 为空或不可读")
    else:
        sums = {}
        for row in summary_rows:
            lambda_value = parse_lambda(row, summary_path)
            alpha_value = parse_alpha(row, summary_path)
            final_loss = parse_float(get_field(row, ["final_loss", "final_metric", "loss"]))
            if lambda_value is None or alpha_value is None or final_loss is None:
                continue
            key = (lambda_value, alpha_value)
            sums.setdefault(key, []).append(final_loss)

        if not sums:
            print("[跳过] fig4: 汇总 CSV 中没有可用的 lambda/alpha/final_loss 数据")
        else:
            means = {key: sum(values) / len(values) for key, values in sums.items()}
            lambda_values = sorted({key[0] for key in sums})
            alpha_values = sorted({key[1] for key in sums})

            if len(alpha_values) == 3 and len(lambda_values) == 2:
                row_values = lambda_values
                col_values = alpha_values
                row_label = "lambda"
                col_label = "alpha"
                orientation = "lambda_rows"
            elif len(lambda_values) == 3 and len(alpha_values) == 2:
                row_values = alpha_values
                col_values = lambda_values
                row_label = "alpha"
                col_label = "lambda"
                orientation = "alpha_rows"
            else:
                row_values = lambda_values
                col_values = alpha_values
                row_label = "lambda"
                col_label = "alpha"
                orientation = "lambda_rows"

            cell_values = {}
            for (lambda_value, alpha_value), mean_value in means.items():
                if orientation == "lambda_rows":
                    row_index = row_values.index(lambda_value)
                    col_index = col_values.index(alpha_value)
                else:
                    row_index = row_values.index(alpha_value)
                    col_index = col_values.index(lambda_value)
                cell_values[(row_index, col_index)] = mean_value

            output = os.path.join(FIG_DIR, "fig4_lambda_alpha_heatmap_schematic.svg")
            if write_heatmap_svg(
                output,
                "图4 lambda-alpha 平均 final_loss 热图（示意数据）",
                row_values,
                col_values,
                cell_values,
                row_label,
                col_label,
            ):
                _append_generated(output, generated)
            else:
                print("[跳过] fig4: SVG 生成失败")

    if not summary_rows:
        print("[跳过] fig5: 汇总 CSV 为空或不可读")
    else:
        sums = {}
        for row in summary_rows:
            budget = parse_budget(row, summary_path)
            lambda_value = parse_lambda(row, summary_path)
            final_loss = parse_float(get_field(row, ["final_loss", "final_metric", "loss"]))
            if budget is None or lambda_value is None or final_loss is None:
                continue
            key = (budget, lambda_value)
            sums.setdefault(key, []).append(final_loss)

        if not sums:
            print("[跳过] fig5: 汇总 CSV 中没有可用的 budget/lambda/final_loss 数据")
        else:
            means = {key: sum(values) / len(values) for key, values in sums.items()}
            best_by_budget = {}
            for (budget, lambda_value), mean_value in means.items():
                if budget not in best_by_budget or mean_value < best_by_budget[budget][1]:
                    best_by_budget[budget] = (lambda_value, mean_value)

            points = []
            for budget in sorted(best_by_budget):
                lambda_value, _ = best_by_budget[budget]
                points.append((budget, lambda_value))

            if not points:
                print("[跳过] fig5: 无法确定最佳 lambda")
            else:
                series = [("best lambda", "#9467bd", points)]
                output = os.path.join(FIG_DIR, "fig5_best_lambda_vs_budget_schematic.svg")
                if write_line_chart_svg(
                    output,
                    "图5 最佳 lambda 随 budget 变化（示意数据）",
                    series,
                    "budget",
                    "best lambda",
                    annotate_points=True,
                ):
                    _append_generated(output, generated)
                else:
                    print("[跳过] fig5: SVG 生成失败")

    table1_records = []
    for path in glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True):
        table1_records.extend(extract_final_records(path))

    if not table1_records:
        print("[跳过] table1: 没有可汇总的 final_loss 记录")
    else:
        groups = {}
        for record in table1_records:
            key = (record["task"], record["method"], record["lambda"], record["alpha"])
            groups.setdefault(key, []).append(record["final_loss"])

        rows_out = []
        for key, values in sorted(
            groups.items(),
            key=lambda item: (
                str(item[0][0]),
                str(item[0][1]),
                str(item[0][2]),
                str(item[0][3]),
            ),
        ):
            task, method, lambda_value, alpha_value = key
            count = len(values)
            mean_loss = sum(values) / count
            std_loss = _std_sample(values)
            rows_out.append(
                [
                    task,
                    method,
                    _fmt_num(lambda_value),
                    _fmt_num(alpha_value),
                    count,
                    f"{mean_loss:.6f}",
                    f"{std_loss:.6f}",
                ]
            )

        output = os.path.join(FIG_DIR, "table1_final_metric_summary_schematic.csv")
        write_table_csv(
            output,
            ["task", "method", "lambda", "alpha", "count", "mean_final_loss", "std_final_loss"],
            rows_out,
        )
        _append_generated(output, generated)

    framework_path = os.path.join(DATA_DIR, "framework_consistency_schematic.csv")
    if not os.path.isfile(framework_path):
        print(f"[跳过] table2: 未找到文件: {framework_path}")
    else:
        framework_rows = read_csv_rows(framework_path)
        if not framework_rows:
            print("[跳过] table2: framework 一致性 CSV 为空或不可读")
        else:
            groups = {}
            for row in framework_rows:
                framework_value = get_field(row, ["framework", "backend"])
                if framework_value is None or str(framework_value).strip() == "":
                    framework_value = "unknown"
                method = parse_method(row, framework_path)
                lambda_value = parse_lambda(row, framework_path)
                alpha_value = parse_alpha(row, framework_path)
                final_loss = parse_float(get_field(row, ["final_loss", "final_metric", "loss"]))
                if final_loss is None:
                    continue
                key = (str(framework_value).strip(), method, lambda_value, alpha_value)
                groups.setdefault(key, []).append(final_loss)

            if not groups:
                print("[跳过] table2: 没有可汇总的 framework 记录")
            else:
                rows_out = []
                for key, values in sorted(
                    groups.items(),
                    key=lambda item: (
                        str(item[0][0]),
                        str(item[0][1]),
                        str(item[0][2]),
                        str(item[0][3]),
                    ),
                ):
                    framework_value, method, lambda_value, alpha_value = key
                    count = len(values)
                    mean_loss = sum(values) / count
                    rows_out.append(
                        [
                            framework_value,
                            method,
                            _fmt_num(lambda_value),
                            _fmt_num(alpha_value),
                            f"{mean_loss:.6f}",
                            count,
                        ]
                    )

                output = os.path.join(FIG_DIR, "table2_framework_consistency_schematic.csv")
                write_table_csv(
                    output,
                    ["framework", "method", "lambda", "alpha", "mean_final_loss", "count"],
                    rows_out,
                )
                _append_generated(output, generated)

    print(f"生成文件总数: {len(generated)}")


if __name__ == "__main__":
    main()