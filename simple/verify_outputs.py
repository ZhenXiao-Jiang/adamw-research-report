# 校验 simple 目录产出的 SVG 与 CSV，仅用标准库。
import csv
import os
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
DATA = os.path.join(BASE, "data")


def check_svg(name):
    path = os.path.join(FIG, name)
    try:
        tree = ET.parse(path)
    except Exception as exc:
        print("SVG_FAIL", name, exc)
        return
    root = tree.getroot()
    tags = [el.tag.split("}")[-1] for el in root.iter()]
    lines = sum(1 for t in tags if t in ("line", "polyline", "path"))
    axes = sum(1 for t in tags if t in ("line", "polyline"))
    print("SVG_OK", name, "size", os.path.getsize(path), "elements", len(tags), "lines", lines, "axes_candidates", axes)


def check_csv(name, folder, expect_header=None):
    path = os.path.join(folder, name)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0] if rows else []
    empties = 0
    widths = set()
    for row in rows[1:]:
        widths.add(len(row))
        empties += sum(1 for c in row if c.strip() == "")
    print("CSV", name, "rows", len(rows) - 1, "cols", len(header), "widths", sorted(widths), "empties", empties, "header", header)
    if expect_header:
        missing = [c for c in expect_header if c not in header]
        print("  missing_fields", missing)


for svg in ["fig1_func_loss_schematic.svg", "fig2_func_weight_l2_schematic.svg", "fig3_cifar_test_error_schematic.svg", "fig4_lambda_alpha_heatmap_schematic.svg", "fig5_best_lambda_vs_budget_schematic.svg"]:
    check_svg(svg)

check_csv("table1_final_metric_summary_schematic.csv", FIG, ["task", "method", "lambda", "alpha"])
check_csv("table2_framework_consistency_schematic.csv", FIG, ["framework", "method"])
check_csv("synthetic_summary_schematic.csv", DATA)
