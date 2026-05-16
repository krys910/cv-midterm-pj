import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser("Generate report-ready markdown from task1-3 outputs.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--output_md", type=str, default="results_for_report.md")
    return parser.parse_args()


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt_num(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (ValueError, TypeError):
        return str(value)


def collect_task1(project_root: Path) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    warnings: List[str] = []
    task1_root = project_root / "task1" / "outputs"
    summary_path = task1_root / "task1_full" / "summary.csv"
    summary_rows = read_csv_rows(summary_path)

    lines.append("## Task1 分类（ResNet 微调）")
    lines.append("")

    if summary_rows:
        lines.append("| run_dir | model | attention | init | best_val_acc | test_acc | test_loss |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in summary_rows:
            lines.append(
                f"| `{row.get('run_dir', '-')}` | {row.get('model', '-')} | {row.get('attention', '-')} | "
                f"{row.get('init', '-')} | {fmt_num(row.get('best_val_acc'))} | "
                f"{fmt_num(row.get('test_acc'))} | {fmt_num(row.get('test_loss'))} |"
            )
        lines.append("")
        best = max(summary_rows, key=lambda r: float(r.get("test_acc", -1)))
        lines.append(
            f"- 最优 `test_acc`：**{fmt_num(best.get('test_acc'))}**（`{best.get('run_dir', '-')}`）"
        )
        lines.append("")
    else:
        warnings.append("Task1 未找到 `task1/outputs/task1_full/summary.csv`。")
        baseline_metrics = read_json(task1_root / "baseline_resnet18" / "metrics.json")
        if baseline_metrics:
            lines.append("- 找到 baseline 指标：")
            lines.append(
                f"  - best_val_acc: {fmt_num(baseline_metrics.get('best_val_acc'))}, "
                f"test_acc: {fmt_num(baseline_metrics.get('test_acc'))}, "
                f"test_loss: {fmt_num(baseline_metrics.get('test_loss'))}"
            )
            lines.append("")
        else:
            lines.append("- 暂无可读的 Task1 指标文件。")
            lines.append("")

    lines.append("### Task1 报告可用曲线（wandb）")
    lines.append("- train/loss vs epoch")
    lines.append("- val/loss vs epoch")
    lines.append("- val/acc vs epoch")
    lines.append("")
    return lines, warnings


def collect_task2(project_root: Path) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    warnings: List[str] = []
    task2_candidates = [
        project_root / "outputs" / "task2_cloud",
        project_root / "outputs" / "task2",
        project_root / "runs" / "detect" / "outputs" / "task2",
    ]
    existing_candidates = [p for p in task2_candidates if p.exists()]
    if existing_candidates:
        def score_task2_dir(path: Path) -> Tuple[int, int]:
            run_dirs = [d for d in path.iterdir() if d.is_dir()]
            marker = 0
            for d in run_dirs:
                if (d / "train_args.json").exists() or (d / "args.yaml").exists():
                    marker += 1
                if (d / "final_metrics.json").exists():
                    marker += 2
                if (d / "tracking_summary.json").exists():
                    marker += 2
                if (d / "results.csv").exists():
                    marker += 2
            return marker, len(run_dirs)

        task2_root = max(existing_candidates, key=score_task2_dir)
    else:
        task2_root = task2_candidates[0]
    lines.append("## Task2 检测 + 跟踪")
    lines.append("")

    if not task2_root.exists():
        warnings.append("Task2 输出目录不存在（已检查 `outputs/task2` 与 `runs/detect/outputs/task2`）。")
        lines.append("- 暂无 Task2 输出目录。")
        lines.append("")
        return lines, warnings

    lines.append(f"- 检测到 Task2 输出目录：`{task2_root.relative_to(project_root)}`")
    lines.append("")

    run_dirs = sorted([p for p in task2_root.iterdir() if p.is_dir()])
    if not run_dirs:
        warnings.append("Task2 没有训练 run 子目录。")
        lines.append("- `outputs/task2` 下没有 run。")
        lines.append("")
        return lines, warnings

    lines.append("| run_name | epochs | imgsz | batch | device |")
    lines.append("|---|---:|---:|---:|---|")
    for run in run_dirs:
        args_json = read_json(run / "train_args.json")
        if args_json is not None:
            lines.append(
                f"| {run.name} | {args_json.get('epochs', '-')} | {args_json.get('imgsz', '-')} | "
                f"{args_json.get('batch', '-')} | {args_json.get('device', '-')} |"
            )
            continue

        args_yaml = run / "args.yaml"
        if args_yaml.exists():
            # args.yaml is produced by ultralytics; parse minimally without extra deps.
            kv = {}
            for raw_line in args_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                kv[key.strip()] = value.strip()
            lines.append(
                f"| {run.name} | {kv.get('epochs', '-')} | {kv.get('imgsz', '-')} | "
                f"{kv.get('batch', '-')} | {kv.get('device', '-')} |"
            )
            continue

        warnings.append(f"Task2 `{run.name}` 缺少 `train_args.json/args.yaml`。")
        lines.append(f"| {run.name} | - | - | - | - |")
    lines.append("")

    found_results_csv = False
    found_tracking_summary = False
    found_final_metrics = False
    for run in run_dirs:
        results_csv = run / "results.csv"
        if results_csv.exists():
            found_results_csv = True
        tracking_summary = run / "tracking_summary.json"
        if tracking_summary.exists():
            found_tracking_summary = True
        if (run / "final_metrics.json").exists():
            found_final_metrics = True

    if not found_results_csv:
        warnings.append("Task2 尚未发现 `results.csv`（无法自动提取 mAP/Precision/Recall）。")
        lines.append("- 暂未检测到 `results.csv`，请从 wandb 或 YOLO 训练日志补 `mAP` 指标。")
    if not found_final_metrics:
        warnings.append("Task2 尚未发现 `final_metrics.json`（无法自动提取最终检测指标）。")
        lines.append("- 暂未检测到 `final_metrics.json`，建议用 `task2/train_visdrone.py` 训练产生该文件。")
    if not found_tracking_summary:
        warnings.append("Task2 尚未发现 `tracking_summary.json`（无法自动提取越线计数与 ID 统计）。")
        lines.append("- 暂未检测到 `tracking_summary.json`，请在跟踪脚本输出目录补充统计。")
    lines.append("")

    lines.append("### Task2 报告可用曲线（wandb）")
    lines.append("- metrics/mAP50(B), metrics/mAP50-95(B)")
    lines.append("- metrics/precision(B), metrics/recall(B)")
    lines.append("- train/box_loss, train/cls_loss, train/dfl_loss")
    lines.append("")
    return lines, warnings


def collect_task3(project_root: Path) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    warnings: List[str] = []
    task3_candidates = [
        project_root / "outputs" / "task3",
        project_root / "outputs" / "task3_cloud",
    ]
    task3_root = next((p for p in task3_candidates if p.exists()), task3_candidates[0])
    lines.append("## Task3 分割（U-Net + Loss Ablation）")
    lines.append("")

    if not task3_root.exists():
        warnings.append("Task3 输出目录不存在（已检查 `outputs/task3` 与 `outputs/task3_cloud`）。")
        lines.append("- 暂无 Task3 输出目录。")
        lines.append("")
        return lines, warnings

    lines.append(f"- 检测到 Task3 输出目录：`{task3_root.relative_to(project_root)}`")
    lines.append("")

    final_metrics_files = sorted(task3_root.glob("**/final_metrics.json"))
    if final_metrics_files:
        lines.append("| run_dir | best_val_miou | test_miou | test_loss |")
        lines.append("|---|---:|---:|---:|")
        parsed = []
        for metrics_path in final_metrics_files:
            m = read_json(metrics_path) or {}
            run_dir = metrics_path.parent
            lines.append(
                f"| `{run_dir.relative_to(project_root)}` | {fmt_num(m.get('best_val_miou'))} | "
                f"{fmt_num(m.get('test_miou'))} | {fmt_num(m.get('test_loss'))} |"
            )
            try:
                parsed.append((float(m.get("test_miou", -1)), str(run_dir.relative_to(project_root))))
            except (TypeError, ValueError):
                pass
        lines.append("")
        if parsed:
            parsed.sort(reverse=True, key=lambda x: x[0])
            lines.append(f"- 最优 `test_miou`：**{parsed[0][0]:.6f}**（`{parsed[0][1]}`）")
            lines.append("")
    else:
        warnings.append("Task3 未发现 `outputs/task3/**/final_metrics.json`。")
        lines.append("- 暂无 Task3 指标文件（请先运行 `task3/run_loss_ablation.py`）。")
        lines.append("")

    lines.append("### Task3 报告可用曲线（wandb）")
    lines.append("- train_loss / val_loss")
    lines.append("- train_miou / val_miou")
    lines.append("")
    return lines, warnings


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_md = Path(args.output_md).resolve()

    lines: List[str] = []
    warnings: List[str] = []

    lines.append("# 实验结果自动汇总（Task1-3）")
    lines.append("")
    lines.append(f"- 生成时间目录：`{project_root}`")
    lines.append("")

    t1_lines, t1_warnings = collect_task1(project_root)
    t2_lines, t2_warnings = collect_task2(project_root)
    t3_lines, t3_warnings = collect_task3(project_root)
    lines.extend(t1_lines)
    lines.extend(t2_lines)
    lines.extend(t3_lines)
    warnings.extend(t1_warnings + t2_warnings + t3_warnings)

    lines.append("## 报告填表建议")
    lines.append("- Task1：可直接复制本文件表格中的数值。")
    lines.append("- Task2：若缺 mAP/跟踪统计，请先补齐 `results.csv` 与 `tracking_summary.json`。")
    lines.append("- Task3：补齐 `final_metrics.json` 后重跑本脚本自动更新。")
    lines.append("")

    if warnings:
        lines.append("## 待补充项（自动检测）")
        for w in warnings:
            lines.append(f"- [ ] {w}")
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report summary to: {output_md}")


if __name__ == "__main__":
    main()

