import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Summarize Task-1 experiment outputs")
    parser.add_argument("--output_root", type=str, default="task1/outputs/task1_full")
    parser.add_argument("--save_csv", type=str, default="task1/outputs/task1_full/summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    rows = []

    for metrics_path in root.glob("**/metrics.json"):
        run_dir = metrics_path.parent
        args_path = run_dir / "args.json"
        if not args_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        with args_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        rows.append(
            {
                "run_dir": str(run_dir),
                "model": cfg.get("model"),
                "attention": cfg.get("attention"),
                "init": cfg.get("init"),
                "epochs": cfg.get("epochs"),
                "base_lr": cfg.get("base_lr"),
                "head_lr": cfg.get("head_lr"),
                "best_val_acc": metrics.get("best_val_acc"),
                "test_acc": metrics.get("test_acc"),
                "test_loss": metrics.get("test_loss"),
                "elapsed_minutes": metrics.get("elapsed_minutes"),
            }
        )

    if not rows:
        print("No completed runs found.")
        return

    df = pd.DataFrame(rows).sort_values(by="test_acc", ascending=False)
    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved summary to: {save_path}")


if __name__ == "__main__":
    main()

