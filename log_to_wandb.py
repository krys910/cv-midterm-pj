import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Upload finished runs to Weights & Biases")
    parser.add_argument("--output_root", type=str, default="outputs/task1_full")
    parser.add_argument("--project", type=str, default="cv-midterm-pet")
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.offline:
        import os

        os.environ["WANDB_MODE"] = "offline"

    import wandb

    output_root = Path(args.output_root)
    run_dirs = sorted({p.parent for p in output_root.glob("**/history.csv")})
    if not run_dirs:
        print("No history.csv found. Nothing to upload.")
        return

    print(f"Found {len(run_dirs)} runs under: {output_root}")
    for run_dir in run_dirs:
        args_path = run_dir / "args.json"
        metrics_path = run_dir / "metrics.json"
        hist_path = run_dir / "history.csv"
        if not (args_path.exists() and metrics_path.exists() and hist_path.exists()):
            continue

        with args_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        with metrics_path.open("r", encoding="utf-8") as f:
            final_metrics = json.load(f)

        run_name = run_dir.name
        run = wandb.init(
            project=args.project,
            entity=args.entity if args.entity else None,
            name=run_name,
            config=cfg,
            reinit=True,
        )

        history_df = pd.read_csv(hist_path)
        for _, row in history_df.iterrows():
            wandb.log(
                {
                    "epoch": int(row["epoch"]),
                    "train/loss": float(row["train_loss"]),
                    "train/acc": float(row["train_acc"]),
                    "val/loss": float(row["val_loss"]),
                    "val/acc": float(row["val_acc"]),
                    "lr/backbone": float(row["lr_backbone"]),
                    "lr/head": float(row["lr_head"]),
                },
                step=int(row["epoch"]),
            )

        wandb.summary["best_val_acc"] = final_metrics.get("best_val_acc")
        wandb.summary["test_acc"] = final_metrics.get("test_acc")
        wandb.summary["test_loss"] = final_metrics.get("test_loss")
        wandb.summary["elapsed_minutes"] = final_metrics.get("elapsed_minutes")
        run.finish()
        print(f"Uploaded: {run_dir}")


if __name__ == "__main__":
    main()

