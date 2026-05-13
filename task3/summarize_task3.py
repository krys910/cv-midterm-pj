import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Summarize Task3 runs.")
    parser.add_argument("--output_root", type=str, default="outputs/task3")
    parser.add_argument("--save_csv", type=str, default="outputs/task3/summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []
    for metric_path in output_root.glob("**/final_metrics.json"):
        run_dir = metric_path.parent
        args_path = run_dir / "train_args.json"
        if not args_path.exists():
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        cfg = json.loads(args_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": str(run_dir),
                "loss_type": cfg.get("loss_type"),
                "epochs": cfg.get("epochs"),
                "image_size": cfg.get("image_size"),
                "batch_size": cfg.get("batch_size"),
                "lr": cfg.get("lr"),
                "best_val_miou": metrics.get("best_val_miou"),
                "test_miou": metrics.get("test_miou"),
                "test_loss": metrics.get("test_loss"),
                "device": metrics.get("device"),
            }
        )

    if not rows:
        print("No Task3 runs found.")
        return

    df = pd.DataFrame(rows).sort_values(by="best_val_miou", ascending=False)
    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()

