import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Run Task3 ablation for CE / Dice / CE+Dice losses.")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_root", type=str, default="outputs/task3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device_hint", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_entity", type=str, default="")
    return parser.parse_args()


def run(cmd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    loss_types = ["ce", "dice", "ce_dice"]
    for loss_type in loss_types:
        run_name = f"task3_unet_{loss_type}"
        cmd = [
            sys.executable,
            "task3/train_unet.py",
            "--data_root",
            args.data_root,
            "--output_dir",
            str(out_root / run_name),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--image_size",
            str(args.image_size),
            "--lr",
            str(args.lr),
            "--loss_type",
            loss_type,
            "--wandb_project",
            args.wandb_project,
            "--wandb_entity",
            args.wandb_entity,
            "--wandb_run_name",
            run_name,
        ]
        run(cmd)
    print("[done] Task3 loss ablation finished.")


if __name__ == "__main__":
    main()

