import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Run experiment sweep for task-1")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_root", type=str, default="task1/outputs/task1_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--models", type=str, default="resnet18,resnet34")
    parser.add_argument("--base_lrs", type=str, default="1e-4,3e-4")
    parser.add_argument("--head_lrs", type=str, default="1e-3,3e-3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_attention", action="store_true")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    return parser.parse_args()


def run(cmd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base_lrs = [float(v.strip()) for v in args.base_lrs.split(",") if v.strip()]
    head_lrs = [float(v.strip()) for v in args.head_lrs.split(",") if v.strip()]

    records = []

    # 1) Hyper-parameter analysis on pretrained baselines.
    for model_name, base_lr, head_lr in product(models, base_lrs, head_lrs):
        tag = f"{model_name}_pretrained_blr{base_lr}_hlr{head_lr}"
        out_dir = output_root / "hyper" / tag
        cmd = [
            sys.executable,
            str(train_script),
            "--data_root",
            args.data_root,
            "--output_dir",
            str(out_dir),
            "--model",
            model_name,
            "--attention",
            "none",
            "--init",
            "pretrained",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--base_lr",
            str(base_lr),
            "--head_lr",
            str(head_lr),
            "--seed",
            str(args.seed),
        ]
        if args.use_wandb:
            cmd.append("--use_wandb")
        run(cmd)
        records.append({"type": "hyper", "tag": tag, "dir": str(out_dir)})

    # 2) Pretraining ablation: pretrained vs scratch.
    for model_name in models:
        for init_mode in ["pretrained", "scratch"]:
            tag = f"{model_name}_{init_mode}_ablation"
            out_dir = output_root / "ablation" / tag
            cmd = [
                sys.executable,
                str(train_script),
                "--data_root",
                args.data_root,
                "--output_dir",
                str(out_dir),
                "--model",
                model_name,
                "--attention",
                "none",
                "--init",
                init_mode,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--base_lr",
                str(base_lrs[0]),
                "--head_lr",
                str(head_lrs[0]),
                "--seed",
                str(args.seed),
            ]
            if args.use_wandb:
                cmd.append("--use_wandb")
            run(cmd)
            records.append({"type": "ablation", "tag": tag, "dir": str(out_dir)})

    # 3) Attention experiments.
    if args.with_attention:
        for model_name in models:
            for attention in ["se", "cbam"]:
                tag = f"{model_name}_{attention}_attention"
                out_dir = output_root / "attention" / tag
                cmd = [
                    sys.executable,
                    str(train_script),
                    "--data_root",
                    args.data_root,
                    "--output_dir",
                    str(out_dir),
                    "--model",
                    model_name,
                    "--attention",
                    attention,
                    "--init",
                    "pretrained",
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    "--base_lr",
                    str(base_lrs[0]),
                    "--head_lr",
                    str(head_lrs[0]),
                    "--seed",
                    str(args.seed),
                ]
                if args.use_wandb:
                    cmd.append("--use_wandb")
                run(cmd)
                records.append({"type": "attention", "tag": tag, "dir": str(out_dir)})

    with (output_root / "runs.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[done] total runs: {len(records)}")


if __name__ == "__main__":
    main()

