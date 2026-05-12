import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("One-click Task2 pipeline")
    parser.add_argument("--visdrone_root", type=str, required=True)
    parser.add_argument("--test_video", type=str, required=True)
    parser.add_argument("--workspace", type=str, default="outputs/task2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_entity", type=str, default="")
    return parser.parse_args()


def run(cmd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    workspace = Path(args.workspace)
    dataset_root = workspace / "dataset"
    train_name = "visdrone_yolov8n"
    train_dir = workspace / train_name
    best_pt = train_dir / "weights" / "best.pt"
    track_dir = workspace / "tracking"
    analysis_dir = workspace / "id_analysis"

    run(
        [
            sys.executable,
            "task2/prepare_visdrone.py",
            "--visdrone_root",
            args.visdrone_root,
            "--output_root",
            str(dataset_root),
        ]
    )
    run(
        [
            sys.executable,
            "task2/train_visdrone.py",
            "--data",
            str(dataset_root / "visdrone.yaml"),
            "--model",
            "yolov8n.pt",
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--project",
            str(workspace),
            "--name",
            train_name,
            "--wandb_project",
            args.wandb_project,
            "--wandb_entity",
            args.wandb_entity,
        ]
    )
    run(
        [
            sys.executable,
            "task2/track_and_count.py",
            "--model",
            str(best_pt),
            "--source",
            args.test_video,
            "--output_dir",
            str(track_dir),
            "--wandb_run_name",
            "task2_tracking",
            "--wandb_project",
            args.wandb_project,
            "--wandb_entity",
            args.wandb_entity,
        ]
    )
    run(
        [
            sys.executable,
            "task2/analyze_id_switch.py",
            "--track_csv",
            str(track_dir / "tracks.csv"),
            "--source_video",
            args.test_video,
            "--output_dir",
            str(analysis_dir),
            "--wandb_run_name",
            "task2_id_switch_analysis",
            "--wandb_project",
            args.wandb_project,
            "--wandb_entity",
            args.wandb_entity,
        ]
    )
    print("[done] Task2 pipeline finished.")


if __name__ == "__main__":
    main()

