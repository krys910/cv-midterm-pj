import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser("Run detection+tracking and line crossing counting.")
    parser.add_argument("--model", type=str, required=True, help="best.pt path")
    parser.add_argument("--source", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="outputs/task2/tracking")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--line",
        type=str,
        default="",
        help="Line in x1,y1,x2,y2 format. Empty means horizontal center line.",
    )
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="task2_tracking")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    return parser.parse_args()


def parse_line(line_arg: str, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if line_arg.strip():
        x1, y1, x2, y2 = [int(v.strip()) for v in line_arg.split(",")]
        return (x1, y1), (x2, y2)
    return (0, height // 2), (width - 1, height // 2)


def point_side(p: Tuple[float, float], a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    line_p1, line_p2 = parse_line(args.line, width, height)
    save_video_path = out_dir / f"{Path(args.source).stem}_tracked.mp4"
    writer = cv2.VideoWriter(
        str(save_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_run_name,
            config=vars(args),
        )

    model = YOLO(args.model)
    names = model.model.names

    prev_side: Dict[int, float] = {}
    counted_ids = set()
    frame_logs = []
    total_crossings = 0

    results = model.track(
        source=args.source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )

    for frame_idx, res in enumerate(results):
        frame = res.orig_img.copy()
        cv2.line(frame, line_p1, line_p2, (0, 255, 255), 2)

        boxes = res.boxes
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=float)
            ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.full((len(xyxy),), -1)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
                track_id = int(ids[i])
                class_id = int(cls[i])
                cf = float(confs[i])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                side = point_side((cx, cy), line_p1, line_p2)
                if track_id >= 0 and track_id in prev_side:
                    if prev_side[track_id] * side < 0 and track_id not in counted_ids:
                        total_crossings += 1
                        counted_ids.add(track_id)
                if track_id >= 0:
                    prev_side[track_id] = side

                color = (0, 200, 0) if track_id not in counted_ids else (0, 128, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID{track_id} {names[class_id]} {cf:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                frame_logs.append(
                    {
                        "frame": frame_idx,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "conf": cf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                    }
                )

        cv2.putText(
            frame,
            f"Crossings: {total_crossings}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        writer.write(frame)

        if wandb_run is not None and frame_idx % 30 == 0:
            import wandb

            wandb.log({"tracking/frame": frame_idx, "tracking/crossings": total_crossings}, step=frame_idx)

    writer.release()

    tracks_csv = out_dir / "tracks.csv"
    pd.DataFrame(frame_logs).to_csv(tracks_csv, index=False)

    summary = {
        "source_video": str(Path(args.source).resolve()),
        "output_video": str(save_video_path.resolve()),
        "tracks_csv": str(tracks_csv.resolve()),
        "line": {"p1": line_p1, "p2": line_p2},
        "total_crossings": total_crossings,
        "unique_ids": len(set([int(r["track_id"]) for r in frame_logs if int(r["track_id"]) >= 0])),
        "total_boxes": len(frame_logs),
    }
    summary_path = out_dir / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if wandb_run is not None:
        import wandb

        wandb.log({"tracking/total_crossings": total_crossings})
        if save_video_path.exists():
            wandb.log({"tracking/video": wandb.Video(str(save_video_path), fps=fps, format="mp4")})
        wandb.save(str(summary_path))
        wandb.save(str(tracks_csv))
        wandb.finish()


if __name__ == "__main__":
    main()

