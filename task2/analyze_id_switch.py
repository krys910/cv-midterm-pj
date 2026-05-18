import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Analyze potential ID switches and export 3-4 frames.")
    parser.add_argument("--track_csv", type=str, required=True)
    parser.add_argument("--source_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/task2/id_analysis")
    parser.add_argument("--start_frame", type=int, default=-1, help="If -1, auto-pick from detected events.")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument(
        "--pick_mode",
        type=str,
        default="crowded",
        choices=["first", "best_iou", "crowded"],
        help="How to auto pick event window when start_frame is -1.",
    )
    parser.add_argument(
        "--crowd_dist_thresh",
        type=float,
        default=120.0,
        help="Center distance threshold (pixels) for crowd score.",
    )
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="task2_id_switch_analysis")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    return parser.parse_args()


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def find_switch_events(df: pd.DataFrame, iou_thresh: float) -> List[Dict]:
    events = []
    max_frame = int(df["frame"].max())
    for f in range(max_frame):
        a = df[df["frame"] == f]
        b = df[df["frame"] == f + 1]
        if a.empty or b.empty:
            continue
        for _, ra in a.iterrows():
            same_cls = b[b["class_id"] == ra["class_id"]]
            if same_cls.empty:
                continue
            box_a = np.array([ra["x1"], ra["y1"], ra["x2"], ra["y2"]], dtype=float)
            best_iou = 0.0
            best_row = None
            for _, rb in same_cls.iterrows():
                box_b = np.array([rb["x1"], rb["y1"], rb["x2"], rb["y2"]], dtype=float)
                v = iou_xyxy(box_a, box_b)
                if v > best_iou:
                    best_iou = v
                    best_row = rb
            if best_row is not None and best_iou >= iou_thresh and int(best_row["track_id"]) != int(ra["track_id"]):
                events.append(
                    {
                        "frame": int(f),
                        "next_frame": int(f + 1),
                        "class_name": str(ra["class_name"]),
                        "old_id": int(ra["track_id"]),
                        "new_id": int(best_row["track_id"]),
                        "iou": float(best_iou),
                    }
                )
    return events


def frame_crowd_score(frame_df: pd.DataFrame, crowd_dist_thresh: float) -> float:
    if len(frame_df) <= 1:
        return 0.0
    boxes = frame_df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    centers = frame_df[["cx", "cy"]].to_numpy(dtype=float)
    max_iou = 0.0
    close_pairs = 0
    n = len(frame_df)
    for i in range(n):
        for j in range(i + 1, n):
            max_iou = max(max_iou, iou_xyxy(boxes[i], boxes[j]))
            if np.linalg.norm(centers[i] - centers[j]) <= crowd_dist_thresh:
                close_pairs += 1
    # Weighted sum: overlapping boxes + nearby targets + frame target density.
    return float(max_iou + 0.05 * close_pairs + 0.01 * n)


def pick_start_frame(
    df: pd.DataFrame,
    events: List[Dict],
    mode: str,
    crowd_dist_thresh: float,
) -> Tuple[int, Optional[Dict], float]:
    if not events:
        return int(df["frame"].min()), None, 0.0
    if mode == "first":
        return int(events[0]["frame"]), events[0], float(events[0]["iou"])
    if mode == "best_iou":
        best = max(events, key=lambda e: float(e["iou"]))
        return int(best["frame"]), best, float(best["iou"])

    # crowded: prefer switches in crowded/occluded scenes.
    best_event = None
    best_score = -1.0
    for e in events:
        f = int(e["frame"])
        frame_df = df[df["frame"] == f]
        cscore = frame_crowd_score(frame_df, crowd_dist_thresh=crowd_dist_thresh)
        score = float(e["iou"]) + cscore
        if score > best_score:
            best_score = score
            best_event = e
    assert best_event is not None
    return int(best_event["frame"]), best_event, float(best_score)


def draw_frame(
    frame: np.ndarray,
    frame_df: pd.DataFrame,
    highlight_ids: Optional[Set[int]] = None,
) -> np.ndarray:
    out = frame.copy()
    highlight_ids = highlight_ids or set()
    for _, r in frame_df.iterrows():
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        tid = int(r["track_id"])
        cls_name = str(r["class_name"])
        is_focus = tid in highlight_ids
        color = (0, 0, 255) if is_focus else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"ID{tid} {cls_name}" + (" *switch*" if is_focus else ""),
            (x1, max(20, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.track_csv)
    if df.empty:
        raise RuntimeError("track_csv 为空，无法分析 ID 跳变。")

    events = find_switch_events(df, iou_thresh=args.iou_thresh)

    if args.start_frame >= 0:
        start_frame = args.start_frame
        selected_event = None
        selection_score = None
    elif events:
        start_frame, selected_event, selection_score = pick_start_frame(
            df=df,
            events=events,
            mode=args.pick_mode,
            crowd_dist_thresh=args.crowd_dist_thresh,
        )
    else:
        start_frame = int(df["frame"].min())
        selected_event = None
        selection_score = None

    end_frame = start_frame + args.num_frames - 1
    window_events = [e for e in events if start_frame <= e["frame"] <= end_frame]
    highlight_by_frame: Dict[int, Set[int]] = {}
    for e in window_events:
        f = int(e["frame"])
        highlight_by_frame.setdefault(f, set()).add(int(e["old_id"]))
        highlight_by_frame.setdefault(f + 1, set()).add(int(e["new_id"]))

    cap = cv2.VideoCapture(args.source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.source_video}")

    saved_images = []
    for fidx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_df = df[df["frame"] == fidx]
        vis = draw_frame(frame, frame_df, highlight_ids=highlight_by_frame.get(fidx, set()))
        save_path = out_dir / f"frame_{fidx:06d}.jpg"
        cv2.imwrite(str(save_path), vis)
        saved_images.append(str(save_path.resolve()))
    cap.release()

    report = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_saved_frames": len(saved_images),
        "auto_pick_mode": args.pick_mode if args.start_frame < 0 else "manual",
        "selected_event": selected_event,
        "selection_score": selection_score,
        "events_in_window": window_events,
        "all_detected_switch_events": events,
        "note": "事件由相邻帧同类目标高 IoU 但 Track ID 改变近似得到；默认优先挑更拥挤场景，便于遮挡/交汇分析。",
    }
    report_path = out_dir / "id_switch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wandb.log(
            {
                "id_switch/num_events_total": len(events),
                "id_switch/num_events_window": len(window_events),
                "id_switch/start_frame": start_frame,
                "id_switch/end_frame": end_frame,
            }
        )
        for img_path in saved_images:
            wandb.log({"id_switch/frame": wandb.Image(img_path)})
        wandb.save(str(report_path))
        run.finish()


if __name__ == "__main__":
    main()

