import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


# VisDrone DET: 0=ignored regions, 1..10 are valid categories.
VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def parse_args():
    parser = argparse.ArgumentParser("Convert VisDrone DET labels to YOLO format.")
    parser.add_argument(
        "--visdrone_root",
        type=str,
        required=True,
        help="VisDrone dataset root. E.g. ./VisDrone2019-DET",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="datasets/visdrone_yolo",
        help="Output root of converted YOLO dataset.",
    )
    parser.add_argument(
        "--train_dir_name",
        type=str,
        default="VisDrone2019-DET-train",
        help="Train split folder name under visdrone_root.",
    )
    parser.add_argument(
        "--val_dir_name",
        type=str,
        default="VisDrone2019-DET-val",
        help="Val split folder name under visdrone_root.",
    )
    return parser.parse_args()


def _load_image_shape(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = img.shape[:2]
    return h, w


def _convert_one_annotation_line(line: str, img_w: int, img_h: int):
    parts = line.strip().split(",")
    if len(parts) < 8:
        return None
    x, y, w, h, score, category, trunc, occ = [int(float(v)) for v in parts[:8]]
    if category <= 0 or category > 10:
        return None
    if w <= 1 or h <= 1:
        return None

    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        return None

    cls_id = category - 1
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def convert_split(split_root: Path, output_root: Path, split_name: str) -> Dict[str, int]:
    image_dir = split_root / "images"
    ann_dir = split_root / "annotations"
    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "boxes": 0}
    ann_files = sorted(ann_dir.glob("*.txt"))
    for ann_path in ann_files:
        stem = ann_path.stem
        img_path = image_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue

        h, w = _load_image_shape(img_path)
        yolo_lines: List[str] = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            item = _convert_one_annotation_line(line, w, h)
            if item is not None:
                yolo_lines.append(item)

        # Create a symbolic link to save disk space.
        link_path = out_img_dir / img_path.name
        if not link_path.exists():
            link_path.symlink_to(img_path.resolve())

        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(yolo_lines)

    return stats


def resolve_split_root(split_root: Path) -> Path:
    cur = split_root
    for _ in range(3):
        if (cur / "images").exists() and (cur / "annotations").exists():
            return cur
        subdirs = [p for p in cur.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            cur = subdirs[0]
            continue
        break
    raise FileNotFoundError(f"无法在 {split_root} 下找到 images/ 与 annotations/ 目录。")


def main():
    args = parse_args()
    visdrone_root = Path(args.visdrone_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_root = visdrone_root / args.train_dir_name
    val_root = visdrone_root / args.val_dir_name
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            "未找到 train/val 目录，请检查 --visdrone_root 与 split 目录名参数。"
        )

    train_root = resolve_split_root(train_root)
    val_root = resolve_split_root(val_root)

    train_stats = convert_split(train_root, output_root, "train")
    val_stats = convert_split(val_root, output_root, "val")

    yaml_path = output_root / "visdrone.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(VISDRONE_CLASS_NAMES)}",
            f"names: {VISDRONE_CLASS_NAMES}",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    summary = {
        "dataset_yaml": str(yaml_path.resolve()),
        "train": train_stats,
        "val": val_stats,
        "class_names": VISDRONE_CLASS_NAMES,
    }
    (output_root / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

