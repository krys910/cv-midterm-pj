import argparse
import json
from pathlib import Path
from typing import Any, Dict
import shutil

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser("Train YOLOv8 on VisDrone with wandb sync.")
    parser.add_argument("--data", type=str, required=True, help="Path to visdrone.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--project", type=str, default="outputs/task2")
    parser.add_argument("--name", type=str, default="visdrone_yolov8")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    return parser.parse_args()


def _to_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def main():
    args = parse_args()
    save_root = Path(args.project) / args.name
    save_root.mkdir(parents=True, exist_ok=True)

    with (save_root / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.name,
            config=vars(args),
        )

    model = YOLO(args.model)

    if wandb_run is not None:
        import wandb

        def on_fit_epoch_end(trainer):
            metrics = _to_float_dict(getattr(trainer, "metrics", {}))
            metrics["epoch"] = int(getattr(trainer, "epoch", -1)) + 1
            if getattr(trainer, "optimizer", None) is not None:
                pg = trainer.optimizer.param_groups
                if len(pg) > 0:
                    metrics["lr/pg0"] = float(pg[0]["lr"])
            wandb.log(metrics, step=metrics["epoch"])

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    train_ret = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        seed=args.seed,
        exist_ok=True,
        val=True,
        pretrained=True,
        amp=args.amp,
    )

    actual_save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else None
    metrics_out = {}
    if hasattr(train_ret, "results_dict"):
        metrics_out.update(_to_float_dict(train_ret.results_dict))
    if hasattr(model, "trainer") and hasattr(model.trainer, "metrics"):
        metrics_out.update(_to_float_dict(model.trainer.metrics))

    if actual_save_dir is not None:
        metrics_out["actual_save_dir"] = str(actual_save_dir.resolve())
        src_weights = actual_save_dir / "weights"
        dst_weights = save_root / "weights"
        dst_weights.mkdir(parents=True, exist_ok=True)
        for w in ("best.pt", "last.pt"):
            src = src_weights / w
            if src.exists():
                shutil.copy2(src, dst_weights / w)

    metrics_path = save_root / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics_out, ensure_ascii=False, indent=2))

    if wandb_run is not None:
        import wandb

        if metrics_out:
            wandb.log(metrics_out)
        best_pt = save_root / "weights" / "best.pt"
        if best_pt.exists():
            wandb.save(str(best_pt))
        wandb.finish()


if __name__ == "__main__":
    main()

