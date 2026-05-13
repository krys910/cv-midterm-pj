import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from task3.data import IGNORE_INDEX, make_seg_dataloaders
from task3.losses import build_loss
from task3.models import UNet


def parse_args():
    parser = argparse.ArgumentParser("Train U-Net from scratch on Oxford-IIIT Pet segmentation.")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs/task3/unet_ce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "dice", "ce_dice"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_batch_iou(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    pred = pred.view(-1)
    target = target.view(-1)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return 0.0

    ious = []
    for cls in [0, 1]:
        pred_c = pred == cls
        target_c = target == cls
        union = torch.logical_or(pred_c, target_c).sum().item()
        if union == 0:
            continue
        inter = torch.logical_and(pred_c, target_c).sum().item()
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_miou = 0.0
    total_count = 0

    bar = tqdm(loader, ncols=100, leave=False)
    for images, masks in bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        if train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        miou = compute_batch_iou(preds, masks, ignore_index=IGNORE_INDEX)

        bs = images.size(0)
        total_count += bs
        total_loss += loss.item() * bs
        total_miou += miou * bs
        bar.set_postfix(loss=f"{total_loss / total_count:.4f}", miou=f"{total_miou / total_count:.4f}")

    return total_loss / total_count, total_miou / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    return run_epoch(model, loader, criterion, optimizer=None, device=device, train=False)


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    train_loader, val_loader, test_loader = make_seg_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = UNet(in_channels=3, num_classes=2, base_ch=args.base_channels)
    device = pick_device()
    model = model.to(device)
    criterion = build_loss(args.loss_type, ignore_index=IGNORE_INDEX)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_run_name if args.wandb_run_name else None,
            config=vars(args),
        )

    best_val_miou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_miou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_miou": float(train_miou),
            "val_loss": float(val_loss),
            "val_miou": float(val_miou),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_mIoU {train_miou:.4f} | "
            f"val_loss {val_loss:.4f} val_mIoU {val_miou:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(row, step=epoch)

        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_loss, test_miou = evaluate(model, test_loader, criterion, device)
    final_metrics = {
        "best_val_miou": float(best_val_miou),
        "test_loss": float(test_loss),
        "test_miou": float(test_miou),
        "device": str(device),
    }
    (out_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))

    if wandb_run is not None:
        import wandb

        wandb_run.log(final_metrics)
        wandb.save(str(out_dir / "best.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()

