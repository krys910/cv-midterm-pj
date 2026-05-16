import argparse
import csv
import json
import random
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data import make_dataloaders
from src.models import build_model, build_param_groups


def parse_args():
    parser = argparse.ArgumentParser("Oxford-IIIT Pet fine-tune training")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--attention", type=str, default="none", choices=["none", "se", "cbam"])
    parser.add_argument("--init", type=str, default="pretrained", choices=["pretrained", "scratch"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--wandb_project", type=str, default="cv-midterm-pet")
    parser.add_argument("--wandb_run_name", type=str, default="")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    bar = tqdm(loader, ncols=100, leave=False)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_count += bs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        bar.set_postfix(loss=f"{total_loss / total_count:.4f}", acc=f"{total_acc / total_count:.4f}")

    return total_loss / total_count, total_acc / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    return run_epoch(model, loader, criterion, optimizer=None, device=device, train=False)


def write_history(history, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr_backbone", "lr_head"]
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_loader, val_loader, test_loader, num_classes = make_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=(args.init == "pretrained"),
        attention=args.attention,
        dropout=args.dropout,
    )
    device = pick_device()
    model = model.to(device)

    optimizer = SGD(
        build_param_groups(model, base_lr=args.base_lr, head_lr=args.head_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name if args.wandb_run_name else None,
                config=vars(args),
            )
        except Exception as e:
            print(f"[warn] wandb 初始化失败，改为本地日志: {e}")

    best_val = -1.0
    history = []
    start = time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(row)

        torch.save(model.state_dict(), output_dir / "last.pt")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    total_minutes = (time() - start) / 60.0
    metrics = {
        "best_val_acc": round(best_val, 6),
        "test_acc": round(test_acc, 6),
        "test_loss": round(test_loss, 6),
        "elapsed_minutes": round(total_minutes, 2),
        "device": str(device),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_history(history, output_dir / "history.csv")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()


if __name__ == "__main__":
    main()

