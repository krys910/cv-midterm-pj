import random
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_IMAGES_ARCHIVE = "images.tar.gz"
_ANN_ARCHIVE = "annotations.tar.gz"


def maybe_extract_archives(data_root: str) -> None:
    root = Path(data_root)
    images_dir = root / "images"
    ann_dir = root / "annotations"
    if images_dir.exists() and ann_dir.exists():
        return

    images_tar = root / _IMAGES_ARCHIVE
    ann_tar = root / _ANN_ARCHIVE
    if not images_tar.exists() or not ann_tar.exists():
        raise FileNotFoundError(
            "未找到 images/annotations 目录，且缺少 images.tar.gz 或 annotations.tar.gz。"
        )

    print("[data] 正在解压 Oxford-IIIT Pet 数据集...")
    with tarfile.open(images_tar, "r:gz") as tar:
        tar.extractall(path=root)
    with tarfile.open(ann_tar, "r:gz") as tar:
        tar.extractall(path=root)
    print("[data] 解压完成。")


def _read_split_file(path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # image_id class_id species breed_id
            image_id, class_id, *_ = line.split()
            items.append((image_id, int(class_id) - 1))
    return items


def build_train_val_split(
    trainval_items: Sequence[Tuple[str, int]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for item in trainval_items:
        by_class.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train_items: List[Tuple[str, int]] = []
    val_items: List[Tuple[str, int]] = []
    for _, cls_items in by_class.items():
        cls_items = list(cls_items)
        rng.shuffle(cls_items)
        n_val = max(1, int(len(cls_items) * val_ratio))
        val_items.extend(cls_items[:n_val])
        train_items.extend(cls_items[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


class OxfordPetDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        items: Sequence[Tuple[str, int]],
        transform: transforms.Compose,
    ) -> None:
        self.data_root = Path(data_root)
        self.items = list(items)
        self.transform = transform
        self.images_dir = self.data_root / "images"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_id, label = self.items[idx]
        image_path = self.images_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def build_transforms(image_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    return train_tf, eval_tf


def make_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    maybe_extract_archives(data_root)
    root = Path(data_root)
    ann_dir = root / "annotations"
    trainval = _read_split_file(ann_dir / "trainval.txt")
    test = _read_split_file(ann_dir / "test.txt")
    train_items, val_items = build_train_val_split(trainval, val_ratio=val_ratio, seed=seed)

    train_tf, eval_tf = build_transforms(image_size=image_size)
    train_ds = OxfordPetDataset(data_root, train_items, train_tf)
    val_ds = OxfordPetDataset(data_root, val_items, eval_tf)
    test_ds = OxfordPetDataset(data_root, test, eval_tf)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, 37

