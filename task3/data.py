import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IGNORE_INDEX = 255


def _read_split_file(path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_id, class_id, *_ = line.split()
        items.append((image_id, int(class_id) - 1))
    return items


def split_train_val(
    trainval_items: Sequence[Tuple[str, int]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    by_class = {}
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


def _map_mask_values(mask_array: np.ndarray) -> np.ndarray:
    # Oxford trimap labels: 1=foreground, 2=background, 3=border/unknown
    mapped = np.full_like(mask_array, fill_value=IGNORE_INDEX, dtype=np.uint8)
    mapped[mask_array == 2] = 0
    mapped[mask_array == 1] = 1
    mapped[mask_array == 3] = IGNORE_INDEX
    return mapped


class OxfordPetSegDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        items: Sequence[Tuple[str, int]],
        image_size: int = 256,
        train: bool = True,
    ) -> None:
        self.root = Path(data_root)
        self.items = list(items)
        self.image_size = image_size
        self.train = train
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "annotations" / "trimaps"

    def __len__(self) -> int:
        return len(self.items)

    def _transform(self, image: Image.Image, mask: Image.Image):
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = _map_mask_values(mask_np)
        mask_t = torch.from_numpy(mask_np).long()
        return image_t, mask_t

    def __getitem__(self, idx: int):
        image_id, _ = self.items[idx]
        image = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.masks_dir / f"{image_id}.png").convert("L")
        return self._transform(image, mask)


def make_seg_dataloaders(
    data_root: str,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    root = Path(data_root)
    trainval_items = _read_split_file(root / "annotations" / "trainval.txt")
    test_items = _read_split_file(root / "annotations" / "test.txt")
    train_items, val_items = split_train_val(trainval_items, val_ratio=val_ratio, seed=seed)

    train_ds = OxfordPetSegDataset(data_root, train_items, image_size=image_size, train=True)
    val_ds = OxfordPetSegDataset(data_root, val_items, image_size=image_size, train=False)
    test_ds = OxfordPetSegDataset(data_root, test_items, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader

