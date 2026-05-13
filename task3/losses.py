import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # foreground probability
        valid_mask = (target != self.ignore_index).float()
        target_fg = (target == 1).float()

        probs = probs * valid_mask
        target_fg = target_fg * valid_mask

        intersection = torch.sum(probs * target_fg, dim=(1, 2))
        union = torch.sum(probs, dim=(1, 2)) + torch.sum(target_fg, dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(loss_type: str, ignore_index: int = 255):
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if loss_type == "dice":
        return DiceLoss(ignore_index=ignore_index)
    if loss_type == "ce_dice":
        return CombinedLoss(ignore_index=ignore_index)
    raise ValueError(f"Unsupported loss_type: {loss_type}")

