from typing import List

import torch
import torch.nn as nn
from torchvision import models


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class CBAMModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool, _ = torch.max(x, dim=2, keepdim=True)
        max_pool, _ = torch.max(max_pool, dim=3, keepdim=True)
        channel_att = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class AttentionResNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        attention: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        weight_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        }
        base = getattr(models, model_name)(weights=weight_map[model_name] if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        out_channels = base.fc.in_features

        if attention == "se":
            self.attention = SEModule(out_channels)
        elif attention == "cbam":
            self.attention = CBAMModule(out_channels)
        else:
            raise ValueError(f"不支持的 attention 模式: {attention}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    attention: str = "none",
    dropout: float = 0.0,
) -> nn.Module:
    if model_name not in {"resnet18", "resnet34"}:
        raise ValueError("model_name 必须是 resnet18 或 resnet34")
    if attention == "none":
        weight_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        }
        model = getattr(models, model_name)(weights=weight_map[model_name] if pretrained else None)
        in_features = model.fc.in_features
        layers: List[nn.Module] = [nn.Linear(in_features, num_classes)]
        if dropout > 0:
            layers = [nn.Dropout(dropout)] + layers
        model.fc = nn.Sequential(*layers)
        return model

    return AttentionResNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        attention=attention,
        dropout=dropout,
    )


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    head_lr: float,
):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ("fc", "classifier", "attention")):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": head_lr},
    ]

