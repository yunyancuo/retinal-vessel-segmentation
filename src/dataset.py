"""PyTorch 数据集：读取预处理输出，提供归一化、翻转/旋转增强与 16 倍数方形填充。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from io_utils import load_color, load_gray

__all__ = ["DRIVEDataset", "load_color", "load_gray", "pad16"]


def pad16(tensor: torch.Tensor) -> torch.Tensor:
    """将张量零填充为 16 倍数边长的正方形，保证 U-Net 四次下采样严格对齐。

    按 ``max(H, W)`` 向上取整至 16 的倍数而非分别填充宽与高：随机 90° 旋转会交换
    H/W，分别填充会导致同一 batch 内尺寸不一致而无法 ``stack``；正方形目标可确保
    旋转前后尺寸一致。

    Args:
        tensor: 形状为 ``(C, H, W)`` 或 ``(B, C, H, W)`` 的图像/掩膜张量。

    Returns:
        右下方零填充后的张量。
    """
    if tensor.ndim not in (3, 4):
        raise ValueError(f"pad16 仅支持 3D/4D 张量，收到形状: {tuple(tensor.shape)}")
    h, w = tensor.shape[-2], tensor.shape[-1]
    side = max(h, w)
    side = side + (16 - side % 16) % 16
    if side == h and side == w:
        return tensor
    return F.pad(tensor, (0, side - w, 0, side - h))


class DRIVEDataset(Dataset):
    """通用眼底血管分割数据集（兼容 FIVES 与 DRIVE 预处理目录）。

    Args:
        root: 指向 ``processed`` 目录，其下包含 ``{split}/{images,masks,fov}``。
        split: 数据子集名，通常为 ``"training"`` 或 ``"test"``。
        augment: 是否启用随机水平/垂直翻转与 90° 旋转增强。
        keys: 可选样本键列表；为 ``None`` 时自动扫描 ``images/*.png``。
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        keys: Sequence[str] | None = None,
    ) -> None:
        self.split_dir = Path(root) / split
        img_dir = self.split_dir / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        all_keys = sorted(p.stem for p in img_dir.glob("*.png"))
        self.keys = all_keys if keys is None else list(keys)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.keys)

    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对图像与掩膜施加一致的随机几何增强（翻转 + 90° 旋转）。"""
        if random.random() < 0.5:  # 水平翻转
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:  # 垂直翻转
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)  # 随机旋转 90° × k
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        key = self.keys[idx]
        img = load_color(self.split_dir / "images" / f"{key}.png")
        mask = load_gray(self.split_dir / "masks" / f"{key}.png")
        img_f = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        mask_f = (mask > 127).astype(np.float32)
        if self.augment:
            img_f, mask_f = self._augment(img_f, mask_f)
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_f).float()[None]
        return pad16(img_t), pad16(mask_t), key
