"""PyTorch 数据集：读取 preprocess.py 的输出，含归一化、翻转/旋转增强与 /16 padding。"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_gray(path):
    """兼容中文/特殊路径的灰度图读取。"""
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"图像读取失败: {path}")
    return m


def load_color(path):
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if m is None:
        raise IOError(f"图像读取失败: {path}")
    return m


def pad16(t: torch.Tensor) -> torch.Tensor:
    """向右/向下 pad 到 16 的倍数，保证 U-Net 四次下采样后尺寸能对齐。"""
    _, h, w = t.shape
    return F.pad(t, (0, (16 - w % 16) % 16, 0, (16 - h % 16) % 16))


class DRIVEDataset(Dataset):
    """root 指向 processed 目录（含 images/ masks/ fov/），split 为 training 或 test。"""

    def __init__(self, root, split, augment=False, keys=None):
        self.split_dir = Path(root) / split
        all_keys = sorted(p.stem for p in (self.split_dir / "images").glob("*.png"))
        self.keys = all_keys if keys is None else list(keys)
        self.augment = augment

    def __len__(self):
        return len(self.keys)

    def _augment(self, img, mask):
        if random.random() < 0.5:  # 水平翻转
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:  # 垂直翻转
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)  # 随机旋转 90° x k
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = load_color(self.split_dir / "images" / f"{key}.png")
        mask = load_gray(self.split_dir / "masks" / f"{key}.png")
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        mask = (mask > 127).astype(np.float32)
        if self.augment:
            img, mask = self._augment(img, mask)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float()[None]
        return pad16(img), pad16(mask), key
