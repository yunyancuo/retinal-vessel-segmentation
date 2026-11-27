"""DRIVE 预处理：提取绿色通道 + CLAHE 局部对比度增强，统一导出 PNG 图像与掩膜。

用法：
    python src/preprocess.py --drive-dir data/DRIVE --out-dir data/DRIVE/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from io_utils import load_color, save_image


def _imread_bgr(path: str | Path) -> np.ndarray:
    """兼容中文路径读取 BGR 彩色眼底图像。"""
    return load_color(path)


def _load_gray(path: str | Path) -> np.ndarray:
    """使用 PIL 读取灰度掩膜（兼容 DRIVE 的 GIF 标注格式）。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"掩膜文件不存在: {file_path}")
    with Image.open(file_path) as im:
        return np.array(im.convert("L"), dtype=np.uint8)


def _to_uint8_channel(channel: np.ndarray) -> np.ndarray:
    """安全将单通道数组规整为 uint8，防范浮点或越界 dtype 导致 CLAHE 报错。"""
    arr = np.asarray(channel)
    if arr.size == 0:
        raise ValueError("输入图像为空数组，无法执行预处理")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        finite_mask = np.isfinite(arr)
        if not finite_mask.all():
            raise ValueError("输入图像包含 NaN 或 Inf，无法执行 CLAHE 增强")
        min_v, max_v = float(arr.min()), float(arr.max())
        if 0.0 <= min_v and max_v <= 1.0:
            return np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
        return np.rint(arr).clip(0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)
    raise TypeError(f"不支持的图像数据类型: {arr.dtype}")


def _clahe_green(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """提取绿色通道并施加 CLAHE 局部对比度增强。

    眼底彩色图像中绿色通道的血管与背景对比度最高；若传入二维灰度图则直接对其增强。

    Args:
        bgr: 输入 BGR/BGRA 彩色图 ``(H, W, 3/4)`` 或灰度图 ``(H, W)``。
        clip_limit: CLAHE 对比度截断阈值，须大于 0。
        tile_grid_size: CLAHE 分块网格尺寸 ``(nx, ny)``，两分量均须大于 0。

    Returns:
        增强后的二维 ``uint8`` 灰度图，形状为 ``(H, W)``。
    """
    arr = np.asarray(bgr)
    if arr.size == 0:
        raise ValueError("输入图像为空，无法执行绿色通道 CLAHE 增强")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit 须大于 0，收到: {clip_limit}")
    if len(tile_grid_size) != 2 or tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError(f"tile_grid_size 须为两个正整数，收到: {tile_grid_size}")

    if arr.ndim == 2:
        green = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        green = arr[:, :, 1]
    else:
        raise ValueError(
            f"期望二维灰度图或至少 3 通道的 BGR 图像，收到形状: {arr.shape}"
        )

    green_u8 = _to_uint8_channel(green)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(green_u8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """将图像归一化至 ``[-1.0, 1.0]`` 的 ``float32`` 张量表示（与训练管线一致）。

    Args:
        image: 输入图像数组（支持 ``uint8`` 或数值范围在 ``[0, 255]`` / ``[0, 1]`` 的浮点图）。

    Returns:
        ``float32`` 数组，值域严格裁剪至 ``[-1.0, 1.0]``。
    """
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("输入图像为空，无法执行归一化")
    if arr.dtype == np.uint8:
        scaled = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("输入图像包含 NaN 或 Inf，无法归一化")
        min_v, max_v = float(arr.min()), float(arr.max())
        scaled = (
            arr.astype(np.float32)
            if (0.0 <= min_v and max_v <= 1.0)
            else np.clip(arr.astype(np.float32), 0.0, 255.0) / 255.0
        )
    else:
        scaled = np.clip(arr.astype(np.float32), 0.0, 255.0) / 255.0
    return ((scaled - 0.5) / 0.5).astype(np.float32, copy=False)


def _find_dir(split_dir: Path, keyword: str) -> Path:
    """在数据子集目录下按关键词大小写无关查找子目录。"""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"数据子集目录不存在: {split_dir}")
    for d in sorted(split_dir.iterdir()):
        if d.is_dir() and keyword in d.name.lower():
            return d
    raise FileNotFoundError(f"{split_dir} 下没有包含 '{keyword}' 的子目录")


def _find_matching_file(directory: Path, prefix: str) -> Path:
    """按样本键前缀稳定查找对应掩膜文件。"""
    matches = sorted(directory.glob(f"{prefix}_*"))
    if not matches:
        raise FileNotFoundError(f"{directory} 下未找到前缀为 '{prefix}_' 的匹配文件")
    return matches[0]


def process_split(split_dir: Path, out_dir: Path) -> int:
    """处理单个 DRIVE 子集（training 或 test）并导出 PNG。

    Returns:
        成功处理的图像数量。
    """
    img_dir = split_dir / "images"
    img_files = sorted(img_dir.glob("*.tif"))
    if not img_files:
        raise FileNotFoundError(f"{img_dir} 下没找到 .tif 图像")
    manual_dir = _find_dir(split_dir, "manual")
    fov_dir = _find_dir(split_dir, "mask")
    for sub in ("images", "masks", "fov"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for f in img_files:
        key = f.name.split("_")[0]
        manual = _find_matching_file(manual_dir, key)
        fov = _find_matching_file(fov_dir, key)
        enhanced = _clahe_green(_imread_bgr(f))
        vessel = (_load_gray(manual) > 0).astype(np.uint8) * 255
        fovm = (_load_gray(fov) > 0).astype(np.uint8) * 255
        save_image(out_dir / "images" / f"{key}.png", enhanced)
        save_image(out_dir / "masks" / f"{key}.png", vessel)
        save_image(out_dir / "fov" / f"{key}.png", fovm)
        print(f"[{split_dir.name}] {key} 完成")
    print(f"{split_dir.name}: 共 {len(img_files)} 张 -> {out_dir}")
    return len(img_files)


def main() -> None:
    """CLI 入口：批量预处理 DRIVE 训练集与测试集。"""
    ap = argparse.ArgumentParser(description="DRIVE 眼底图像绿色通道 CLAHE 预处理")
    ap.add_argument("--drive-dir", default="data/DRIVE")
    ap.add_argument("--out-dir", default="data/DRIVE/processed")
    args = ap.parse_args()
    drive = Path(args.drive_dir)
    for split in ("training", "test"):
        process_split(drive / split, Path(args.out_dir) / split)


if __name__ == "__main__":
    main()
