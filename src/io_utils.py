"""跨平台图像读写工具：兼容中文及特殊字符路径，不依赖 PyTorch。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_gray(path: str | Path) -> np.ndarray:
    """以灰度模式读取图像，兼容含中文或空格的路径。

    Args:
        path: 图像文件路径。

    Returns:
        二维 uint8 灰度图像数组，形状为 ``(H, W)``。

    Raises:
        FileNotFoundError: 路径不存在或不是有效文件。
        OSError: 图像解码失败或文件内容损坏。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"图像文件不存在: {file_path}")
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"灰度图像解码失败: {file_path}")
    return img


def load_color(path: str | Path) -> np.ndarray:
    """以 BGR 彩色模式读取图像，兼容含中文或空格的路径。

    Args:
        path: 图像文件路径。

    Returns:
        三维 uint8 BGR 图像数组，形状为 ``(H, W, 3)``。

    Raises:
        FileNotFoundError: 路径不存在或不是有效文件。
        OSError: 图像解码失败或文件内容损坏。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"图像文件不存在: {file_path}")
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"彩色图像解码失败: {file_path}")
    return img


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """将图像写入磁盘，使用 ``cv2.imencode`` 兼容 Windows 中文路径。

    Args:
        path: 目标输出路径，根据扩展名自动决定编码格式（默认 ``.png``）。
        image: 待保存的图像数组（二维灰度或三维 BGR/BGRA）。

    Returns:
        实际写入的 :class:`pathlib.Path` 对象。

    Raises:
        ValueError: 图像数组为空或维度非法。
        OSError: 图像编码或落盘失败。
    """
    arr = np.asarray(image)
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError(f"待保存图像形状非法: {arr.shape}")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, arr)
    if not ok:
        raise OSError(f"图像编码失败（格式 {ext}）: {out_path}")
    encoded.tofile(str(out_path))
    return out_path
