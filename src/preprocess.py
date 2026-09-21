"""DRIVE 预处理：取绿色通道 + CLAHE 对比度增强，掩膜统一导出 PNG。

用法：
    python src/preprocess.py --drive-dir data/DRIVE --out-dir data/DRIVE/processed
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _imread_bgr(path: Path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"图像读取失败: {path}")
    return img


def _load_gray(path: Path):
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def _clahe_green(bgr):
    """绿色通道的血管/背景对比度最高，再做 CLAHE 局部对比度增强。"""
    green = bgr[:, :, 1]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)


def _find_dir(split_dir: Path, keyword: str) -> Path:
    for d in sorted(split_dir.iterdir()):
        if d.is_dir() and keyword in d.name.lower():
            return d
    raise FileNotFoundError(f"{split_dir} 下没有包含 '{keyword}' 的子目录")


def process_split(split_dir: Path, out_dir: Path):
    img_files = sorted((split_dir / "images").glob("*.tif"))
    if not img_files:
        raise FileNotFoundError(f"{split_dir / 'images'} 下没找到 .tif 图像")
    manual_dir = _find_dir(split_dir, "manual")
    fov_dir = _find_dir(split_dir, "mask")
    for sub in ("images", "masks", "fov"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for f in img_files:
        key = f.name.split("_")[0]
        manual = next(manual_dir.glob(f"{key}_*"))
        fov = next(fov_dir.glob(f"{key}_*"))
        enhanced = _clahe_green(_imread_bgr(f))
        vessel = (_load_gray(manual) > 0).astype(np.uint8) * 255
        fovm = (_load_gray(fov) > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / "images" / f"{key}.png"), enhanced)
        cv2.imwrite(str(out_dir / "masks" / f"{key}.png"), vessel)
        cv2.imwrite(str(out_dir / "fov" / f"{key}.png"), fovm)
        print(f"[{split_dir.name}] {key} 完成")
    print(f"{split_dir.name}: 共 {len(img_files)} 张 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-dir", default="data/DRIVE")
    ap.add_argument("--out-dir", default="data/DRIVE/processed")
    args = ap.parse_args()
    drive = Path(args.drive_dir)
    for split in ("training", "test"):
        process_split(drive / split, Path(args.out_dir) / split)


if __name__ == "__main__":
    main()
