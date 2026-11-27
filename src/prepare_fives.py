"""FIVES 数据准备：parquet → 缩放 1024 → 512×512 裁块 → processed 目录。

输出目录结构与 DRIVE 的 ``preprocess.py`` 完全一致
（``processed/{training,test}/{images,masks,fov}/*.png``），
下游 ``dataset`` / ``train`` / ``evaluate`` / ``morphology`` 脚本可直接复用。

FIVES（Jin et al., 2022）：800 张 2048×2048 彩色眼底图（600 训练 / 200 测试），
图像与血管标注一一对应。统一双三次缩放到 1024 后裁成 2×2 不重叠 512 块
（训练 2400 块 / 测试 800 块）；FOV 掩膜按“灰度 > 10”自动生成。

用法：
    python src/prepare_fives.py --src-dir data_src --out-dir data/FIVES/processed
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from io_utils import save_image
from preprocess import _clahe_green

FULL = 1024  # 缩放后的整图尺寸
PATCH = 512  # 裁块尺寸


def to_bgr(raw_bytes: bytes) -> np.ndarray:
    """将压缩图像字节流安全解码为 BGR 格式的 uint8 数组。"""
    with Image.open(io.BytesIO(raw_bytes)) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clahe_green(bgr: np.ndarray) -> np.ndarray:
    """提取绿色通道并施加 CLAHE 局部对比度增强。"""
    return _clahe_green(bgr)


def make_fov(bgr: np.ndarray) -> np.ndarray:
    """根据灰度阈值与形态学闭运算/中值滤波自动生成视野（FOV）掩膜。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return cv2.medianBlur(mask, 15)


def process_split(shards: Sequence[Path], out_dir: Path) -> tuple[int, int]:
    """还原单个 FIVES 子集 parquet 分片并输出增强裁块与掩膜。

    Returns:
        二元组 ``(全图数量, 生成裁块数量)``。
    """
    for sub in ("images", "masks", "fov"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    n_img = 0
    n_patch = 0
    for sp in shards:
        tbl = pq.read_table(sp, columns=["image", "mask", "image_id"])
        for img, mask, iid in zip(
            tbl.column("image").to_pylist(),
            tbl.column("mask").to_pylist(),
            tbl.column("image_id").to_pylist(),
            strict=True,
        ):
            bgr = to_bgr(img["bytes"])
            with Image.open(io.BytesIO(mask["bytes"])) as mask_im:
                vessel = np.array(mask_im.convert("L"), dtype=np.uint8)
            bgr = cv2.resize(bgr, (FULL, FULL), interpolation=cv2.INTER_CUBIC)
            vessel = cv2.resize(vessel, (FULL, FULL), interpolation=cv2.INTER_NEAREST)
            fov = make_fov(bgr)
            enhanced = clahe_green(bgr)
            key = Path(img["path"] or str(iid)).stem
            for r in range(FULL // PATCH):
                for c in range(FULL // PATCH):
                    sl = np.s_[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
                    pk = f"{key}_r{r}c{c}"
                    save_image(out_dir / "images" / f"{pk}.png", enhanced[sl])
                    save_image(out_dir / "masks" / f"{pk}.png", vessel[sl])
                    save_image(out_dir / "fov" / f"{pk}.png", fov[sl])
                    n_patch += 1
            n_img += 1
            if n_img % 50 == 0:
                print(f"  已处理 {n_img} 张全图", flush=True)
    print(f"{out_dir.parent.name}/{out_dir.name}: {n_img} 张全图 → {n_patch} 个 {PATCH}×{PATCH} 块")
    return n_img, n_patch


def main() -> None:
    """CLI 入口：准备 FIVES 训练集与测试集。"""
    ap = argparse.ArgumentParser(description="FIVES parquet 数据集还原与裁块预处理")
    ap.add_argument("--src-dir", default="data_src")
    ap.add_argument("--out-dir", default="data/FIVES/processed")
    args = ap.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    train_shards = sorted(src.rglob("train-*.parquet"))
    test_shards = sorted(src.rglob("test-*.parquet"))
    if not train_shards or not test_shards:
        raise FileNotFoundError(
            f"{src} 下未同时找到 train-*.parquet 与 test-*.parquet 分片"
        )
    process_split(train_shards, out / "training")  # 与 DRIVE 惯例一致，下游脚本免改
    process_split(test_shards, out / "test")


if __name__ == "__main__":
    main()
