"""FIVES 数据准备：parquet → 缩放 1024 → 512×512 裁块 → processed 目录。

输出目录结构与 DRIVE 的 preprocess.py 完全一致
（processed/{train,test}/{images,masks,fov}/*.png），
下游 dataset/train/evaluate/morphology 全部无需修改。

FIVES（Jin et al., 2022）：800 张 2048×2048 彩色眼底图（600 训练 / 200 测试），
图像与血管标注一一对应。2048 直接训练太慢，统一双三次缩放到 1024 再裁
2×2 不重叠 512 块（训练 2400 块 / 测试 800 块）；
FOV 掩膜按"灰度>10"自动生成（黑色背景不含视野）。

用法：
    python src/prepare_fives.py --src-dir data_src --out-dir data/FIVES/processed
"""
import argparse
import glob
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

FULL = 1024   # 缩放后的整图尺寸
PATCH = 512   # 裁块尺寸


def to_bgr(b):
    rgb = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clahe_green(bgr):
    green = bgr[:, :, 1]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)


def make_fov(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = (gray > 10).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return cv2.medianBlur(m, 15)


def process_split(shards, out_dir):
    for sub in ("images", "masks", "fov"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    n_img = n_patch = 0
    for sp in shards:
        tbl = pq.read_table(sp, columns=["image", "mask", "image_id"])
        for img, mask, iid in zip(tbl.column("image").to_pylist(),
                                  tbl.column("mask").to_pylist(),
                                  tbl.column("image_id").to_pylist()):
            bgr = to_bgr(img["bytes"])
            vessel = np.array(Image.open(io.BytesIO(mask["bytes"])).convert("L"))
            bgr = cv2.resize(bgr, (FULL, FULL), interpolation=cv2.INTER_CUBIC)
            vessel = cv2.resize(vessel, (FULL, FULL), interpolation=cv2.INTER_NEAREST)
            fov = make_fov(bgr)
            enhanced = clahe_green(bgr)
            key = Path(img["path"] or iid).stem
            for r in range(FULL // PATCH):
                for c in range(FULL // PATCH):
                    sl = np.s_[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
                    pk = f"{key}_r{r}c{c}"
                    cv2.imwrite(str(out_dir / "images" / f"{pk}.png"), enhanced[sl])
                    cv2.imwrite(str(out_dir / "masks" / f"{pk}.png"), vessel[sl])
                    cv2.imwrite(str(out_dir / "fov" / f"{pk}.png"), fov[sl])
                    n_patch += 1
            n_img += 1
            if n_img % 50 == 0:
                print(f"  已处理 {n_img} 张全图", flush=True)
    print(f"{out_dir.parent.name}/{out_dir.name}: {n_img} 张全图 → {n_patch} 个 {PATCH}×{PATCH} 块")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="data_src")
    ap.add_argument("--out-dir", default="data/FIVES/processed")
    args = ap.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    train_shards = sorted(src.rglob("train-*.parquet"))
    test_shards = sorted(src.rglob("test-*.parquet"))
    assert train_shards and test_shards, f"{src} 下没找到 parquet"
    process_split(train_shards, out / "training")  # 与 DRIVE 惯例一致，下游脚本免改
    process_split(test_shards, out / "test")


if __name__ == "__main__":
    main()
