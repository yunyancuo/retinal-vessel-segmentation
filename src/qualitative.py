"""定性对比图：每个测试图一行（增强图 / 金标准 / 各模型预测），供报告使用。

用法：
    python src/qualitative.py --keys 01 05 12 \
        --preds outputs/unet_bce_dice/predictions/test outputs/attn_unet_tversky/predictions/test \
        --names "U-Net" "Attention U-Net" --out outputs/figures/qualitative.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt

from dataset import load_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/DRIVE/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/qualitative.png")
    args = ap.parse_args()

    ncol = 2 + len(args.preds)
    fig, axes = plt.subplots(len(args.keys), ncol, figsize=(2.6 * ncol, 2.6 * len(args.keys)))
    if len(args.keys) == 1:
        axes = axes[None]
    for r, key in enumerate(args.keys):
        img = load_gray(Path(args.data_root) / args.split / "images" / f"{key}.png")
        gt = load_gray(Path(args.data_root) / args.split / "masks" / f"{key}.png")
        row = [img, gt] + [load_gray(Path(p) / f"{key}_pred.png") for p in args.preds]
        titles = ["增强图", "金标准"] + list(args.names)
        for ax, im, t in zip(axes[r], row, titles):
            ax.imshow(im, cmap="gray")
            ax.set_title(t, fontsize=10)
            ax.axis("off")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
