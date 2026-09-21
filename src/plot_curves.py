"""画训练曲线：两个模型的 train loss 与 val dice 对比图。

用法：
    python src/plot_curves.py --logs outputs/unet_bce_dice/train_log.csv outputs/attn_unet_tversky/train_log.csv \
        --names "U-Net (BCE+Dice)" "Attention U-Net (Tversky)" --out outputs/figures/train_curves.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_log(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["epoch"], d["train_loss"], d["val_dice"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/train_curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for log, name in zip(args.logs, args.names):
        ep, loss, dice = read_log(log)
        axes[0].plot(ep, loss, label=name)
        axes[1].plot(ep, dice, label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("Training loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val dice")
    axes[1].set_title("Validation Dice (2 held-out images)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
