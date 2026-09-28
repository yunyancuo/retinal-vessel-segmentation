"""画训练曲线：两个模型的 train loss 与 val dice 对比图。

用法：
    python src/plot_curves.py --logs outputs/unet_bce_dice/train_log.csv outputs/attn_unet_tversky/train_log.csv \
        --names "U-Net (BCE+Dice)" "Attention U-Net (Tversky)" --out outputs/figures/train_curves.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt
import numpy as np


def read_log(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    from scipy.ndimage import gaussian_filter1d
    sig = 2.0  # 高斯平滑窗口约 ±2 个轮次，只抹平随机抖动，不改变趋势
    return d["epoch"], gaussian_filter1d(d["train_loss"], sig), gaussian_filter1d(d["val_dice"], sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/train_curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for log, name in zip(args.logs, args.names):
        ep, loss, dice = read_log(log)
        axes[0].plot(ep, loss, label=name)
        axes[1].plot(ep, dice, label=name)
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("训练损失")
    axes[0].set_title("训练损失曲线")
    axes[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.9)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("验证 Dice")
    axes[1].set_title("验证 Dice（留出 2 张）")
    axes[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
