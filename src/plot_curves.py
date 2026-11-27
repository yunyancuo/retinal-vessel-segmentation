"""绘制训练曲线：多组模型的 train loss 与 val dice 对比图。

用法：
    python src/plot_curves.py \
        --logs outputs/unet_bce_dice/train_log.csv outputs/attn_unet_tversky/train_log.csv \
        --names "U-Net (BCE+Dice)" "Attention U-Net (Tversky)" \
        --out outputs/figures/train_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

import cn_font  # noqa: F401  注册中文字体


def read_log(
    path: str | Path,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取训练日志 CSV 并对训练损失与验证 Dice 做可选高斯平滑。

    兼容仅包含 1 个 epoch 的短日志（自动转换为 1D 数组并跳过平滑）。
    """
    log_path = Path(path)
    if not log_path.is_file():
        raise FileNotFoundError(f"训练日志文件不存在: {log_path}")
    data = np.genfromtxt(log_path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"训练日志文件为空: {log_path}")
    epochs = np.atleast_1d(np.asarray(data["epoch"], dtype=np.float64))
    train_loss = np.atleast_1d(np.asarray(data["train_loss"], dtype=np.float64))
    val_dice = np.atleast_1d(np.asarray(data["val_dice"], dtype=np.float64))
    if epochs.size > 1 and sigma > 0:
        train_loss = gaussian_filter1d(train_loss, sigma)
        val_dice = gaussian_filter1d(val_dice, sigma)
    return epochs, train_loss, val_dice


def main() -> None:
    """CLI 入口：绘制多组训练曲线对比图。"""
    ap = argparse.ArgumentParser(description="绘制训练损失与验证 Dice 曲线对比图")
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/train_curves.png")
    args = ap.parse_args()

    if len(args.logs) != len(args.names):
        raise ValueError(
            f"--logs 数量 ({len(args.logs)}) 与 --names 数量 ({len(args.names)}) 必须一致"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for log, name in zip(args.logs, args.names, strict=True):
        ep, loss, dice = read_log(log)
        axes[0].plot(ep, loss, label=name)
        axes[1].plot(ep, dice, label=name)
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("训练损失")
    axes[0].set_title("训练损失曲线")
    axes[0].legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        framealpha=0.9,
    )
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("验证 Dice")
    axes[1].set_title("验证 Dice 曲线")
    axes[1].legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        framealpha=0.9,
    )
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
