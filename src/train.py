"""视网膜血管分割模型训练脚本。

示例：
    python src/train.py --model unet --epochs 100 --batch-size 2
    python src/train.py --model attn_unet --loss tversky
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DRIVEDataset
from losses import make_loss
from models import AttentionUNet, UNet


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子以保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    """计算二值化后的硬 Dice 系数，空前景时通过 ``clamp(min=1.0)`` 防除零。"""
    if logits.ndim == 4 and target.ndim == 3 and logits.shape[1] == 1:
        target = target.unsqueeze(1)
    if logits.shape != target.shape:
        raise ValueError(
            f"dice_score 形状不匹配: logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
    pred = (torch.sigmoid(logits.float()) > thr).float()
    target_f = target.to(device=pred.device, dtype=torch.float32)
    den = (pred.sum() + target_f.sum()).clamp(min=1.0)
    return float((2.0 * (pred * target_f).sum() / den).item())


def split_keys(all_keys: list[str], val_count: int) -> tuple[list[str], list[str]]:
    """从排序后的样本键列表末尾划分验证集，并校验合法边界。"""
    if not all_keys:
        raise FileNotFoundError("训练集图像列表为空，请先运行数据预处理脚本")
    if val_count < 0 or val_count >= len(all_keys):
        raise ValueError(
            f"val_count 须满足 0 <= val_count < 样本总数({len(all_keys)})，收到: {val_count}"
        )
    val_keys = all_keys[-val_count:] if val_count > 0 else []
    train_keys = all_keys[:-val_count] if val_count > 0 else all_keys
    return train_keys, val_keys


def main() -> None:
    """CLI 入口：训练 U-Net 或 Attention U-Net 并保存最优权重与训练日志。"""
    ap = argparse.ArgumentParser(description="视网膜血管分割模型训练")
    ap.add_argument("--data-root", default="data/DRIVE/processed")
    ap.add_argument("--model", choices=["unet", "attn_unet"], default="unet")
    ap.add_argument("--width", type=int, default=32, help="首层通道数，显存不足可调小")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["bce", "dice", "bce_dice", "tversky"], default="bce_dice")
    ap.add_argument("--val-count", type=int, default=2, help="训练集末尾留出几张做验证")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.width <= 0 or args.lr <= 0:
        raise ValueError("epochs、batch_size、width 与 lr 均必须大于 0")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模型: {args.model} | 损失: {args.loss}")

    img_dir = Path(args.data_root) / "training" / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"训练图像目录不存在: {img_dir}")
    all_keys = sorted(p.stem for p in img_dir.glob("*.png"))
    train_keys, val_keys = split_keys(all_keys, args.val_count)

    train_set = DRIVEDataset(args.data_root, "training", augment=True, keys=train_keys)
    val_set = DRIVEDataset(args.data_root, "training", augment=False, keys=val_keys)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0)

    model_cls = UNet if args.model == "unet" else AttentionUNet
    model = model_cls(width=args.width).to(device)
    crit = make_loss(args.loss)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice"])

    has_val = len(val_set) > 0
    best_metric = -1.0  # 有验证集时按最大 val_dice；无验证集时按最小 train_loss（负值越大越好）
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for img, mask, _ in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(img), mask)
            loss.backward()
            opt.step()
            total += float(loss.item()) * img.size(0)
        sched.step()

        model.eval()
        dices: list[float] = []
        with torch.no_grad():
            for img, mask, _ in val_loader:
                dices.append(dice_score(model(img.to(device)).cpu(), mask))
        val_dice = float(np.mean(dices)) if dices else float("nan")
        avg_loss = total / max(len(train_set), 1)
        print(
            f"epoch {epoch:3d}/{args.epochs} | train loss {avg_loss:.4f} | val dice {val_dice:.4f}"
        )
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.4f}", f"{val_dice:.4f}"])

        score = val_dice if has_val else -avg_loss
        if (has_val and score > best_metric) or (
            not has_val and (epoch == 1 or score > best_metric)
        ):
            best_metric = score
            torch.save(
                {"model": args.model, "width": args.width, "state": model.state_dict()},
                ckpt_dir / "best.pth",
            )

    if has_val:
        print(f"完成，最佳 val dice {best_metric:.4f}，权重在 {ckpt_dir / 'best.pth'}")
    else:
        print(f"完成，最低 train loss {-best_metric:.4f}，权重在 {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
