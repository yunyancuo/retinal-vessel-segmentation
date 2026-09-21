"""训练脚本。

示例：
    python src/train.py --model unet --epochs 100 --batch-size 2
    python src/train.py --model attn_unet --loss tversky
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DRIVEDataset
from losses import make_loss
from models import UNet, AttentionUNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def dice_score(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    return (2 * (pred * target).sum() / (pred.sum() + target.sum()).clamp(min=1.0)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/DRIVE/processed")
    ap.add_argument("--model", choices=["unet", "attn_unet"], default="unet")
    ap.add_argument("--width", type=int, default=32, help="首层通道数，显存不足就调小")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["bce", "dice", "bce_dice", "tversky"], default="bce_dice")
    ap.add_argument("--val-count", type=int, default=2, help="训练集末尾留出几张做验证")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模型: {args.model} | 损失: {args.loss}")

    all_keys = sorted(p.stem for p in (Path(args.data_root) / "training" / "images").glob("*.png"))
    val_keys = all_keys[-args.val_count:] if args.val_count else []
    train_keys = all_keys[:-args.val_count] if args.val_count else all_keys
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
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_dice\n")

    best = -1.0  # 初始 -1 而非 0：未收敛时 val dice 可能为 0，也要能存下权重
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for img, mask, _ in train_loader:
            img, mask = img.to(device), mask.to(device)
            loss = crit(model(img), mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * img.size(0)
        sched.step()

        model.eval()
        dices = []
        with torch.no_grad():
            for img, mask, _ in val_loader:
                dices.append(dice_score(model(img.to(device)).cpu(), mask))
        val_dice = float(np.mean(dices)) if dices else float("nan")
        avg_loss = total / max(len(train_set), 1)
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {avg_loss:.4f} | val dice {val_dice:.4f}")
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.4f},{val_dice:.4f}\n")
        if val_dice > best:
            best = val_dice
            torch.save(
                {"model": args.model, "width": args.width, "state": model.state_dict()},
                ckpt_dir / "best.pth",
            )
    print(f"完成，最佳 val dice {best:.4f}，权重在 {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
