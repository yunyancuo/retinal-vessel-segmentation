"""测试集评估：Dice / IoU / 敏感性 / 特异性 / 准确率 / AUC + 分割可视化。

指标只在 FOV（视野掩膜）内统计，避免图像黑边拉高准确率。
用法：
    python src/evaluate.py --weights outputs/checkpoints/best.pth
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import DRIVEDataset, load_gray
from models import UNet, AttentionUNet


def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device)
    cls = UNet if ckpt["model"] == "unet" else AttentionUNet
    model = cls(width=ckpt["width"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()


def seg_metrics(pred, gt):
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    eps = 1e-6
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def overlay(base_gray, gt, prob, thr):
    """TP 红色、FN 蓝色、FP 绿色，方便在报告里分析错误类型。"""
    out = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    p = prob > thr
    g = gt > 0.5
    out[p & g] = (0, 0, 255)
    out[~p & g] = (255, 0, 0)
    out[p & ~g] = (0, 255, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/checkpoints/best.pth")
    ap.add_argument("--data-root", default="data/DRIVE/processed")
    ap.add_argument("--split", choices=["training", "test"], default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-dir", default="outputs/predictions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    dataset = DRIVEDataset(args.data_root, args.split)
    save_dir = Path(args.save_dir) / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    fields = ["key", "dice", "iou", "sensitivity", "specificity", "accuracy", "auc"]
    rows = []
    with torch.no_grad():
        for img, mask, key in dataset:  # 逐张处理
            prob = torch.sigmoid(model(img[None].to(device)))[0, 0].cpu().numpy()
            gt = mask[0].numpy()
            fov = load_gray(Path(args.data_root) / args.split / "fov" / f"{key}.png")
            h, w = fov.shape
            prob, gt = prob[:h, :w], gt[:h, :w]
            fovb = fov > 127
            pred = prob > args.threshold
            m = seg_metrics(pred[fovb], gt[fovb] > 0.5)
            m["auc"] = float(roc_auc_score(gt[fovb] > 0.5, prob[fovb]))
            m["key"] = key
            rows.append(m)
            cv2.imwrite(str(save_dir / f"{key}_pred.png"), (pred * 255).astype(np.uint8))
            base = load_gray(Path(args.data_root) / args.split / "images" / f"{key}.png")
            cv2.imwrite(str(save_dir / f"{key}_overlay.png"),
                        overlay(base, gt, prob, args.threshold))

    with open(Path(args.save_dir) / "eval_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {args.split} 集结果（mean ± std, n={len(rows)}，阈值 {args.threshold}）===")
    for name in fields[1:]:
        col = np.array([r[name] for r in rows], dtype=float)
        n_valid = int(np.isfinite(col).sum())
        col = col[np.isfinite(col)]  # 个别块无血管像素算不了 AUC，跳过
        print(f"{name:>12}: {col.mean():.4f} ± {col.std():.4f}  (n={n_valid})")
    print(f"\n逐图明细: {Path(args.save_dir) / 'eval_report.csv'}")
    print(f"预测与可视化: {save_dir}")


if __name__ == "__main__":
    main()
