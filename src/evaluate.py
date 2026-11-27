"""测试集评估：Dice / IoU / 敏感性 / 特异性 / 准确率 / AUC + 分割可视化。

指标只在 FOV（视野掩膜）内统计，避免图像黑边虚高准确率与特异性。
核心指标函数 ``seg_metrics``、``compute_auc``、``overlay`` 不依赖 PyTorch，可独立测试复用。

用法：
    python src/evaluate.py --weights outputs/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from io_utils import load_gray, save_image


def load_model(weights: str | Path, device: Any) -> Any:
    """加载训练保存的 checkpoint 权重并返回 ``eval`` 模式模型。"""
    import torch

    from models import AttentionUNet, UNet

    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # 兼容旧版 PyTorch 无 weights_only 参数
        ckpt = torch.load(weights_path, map_location=device)
    model_name = ckpt.get("model")
    if model_name == "unet":
        cls = UNet
    elif model_name == "attn_unet":
        cls = AttentionUNet
    else:
        raise ValueError(f"未知模型类型: {model_name!r}")
    model = cls(width=int(ckpt["width"]))
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """计算二值分割核心指标（Dice / IoU / Sensitivity / Specificity / Accuracy）。

    输入会自动规范为布尔数组，杜绝 ``uint8`` 按位取反陷阱；
    当输入数组为空（0 像素）时安全返回全 ``0.0``；
    平滑项 ``eps=1e-6`` 与既有 FIVES/DRIVE 评测口径完全一致。

    Args:
        pred: 预测二值掩膜（任意形状，非零视为前景）。
        gt: 真值二值掩膜（须与 ``pred`` 形状一致）。

    Returns:
        含 ``dice``、``iou``、``sensitivity``、``specificity``、``accuracy`` 的字典。
    """
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred 形状 {pred_arr.shape} 与 gt 形状 {gt_arr.shape} 不匹配")
    if pred_arr.size == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "accuracy": 0.0,
        }

    pred_b = pred_arr.astype(bool, copy=False)
    gt_b = gt_arr.astype(bool, copy=False)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    tn = float(np.logical_and(~pred_b, ~gt_b).sum())
    eps = 1e-6
    return {
        "dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def _auc_mann_whitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """基于 Mann-Whitney U 平均秩的精确 ROC AUC 计算（与 sklearn 数值等价）。"""
    from scipy.stats import rankdata

    ranks = rankdata(y_score, method="average")
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    pos_rank_sum = float(ranks[y_true].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_auc(gt: np.ndarray, prob: np.ndarray) -> float:
    """安全计算 ROC AUC；空样本或单类别真值时返回 ``nan`` 而非抛出异常。

    Args:
        gt: 真值标签（非零为正类）。
        prob: 连续预测概率/分数（须与 ``gt`` 形状一致）。

    Returns:
        ROC AUC 浮点值；当无正类或无负类时返回 ``nan``。
    """
    gt_arr = np.asarray(gt).ravel()
    prob_arr = np.asarray(prob, dtype=np.float64).ravel()
    if gt_arr.shape != prob_arr.shape:
        raise ValueError(f"gt 元素数 {gt_arr.size} 与 prob 元素数 {prob_arr.size} 不匹配")
    if gt_arr.size == 0 or not np.isfinite(prob_arr).all():
        return float("nan")

    y_true = gt_arr.astype(bool)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == y_true.size:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, prob_arr))
    except ImportError:
        return float(_auc_mann_whitney(y_true, prob_arr))


def overlay(
    base_gray: np.ndarray,
    gt: np.ndarray,
    prob: np.ndarray,
    thr: float,
) -> np.ndarray:
    """生成错误类型着色叠加图（TP 红色、FN 蓝色、FP 绿色，BGR 通道顺序）。"""
    base = np.asarray(base_gray)
    if base.ndim == 2:
        base_u8 = np.clip(base, 0, 255).astype(np.uint8, copy=False)
        out = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)
    elif base.ndim == 3 and base.shape[2] == 3:
        out = np.clip(base, 0, 255).astype(np.uint8, copy=True)
    else:
        raise ValueError(f"base_gray 期望 2D 灰度图或 3 通道 BGR 图，收到: {base.shape}")

    gt_arr = np.asarray(gt)
    prob_arr = np.asarray(prob)
    if gt_arr.shape != out.shape[:2] or prob_arr.shape != out.shape[:2]:
        raise ValueError(
            f"叠加图尺寸不一致: base={out.shape[:2]}, gt={gt_arr.shape}, prob={prob_arr.shape}"
        )

    p = prob_arr > thr
    g = gt_arr > 0.5
    out[p & g] = (0, 0, 255)
    out[~p & g] = (255, 0, 0)
    out[p & ~g] = (0, 255, 0)
    return out


def main() -> None:
    """CLI 入口：在指定划分上执行模型评估、保存预测与误差可视化。"""
    import torch

    from dataset import DRIVEDataset

    ap = argparse.ArgumentParser(description="视网膜血管分割测试集评估与可视化")
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
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for img, mask, key in dataset:  # 逐张推理并在原图 FOV 内统计
            prob = torch.sigmoid(model(img[None].to(device)))[0, 0].cpu().numpy()
            gt = mask[0].numpy()
            fov = load_gray(Path(args.data_root) / args.split / "fov" / f"{key}.png")
            h, w = fov.shape
            prob, gt = prob[:h, :w], gt[:h, :w]
            fovb = fov > 127
            pred = prob > args.threshold
            m: dict[str, Any] = seg_metrics(pred[fovb], gt[fovb] > 0.5)
            m["auc"] = compute_auc(gt[fovb] > 0.5, prob[fovb])
            m["key"] = key
            rows.append(m)
            save_image(save_dir / f"{key}_pred.png", (pred.astype(np.uint8) * 255))
            base = load_gray(Path(args.data_root) / args.split / "images" / f"{key}.png")
            save_image(save_dir / f"{key}_overlay.png", overlay(base, gt, prob, args.threshold))

    report_path = Path(args.save_dir) / "eval_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {args.split} 集结果（mean ± std, n={len(rows)}，阈值 {args.threshold}）===")
    for name in fields[1:]:
        col = np.array([r[name] for r in rows], dtype=float)
        valid = col[np.isfinite(col)]  # 个别纯背景块无血管像素无法算 AUC，安全跳过
        n_valid = int(valid.size)
        if n_valid > 0:
            print(f"{name:>12}: {valid.mean():.4f} ± {valid.std():.4f}  (n={n_valid})")
        else:
            print(f"{name:>12}: nan ± nan  (n=0)")
    print(f"\n逐图明细: {report_path}")
    print(f"预测与可视化: {save_dir}")


if __name__ == "__main__":
    main()
