"""分割损失函数：BCE / Dice / BCE+Dice / Tversky。

视网膜血管分割中前景（血管）占比仅约 9%–12%，单独使用 BCE 易偏向保守预测背景；
Dice 与 Tversky 直接优化区域重叠度，Tversky 可通过增大 ``beta`` 加大漏检（FN）惩罚，
从而提升细小血管末梢的召回率。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _align_logits_and_target(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对齐 ``logits`` 与 ``target`` 的形状和数据类型，防止隐式错误广播。

    支持 2D ``(H, W)``、3D ``(B, H, W)`` 与 4D ``(B, C, H, W)`` 输入；
    当 ``logits`` 为单通道 4D ``(B, 1, H, W)`` 而 ``target`` 为 3D ``(B, H, W)`` 时，
    自动为 ``target`` 补齐通道维，避免广播成 ``(B, B, H, W)``。
    """
    if logits.ndim == 4 and target.ndim == 3 and logits.shape[1] == 1:
        target = target.unsqueeze(1)
    elif logits.ndim == 3 and target.ndim == 4 and target.shape[1] == 1:
        logits = logits.unsqueeze(1)
    elif logits.ndim == 2 and target.ndim == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif logits.ndim == 3 and target.ndim == 3:
        logits = logits.unsqueeze(1)
        target = target.unsqueeze(1)

    if logits.shape != target.shape:
        raise ValueError(
            f"logits 与 target 形状不匹配: logits={tuple(logits.shape)}, "
            f"target={tuple(target.shape)}"
        )

    logits_f = logits.float()
    target_f = target.to(device=logits.device, dtype=torch.float32)
    return logits_f, target_f


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """数值稳定的二元交叉熵损失（基于 logits）。"""
    logits_f, target_f = _align_logits_and_target(logits, target)
    return F.binary_cross_entropy_with_logits(logits_f, target_f)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice 损失，逐样本计算空间重叠度后取均值。

    Args:
        logits: 未经过 Sigmoid 的网络输出，常规形状为 ``(B, C, H, W)``。
        target: 二值标注掩膜，值域 ``[0, 1]``。
        eps: 平滑项，防止空掩膜除零；须满足 ``eps >= 0``。

    Returns:
        标量 Dice 损失张量，值域约 ``[0, 1]``。
    """
    if eps < 0:
        raise ValueError(f"eps 不能为负数，收到: {eps}")
    logits_f, target_f = _align_logits_and_target(logits, target)
    probs = torch.sigmoid(logits_f)
    spatial_dims = tuple(range(2, probs.ndim))
    num = 2.0 * (probs * target_f).sum(dim=spatial_dims) + eps
    den = probs.sum(dim=spatial_dims) + target_f.sum(dim=spatial_dims) + eps
    score = num / den.clamp_min(1e-7)
    return 1.0 - score.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1.0,
) -> torch.Tensor:
    """Tversky 损失；当 ``beta > alpha`` 时惩罚漏检（FN）更重，提升细血管召回。

    Args:
        logits: 未经过 Sigmoid 的网络输出，常规形状为 ``(B, C, H, W)``。
        target: 二值标注掩膜，值域 ``[0, 1]``。
        alpha: 假阳性（FP）惩罚权重，须 ``>= 0``。
        beta: 假阴性（FN）惩罚权重，须 ``>= 0``。
        eps: 平滑项，防止空掩膜除零；须 ``>= 0``。

    Returns:
        标量 Tversky 损失张量，值域约 ``[0, 1]``。
    """
    if alpha < 0 or beta < 0 or (alpha == 0 and beta == 0):
        raise ValueError(f"alpha 与 beta 须非负且不能同时为 0，收到: alpha={alpha}, beta={beta}")
    if eps < 0:
        raise ValueError(f"eps 不能为负数，收到: {eps}")
    logits_f, target_f = _align_logits_and_target(logits, target)
    probs = torch.sigmoid(logits_f)
    spatial_dims = tuple(range(2, probs.ndim))
    tp = (probs * target_f).sum(dim=spatial_dims)
    fp = (probs * (1.0 - target_f)).sum(dim=spatial_dims)
    fn = ((1.0 - probs) * target_f).sum(dim=spatial_dims)
    den = (tp + alpha * fp + beta * fn + eps).clamp_min(1e-7)
    score = (tp + eps) / den
    return (1.0 - score).mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """BCE 与 Soft Dice 的复合分割损失。"""
    return bce_loss(logits, target) + dice_loss(logits, target, eps=eps)


def make_loss(name: str) -> LossFn:
    """根据名称构造分割损失函数。

    Args:
        name: 支持 ``"bce"``、``"dice"``、``"bce_dice"``、``"tversky"``。

    Returns:
        可调用的损失函数 ``(logits, target) -> loss``。

    Raises:
        ValueError: 传入未支持的损失名称。
    """
    losses: dict[str, LossFn] = {
        "bce": bce_loss,
        "dice": dice_loss,
        "bce_dice": bce_dice_loss,
        "tversky": tversky_loss,
    }
    if name not in losses:
        raise ValueError(f"未知损失: {name}，可选: {sorted(losses)}")
    return losses[name]
