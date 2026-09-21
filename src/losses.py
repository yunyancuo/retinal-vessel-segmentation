"""分割损失：BCE / Dice / BCE+Dice / Tversky。

血管分割中前景（血管）占比只有 ~12%，单独 BCE 容易偏向预测背景；
Dice/Tversky 直接优化重叠度，Tversky 可通过加权强调漏检（FN）。
"""
import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    """beta > alpha：惩罚漏检（FN）更多，用于提升细血管召回。"""
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(2, 3))
    fp = (p * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - p) * target).sum(dim=(2, 3))
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - t).mean()


def make_loss(name):
    if name == "bce":
        return lambda l, t: F.binary_cross_entropy_with_logits(l, t)
    if name == "dice":
        return dice_loss
    if name == "bce_dice":
        return lambda l, t: F.binary_cross_entropy_with_logits(l, t) + dice_loss(l, t)
    if name == "tversky":
        return tversky_loss
    raise ValueError(f"未知损失: {name}")
