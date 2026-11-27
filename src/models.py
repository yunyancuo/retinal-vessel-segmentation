"""模型定义：U-Net 基线与 Attention U-Net 改进版。

网络输出均为未经过 Sigmoid 的 logits，训练时配合 BCEWithLogits / Dice / Tversky 损失，
推理评估阶段再通过 Sigmoid 映射为概率图。跳跃连接内置空间尺寸自动对齐，
支持任意奇偶分辨率输入，同时保持与既有 ``state_dict`` 权重完全兼容。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _match_spatial_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """将 ``src`` 的空间尺寸 ``(H, W)`` 对齐到 ``ref``。

    当输入分辨率不是 16 的倍数（或含奇数边长）时，``MaxPool2d`` 向下取整会导致
    ``ConvTranspose2d`` 上采样结果比对应编码特征小 1 个像素。此处采用对称零填充
    （或在必要时中心裁剪）实现精确对齐；当尺寸本身匹配时直接零开销返回原张量。
    """
    dh = ref.shape[-2] - src.shape[-2]
    dw = ref.shape[-1] - src.shape[-1]
    if dh == 0 and dw == 0:
        return src
    if dh > 0 or dw > 0:
        pad_top = max(dh, 0) // 2
        pad_bottom = max(dh, 0) - pad_top
        pad_left = max(dw, 0) // 2
        pad_right = max(dw, 0) - pad_left
        src = F.pad(src, (pad_left, pad_right, pad_top, pad_bottom))
    if src.shape[-2] > ref.shape[-2] or src.shape[-1] > ref.shape[-1]:
        start_h = (src.shape[-2] - ref.shape[-2]) // 2
        start_w = (src.shape[-1] - ref.shape[-1]) // 2
        src = src[
            ...,
            start_h : start_h + ref.shape[-2],
            start_w : start_w + ref.shape[-1],
        ]
    return src


class DoubleConv(nn.Sequential):
    """两次 3×3 卷积 + BatchNorm + ReLU 的基础特征提取块。"""

    def __init__(self, cin: int, cout: int) -> None:
        if cin <= 0 or cout <= 0:
            raise ValueError(f"通道数须为正整数，收到: cin={cin}, cout={cout}")
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """经典 U-Net：4 级编码器-解码器 + 跳跃连接，输出分割 logits。"""

    def __init__(self, in_ch: int = 3, num_classes: int = 1, width: int = 32) -> None:
        super().__init__()
        if in_ch <= 0 or num_classes <= 0 or width <= 0:
            raise ValueError(
                f"in_ch / num_classes / width 须为正整数，"
                f"收到: in_ch={in_ch}, num_classes={num_classes}, width={width}"
            )
        w = width
        self.inc = DoubleConv(in_ch, w)
        self.down1 = DoubleConv(w, w * 2)
        self.down2 = DoubleConv(w * 2, w * 4)
        self.down3 = DoubleConv(w * 4, w * 8)
        self.down4 = DoubleConv(w * 8, w * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(w * 16, w * 8, 2, stride=2)
        self.dec4 = DoubleConv(w * 16, w * 8)
        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.dec3 = DoubleConv(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = DoubleConv(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = DoubleConv(w * 2, w)
        self.outc = nn.Conv2d(w, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出与输入同空间尺寸的 logits 张量 ``(B, num_classes, H, W)``。"""
        if x.ndim != 4:
            raise ValueError(f"期望 4D 输入 (B, C, H, W)，收到形状: {tuple(x.shape)}")
        if x.shape[-2] < 16 or x.shape[-1] < 16:
            raise ValueError(
                f"4 级下采样要求输入高宽均至少为 16，收到: {tuple(x.shape[-2:])}"
            )
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        u4 = _match_spatial_size(self.up4(x5), x4)
        x = self.dec4(torch.cat([u4, x4], dim=1))
        u3 = _match_spatial_size(self.up3(x), x3)
        x = self.dec3(torch.cat([u3, x3], dim=1))
        u2 = _match_spatial_size(self.up2(x), x2)
        x = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = _match_spatial_size(self.up1(x), x1)
        x = self.dec1(torch.cat([u1, x1], dim=1))
        return self.outc(x)


class AttentionGate(nn.Module):
    """注意力门：利用深层门控特征 ``g`` 抑制浅层跳跃特征 ``s`` 中的背景干扰。"""

    def __init__(self, g_ch: int, s_ch: int, inter: int) -> None:
        super().__init__()
        if g_ch <= 0 or s_ch <= 0 or inter <= 0:
            raise ValueError(
                f"注意力门通道数须为正整数，收到: g_ch={g_ch}, s_ch={s_ch}, inter={inter}"
            )
        self.theta = nn.Conv2d(g_ch, inter, 1)
        self.phi = nn.Conv2d(s_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """计算空间注意力权重并加权浅层跳跃特征 ``s``。"""
        g_aligned = _match_spatial_size(g, s)
        a = torch.relu(self.theta(g_aligned) + self.phi(s))
        return s * torch.sigmoid(self.psi(a))


class AttentionUNet(UNet):
    """在每级跳跃连接处嵌入 :class:`AttentionGate` 的改进版 U-Net。"""

    def __init__(self, in_ch: int = 3, num_classes: int = 1, width: int = 32) -> None:
        super().__init__(in_ch, num_classes, width)
        w = width
        self.ag4 = AttentionGate(w * 8, w * 8, max(w // 2, 8))
        self.ag3 = AttentionGate(w * 4, w * 4, max(w // 2, 8))
        self.ag2 = AttentionGate(w * 2, w * 2, max(w // 2, 8))
        self.ag1 = AttentionGate(w, w, max(w // 2, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出与输入同空间尺寸的 logits 张量 ``(B, num_classes, H, W)``。"""
        if x.ndim != 4:
            raise ValueError(f"期望 4D 输入 (B, C, H, W)，收到形状: {tuple(x.shape)}")
        if x.shape[-2] < 16 or x.shape[-1] < 16:
            raise ValueError(
                f"4 级下采样要求输入高宽均至少为 16，收到: {tuple(x.shape[-2:])}"
            )
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        g4 = _match_spatial_size(self.up4(x5), x4)
        x = self.dec4(torch.cat([g4, self.ag4(g4, x4)], dim=1))
        g3 = _match_spatial_size(self.up3(x), x3)
        x = self.dec3(torch.cat([g3, self.ag3(g3, x3)], dim=1))
        g2 = _match_spatial_size(self.up2(x), x2)
        x = self.dec2(torch.cat([g2, self.ag2(g2, x2)], dim=1))
        g1 = _match_spatial_size(self.up1(x), x1)
        x = self.dec1(torch.cat([g1, self.ag1(g1, x1)], dim=1))
        return self.outc(x)
