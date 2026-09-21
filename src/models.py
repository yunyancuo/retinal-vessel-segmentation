"""模型定义：U-Net 基线与 Attention U-Net 改进版。

输出均为 logits（未过 sigmoid），训练用 BCEWithLogits 类损失，评估时再 sigmoid。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """经典 U-Net：4 次下采样 + 跳跃连接，输出单通道分割 logits。"""

    def __init__(self, in_ch=3, num_classes=1, width=32):
        super().__init__()
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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x1], dim=1))
        return self.outc(x)


class AttentionGate(nn.Module):
    """注意力门：用深层特征 g 引导浅层跳跃特征 s 的空间筛选，抑制无关背景响应。"""

    def __init__(self, g_ch, s_ch, inter):
        super().__init__()
        self.theta = nn.Conv2d(g_ch, inter, 1)
        self.phi = nn.Conv2d(s_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, g, s):
        a = torch.relu(self.theta(g) + self.phi(s))
        return s * torch.sigmoid(self.psi(a))


class AttentionUNet(UNet):
    """在每级跳跃连接处插入 AttentionGate 的改进版 U-Net。"""

    def __init__(self, in_ch=3, num_classes=1, width=32):
        super().__init__(in_ch, num_classes, width)
        w = width
        self.ag4 = AttentionGate(w * 8, w * 8, max(w // 2, 8))
        self.ag3 = AttentionGate(w * 4, w * 4, max(w // 2, 8))
        self.ag2 = AttentionGate(w * 2, w * 2, max(w // 2, 8))
        self.ag1 = AttentionGate(w, w, max(w // 2, 8))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        g4 = self.up4(x5)
        x = self.dec4(torch.cat([g4, self.ag4(g4, x4)], dim=1))
        g3 = self.up3(x)
        x = self.dec3(torch.cat([g3, self.ag3(g3, x3)], dim=1))
        g2 = self.up2(x)
        x = self.dec2(torch.cat([g2, self.ag2(g2, x2)], dim=1))
        g1 = self.up1(x)
        x = self.dec1(torch.cat([g1, self.ag1(g1, x1)], dim=1))
        return self.outc(x)
