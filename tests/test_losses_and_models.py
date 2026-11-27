"""PyTorch 模块单元测试：losses / models / dataset / train。

当运行环境未安装 PyTorch 时，本测试模块会自动通过 ``pytest.importorskip("torch")`` 跳过；
在 CI 或已装 PyTorch 的本地环境中将全量执行前向传播、奇偶尺寸对齐、损失梯度与数据集测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataset import DRIVEDataset, pad16  # noqa: E402
from io_utils import save_image  # noqa: E402
from losses import bce_dice_loss, bce_loss, dice_loss, make_loss, tversky_loss  # noqa: E402
from models import AttentionGate, AttentionUNet, UNet  # noqa: E402
from train import dice_score, set_seed, split_keys  # noqa: E402


class TestLosses:
    """BCE / Dice / BCE+Dice / Tversky 数值稳定性、形状对齐与梯度测试。"""

    @pytest.mark.parametrize("name", ["bce", "dice", "bce_dice", "tversky"])
    def test_make_loss_and_gradient_flow(self, name: str) -> None:
        crit = make_loss(name)
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = (torch.rand(2, 1, 32, 32) > 0.8).float()
        loss = crit(logits, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_unknown_loss_name_raises(self) -> None:
        with pytest.raises(ValueError):
            make_loss("focal_unknown")

    def test_empty_mask_and_extreme_logits_numerical_stability(self) -> None:
        empty_target = torch.zeros(2, 1, 16, 16)
        neg_logits = torch.full((2, 1, 16, 16), -20.0, requires_grad=True)
        d = dice_loss(neg_logits, empty_target)
        t = tversky_loss(neg_logits, empty_target)
        b = bce_loss(neg_logits, empty_target)
        bd = bce_dice_loss(neg_logits, empty_target)
        for val in (d, t, b, bd):
            assert torch.isfinite(val)
        assert float(d.item()) < 1e-4
        assert float(t.item()) < 1e-4

        # eps=0 全零边界仍不产生 NaN
        assert torch.isfinite(dice_loss(neg_logits, empty_target, eps=0.0))
        assert torch.isfinite(tversky_loss(neg_logits, empty_target, eps=0.0))

    def test_shape_alignment_prevents_silent_broadcasting(self) -> None:
        logits_4d = torch.zeros(3, 1, 16, 16)
        target_3d = torch.ones(3, 16, 16, dtype=torch.uint8)
        assert torch.isfinite(dice_loss(logits_4d, target_3d))
        assert torch.isfinite(bce_dice_loss(logits_4d, target_3d))

        # 明显不匹配的 batch 大小必须显式报错
        with pytest.raises(ValueError):
            dice_loss(torch.zeros(3, 1, 16, 16), torch.zeros(2, 1, 16, 16))

    def test_tversky_penalizes_fn_more_than_fp_when_beta_gt_alpha(self) -> None:
        # 构造相同像素数量的纯漏检（FN）与纯误检（FP）场景
        gt_fn = torch.zeros(1, 1, 16, 16)
        gt_fn[:, :, :4, :4] = 1.0  # 16 个真血管
        gt_fn[:, :, 8:12, 8:12] = 1.0  # 另外 16 个真血管
        logits_fn = torch.full((1, 1, 16, 16), -12.0)
        logits_fn[:, :, :4, :4] = 12.0  # 命中 16，漏检 16

        gt_fp = torch.zeros(1, 1, 16, 16)
        gt_fp[:, :, :4, :4] = 1.0  # 16 个真血管
        logits_fp = torch.full((1, 1, 16, 16), -12.0)
        logits_fp[:, :, :4, :4] = 12.0  # 命中 16
        logits_fp[:, :, 8:12, 8:12] = 12.0  # 误检 16

        loss_fn = tversky_loss(logits_fn, gt_fn, alpha=0.3, beta=0.7)
        loss_fp = tversky_loss(logits_fp, gt_fp, alpha=0.3, beta=0.7)
        assert float(loss_fn.item()) > float(loss_fp.item())

    def test_invalid_hyperparams_raise(self) -> None:
        z = torch.zeros(1, 1, 8, 8)
        with pytest.raises(ValueError):
            dice_loss(z, z, eps=-0.1)
        with pytest.raises(ValueError):
            tversky_loss(z, z, alpha=0.0, beta=0.0)
        with pytest.raises(ValueError):
            tversky_loss(z, z, alpha=-0.2, beta=0.7)


class TestModels:
    """U-Net 与 Attention U-Net 任意奇偶分辨率前向传播与梯度测试。"""

    @pytest.mark.parametrize("model_cls", [UNet, AttentionUNet])
    @pytest.mark.parametrize("hw", [(64, 64), (53, 71), (49, 33), (80, 66)])
    def test_arbitrary_input_sizes_forward_and_backward(
        self,
        model_cls: type[UNet],
        hw: tuple[int, int],
    ) -> None:
        h, w = hw
        model = model_cls(in_ch=3, num_classes=1, width=8)
        x = torch.randn(2, 3, h, w, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 1, h, w)
        assert torch.isfinite(out).all()
        out.mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_attention_gate_mismatched_spatial_size(self) -> None:
        gate = AttentionGate(g_ch=16, s_ch=16, inter=8)
        g = torch.randn(2, 16, 14, 15)
        s = torch.randn(2, 16, 15, 16)
        y = gate(g, s)
        assert y.shape == s.shape

    def test_too_small_or_invalid_ndim_input_raises(self) -> None:
        model = UNet(width=8)
        with pytest.raises(ValueError):
            model(torch.randn(1, 3, 12, 12))
        with pytest.raises(ValueError):
            model(torch.randn(3, 32, 32))


class TestDatasetAndTrainUtils:
    """pad16、DRIVEDataset 与训练辅助函数测试。"""

    def test_pad16_square_alignment(self) -> None:
        t3 = torch.zeros(3, 51, 37)
        p3 = pad16(t3)
        assert p3.shape == (3, 64, 64)

        t4 = torch.zeros(2, 1, 64, 64)
        assert pad16(t4).shape == (2, 1, 64, 64)

    def test_drive_dataset_loading_and_augmentation(self, tmp_path: Path) -> None:
        root = tmp_path / "processed"
        img = np.full((50, 40, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 40), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        save_image(root / "training" / "images" / "01.png", img)
        save_image(root / "training" / "masks" / "01.png", mask)

        ds = DRIVEDataset(root, "training", augment=True)
        assert len(ds) == 1
        set_seed(42)
        img_t, mask_t, key = ds[0]
        assert key == "01"
        assert img_t.shape == (3, 64, 64)
        assert mask_t.shape == (1, 64, 64)
        assert float(mask_t.sum().item()) == 100.0

    def test_split_keys_and_dice_score(self) -> None:
        tr, va = split_keys(["a", "b", "c", "d"], val_count=1)
        assert tr == ["a", "b", "c"] and va == ["d"]

        tr0, va0 = split_keys(["a", "b"], val_count=0)
        assert tr0 == ["a", "b"] and va0 == []

        with pytest.raises(ValueError):
            split_keys(["a", "b"], val_count=2)
        with pytest.raises(ValueError):
            split_keys(["a", "b"], val_count=-1)

        logits = torch.full((1, 1, 8, 8), 10.0)
        target = torch.ones(1, 1, 8, 8)
        assert dice_score(logits, target) == pytest.approx(1.0)
        assert dice_score(torch.full((1, 1, 8, 8), -10.0), torch.zeros(1, 1, 8, 8)) == 0.0
