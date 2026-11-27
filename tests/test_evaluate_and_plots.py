"""分割评价指标（evaluate）与绘图工具（plot_curves / cn_font）离线单元测试。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import cn_font  # noqa: F401
import plot_curves
import qualitative
from evaluate import compute_auc, overlay, seg_metrics
from io_utils import save_image
from plot_curves import read_log


class TestSegMetrics:
    """Dice / IoU / Sensitivity / Specificity / Accuracy 边界与类型安全性测试。"""

    def test_perfect_and_all_wrong_predictions(self) -> None:
        gt = np.array([True, True, False, False, False])
        m_perfect = seg_metrics(gt, gt)
        assert m_perfect["dice"] == pytest.approx(1.0, abs=1e-5)
        assert m_perfect["iou"] == pytest.approx(1.0, abs=1e-5)
        assert m_perfect["sensitivity"] == pytest.approx(1.0, abs=1e-5)
        assert m_perfect["specificity"] == pytest.approx(1.0, abs=1e-5)
        assert m_perfect["accuracy"] == pytest.approx(1.0, abs=1e-5)

        m_wrong = seg_metrics(~gt, gt)
        assert m_wrong["dice"] == pytest.approx(0.0, abs=1e-6)
        assert m_wrong["iou"] == pytest.approx(0.0, abs=1e-6)
        assert m_wrong["sensitivity"] == pytest.approx(0.0, abs=1e-6)
        assert m_wrong["specificity"] == pytest.approx(0.0, abs=1e-6)
        assert m_wrong["accuracy"] == pytest.approx(0.0, abs=1e-6)

    def test_exact_confusion_matrix_values(self) -> None:
        # TP=2, FP=1, FN=1, TN=4
        gt = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        pred = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
        m = seg_metrics(pred, gt)
        assert m["dice"] == pytest.approx(4.0 / 6.0, abs=1e-5)
        assert m["iou"] == pytest.approx(2.0 / 4.0, abs=1e-5)
        assert m["sensitivity"] == pytest.approx(2.0 / 3.0, abs=1e-5)
        assert m["specificity"] == pytest.approx(4.0 / 5.0, abs=1e-5)
        assert m["accuracy"] == pytest.approx(6.0 / 8.0, abs=1e-5)

    def test_uint8_non_bool_inputs_avoid_bitwise_invert_bug(self) -> None:
        """回归测试：传入 uint8 (0/255 或 0/1) 不应触发 ~uint8 按位取反错误。"""
        gt_u8 = np.array([255, 255, 0, 0], dtype=np.uint8)
        pred_u8 = np.array([1, 1, 0, 0], dtype=np.uint8)
        m = seg_metrics(pred_u8, gt_u8)
        assert m["dice"] == pytest.approx(1.0, abs=1e-5)
        assert m["specificity"] == pytest.approx(1.0, abs=1e-5)
        assert m["accuracy"] == pytest.approx(1.0, abs=1e-5)

    def test_empty_label_and_empty_array_boundaries(self) -> None:
        # 空前景（全 0 真值 + 全 0 预测）
        zeros = np.zeros((16, 16), dtype=bool)
        m_zero = seg_metrics(zeros, zeros)
        assert m_zero["dice"] == pytest.approx(0.0, abs=1e-6)
        assert m_zero["specificity"] == pytest.approx(1.0, abs=1e-5)
        assert m_zero["accuracy"] == pytest.approx(1.0, abs=1e-5)

        # 0 元素数组（例如空 FOV 切片）
        empty = np.array([], dtype=bool)
        m_empty = seg_metrics(empty, empty)
        assert all(v == 0.0 for v in m_empty.values())

        # 形状不匹配报错
        with pytest.raises(ValueError):
            seg_metrics(np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool))


class TestComputeAUCAndOverlay:
    """ROC AUC 单类别边界、ties 精度与误差着色图测试。"""

    def test_auc_perfect_inverted_and_ties(self) -> None:
        gt = np.array([0, 0, 1, 1], dtype=bool)
        assert compute_auc(gt, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
        assert compute_auc(gt, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
        assert compute_auc(gt, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)

        # 含并列分数的解析手算验证：
        # gt = [0, 0, 1, 1], score = [0.1, 0.5, 0.5, 0.9]
        # 秩 = [1, 2.5, 2.5, 4]，正类秩和 = 6.5 -> (6.5 - 3) / 4 = 0.875
        assert compute_auc(gt, np.array([0.1, 0.5, 0.5, 0.9])) == pytest.approx(0.875, abs=1e-12)

    def test_auc_single_class_and_empty_return_nan(self) -> None:
        assert math.isnan(compute_auc(np.zeros(10, dtype=bool), np.linspace(0, 1, 10)))
        assert math.isnan(compute_auc(np.ones(10, dtype=bool), np.linspace(0, 1, 10)))
        assert math.isnan(compute_auc(np.array([], dtype=bool), np.array([], dtype=float)))
        assert math.isnan(compute_auc(np.array([0, 1], dtype=bool), np.array([0.2, np.nan])))

    def test_overlay_color_coding(self) -> None:
        base = np.full((2, 2), 100, dtype=np.uint8)
        gt = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        prob = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)

        vis = overlay(base, gt, prob, thr=0.5)
        assert vis.shape == (2, 2, 3)
        assert tuple(vis[0, 0]) == (0, 0, 255)  # TP -> 红
        assert tuple(vis[0, 1]) == (255, 0, 0)  # FN -> 蓝
        assert tuple(vis[1, 0]) == (0, 255, 0)  # FP -> 绿
        assert tuple(vis[1, 1]) == (100, 100, 100)  # TN -> 原灰度

        with pytest.raises(ValueError):
            overlay(base, np.zeros((3, 3)), prob, thr=0.5)


class TestPlotCurvesAndFont:
    """训练日志读取（含单行日志边界）与中文字体配置测试。"""

    def test_read_log_single_and_multi_epoch(self, tmp_path: Path) -> None:
        single_csv = tmp_path / "single.csv"
        single_csv.write_text("epoch,train_loss,val_dice\n1,0.4500,0.8000\n", encoding="utf-8")
        ep, loss, dice = read_log(single_csv)
        assert ep.shape == (1,)
        assert loss[0] == pytest.approx(0.45)
        assert dice[0] == pytest.approx(0.80)

        # 读取仓库既有真实 FIVES 训练日志验证端到端兼容性
        real_log = (
            Path(__file__).resolve().parent.parent
            / "outputs_fives"
            / "unet_bce_dice"
            / "train_log.csv"
        )
        if real_log.is_file():
            ep_r, loss_r, dice_r = read_log(real_log)
            assert len(ep_r) == 15
            assert np.all(np.isfinite(loss_r)) and np.all(np.isfinite(dice_r))

    def test_cn_font_disables_unicode_minus(self) -> None:
        assert matplotlib.rcParams["axes.unicode_minus"] is False

    def test_plot_curves_and_qualitative_cli(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        log_csv = tmp_path / "train_log.csv"
        log_csv.write_text(
            "epoch,train_loss,val_dice\n1,0.50,0.75\n2,0.32,0.84\n3,0.25,0.88\n",
            encoding="utf-8",
        )
        curve_out = tmp_path / "curves.png"
        monkeypatch.setattr(
            sys,
            "argv",
            ["plot_curves.py", "--logs", str(log_csv), "--names", "U-Net", "--out", str(curve_out)],
        )
        plot_curves.main()
        assert curve_out.is_file()

        # logs 与 names 数量不匹配应显式报错
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "plot_curves.py",
                "--logs",
                str(log_csv),
                "--names",
                "A",
                "B",
                "--out",
                str(curve_out),
            ],
        )
        with pytest.raises(ValueError):
            plot_curves.main()

        # qualitative 端到端拼图测试
        data_root = tmp_path / "processed"
        pred_dir = tmp_path / "preds"
        save_image(data_root / "test" / "images" / "k1.png", np.full((32, 32), 120, dtype=np.uint8))
        save_image(data_root / "test" / "masks" / "k1.png", np.full((32, 32), 255, dtype=np.uint8))
        save_image(pred_dir / "k1_pred.png", np.full((32, 32), 255, dtype=np.uint8))
        qual_out = tmp_path / "qual.png"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "qualitative.py",
                "--data-root",
                str(data_root),
                "--keys",
                "k1",
                "--preds",
                str(pred_dir),
                "--names",
                "U-Net",
                "--out",
                str(qual_out),
            ],
        )
        qualitative.main()
        assert qual_out.is_file()
