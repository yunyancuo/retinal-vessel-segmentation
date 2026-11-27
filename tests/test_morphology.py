"""血管形态特征分析（morphology / morphology_batch）离线单元测试。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from io_utils import save_image
from morphology import (
    _zhang_suen_skeletonize,
    analyze,
    count_topology_points,
    fractal_dimension,
    skeletonize_mask,
    tortuosity,
)
from morphology_batch import parse_source_spec, run_batch


def _make_straight_vessel(
    size: int = 128,
    thickness: int = 7,
    touch_border: bool = False,
) -> np.ndarray:
    """合成水平直线血管二值掩膜。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    x0, x1 = (0, size - 1) if touch_border else (16, size - 17)
    cv2.line(mask, (x0, size // 2), (x1, size // 2), 255, thickness=thickness)
    return mask


def _make_branched_vessel(size: int = 160, thickness: int = 6) -> np.ndarray:
    """合成带清晰 T/Y 形分叉的血管二值掩膜。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    # 主干水平血管
    cv2.line(mask, (16, mid), (size - 17, mid), 255, thickness=thickness)
    # 两条上下分支
    cv2.line(mask, (mid, mid), (mid + 36, mid - 44), 255, thickness=max(thickness - 2, 3))
    cv2.line(mask, (mid - 16, mid), (mid - 42, mid + 46), 255, thickness=max(thickness - 2, 3))
    return mask


def _make_wavy_vessel(size: int = 160, thickness: int = 5) -> np.ndarray:
    """合成大幅弯曲的正弦波血管，用于验证迂曲度高于直线血管。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    xs = np.linspace(16, size - 17, 400)
    ys = size / 2.0 + 36.0 * np.sin((xs - 16) / (size - 33) * 4.0 * np.pi)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


class TestMorphologyCore:
    """骨架化、管径估算、迂曲度、拓扑点与分形维数边界测试。"""

    def test_empty_mask_and_empty_fov_do_not_crash_or_warn(self) -> None:
        empty = np.zeros((128, 128), dtype=np.uint8)
        res = analyze(empty)
        assert res["vessel_density"] == 0.0
        assert res["mean_diameter_px"] == 0.0
        assert res["median_diameter_px"] == 0.0
        assert res["total_length_px"] == 0.0
        assert res["tortuosity"] == 0.0
        assert math.isnan(res["fractal_dimension"])
        assert res["endpoints"] == 0
        assert res["branch_points"] == 0
        assert res["widths"].size == 0
        assert not res["skel"].any()

        # 非空掩膜但全黑 FOV 也必须安全返回 0 密度且不产生空切片警告
        vessel = _make_straight_vessel(128, thickness=7)
        empty_fov = np.zeros((128, 128), dtype=bool)
        res_fov = analyze(vessel, fov=empty_fov)
        assert res_fov["vessel_density"] == 0.0
        assert res_fov["total_length_px"] == 0.0
        assert math.isnan(res_fov["fractal_dimension"])

    def test_straight_vessel_diameter_length_and_tortuosity(self) -> None:
        # 精确构造 7px 宽的水平条带（y in [61, 67]），中心行到背景距离恰为 4.0 → 2 * dt ≈ 8.0
        thin = np.zeros((128, 128), dtype=np.uint8)
        thin[62:67, 16:112] = 255  # 5px 宽
        medium = np.zeros((128, 128), dtype=np.uint8)
        medium[61:68, 16:112] = 255  # 7px 宽
        thick = np.zeros((128, 128), dtype=np.uint8)
        thick[59:70, 16:112] = 255  # 11px 宽

        res_thin = analyze(thin)
        res = analyze(medium)
        res_thick = analyze(thick)

        # 距离变换估算管径应精确对应带宽，且随真实管径严格单调递增
        assert 7.0 <= res["mean_diameter_px"] <= 8.5
        assert 7.5 <= res["median_diameter_px"] <= 8.5
        assert (
            res_thin["mean_diameter_px"]
            < res["mean_diameter_px"]
            < res_thick["mean_diameter_px"]
        )
        assert 85.0 <= res["total_length_px"] <= 105.0
        # 直线血管迂曲度接近 1.0
        assert 0.95 <= res["tortuosity"] <= 1.15
        # 直线血管仅 2 个端点、无分叉点
        assert res["endpoints"] == 2
        assert res["branch_points"] == 0
        # 直线段分形维数接近 1.0
        assert 0.85 <= res["fractal_dimension"] <= 1.25

    def test_border_touching_skeleton_preserves_endpoints(self) -> None:
        """回归测试：血管骨架触碰图像边界时不应因镜像填充丢失端点。"""
        skel = np.zeros((64, 64), dtype=bool)
        skel[32, :] = True  # 贯通左右边界的 1px 直线骨架
        endpoints, branch_points = count_topology_points(skel)
        assert endpoints == 2
        assert branch_points == 0
        assert 0.98 <= tortuosity(skel) <= 1.05

    def test_branched_vessel_topology_and_fractal_complexity(self) -> None:
        straight = _make_straight_vessel(160, thickness=6)
        branched = _make_branched_vessel(160, thickness=6)

        res_s = analyze(straight)
        res_b = analyze(branched)

        # 分叉血管应有 >= 4 个末梢端点和 >= 2 个分叉点，且总长度显著更长
        assert res_b["endpoints"] >= 4
        assert res_b["branch_points"] >= 2
        assert res_b["total_length_px"] > res_s["total_length_px"]
        assert 1.0 < res_b["fractal_dimension"] < 2.0

    def test_wavy_vessel_has_higher_tortuosity_than_straight(self) -> None:
        res_s = analyze(_make_straight_vessel(160, thickness=5))
        res_w = analyze(_make_wavy_vessel(160, thickness=5))
        assert res_w["tortuosity"] > res_s["tortuosity"] * 1.4

    def test_closed_loop_skeleton_tortuosity(self) -> None:
        """无端点的闭环骨架应安全退化为外接矩形对角线计算，不除零不崩溃。"""
        loop = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(loop, (48, 48), 24, 255, thickness=1)
        t = tortuosity(loop > 0)
        assert math.isfinite(t) and t > 1.5

    def test_fractal_dimension_small_images_do_not_crash(self) -> None:
        """回归测试：图像边长小于最大盒尺度 64 时不触发 log(0) 或 LinAlgError。"""
        small = np.zeros((24, 24), dtype=bool)
        small[12, 2:22] = True
        fd = fractal_dimension(small)
        assert math.isfinite(fd) and 0.5 <= fd <= 1.6

        tiny = np.ones((2, 2), dtype=bool)
        assert math.isnan(fractal_dimension(tiny))

    def test_zhang_suen_fallback_matches_basic_invariants(self) -> None:
        vessel = _make_branched_vessel(128, thickness=6) > 0
        skel_zs = _zhang_suen_skeletonize(vessel)
        skel_api = skeletonize_mask(vessel)
        assert skel_zs.dtype == bool and skel_zs.shape == vessel.shape
        assert skel_zs.any() and skel_api.any()
        # 骨架必须是原血管掩膜的子集
        assert np.all(vessel[skel_zs])
        assert np.all(vessel[skel_api])

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            analyze(np.zeros((8, 8, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            analyze(np.zeros((16, 16), dtype=np.uint8), fov=np.ones((8, 8), dtype=bool))


class TestMorphologyBatch:
    """批量形态分析 CLI 逻辑与空掩膜聚合测试。"""

    def test_parse_source_spec_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            parse_source_spec("missing_equals_sign")
        with pytest.raises(ValueError):
            parse_source_spec(f"={tmp_path}")
        with pytest.raises(FileNotFoundError):
            parse_source_spec(f"gt={tmp_path / 'nonexistent'}")
        name, p = parse_source_spec(f"gt={tmp_path}")
        assert name == "gt" and p == tmp_path

    def test_run_batch_end_to_end_with_empty_patch(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        fov_dir = tmp_path / "fov"
        out_dir = tmp_path / "out"

        fov = np.full((128, 128), 255, dtype=np.uint8)
        save_image(fov_dir / "s1.png", fov)
        save_image(fov_dir / "s2.png", fov)

        # s1 为真实血管，s2 为纯背景空块（测试 nanmean 安全聚合与不崩溃）
        save_image(gt_dir / "s1.png", _make_branched_vessel(128, thickness=6))
        save_image(gt_dir / "s2.png", np.zeros((128, 128), dtype=np.uint8))
        save_image(pred_dir / "s1_pred.png", _make_straight_vessel(128, thickness=6))
        save_image(pred_dir / "s2_pred.png", np.zeros((128, 128), dtype=np.uint8))

        rows, agg = run_batch(
            sources=[f"金标准={gt_dir}", f"预测={pred_dir}"],
            fov_dir=fov_dir,
            out_dir=out_dir,
        )
        assert len(rows) == 4
        assert set(agg.keys()) == {"金标准", "预测"}
        assert (out_dir / "morphology_all.csv").is_file()
        assert (out_dir / "morphology_mean.csv").is_file()
        assert (out_dir / "diameter_dist.png").is_file()

        with open(out_dir / "morphology_mean.csv", encoding="utf-8") as f:
            mean_rows = list(csv.DictReader(f))
        assert len(mean_rows) == 2
        assert all(math.isfinite(float(r["fractal_dimension"])) for r in mean_rows)
