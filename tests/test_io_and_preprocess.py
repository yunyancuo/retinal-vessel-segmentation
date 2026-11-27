"""图像读写（io_utils）与预处理管线（preprocess / prepare_fives）单元测试。"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from io_utils import load_color, load_gray, save_image
from prepare_fives import clahe_green, make_fov, to_bgr
from prepare_fives import process_split as process_fives_split
from preprocess import _clahe_green, normalize_image, process_split


def _make_synthetic_fundus(height: int = 128, width: int = 128) -> np.ndarray:
    """构造带暗角与血管纹理的合成 BGR 眼底图像（低对比度绿色通道）。"""
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height / 2.0, width / 2.0
    radius = min(height, width) * 0.44
    fov = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    # 模拟眼底红通道偏亮、绿通道含清晰血管暗线、蓝通道偏暗的特征
    bgr[:, :, 2] = np.where(fov, 180, 2)
    green = np.where(fov, 115, 2).astype(np.uint8)
    # 在绿色通道画几条稍暗的血管
    cv2.line(green, (24, 64), (104, 64), 85, thickness=4)
    cv2.line(green, (64, 24), (64, 104), 88, thickness=3)
    bgr[:, :, 1] = green
    bgr[:, :, 0] = np.where(fov, 40, 2)
    return bgr


class TestIOUtils:
    """跨平台中文路径读写与异常处理测试。"""

    def test_save_and_load_unicode_path(self, tmp_path: Path) -> None:
        cn_dir = tmp_path / "眼底样本_测试"
        gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
        color = _make_synthetic_fundus(32, 32)

        gray_path = save_image(cn_dir / "灰度掩膜.png", gray)
        color_path = save_image(cn_dir / "彩色眼底.png", color)

        assert np.array_equal(load_gray(gray_path), gray)
        assert np.array_equal(load_color(color_path), color)

    def test_missing_or_corrupted_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gray(tmp_path / "missing.png")
        with pytest.raises(FileNotFoundError):
            load_color(tmp_path / "missing.png")

        bad = tmp_path / "broken.png"
        bad.write_bytes(b"not-a-valid-png")
        with pytest.raises(OSError):
            load_gray(bad)
        with pytest.raises(OSError):
            load_color(bad)

    def test_save_invalid_array_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            save_image(tmp_path / "empty.png", np.empty((0, 0), dtype=np.uint8))
        with pytest.raises(ValueError):
            save_image(tmp_path / "1d.png", np.zeros(10, dtype=np.uint8))


class TestPreprocess:
    """绿色通道 CLAHE、归一化与 DRIVE 预处理目录流测试。"""

    def test_clahe_green_enhances_contrast_and_preserves_shape(self) -> None:
        bgr = _make_synthetic_fundus(128, 128)
        enhanced = _clahe_green(bgr)
        assert enhanced.shape == (128, 128)
        assert enhanced.dtype == np.uint8
        # CLAHE 应保持血管区域仍暗于视野背景，且动态范围不低于原绿通道
        assert int(enhanced.max()) - int(enhanced.min()) >= int(bgr[:, :, 1].max()) - int(
            bgr[:, :, 1].min()
        )

    def test_clahe_green_accepts_gray_and_float_inputs(self) -> None:
        gray_u8 = _make_synthetic_fundus(64, 64)[:, :, 1]
        gray_f01 = gray_u8.astype(np.float32) / 255.0
        gray_f255 = gray_u8.astype(np.float64)

        out_u8 = _clahe_green(gray_u8)
        out_f01 = _clahe_green(gray_f01)
        out_f255 = _clahe_green(gray_f255)

        assert out_u8.dtype == np.uint8
        assert np.array_equal(out_u8, out_f01)
        assert np.array_equal(out_u8, out_f255)

    def test_clahe_green_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            _clahe_green(np.empty((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            _clahe_green(np.zeros((16, 16, 2), dtype=np.uint8))
        with pytest.raises(ValueError):
            _clahe_green(np.full((16, 16), np.nan, dtype=np.float32))
        with pytest.raises(ValueError):
            _clahe_green(np.zeros((16, 16), dtype=np.uint8), clip_limit=0.0)
        with pytest.raises(ValueError):
            _clahe_green(np.zeros((16, 16), dtype=np.uint8), tile_grid_size=(0, 8))

    def test_normalize_image_bounds_and_dtype(self) -> None:
        arr_u8 = np.array([[0, 127, 255]], dtype=np.uint8)
        norm_u8 = normalize_image(arr_u8)
        assert norm_u8.dtype == np.float32
        assert np.isclose(norm_u8[0, 0], -1.0)
        assert np.isclose(norm_u8[0, 2], 1.0)
        assert -1.0 <= float(norm_u8.min()) <= float(norm_u8.max()) <= 1.0

        arr_f01 = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        norm_f01 = normalize_image(arr_f01)
        assert norm_f01.dtype == np.float32
        assert np.allclose(norm_f01, [[-1.0, 0.0, 1.0]])

        with pytest.raises(ValueError):
            normalize_image(np.array([np.inf], dtype=np.float32))

    def test_drive_process_split_end_to_end(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "DRIVE" / "training"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "1st_manual").mkdir(parents=True)
        (split_dir / "mask").mkdir(parents=True)

        bgr = _make_synthetic_fundus(64, 64)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(split_dir / "images" / "21_training.tif")

        vessel = np.zeros((64, 64), dtype=np.uint8)
        vessel[30:34, 10:54] = 255
        Image.fromarray(vessel).save(split_dir / "1st_manual" / "21_manual1.gif")

        fov = np.full((64, 64), 255, dtype=np.uint8)
        Image.fromarray(fov).save(split_dir / "mask" / "21_training_mask.gif")

        out_dir = tmp_path / "processed" / "training"
        count = process_split(split_dir, out_dir)
        assert count == 1
        assert load_gray(out_dir / "images" / "21.png").shape == (64, 64)
        assert set(np.unique(load_gray(out_dir / "masks" / "21.png"))).issubset({0, 255})
        assert np.all(load_gray(out_dir / "fov" / "21.png") == 255)


class TestPrepareFives:
    """FIVES parquet 还原与 2×2 裁块预处理测试。"""

    def test_make_fov_and_to_bgr(self) -> None:
        bgr = _make_synthetic_fundus(128, 128)
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        decoded = to_bgr(buf.getvalue())
        assert np.array_equal(decoded, bgr)

        fov = make_fov(bgr)
        assert fov.shape == (128, 128)
        assert fov[64, 64] == 255  # 视野中心为亮区
        assert fov[0, 0] == 0  # 四角黑边为视野外
        assert clahe_green(bgr).dtype == np.uint8

    def test_process_fives_parquet_split(self, tmp_path: Path) -> None:
        bgr = _make_synthetic_fundus(256, 256)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(mask, (40, 128), (216, 128), 255, thickness=6)

        img_buf, mask_buf = io.BytesIO(), io.BytesIO()
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(img_buf, format="PNG")
        Image.fromarray(mask).save(mask_buf, format="PNG")

        table = pa.table(
            {
                "image": [{"bytes": img_buf.getvalue(), "path": "sample_001.png"}],
                "mask": [{"bytes": mask_buf.getvalue(), "path": "sample_001_gt.png"}],
                "image_id": ["sample_001"],
            }
        )
        shard_path = tmp_path / "test-00000.parquet"
        pq.write_table(table, shard_path)

        out_dir = tmp_path / "processed" / "test"
        n_img, n_patch = process_fives_split([shard_path], out_dir)
        assert n_img == 1
        assert n_patch == 4
        for r in range(2):
            for c in range(2):
                pk = f"sample_001_r{r}c{c}"
                assert load_gray(out_dir / "images" / f"{pk}.png").shape == (512, 512)
                assert load_gray(out_dir / "masks" / f"{pk}.png").shape == (512, 512)
                assert load_gray(out_dir / "fov" / f"{pk}.png").shape == (512, 512)
