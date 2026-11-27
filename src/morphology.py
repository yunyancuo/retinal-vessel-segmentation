"""血管形态特征分析：密度、直径分布、总长度、迂曲度（简化口径）、分形维数与拓扑关键点。

用法：
    python src/morphology.py --mask outputs/predictions/test/01_pred.png \
        --fov data/DRIVE/processed/test/fov/01.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cn_font  # noqa: F401  注册中文字体
from io_utils import load_gray, save_image

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except ImportError:  # 轻量无 scikit-image 环境回退至纯 NumPy Zhang-Suen 细化
    _skimage_skeletonize = None


def _zhang_suen_skeletonize(binary: np.ndarray) -> np.ndarray:
    """纯 NumPy 向量化 Zhang-Suen 二值细化算法（无 scikit-image 时的精确回退）。"""
    mask = np.asarray(binary, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"骨架化仅支持二维掩膜，收到形状: {mask.shape}")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
    while True:
        changed = False
        for step in (0, 1):
            p1 = padded[1:-1, 1:-1]
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                cond_step = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond_step = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            to_remove = (
                (p1 == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & cond_step
            )
            if np.any(to_remove):
                p1[to_remove] = 0
                changed = True
        if not changed:
            break
    return padded[1:-1, 1:-1].astype(bool)


def skeletonize_mask(binary: np.ndarray) -> np.ndarray:
    """对二维二值血管掩膜执行骨架化。

    环境安装有 ``scikit-image`` 时优先调用 :func:`skimage.morphology.skeletonize`，
    与既有 FIVES 实验保持一致；缺失时自动回退至纯 NumPy Zhang-Suen 细化算法。
    """
    mask = np.asarray(binary, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"骨架化仅支持二维掩膜，收到形状: {mask.shape}")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if _skimage_skeletonize is not None:
        return np.asarray(_skimage_skeletonize(mask), dtype=bool)
    return _zhang_suen_skeletonize(mask)


def fractal_dimension(binary: np.ndarray) -> float:
    """盒计数法（Box-Counting）估算血管分形维数。

    统计多个盒子尺度下的非空盒子数量，对 ``log(N(s)) ~ -D * log(s)`` 线性拟合求斜率。
    当掩膜为空或图像尺寸过小（有效尺度不足 2 个）时安全返回 ``nan``，不产生除零警告。
    """
    arr = np.asarray(binary, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"分形维数仅支持二维掩膜，收到形状: {arr.shape}")
    if not arr.any():
        return float("nan")

    h, w = arr.shape
    min_dim = min(h, w)
    sizes = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    valid_sizes: list[int] = []
    valid_counts: list[int] = []
    for s in sizes:
        if s > min_dim:
            continue
        hh, ww = h - h % s, w - w % s
        if hh == 0 or ww == 0:
            continue
        boxes = arr[:hh, :ww].reshape(hh // s, s, ww // s, s)
        cnt = int(boxes.any(axis=(1, 3)).sum())
        if cnt > 0:
            valid_sizes.append(s)
            valid_counts.append(cnt)

    if len(valid_sizes) < 2:
        return float("nan")
    logs = np.log(np.array(valid_sizes, dtype=np.float64))
    logn = np.log(np.array(valid_counts, dtype=np.float64))
    # N(s) ∝ s^-D，拟合斜率取负才是分形维数
    return -float(np.polyfit(logs, logn, 1)[0])


def count_topology_points(skel: np.ndarray) -> tuple[int, int]:
    """统计血管骨架的端点数（8 邻域邻居数 == 1）与分叉点数（8 邻域邻居数 >= 3）。

    使用 ``cv2.BORDER_CONSTANT`` 零填充边界，避免图像边缘处的血管末梢因镜像反射
    被误判为非端点。

    Returns:
        二元组 ``(endpoints, branch_points)``。
    """
    skel_b = np.asarray(skel, dtype=bool)
    if skel_b.ndim != 2:
        raise ValueError(f"骨架拓扑分析仅支持二维掩膜，收到形状: {skel_b.shape}")
    if not skel_b.any():
        return 0, 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neigh = cv2.filter2D(
        skel_b.astype(np.uint8),
        ddepth=cv2.CV_16S,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    endpoints = int(np.count_nonzero(skel_b & (neigh == 1)))
    branch_points = int(np.count_nonzero(skel_b & (neigh >= 3)))
    return endpoints, branch_points


def _max_pairwise_distance(pts: np.ndarray) -> float:
    """借助凸包精确且高效地计算二维点集的最大欧氏距离（直径）。"""
    if len(pts) < 2:
        return 0.0
    if len(pts) == 2:
        return float(np.hypot(*(pts[0] - pts[1])))
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    candidates = hull if len(hull) >= 2 else pts.astype(np.float32)
    diff = candidates[:, None, :] - candidates[None, :, :]
    return float(np.sqrt(np.sum(diff * diff, axis=-1)).max())


def tortuosity(skel: np.ndarray) -> float:
    """计算长度加权平均迂曲度（简化口径：各骨架连通域弧长 / 最远端点弦长）。

    闭环或无端点的连通域自动退化为外接矩形对角线长度；空骨架安全返回 ``0.0``。
    """
    skel_b = np.asarray(skel, dtype=bool)
    if skel_b.ndim != 2:
        raise ValueError(f"迂曲度计算仅支持二维骨架，收到形状: {skel_b.shape}")
    if not skel_b.any():
        return 0.0

    n, labels = cv2.connectedComponents(skel_b.astype(np.uint8), connectivity=8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    per_comp: list[float] = []
    weights: list[float] = []
    for i in range(1, n):
        comp = labels == i
        arc = float(comp.sum())
        if arc <= 0.0:
            continue
        neigh = cv2.filter2D(
            comp.astype(np.uint8),
            ddepth=cv2.CV_16S,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        pts = np.argwhere(comp & (neigh == 1))  # 端点：仅 1 个 8 邻域骨架邻居
        if len(pts) >= 2:
            chord = _max_pairwise_distance(pts)
        else:
            ys, xs = np.where(comp)
            chord = max(float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min())), 1.0)
        per_comp.append(arc / max(chord, 1.0))
        weights.append(arc)
    return float(np.average(per_comp, weights=weights)) if per_comp else 0.0


def analyze(mask: np.ndarray, fov: np.ndarray | None = None) -> dict[str, Any]:
    """对单张二值血管掩膜执行全套形态特征定量分析。

    Args:
        mask: 二维血管掩膜（非零视为前景血管）。
        fov: 可选二维视野（FOV）布尔或灰度掩膜；提供时仅在视野内统计密度与血管结构。

    Returns:
        包含血管密度、平均/中位管径、总长度、迂曲度、分形维数、端点/分叉点数量、
        骨架处管径数组 ``widths`` 及骨架布尔图 ``skel`` 的字典。
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"血管掩膜必须为二维数组，收到形状: {mask_arr.shape}")
    if mask_arr.size == 0:
        raise ValueError("血管掩膜为空数组")

    m = (mask_arr > 0).copy()
    if fov is not None:
        fov_arr = np.asarray(fov)
        if fov_arr.shape != m.shape:
            raise ValueError(f"FOV 形状 {fov_arr.shape} 与掩膜形状 {m.shape} 不匹配")
        fov_mask = fov_arr > 0
        density = float(m[fov_mask].mean()) if fov_mask.any() else 0.0
        m &= fov_mask
    else:
        density = float(m.mean())

    skel = skeletonize_mask(m)
    if skel.any():
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        widths = (dt[skel] * 2.0).astype(np.float32)  # 骨架处 2 倍距离变换 ≈ 局部血管直径
    else:
        widths = np.empty(0, dtype=np.float32)

    endpoints, branch_points = count_topology_points(skel)
    return {
        "vessel_density": density,
        "mean_diameter_px": float(widths.mean()) if widths.size else 0.0,
        "median_diameter_px": float(np.median(widths)) if widths.size else 0.0,
        "total_length_px": float(skel.sum()),
        "tortuosity": tortuosity(skel),
        "fractal_dimension": fractal_dimension(m),
        "endpoints": endpoints,
        "branch_points": branch_points,
        "widths": widths,
        "skel": skel,
    }


def main() -> None:
    """CLI 入口：单图血管形态分析并保存直径直方图与骨架叠加图。"""
    ap = argparse.ArgumentParser(description="单张眼底血管掩膜的形态特征定量分析")
    ap.add_argument("--mask", required=True, help="二值血管掩膜 png（GT 或模型预测）")
    ap.add_argument("--fov", default=None, help="可选 FOV 掩膜，密度只在视野内统计")
    ap.add_argument("--out-dir", default="outputs/morphology")
    args = ap.parse_args()

    mask = load_gray(args.mask)
    fov = load_gray(args.fov) if args.fov else None
    res = analyze(mask, fov > 127 if fov is not None else None)

    print(f"掩膜: {args.mask}")
    print(f"血管密度      : {res['vessel_density']:.4f}")
    print(f"平均直径 (px) : {res['mean_diameter_px']:.2f}")
    print(f"中位直径 (px) : {res['median_diameter_px']:.2f}")
    print(f"血管总长 (px) : {res['total_length_px']:.0f}")
    print(f"迂曲度(简化)  : {res['tortuosity']:.3f}")
    print(f"分形维数      : {res['fractal_dimension']:.3f}")
    print(f"端点 / 分叉点 : {res['endpoints']} / {res['branch_points']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mask).stem

    fig = plt.figure(figsize=(6, 4))
    if res["widths"].size > 0:
        plt.hist(res["widths"], bins=30, color="#c0504d", edgecolor="white")
    plt.xlabel("血管直径（像素）")
    plt.ylabel("数量")
    plt.title("血管直径分布")
    plt.tight_layout()
    hist_path = out / f"{stem}_diameter_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    vis = np.full((*mask.shape, 3), 30, dtype=np.uint8)
    vis[mask > 0] = (90, 90, 90)
    vis[res["skel"]] = (0, 255, 120)
    skel_path = save_image(out / f"{stem}_skeleton.png", vis)
    print(f"图已保存: {hist_path} 与 {skel_path}")


if __name__ == "__main__":
    main()
