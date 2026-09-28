"""血管形态特征分析：密度、直径分布、总长度、迂曲度（简化口径）、分形维数。

用法：
    python src/morphology.py --mask outputs/predictions/test/01_pred.png \
        --fov data/DRIVE/processed/test/fov/01.png
"""
import argparse
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

from dataset import load_gray


def fractal_dimension(binary):
    """盒计数法：统计多个盒子尺寸下非空盒子数，log-log 拟合斜率即分形维数。"""
    sizes = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    counts = []
    h, w = binary.shape
    for s in sizes:
        hh, ww = h - h % s, w - w % s
        boxes = binary[:hh, :ww].reshape(hh // s, s, ww // s, s)
        counts.append(boxes.any(axis=(1, 3)).sum())
    logs = np.log(np.array(sizes, dtype=float))
    logn = np.log(np.array(counts, dtype=float))
    return -float(np.polyfit(logs, logn, 1)[0])  # N(s) ∝ s^-D，拟合斜率取负才是维数


def tortuosity(skel):
    """长度加权平均迂曲度（简化口径）：

    每个骨架连通域取 弧长/端点弦长，闭环结构退化为外接框对角线。
    """
    n, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    per_comp, weights = [], []
    for i in range(1, n):
        comp = labels == i
        arc = float(comp.sum())
        if arc == 0:
            continue
        neigh = cv2.filter2D(comp.astype(np.uint8), -1, kernel)
        pts = np.argwhere(comp & (neigh == 1))  # 端点：只有 1 个邻居的骨架点
        if len(pts) >= 2:
            chord = max(float(np.hypot(*(a - b))) for a, b in combinations(pts, 2))
        else:
            ys, xs = np.where(comp)
            chord = max(float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min())), 1.0)
        per_comp.append(arc / max(chord, 1.0))
        weights.append(arc)
    return float(np.average(per_comp, weights=weights)) if per_comp else 0.0


def analyze(mask, fov=None):
    m = mask > 0
    if fov is not None:
        density = float(m[fov].mean())
        m &= fov
    else:
        density = float(m.mean())
    skel = skeletonize(m)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    widths = dt[skel] * 2.0  # 骨架处 2 倍距离变换 ≈ 该处血管直径（像素）
    return {
        "vessel_density": density,
        "mean_diameter_px": float(widths.mean()) if widths.size else 0.0,
        "median_diameter_px": float(np.median(widths)) if widths.size else 0.0,
        "total_length_px": float(skel.sum()),
        "tortuosity": tortuosity(skel),
        "fractal_dimension": fractal_dimension(m),
        "widths": widths,
        "skel": skel,
    }


def main():
    ap = argparse.ArgumentParser()
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mask).stem

    plt.figure(figsize=(6, 4))
    plt.hist(res["widths"], bins=30, color="#c0504d", edgecolor="white")
    plt.xlabel("血管直径（像素）")
    plt.ylabel("数量")
    plt.title("血管直径分布")
    plt.tight_layout()
    plt.savefig(out / f"{stem}_diameter_hist.png", dpi=150)

    vis = np.full((*mask.shape, 3), 30, np.uint8)
    vis[mask > 0] = (90, 90, 90)
    vis[res["skel"]] = (0, 255, 120)
    cv2.imwrite(str(out / f"{stem}_skeleton.png"), vis)
    print(f"图已保存: {out / (stem + '_diameter_hist.png')} 与 {out / (stem + '_skeleton.png')}")


if __name__ == "__main__":
    main()
