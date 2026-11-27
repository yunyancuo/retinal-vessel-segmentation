"""批量血管形态分析：对金标准与各模型预测逐图计算形态特征，输出汇总 CSV、
均值对比表和直径分布对比图（报告用）。

用法：
    python src/morphology_batch.py \
        --sources gt=data/DRIVE/processed/test/masks \
                  unet=outputs/unet_bce_dice/predictions/test \
                  attn=outputs/attn_unet_tversky/predictions/test \
        --fov-dir data/DRIVE/processed/test/fov --out-dir outputs/morphology
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cn_font  # noqa: F401  注册中文字体
from io_utils import load_gray
from morphology import analyze

FIELDS = [
    "source",
    "key",
    "vessel_density",
    "mean_diameter_px",
    "median_diameter_px",
    "total_length_px",
    "tortuosity",
    "fractal_dimension",
]


def _safe_nanmean(values: np.ndarray) -> float:
    """安全计算忽略 NaN 的均值，全为 NaN 或空数组时返回 ``nan`` 且不发告警。"""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def parse_source_spec(spec: str) -> tuple[str, Path]:
    """解析 ``name=directory`` 格式的数据源参数。"""
    if "=" not in spec:
        raise ValueError(f"--sources 参数格式应为 '名称=目录'，收到: {spec!r}")
    name, dir_str = spec.split("=", 1)
    name = name.strip()
    dir_path = Path(dir_str.strip())
    if not name:
        raise ValueError(f"--sources 数据源名称不能为空: {spec!r}")
    if not dir_path.is_dir():
        raise FileNotFoundError(f"数据源目录不存在: {dir_path}")
    return name, dir_path


def run_batch(
    sources: Sequence[str],
    fov_dir: str | Path,
    out_dir: str | Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, np.ndarray]]]:
    """批量执行多组血管掩膜的形态定量分析并导出 CSV 与分布图。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fov_root = Path(fov_dir)
    if not fov_root.is_dir():
        raise FileNotFoundError(f"FOV 目录不存在: {fov_root}")

    rows: list[dict[str, str]] = []
    widths_by_source: dict[str, list[np.ndarray]] = {}
    for spec in sources:
        name, d = parse_source_spec(spec)
        preds = sorted(d.glob("*_pred.png"))
        if preds:  # 模型预测目录：去掉 _pred 后缀
            keys = [p.stem[:-5] for p in preds]
        else:  # 金标准目录：直接用文件名
            keys = sorted(p.stem for p in d.glob("*.png"))
        widths_by_source.setdefault(name, [])
        for key in keys:
            pred_candidate = d / f"{key}_pred.png"
            mask_path = pred_candidate if pred_candidate.exists() else d / f"{key}.png"
            fov = load_gray(fov_root / f"{key}.png") > 127
            res = analyze(load_gray(mask_path), fov)
            rows.append(
                {
                    "source": name,
                    "key": key,
                    "vessel_density": f"{res['vessel_density']:.4f}",
                    "mean_diameter_px": f"{res['mean_diameter_px']:.2f}",
                    "median_diameter_px": f"{res['median_diameter_px']:.2f}",
                    "total_length_px": f"{res['total_length_px']:.0f}",
                    "tortuosity": f"{res['tortuosity']:.3f}",
                    "fractal_dimension": f"{res['fractal_dimension']:.3f}",
                }
            )
            widths_by_source[name].append(res["widths"])
        print(f"{name}: {len(keys)} 张")

    with open(out / "morphology_all.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== 形态特征均值（{len(widths_by_source)} 组）===")
    header = ["source"] + FIELDS[2:]
    print(",".join(header))
    agg: dict[str, dict[str, np.ndarray]] = {}
    for name in widths_by_source:
        sub = [r for r in rows if r["source"] == name]
        vals = {
            k: np.array([float(r[k]) for r in sub], dtype=np.float64)
            for k in FIELDS[2:]
        }
        agg[name] = vals
        # 个别纯背景块的分形维数为 nan，使用安全 nanmean 聚合
        print(",".join([name] + [f"{_safe_nanmean(v):.4f}" for v in vals.values()]))

    with open(out / "morphology_mean.csv", "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for name, vals in agg.items():
            csv_writer.writerow([name] + [f"{_safe_nanmean(v):.4f}" for v in vals.values()])

    fig = plt.figure(figsize=(7, 4.2))
    bins = np.linspace(0, 16, 33)
    plotted = False
    for name, ws in widths_by_source.items():
        non_empty = [w for w in ws if w.size > 0]
        if not non_empty:
            continue
        allw = np.concatenate(non_empty)
        plt.hist(
            np.clip(allw, bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"{name} (n={len(allw)})",
        )
        plotted = True
    plt.xlabel("血管直径（像素）")
    plt.ylabel("骨架像素占比")
    plt.title("血管直径分布：金标准与模型对比")
    if plotted:
        plt.legend()
    plt.tight_layout()
    fig.savefig(out / "diameter_dist.png", dpi=150)
    plt.close(fig)
    print(f"\n图与表已保存到 {out}")
    return rows, agg


def main() -> None:
    """CLI 入口：批量血管形态定量分析。"""
    ap = argparse.ArgumentParser(description="批量视网膜血管形态特征对比分析")
    ap.add_argument("--sources", nargs="+", required=True, help="name=目录")
    ap.add_argument("--fov-dir", default="data/DRIVE/processed/test/fov")
    ap.add_argument("--out-dir", default="outputs/morphology")
    args = ap.parse_args()
    run_batch(args.sources, args.fov_dir, args.out_dir)


if __name__ == "__main__":
    main()
