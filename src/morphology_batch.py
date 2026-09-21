"""批量血管形态分析：对 GT 与各模型预测逐图计算形态特征，输出汇总 CSV、
均值对比表和直径分布对比图（报告用）。

用法：
    python src/morphology_batch.py \
        --sources gt=data/DRIVE/processed/test/masks \
                  unet=outputs/unet_bce_dice/predictions/test \
                  attn=outputs/attn_unet_tversky/predictions/test \
        --fov-dir data/DRIVE/processed/test/fov --out-dir outputs/morphology
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_gray
from morphology import analyze

FIELDS = [
    "source", "key", "vessel_density", "mean_diameter_px", "median_diameter_px",
    "total_length_px", "tortuosity", "fractal_dimension",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="name=目录")
    ap.add_argument("--fov-dir", default="data/DRIVE/processed/test/fov")
    ap.add_argument("--out-dir", default="outputs/morphology")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fov_dir = Path(args.fov_dir)
    rows, widths_by_source = [], {}
    for spec in args.sources:
        name, d = spec.split("=", 1)
        d = Path(d)
        keys = sorted(p.stem.split("_")[0] for p in d.glob("*_pred.png")) or \
            sorted(p.stem for p in d.glob("*.png"))
        for key in keys:
            mask_path = d / f"{key}_pred.png" if (d / f"{key}_pred.png").exists() else d / f"{key}.png"
            fov = load_gray(fov_dir / f"{key}.png") > 127
            res = analyze(load_gray(mask_path), fov)
            rows.append({
                "source": name, "key": key,
                "vessel_density": f"{res['vessel_density']:.4f}",
                "mean_diameter_px": f"{res['mean_diameter_px']:.2f}",
                "median_diameter_px": f"{res['median_diameter_px']:.2f}",
                "total_length_px": f"{res['total_length_px']:.0f}",
                "tortuosity": f"{res['tortuosity']:.3f}",
                "fractal_dimension": f"{res['fractal_dimension']:.3f}",
            })
            widths_by_source.setdefault(name, []).append(res["widths"])
        print(f"{name}: {len(keys)} 张")

    with open(out / "morphology_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== 形态特征均值（{len(set(r['source'] for r in rows))} 组）===")
    header = ["source"] + FIELDS[2:]
    print(",".join(header))
    agg = {}
    for name in widths_by_source:
        sub = [r for r in rows if r["source"] == name]
        vals = {k: np.array([float(r[k]) for r in sub]) for k in FIELDS[2:]}
        agg[name] = vals
        print(",".join([name] + [f"{v.mean():.4f}" for v in vals.values()]))
    with open(out / "morphology_mean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name, vals in agg.items():
            w.writerow([name] + [f"{v.mean():.4f}" for v in vals.values()])

    plt.figure(figsize=(7, 4.2))
    bins = np.linspace(0, 16, 33)
    for name, ws in widths_by_source.items():
        allw = np.concatenate(ws)
        plt.hist(np.clip(allw, bins[0], bins[-1]), bins=bins, density=True,
                 histtype="step", linewidth=1.8, label=f"{name} (n={len(allw)})")
    plt.xlabel("vessel diameter (px)")
    plt.ylabel("fraction of skeleton pixels")
    plt.title("Vessel diameter distribution: GT vs models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "diameter_dist.png", dpi=150)
    print(f"\n图与表已保存到 {out}")


if __name__ == "__main__":
    main()
