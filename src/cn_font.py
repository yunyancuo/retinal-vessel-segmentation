"""统一中文字体设置：优先加载项目内置 SimHei，回退系统常见中文字体。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

_FONT = Path(__file__).resolve().parent.parent / "fonts" / "simhei.ttf"
_FALLBACK_FONTS = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]

if _FONT.exists():
    font_manager.fontManager.addfont(str(_FONT))
    matplotlib.rcParams["font.sans-serif"] = _FALLBACK_FONTS
else:
    matplotlib.rcParams["font.sans-serif"] = _FALLBACK_FONTS[1:]

matplotlib.rcParams["axes.unicode_minus"] = False
