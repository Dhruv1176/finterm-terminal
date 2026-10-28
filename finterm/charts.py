"""Terminal charts built from Unicode block characters."""
from __future__ import annotations

import numpy as np

SPARK = "▁▂▃▄▅▆▇█"
EIGHTHS = " ▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 24) -> str:
    v = np.asarray(values, float)
    if len(v) == 0:
        return ""
    if len(v) > width:
        v = np.array([b.mean() for b in np.array_split(v, width)])
    lo, hi = v.min(), v.max()
    if hi == lo:
        return SPARK[3] * len(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * 7)] for x in v)


def area_chart(values, width: int = 64, height: int = 12, fmt: str = "{:,.2f}") -> list[str]:
    """Filled area chart with a y-axis. Returns a list of text lines."""
    v = np.asarray(values, float)
    v = np.interp(np.linspace(0, len(v) - 1, width), np.arange(len(v)), v)
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1.0
    levels = (v - lo) / span * (height * 8 - 1) + 1
    lab_w = max(len(fmt.format(lo)), len(fmt.format(hi)))
    lines = []
    for row in range(height, 0, -1):
        base = (row - 1) * 8
        cells = "".join("█" if lv - base >= 8 else EIGHTHS[int(lv - base)] if lv - base > 0 else " " for lv in levels)
        tag = fmt.format(lo + span * row / height) if row in (height, (height + 1) // 2, 1) else ""
        lines.append(f"{tag:>{lab_w}} │{cells}")
    lines.append(" " * lab_w + " └" + "─" * width)
    return lines


def histogram(values, bins: int = 12, width: int = 34, fmt: str = "{:,.0f}") -> list[str]:
    counts, edges = np.histogram(values, bins=bins)
    mx = counts.max() or 1
    return [f"{fmt.format(edges[i]):>10} - {fmt.format(edges[i + 1]):<10} {'█' * int(c / mx * width)} {c / counts.sum():.1%}"
            for i, c in enumerate(counts)]
