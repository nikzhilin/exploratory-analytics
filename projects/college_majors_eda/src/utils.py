import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def cohen_r_magnitude(abs_r: float) -> str:
    if abs_r < 0.1:
        return "тривиальный"
    if abs_r < 0.3:
        return "слабый"
    if abs_r < 0.5:
        return "средний"
    return "большой"


def spearman_magnitude(abs_rho: float) -> str:
    if abs_rho < 0.3:
        return "слабая"
    if abs_rho < 0.5:
        return "умеренная"
    if abs_rho < 0.7:
        return "заметная"
    if abs_rho < 0.9:
        return "высокая"
    return "очень высокая"


def eta_sq_kruskal(h_stat: float, k: int, n: int) -> float:
    return (h_stat - k + 1) / (n - k)


def wilcoxon_r(W: float, n: int) -> float:
    mu_W = n * (n + 1) / 4
    sigma_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu_W) / sigma_W
    return abs(z) / np.sqrt(n)


def save_fig(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    logging.getLogger(__name__).info(f"  Chart saved: {path}")
