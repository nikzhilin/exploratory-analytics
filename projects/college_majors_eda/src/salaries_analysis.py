import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils import cohen_r_magnitude, eta_sq_kruskal, save_fig

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

_STEM_CATEGORIES = [
    "Engineering",
    "Computers & Mathematics",
    "Physical Sciences",
    "Biology & Life Science",
]


def _scatter_with_trend(
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    m, b = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#1f9e89", opacity=0.5, size=6),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=m * x_range + b,
            mode="lines",
            line=dict(color="#440154", width=1.5),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font_size=13),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        width=700,
        height=500,
    )
    return fig


# ── H1: медианные зарплаты по категориям направлений ─────────────────────────


def _h1_salary_by_category(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H1: Медианные зарплаты по категориям направлений")
    order = (
        df.groupby("Major_category", observed=True)["Median"]
        .median()
        .sort_values()
        .index.tolist()
    )
    colors = pc.sample_colorscale("Viridis", len(order))
    fig = px.box(
        df,
        y="Major_category",
        x="Median",
        color="Major_category",
        category_orders={"Major_category": order},
        color_discrete_map=dict(zip(order, colors)),
        labels={"Median": "Медианная зарплата", "Major_category": ""},
        title="<b>Медианные зарплаты по категориям специальностей</b>",
        template="plotly_white",
    )
    fig.update_traces(
        boxpoints="outliers",
        marker=dict(size=4, opacity=0.5),
        line=dict(width=0.9),
        width=0.6,
    )
    fig.update_layout(showlegend=False, title_font_size=13, width=900, height=580)
    save_fig(fig, plots_dir / "h1_median_salary_by_category.html")

    groups = [
        g["Median"].values for _, g in df.groupby("Major_category", observed=True)
    ]
    h_stat, p_val = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta = eta_sq_kruskal(h_stat, k, n)
    log.info("  Краскел–Уоллис: H = %.4f, p = %.2e", h_stat, p_val)
    log.info(
        "  η² = %.4f → %s",
        eta,
        "большой" if eta >= 0.14 else "средний" if eta >= 0.06 else "слабый",
    )
    log.info("  → H0 отвергается: категории значимо различаются по медианной зарплате")


# ── H2: STEM vs Non-STEM ──────────────────────────────────────────────────────


def _h2_stem_vs_nonstems(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H2: STEM зарабатывают больше Non-STEM")
    df = df.copy()
    df["is_stem"] = (
        df["Major_category"]
        .isin(_STEM_CATEGORIES)
        .map({True: "STEM", False: "Not STEM"})
    )
    order = ["Not STEM", "STEM"]
    color_map = {"STEM": "#1f9e89", "Not STEM": "#440154"}
    fig = px.box(
        df,
        y="is_stem",
        x="Median",
        color="is_stem",
        category_orders={"is_stem": order},
        color_discrete_map=color_map,
        labels={"Median": "Медианная зарплата", "is_stem": ""},
        title="<b>Медианные зарплаты: STEM vs Not STEM</b>",
        template="plotly_white",
    )
    fig.update_traces(
        boxpoints="outliers",
        marker=dict(size=4, opacity=0.5),
        line=dict(width=0.9),
        width=0.6,
    )
    fig.update_layout(showlegend=False, title_font_size=13, width=700, height=300)
    save_fig(fig, plots_dir / "h2_stem_vs_nonstems.html")

    stem = df.loc[df["is_stem"] == "STEM", "Median"].values
    not_stem = df.loc[df["is_stem"] == "Not STEM", "Median"].values
    u_stat, p_val = stats.mannwhitneyu(stem, not_stem, alternative="greater")
    n1, n2 = len(stem), len(not_stem)
    r = (2 * u_stat) / (n1 * n2) - 1
    log.info("  Манн–Уитни: U = %.0f, p = %.4f", u_stat, p_val)
    log.info("  r = %.4f (%s по Cohen)", r, cohen_r_magnitude(abs(r)))
    log.info("  → H0 отвергается: STEM зарабатывают значимо больше Non-STEM")


# ── H3: дисперсия зарплат по категориям (тест Левена) ────────────────────────


def _h3_salary_variance_by_category(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H3: Разброс зарплат внутри категорий различается")
    std_df = (
        df.groupby("Major_category", observed=True)["Median"]
        .std()
        .reset_index()
        .rename(columns={"Median": "Salary_std"})
    )
    order = std_df.sort_values("Salary_std")["Major_category"].tolist()
    colors = pc.sample_colorscale("Viridis", len(order))
    fig = px.bar(
        std_df,
        y="Major_category",
        x="Salary_std",
        color="Major_category",
        category_orders={"Major_category": order},
        color_discrete_map=dict(zip(order, colors)),
        labels={
            "Salary_std": "Стандартное отклонение медианной зарплаты",
            "Major_category": "",
        },
        title="<b>Разброс зарплат внутри категорий специальностей</b>",
        template="plotly_white",
        orientation="h",
    )
    fig.update_layout(showlegend=False, title_font_size=13, width=900, height=580)
    save_fig(fig, plots_dir / "h3_salary_std_by_category.html")

    groups = [
        g["Median"].values for _, g in df.groupby("Major_category", observed=True)
    ]
    lev_stat, p_val = stats.levene(*groups)
    k, n = len(groups), sum(len(g) for g in groups)
    eta = (lev_stat * (k - 1)) / (lev_stat * (k - 1) + (n - k))
    log.info("  Тест Левена: F = %.4f, p = %.4f", lev_stat, p_val)
    log.info("  η² = %.4f", eta)
    if p_val < 0.05:
        log.info("  → H0 отвергается: дисперсии значимо различаются между категориями")
    else:
        log.info("  → H0 не отвергается")


# ── H4–H6: корреляции ─────────────────────────────────────────────────────────


def _spearman_hypothesis(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    h_num: int,
    plots_dir: Path,
) -> None:
    log.info("H%d: корреляция %s ~ %s", h_num, x_col, y_col)
    fig = _scatter_with_trend(df[x_col], df[y_col], x_label, y_label, title)
    save_fig(fig, plots_dir / f"h{h_num}_{x_col.lower()}_vs_{y_col.lower()}.html")

    rho, p_val = stats.spearmanr(df[x_col], df[y_col])
    abs_rho = abs(rho)
    log.info("  Спирмен: ρ = %.4f, p = %.4f", rho, p_val)
    log.info("  |ρ| = %.4f (%s по Cohen)", abs_rho, cohen_r_magnitude(abs_rho))
    if p_val < 0.05:
        direction = "отрицательная" if rho < 0 else "положительная"
        log.info("  → H0 отвергается: значимая %s связь", direction)
    else:
        log.info("  → H0 не отвергается: связь незначима")


def run(processed_dir: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(processed_dir / "majors_all_ages_analytics.parquet")
    log.info("Loaded majors_all_ages_analytics: %d rows, %d cols", *df.shape)

    _h1_salary_by_category(df, plots_dir)
    _h2_stem_vs_nonstems(df, plots_dir)
    _h3_salary_variance_by_category(df, plots_dir)

    _spearman_hypothesis(
        df,
        "full_time_rate",
        "Unemployment_rate",
        "Доля занятых полный рабочий день (full_time_rate)",
        "Уровень безработицы (Unemployment_rate)",
        "Unemployment_rate vs full_time_rate",
        4,
        plots_dir,
    )
    _spearman_hypothesis(
        df,
        "Median",
        "full_time_rate",
        "Медианная зарплата (Median)",
        "Доля занятых полный рабочий день (full_time_rate)",
        "Median vs full_time_rate",
        5,
        plots_dir,
    )
    _spearman_hypothesis(
        df,
        "Unemployment_rate",
        "Median",
        "Уровень безработицы (Unemployment_rate)",
        "Медианная зарплата (Median)",
        "Median vs Unemployment_rate",
        6,
        plots_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        PROJECT_DIR / "data" / "processed",
        PROJECT_DIR / "artifacts" / "plots" / "salaries",
    )
