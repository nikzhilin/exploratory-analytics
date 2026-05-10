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


def _spearman_log(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    rho, p_val = stats.spearmanr(df[x_col], df[y_col])
    log.info("  Спирмен: ρ = %.4f, p = %.4f", rho, p_val)
    log.info("  |ρ| = %.4f (%s по Cohen)", abs(rho), cohen_r_magnitude(abs(rho)))
    if p_val < 0.05:
        direction = "отрицательная" if rho < 0 else "положительная"
        log.info("  → H0 отвергается: значимая %s монотонная связь", direction)
    else:
        log.info("  → H0 не отвергается: связь статистически незначима")


# ── H1: ShareWomen ~ Median ───────────────────────────────────────────────────


def _h1_share_women_vs_median(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H1: ShareWomen ~ Median")
    fig = _scatter_with_trend(
        df["ShareWomen"],
        df["Median"],
        "Доля женщин (ShareWomen)",
        "Медианная зарплата (Median)",
        "Median vs ShareWomen",
    )
    save_fig(fig, plots_dir / "h1_median_vs_share_women.html")
    _spearman_log(df, "ShareWomen", "Median")


# ── H2: ShareWomen ~ FTR ──────────────────────────────────────────────────────


def _h2_share_women_vs_ftr(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H2: ShareWomen ~ FTR")
    fig = _scatter_with_trend(
        df["ShareWomen"],
        df["FTR"],
        "Доля женщин (ShareWomen)",
        "Доля занятых полный год (FTR)",
        "FTR vs ShareWomen",
    )
    save_fig(fig, plots_dir / "h2_ftr_vs_share_women.html")
    _spearman_log(df, "ShareWomen", "FTR")


# ── H3: ShareWomen по категориям ──────────────────────────────────────────────


def _h3_share_women_by_category(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H3: ShareWomen различается между категориями")
    order = (
        df.groupby("Major_category", observed=True)["ShareWomen"]
        .median()
        .sort_values()
        .index.tolist()
    )
    colors = pc.sample_colorscale("Viridis", len(order))
    fig = px.box(
        df,
        y="Major_category",
        x="ShareWomen",
        color="Major_category",
        category_orders={"Major_category": order},
        color_discrete_map=dict(zip(order, colors)),
        labels={"ShareWomen": "Доля женщин", "Major_category": ""},
        title="<b>Доля женщин по категориям специальностей</b>",
        template="plotly_white",
    )
    fig.update_traces(
        boxpoints="outliers",
        marker=dict(size=4, opacity=0.5),
        line=dict(width=0.9),
        width=0.6,
    )
    fig.update_layout(showlegend=False, title_font_size=13, width=900, height=580)
    save_fig(fig, plots_dir / "h3_share_women_by_category.html")

    groups = [
        g["ShareWomen"].values for _, g in df.groupby("Major_category", observed=True)
    ]
    h_stat, p_val = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta = eta_sq_kruskal(h_stat, k, n)
    log.info("  Краскел–Уоллис: H = %.4f, p = %.2e", h_stat, p_val)
    log.info("  η² = %.4f", eta)
    log.info("  → H0 отвергается: доля женщин значимо различается между категориями")


# ── H4: зарплаты мужских vs женских специальностей ───────────────────────────


def _h4_salary_male_vs_female_dominated(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H4: Median в специальностях с ShareWomen < 0.3 vs > 0.7")
    male_dom = df[df["ShareWomen"] < 0.3]["Median"]
    female_dom = df[df["ShareWomen"] > 0.7]["Median"]

    fig = go.Figure()
    for label, data, color in [
        (f"ShareWomen < 0.3 (n={len(male_dom)})", male_dom, "#440154"),
        (f"ShareWomen > 0.7 (n={len(female_dom)})", female_dom, "#1f9e89"),
    ]:
        fig.add_trace(
            go.Box(
                y=data,
                name=label,
                marker_color=color,
                boxpoints="outliers",
                marker=dict(size=4, opacity=0.5),
                line=dict(width=0.9),
            )
        )
    fig.update_layout(
        title=dict(
            text="<b>Медианные зарплаты: специальности с преобладанием мужчин vs женщин</b>",
            font_size=13,
        ),
        yaxis_title="Медианная зарплата (Median)",
        template="plotly_white",
        width=700,
        height=500,
    )
    save_fig(fig, plots_dir / "h4_salary_male_vs_female_dominated.html")

    u_stat, p_val = stats.mannwhitneyu(male_dom, female_dom, alternative="two-sided")
    n1, n2 = len(male_dom), len(female_dom)
    r = 1 - 2 * u_stat / (n1 * n2)
    log.info("  Манн–Уитни: U = %.0f, p = %.4f", u_stat, p_val)
    log.info(
        "  Медиана male-dom = %s, female-dom = %s",
        f"{male_dom.median():,.0f}",
        f"{female_dom.median():,.0f}",
    )
    log.info("  r = %.4f (%s по Cohen)", r, cohen_r_magnitude(abs(r)))
    log.info("  → H0 отвергается: мужские специальности зарабатывают значимо больше")


def run(processed_dir: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(processed_dir / "majors_recent_graduates_analytics.parquet")
    log.info("Loaded majors_recent_graduates_analytics: %d rows, %d cols", *df.shape)

    _h1_share_women_vs_median(df, plots_dir)
    _h2_share_women_vs_ftr(df, plots_dir)
    _h3_share_women_by_category(df, plots_dir)
    _h4_salary_male_vs_female_dominated(df, plots_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        PROJECT_DIR / "data" / "processed",
        PROJECT_DIR / "artifacts" / "plots" / "gender",
    )
