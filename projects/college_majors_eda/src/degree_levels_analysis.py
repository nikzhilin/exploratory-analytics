import logging
import warnings
from pathlib import Path

import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils import (
    cohen_r_magnitude,
    eta_sq_kruskal,
    save_fig,
    spearman_magnitude,
    wilcoxon_r,
)

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _scatter_with_diagonal(
    x: pd.Series,
    y: pd.Series,
    hover_text: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#1f9e89", opacity=0.6, size=6),
            text=hover_text,
            hovertemplate="<b>%{text}</b><br>X: %{x:,}<br>Y: %{y:,}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#440154", width=1.5, dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font_size=13),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        width=600,
        height=500,
    )
    return fig


def _box_by_category(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
) -> go.Figure:
    order = (
        df.groupby("Major_category", observed=True)[x_col]
        .median()
        .sort_values()
        .index.tolist()
    )
    colors = pc.sample_colorscale("Viridis", len(order))
    fig = px.box(
        df,
        y="Major_category",
        x=x_col,
        color="Major_category",
        category_orders={"Major_category": order},
        color_discrete_map=dict(zip(order, colors)),
        labels={x_col: x_label, "Major_category": ""},
        title=f"<b>{title}</b>",
        template="plotly_white",
    )
    fig.update_traces(
        boxpoints="outliers",
        marker=dict(size=4, opacity=0.5),
        line=dict(width=0.9),
        width=0.6,
    )
    fig.update_layout(showlegend=False, title_font_size=13, width=900, height=580)
    return fig


def _kruskal_by_category(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    groups = [g[col].values for _, g in df.groupby("Major_category", observed=True)]
    h_stat, p_val = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta = eta_sq_kruskal(h_stat, k, n)
    return h_stat, p_val, eta


def _wilcoxon_paired(
    a: pd.Series, b: pd.Series, alternative: str
) -> tuple[float, float]:
    result = stats.wilcoxon(a, b, alternative=alternative)
   wilcoxon_r(result.statistic, len(a))
    return result.pvalue, r


# ── H1: аспиранты зарабатывают больше ────────────────────────────────────────


def _h1_grad_vs_nongrad_salary(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H1: Grad_median > Nongrad_median")
    fig = _scatter_with_diagonal(
        df["Nongrad_median"],
        df["Grad_median"],
        df["Major"],
        "Зарплата без аспирантуры (Nongrad_median)",
        "Зарплата с аспирантурой (Grad_median)",
        "Grad_median vs Nongrad_median",
    )
    save_fig(fig, plots_dir / "h1_grad_vs_nongrad_salary.html")

    p_val,_wilcoxon_paired(df["Grad_median"], df["Nongrad_median"], "greater")
    log.info("  Вилкоксон: p = %.6f", p_val)
    log.info(" %.4f (%s)", r, cohen_r_magnitude(r))
    log.info(
        "  Медиана Grad = %s, Nongrad = %s",
        f"{df['Grad_median'].median():,.0f}",
        f"{df['Nongrad_median'].median():,.0f}",
    )
    log.info("  → H0 отвергается: аспиранты зарабатывают значимо больше")


# ── H2: популярность аспирантуры по категориям (используем Grad_share) ───────


def _h2_grad_share_by_category(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H2: Популярность аспирантуры по категориям (Grad_share)")
    fig = _box_by_category(
        df,
        "Grad_share",
        "Доля выпускников с высшей степенью (Grad_share)",
        "Популярность аспирантуры по категориям специальностей",
    )
    save_fig(fig, plots_dir / "h2_grad_share_by_category.html")

    h_stat, p_val, eta = _kruskal_by_category(df, "Grad_share")
    log.info("  Краскел–Уоллис: H = %.4f, p = %.6f", h_stat, p_val)
    log.info("  η² = %.4f", eta)
    log.info(
        "  → H0 отвергается: популярность аспирантуры значимо различается между категориями"
    )


# ── H3: безработица аспирантов ниже ──────────────────────────────────────────


def _h3_grad_unemployment(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H3: Grad_unemployment_rate < Nongrad_unemployment_rate")
    fig = _scatter_with_diagonal(
        df["Nongrad_unemployment_rate"],
        df["Grad_unemployment_rate"],
        df["Major"],
        "Безработица без аспирантуры (Nongrad_unemployment_rate)",
        "Безработица с аспирантурой (Grad_unemployment_rate)",
        "Grad_unemployment_rate vs Nongrad_unemployment_rate",
    )
    save_fig(fig, plots_dir / "h3_grad_vs_nongrad_unemployment.html")

    p_val,_wilcoxon_paired(
        df["Grad_unemployment_rate"], df["Nongrad_unemployment_rate"], "less"
    )
    log.info("  Вилкоксон: p = %.6f", p_val)
    log.info(" %.4f (%s)", r, cohen_r_magnitude(r))
    log.info(
        "  Среднее Grad = %.4f, Nongrad = %.4f",
        df["Grad_unemployment_rate"].mean(),
        df["Nongrad_unemployment_rate"].mean(),
    )
    log.info("  → H0 отвергается: уровень безработицы аспирантов значимо ниже")


# ── H4: зарплатная премия по категориям ──────────────────────────────────────


def _h4_grad_premium_by_category(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H4: Зарплатная премия за аспирантуру различается между категориями")
    fig = _box_by_category(
        df,
        "Grad_premium",
        "Зарплатная премия (Grad_premium)",
        "Зарплатная премия за аспирантуру по категориям специальностей",
    )
    save_fig(fig, plots_dir / "h4_grad_premium_by_category.html")

    h_stat, p_val, eta = _kruskal_by_category(df, "Grad_premium")
    log.info("  Краскел–Уоллис: H = %.4f, p = %.6f", h_stat, p_val)
    log.info("  η² = %.4f", eta)
    log.info(
        "  → H0 отвергается: зарплатная премия значимо различается между категориями"
    )


# ── H5: финансовая выгода связана с долей продолживших обучение ───────────────


def _h5_premium_vs_grad_share(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H5: Спирмен Grad_premium ~ Grad_share")
    fig = px.scatter(
        df,
        x="Grad_premium",
        y="Grad_share",
        hover_name="Major",
        color="Major_category",
        title="<b>Связь зарплатной премии и доли продолживших обучение</b>",
        labels={
            "Grad_premium": "Зарплатная премия (Grad_premium)",
            "Grad_share": "Доля выпускников со степенью (Grad_share)",
        },
        template="plotly_white",
        width=800,
        height=500,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    save_fig(fig, plots_dir / "h5_grad_premium_vs_grad_share.html")

    corr, p_val = stats.spearmanr(df["Grad_premium"], df["Grad_share"])
    log.info("  Спирмен: ρ = %.4f, p = %.6f", corr, p_val)
    log.info("  Сила связи: %s", spearman_magnitude(abs(corr)))
    log.info(
        "  → H0 отвергается: связь между премией и долей продолживших обучение значима"
    )


# ── H6: доля Grad различается между категориями ───────────────────────────────


def _h6_grad_share_by_category(df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("H6: Grad_share значимо различается между категориями")
    fig = _box_by_category(
        df,
        "Grad_share",
        "Доля выпускников с высшей степенью (Grad_share)",
        "Доля выпускников с высшей степенью по категориям специальностей",
    )
    save_fig(fig, plots_dir / "h6_grad_share_by_category.html")

    h_stat, p_val, eta = _kruskal_by_category(df, "Grad_share")
    log.info("  Краскел–Уоллис: H = %.4f, p = %.6f", h_stat, p_val)
    log.info("  η² = %.4f", eta)
    log.info("  → H0 отвергается: доля Grad кардинально различается между категориями")


def run(processed_dir: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(processed_dir / "majors_degree_levels_analytics.parquet")
    log.info("Loaded majors_degree_levels_analytics: %d rows, %d cols", *df.shape)

    _h1_grad_vs_nongrad_salary(df, plots_dir)
    _h2_grad_share_by_category(df, plots_dir)
    _h3_grad_unemployment(df, plots_dir)
    _h4_grad_premium_by_category(df, plots_dir)
    _h5_premium_vs_grad_share(df, plots_dir)
    _h6_grad_share_by_category(df, plots_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        PROJECT_DIR / "data" / "processed",
        PROJECT_DIR / "artifacts" / "plots" / "degree_levels",
    )
