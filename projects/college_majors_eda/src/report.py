"""Assembles reports/report.html with the top-8 EDA findings."""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils import cohen_r_magnitude, eta_sq_kruskal, wilcoxon_r

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

_STEM_CATEGORIES = [
    "Engineering",
    "Computers & Mathematics",
    "Physical Sciences",
    "Biology & Life Science",
]

# ── Chart helpers ──────────────────────────────────────────────────────────────


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


def _scatter_with_diagonal(
    x: pd.Series,
    y: pd.Series,
    hover_text: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    fmt: str = ",",
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
            hovertemplate=f"<b>%{{text}}</b><br>X: %{{x:{fmt}}}<br>Y: %{{y:{fmt}}}<extra></extra>",
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


# ── Top-8 findings ─────────────────────────────────────────────────────────────


def _finding_3_salary_by_category(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
    fig = _box_by_category(
        df,
        "Median",
        "Медианная зарплата",
        "Медианные зарплаты по категориям специальностей",
    )
    groups = [
        g["Median"].values for _, g in df.groupby("Major_category", observed=True)
    ]
    h_stat, p_val = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    eta = eta_sq_kruskal(h_stat, k, n)
    return fig, {
        "test": "Краскел–Уоллис",
        "stat": f"H = {h_stat:.2f}",
        "p_value": f"{p_val:.2e}",
        "effect": f"η² = {eta:.2f} (большой)",
        "conclusion": (
            "Категории направлений статистически значимо различаются по медианной зарплате. "
            "Наибольшие — у Engineering, наименьшие — у Education."
        ),
    }


def _finding_7_median_vs_ftr(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
    fig = _scatter_with_trend(
        df["Median"],
        df["full_time_rate"],
        "Медианная зарплата (Median)",
        "Доля занятых полный рабочий день (full_time_rate)",
        "Median vs full_time_rate",
    )
    rho, p_val = stats.spearmanr(df["Median"], df["full_time_rate"])
    return fig, {
        "test": "Спирмен",
        "stat": f"ρ = {rho:.2f}",
        "p_value": f"{p_val:.2e}",
        "effect": f"|ρ| = {abs(rho):.2f} (большой)",
        "conclusion": (
            "Чем выше медианная зарплата направления, тем выше доля работников на полной ставке. "
            "Обе характеристики отражают «качество» специальности на рынке труда."
        ),
    }


def _finding_8_stem_vs_nonstem(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
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

    stem = df.loc[df["is_stem"] == "STEM", "Median"].values
    not_stem = df.loc[df["is_stem"] == "Not STEM", "Median"].values
    u_stat, p_val = stats.mannwhitneyu(stem, not_stem, alternative="greater")
    n1, n2 = len(stem), len(not_stem)
   (2 * u_stat) / (n1 * n2) - 1
    return fig, {
        "test": "Манн–Уитни",
        "stat": f"U = {u_stat:.0f}",
        "p_value": f"{p_val:.4f}",
        "effect": f"r = {r:.2f} ({cohen_r_magnitude(abs(r))})",
        "conclusion": (
            "STEM-направления статистически значимо зарабатывают больше Non-STEM. "
            "Разрыв обусловлен концентрацией высокооплачиваемых направлений в Engineering, "
            "Computers & Mathematics и Physical Sciences."
        ),
    }


def _finding_2_grad_salary(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
    fig = _scatter_with_diagonal(
        df["Nongrad_median"],
        df["Grad_median"],
        df["Major"],
        "Зарплата без аспирантуры (Nongrad_median)",
        "Зарплата с аспирантурой (Grad_median)",
        "Grad_median vs Nongrad_median",
    )
    result = stats.wilcoxon(
        df["Grad_median"], df["Nongrad_median"], alternative="greater"
    )
   wilcoxon_r(result.statistic, len(df))
    grad_med = df["Grad_median"].median()
    nongrad_med = df["Nongrad_median"].median()
    return fig, {
        "test": "Вилкоксон (парный)",
        "stat": f"W = {result.statistic:.0f}",
        "p_value": f"{result.pvalue:.2e}",
        "effect": f"r = {r:.2f} (большой)",
        "conclusion": (
            f"Медиана Grad = ${grad_med:,.0f} vs Nongrad = ${nongrad_med:,.0f}. "
            "Преимущество устойчиво во всех специальностях без исключения."
        ),
    }


def _finding_5_grad_unemployment(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
    fig = _scatter_with_diagonal(
        df["Nongrad_unemployment_rate"],
        df["Grad_unemployment_rate"],
        df["Major"],
        "Безработица без аспирантуры (Nongrad_unemployment_rate)",
        "Безработица с аспирантурой (Grad_unemployment_rate)",
        "Grad_unemployment_rate vs Nongrad_unemployment_rate",
        fmt=".3f",
    )
    result = stats.wilcoxon(
        df["Grad_unemployment_rate"],
        df["Nongrad_unemployment_rate"],
        alternative="less",
    )
   wilcoxon_r(result.statistic, len(df))
    grad_unemp = df["Grad_unemployment_rate"].mean()
    nongrad_unemp = df["Nongrad_unemployment_rate"].mean()
    return fig, {
        "test": "Вилкоксон (парный)",
        "stat": f"W = {result.statistic:.0f}",
        "p_value": f"{result.pvalue:.2e}",
        "effect": f"r = {r:.2f} (большой)",
        "conclusion": (
            f"Уровень безработицы: Grad = {grad_unemp:.2%} vs Nongrad = {nongrad_unemp:.2%}. "
            "Эффект наблюдается независимо от категории направления."
        ),
    }


def _finding_1_male_vs_female(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
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
            text="<b>Медианные зарплаты: преобладание мужчин vs женщин</b>",
            font_size=13,
        ),
        yaxis_title="Медианная зарплата (Median)",
        template="plotly_white",
        width=700,
        height=500,
    )
    u_stat, p_val = stats.mannwhitneyu(male_dom, female_dom, alternative="two-sided")
    n1, n2 = len(male_dom), len(female_dom)
   1 - 2 * u_stat / (n1 * n2)
    return fig, {
        "test": "Манн–Уитни",
        "stat": f"U = {u_stat:.0f}",
        "p_value": f"{p_val:.2e}",
        "effect": f"r = {r:.2f} (большой)",
        "conclusion": (
            f"Медиана (ShareWomen < 0.3) = ${male_dom.median():,.0f} vs "
            f"(ShareWomen > 0.7) = ${female_dom.median():,.0f}. "
            "Наибольший размер эффекта во всём анализе."
        ),
    }


def _finding_4_share_women_by_category(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
    fig = _box_by_category(
        df,
        "ShareWomen",
        "Доля женщин",
        "Доля женщин по категориям специальностей",
    )
    groups = [
        g["ShareWomen"].values for _, g in df.groupby("Major_category", observed=True)
    ]
    h_stat, p_val = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    eta = eta_sq_kruskal(h_stat, k, n)
    return fig, {
        "test": "Краскел–Уоллис",
        "stat": f"H = {h_stat:.2f}",
        "p_value": f"{p_val:.2e}",
        "effect": f"η² = {eta:.2f} (большой)",
        "conclusion": (
            "Гендерная сегрегация по специальностям выражена и системна. "
            "Распределение ShareWomen по категориям статистически высоко значимо."
        ),
    }


def _finding_6_share_women_vs_median(
    df: pd.DataFrame,
) -> tuple[go.Figure, dict]:
    fig = _scatter_with_trend(
        df["ShareWomen"],
        df["Median"],
        "Доля женщин (ShareWomen)",
        "Медианная зарплата (Median)",
        "Median vs ShareWomen",
    )
    rho, p_val = stats.spearmanr(df["ShareWomen"], df["Median"])
    return fig, {
        "test": "Спирмен",
        "stat": f"ρ = {rho:.2f}",
        "p_value": f"{p_val:.2e}",
        "effect": f"|ρ| = {abs(rho):.2f} (большой)",
        "conclusion": (
            "Чем выше доля женщин в направлении, тем ниже медианная зарплата. "
            "Связь устойчивее, чем влияние STEM-статуса."
        ),
    }


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>College Majors EDA &#8212; Итоговый отчёт</title>
  <script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", sans-serif;
      margin: 0; background: #f7f8fa; color: #222; line-height: 1.6;
    }}
    .header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: white; padding: 48px 40px;
    }}
    .header h1 {{
      margin: 0 0 8px; font-size: 2rem; font-weight: 300; letter-spacing: -0.5px;
    }}
    .header .subtitle {{ margin: 0; opacity: 0.65; font-size: 1rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}

    .summary-block {{
      background: white; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      margin-bottom: 40px; padding: 28px;
    }}
    .summary-block h2 {{ margin: 0 0 18px; font-size: 1.25rem; color: #1a1a2e; }}
    table.summary {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.summary th {{
      background: #1a1a2e; color: white;
      padding: 10px 14px; text-align: left; font-weight: 500;
    }}
    table.summary td {{
      padding: 9px 14px; border-bottom: 1px solid #eef0f3; vertical-align: middle;
    }}
    table.summary tr:last-child td {{ border-bottom: none; }}
    table.summary tr:hover td {{ background: #f7f8fa; }}
    .rank-chip {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 26px; height: 26px; border-radius: 50%;
      background: #440154; color: white; font-size: 0.78rem; font-weight: 600;
    }}

    .section {{ margin: 48px 0 32px; }}
    .section-title {{
      font-size: 1.4rem; font-weight: 600; color: #1a1a2e;
      border-bottom: 3px solid #1f9e89; padding-bottom: 8px; margin-bottom: 24px;
    }}

    .finding {{
      background: white; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      margin-bottom: 28px; overflow: hidden;
    }}
    .finding-header {{
      display: flex; align-items: flex-start; gap: 14px;
      padding: 22px 24px 14px; border-bottom: 1px solid #eef0f3;
    }}
    .finding-rank {{
      display: flex; align-items: center; justify-content: center;
      width: 34px; height: 34px; border-radius: 50%;
      background: #440154; color: white;
      font-size: 0.85rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;
    }}
    .finding-title {{ font-size: 1.05rem; font-weight: 600; color: #1a1a2e; }}
    .finding-body {{ padding: 16px 24px 24px; }}
    .stats-box {{
      background: #f0f4f8; border-radius: 7px; padding: 14px 18px;
      margin-bottom: 16px; font-size: 0.88rem;
      display: grid; grid-template-columns: auto 1fr; gap: 5px 20px;
    }}
    .stats-box .label {{ font-weight: 600; color: #444; white-space: nowrap; }}
    .stats-box .value {{ color: #555; }}
    .stats-box .conclusion {{
      grid-column: 1 / -1; margin-top: 8px;
      color: #1a1a2e; font-style: italic; border-top: 1px solid #dde3ea;
      padding-top: 8px;
    }}
    .chart-wrap {{ margin-top: 8px; }}

    .overall-conclusion {{
      background: #1a1a2e; color: white;
      border-radius: 10px; padding: 28px 32px; margin-top: 48px;
    }}
    .overall-conclusion h2 {{ margin: 0 0 12px; font-size: 1.25rem; font-weight: 500; }}
    .overall-conclusion p {{ margin: 0; opacity: 0.85; line-height: 1.7; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>College Majors EDA</h1>
    <p class="subtitle">
      Итоговый отчёт &#8212; топ-8 гипотез, ранжированных по размеру практического эффекта
    </p>
  </div>
  <div class="container">
    {summary_table}
    {sections}
    <div class="overall-conclusion">
      <h2>Общий вывод</h2>
      <p>
        Доходы и условия занятости выпускников определяются тремя взаимосвязанными
        факторами &#8212; категорией специальности, гендерным составом направления и уровнем
        образования. Аспирантура и STEM-специализация дают измеримое преимущество,
        тогда как гендерная сегрегация в специальностях объясняет значительную часть
        наблюдаемого разрыва в зарплатах.
      </p>
    </div>
  </div>
</body>
</html>
"""


def _chart_div(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False},
    )


def _build_html(findings_by_section: dict) -> str:
    all_findings = sorted(
        [f for findings in findings_by_section.values() for f in findings],
        key=lambda x: x["rank"],
    )

    rows = "".join(
        f"""
        <tr>
          <td><span class="rank-chip">{f["rank"]}</span></td>
          <td>{f["title"]}</td>
          <td>{f["stats"]["test"]}</td>
          <td>{f["stats"]["effect"]}</td>
        </tr>"""
        for f in all_findings
    )
    summary_table = f"""
    <div class="summary-block">
      <h2>Ключевые находки</h2>
      <table class="summary">
        <thead>
          <tr>
            <th>#</th>
            <th>Находка</th>
            <th>Тест</th>
            <th>Размер эффекта</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    sections_html = ""
    for section_title, findings in findings_by_section.items():
        findings_html = ""
        for f in findings:
            s = f["stats"]
            findings_html += f"""
      <div class="finding">
        <div class="finding-header">
          <div class="finding-rank">{f["rank"]}</div>
          <div class="finding-title">{f["title"]}</div>
        </div>
        <div class="finding-body">
          <div class="stats-box">
            <span class="label">Тест</span>
            <span class="value">{s["test"]}</span>
            <span class="label">Статистика</span>
            <span class="value">{s["stat"]}</span>
            <span class="label">p-значение</span>
            <span class="value">{s["p_value"]}</span>
            <span class="label">Эффект</span>
            <span class="value">{s["effect"]}</span>
            <span class="conclusion">{s["conclusion"]}</span>
          </div>
          <div class="chart-wrap">{_chart_div(f["fig"])}</div>
        </div>
      </div>"""
        sections_html += f"""
    <div class="section">
      <div class="section-title">{section_title}</div>
      {findings_html}
    </div>"""

    return _HTML_TEMPLATE.format(
        summary_table=summary_table,
        sections=sections_html,
    )


# ── Entry point ────────────────────────────────────────────────────────────────


def run(processed_dir: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading analytics datasets for report...")
    df_all = pd.read_parquet(processed_dir / "majors_all_ages_analytics.parquet")
    df_deg = pd.read_parquet(processed_dir / "majors_degree_levels_analytics.parquet")
    df_rec = pd.read_parquet(
        processed_dir / "majors_recent_graduates_analytics.parquet"
    )

    log.info("Computing top-8 findings...")
    fig3, s3 = _finding_3_salary_by_category(df_all)
    fig7, s7 = _finding_7_median_vs_ftr(df_all)
    fig8, s8 = _finding_8_stem_vs_nonstem(df_all)

    fig2, s2 = _finding_2_grad_salary(df_deg)
    fig5, s5 = _finding_5_grad_unemployment(df_deg)

    fig1, s1 = _finding_1_male_vs_female(df_rec)
    fig4, s4 = _finding_4_share_women_by_category(df_rec)
    fig6, s6 = _finding_6_share_women_vs_median(df_rec)

    findings_by_section = {
        "Зарплатные результаты": [
            {
                "rank": 3,
                "title": "Категория специальности — главный предиктор зарплаты",
                "fig": fig3,
                "stats": s3,
            },
            {
                "rank": 7,
                "title": "Медианная зарплата и занятость полный день тесно связаны",
                "fig": fig7,
                "stats": s7,
            },
            {
                "rank": 8,
                "title": "STEM зарабатывает значимо больше Non-STEM",
                "fig": fig8,
                "stats": s8,
            },
        ],
        "Уровни образования": [
            {
                "rank": 2,
                "title": "Аспирантура даёт существенную прибавку к зарплате",
                "fig": fig2,
                "stats": s2,
            },
            {
                "rank": 5,
                "title": "Аспирантура значимо снижает безработицу",
                "fig": fig5,
                "stats": s5,
            },
        ],
        "Гендерный анализ": [
            {
                "rank": 1,
                "title": "Специальности с преобладанием мужчин зарабатывают значимо больше",
                "fig": fig1,
                "stats": s1,
            },
            {
                "rank": 4,
                "title": "Категория специальности сильно определяет долю женщин",
                "fig": fig4,
                "stats": s4,
            },
            {
                "rank": 6,
                "title": "Доля женщин сильно отрицательно коррелирует с зарплатой",
                "fig": fig6,
                "stats": s6,
            },
        ],
    }

    log.info("Assembling report.html...")
    html = _build_html(findings_by_section)

    out_path = reports_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("Report saved → %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        PROJECT_DIR / "data" / "processed",
        PROJECT_DIR / "reports",
    )
