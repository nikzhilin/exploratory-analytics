import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
pd.options.mode.use_inf_as_na = True

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_clean(processed_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "majors_all_ages_clean",
        "majors_degree_levels_clean",
        "majors_recent_graduates_clean",
    ]
    return {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in names}


def _add_all_ages_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["full_time_rate"] = (
        df["Employed_full_time_year_round"] / df["Employed"]
    ).fillna(0)
    df["salary_range_rate"] = (df["P75th"] / df["P25th"]).fillna(0)
    log.info("  all_ages: added full_time_rate, salary_range_rate")
    return df


def _add_degree_levels_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grad_unemployment_risk_rate"] = (
        df["Grad_unemployment_rate"] / df["Nongrad_unemployment_rate"]
    ).fillna(0)
    df["Grad_FTR"] = (df["Grad_full_time_year_round"] / df["Grad_employed"]).fillna(0)
    df["Nongrad_FTR"] = (
        df["Nongrad_full_time_year_round"] / df["Nongrad_employed"]
    ).fillna(0)
    log.info(
        "  degree_levels: added grad_unemployment_risk_rate, Grad_FTR, Nongrad_FTR"
    )
    return df


def _add_recent_graduates_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diploma_impact_rate"] = (df["College_jobs"] / df["Employed"]).fillna(0)
    df["FTR"] = df["Full_time_year_round"] / df["Employed"]
    log.info("  recent_graduates: added diploma_impact_rate, FTR")
    return df


def _drop_interdisciplinary(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    result = {}
    for name, df in datasets.items():
        before = len(df)
        df = df[df["Major_category"] != "Interdisciplinary"].copy()
        dropped = before - len(df)
        if dropped:
            log.info("  Dropped %d Interdisciplinary row(s) from %s", dropped, name)
        result[name] = df
    return result


def _save(datasets: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    output_map = {
        "majors_all_ages_clean": "majors_all_ages_analytics",
        "majors_degree_levels_clean": "majors_degree_levels_analytics",
        "majors_recent_graduates_clean": "majors_recent_graduates_analytics",
    }
    for src_name, dst_name in output_map.items():
        out = processed_dir / f"{dst_name}.parquet"
        datasets[src_name].to_parquet(out, index=False)
        log.info("  Saved → %s  (%d rows)", out, len(datasets[src_name]))


def run(processed_dir: Path) -> None:
    log.info("Loading clean parquets...")
    datasets = _load_clean(processed_dir)

    log.info("Adding derived features...")
    datasets["majors_all_ages_clean"] = _add_all_ages_features(
        datasets["majors_all_ages_clean"]
    )
    datasets["majors_degree_levels_clean"] = _add_degree_levels_features(
        datasets["majors_degree_levels_clean"]
    )
    datasets["majors_recent_graduates_clean"] = _add_recent_graduates_features(
        datasets["majors_recent_graduates_clean"]
    )

    log.info("Removing Interdisciplinary category...")
    datasets = _drop_interdisciplinary(datasets)

    log.info("Saving analytics parquets...")
    _save(datasets, processed_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(PROJECT_DIR / "data" / "processed")
