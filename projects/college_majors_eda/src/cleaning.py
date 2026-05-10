import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

_DATASET_NAMES = ["majors_all_ages", "majors_degree_levels", "majors_recent_graduates"]

_HIERARCHY_CORRECTIONS = [
    (
        "majors_all_ages",
        "Employed",
        "Employed_full_time_year_round",
        ["MILITARY TECHNOLOGIES"],
    ),
    (
        "majors_degree_levels",
        "Grad_employed",
        "Grad_full_time_year_round",
        ["OCEANOGRAPHY"],
    ),
    (
        "majors_degree_levels",
        "Nongrad_employed",
        "Nongrad_full_time_year_round",
        ["MILITARY TECHNOLOGIES"],
    ),
    (
        "majors_recent_graduates",
        "Employed",
        "Full_time",
        [
            "NAVAL ARCHITECTURE AND MARINE ENGINEERING",
            "NUCLEAR ENGINEERING",
            "ACTUARIAL SCIENCE",
            "OCEANOGRAPHY",
            "MATHEMATICS AND COMPUTER SCIENCE",
            "MILITARY TECHNOLOGIES",
            "EDUCATIONAL ADMINISTRATION AND SUPERVISION",
        ],
    ),
]


def _load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(raw_dir / f"{name}.csv") for name in _DATASET_NAMES}


def _check_key_consistency(datasets: dict[str, pd.DataFrame]) -> None:
    reference = set(datasets["majors_all_ages"]["Major_code"])
    for name, df in datasets.items():
        if set(df["Major_code"]) != reference:
            log.warning("Major_code mismatch between majors_all_ages and %s", name)
        else:
            log.info("  Key consistency OK: %s", name)


def _drop_food_science(datasets: dict[str, pd.DataFrame]) -> None:
    df = datasets["majors_recent_graduates"]
    missing = df[df["Total"].isna()]
    if not missing.empty:
        affected = missing["Major"].tolist()
        log.info(
            "  Dropping rows with missing Total in majors_recent_graduates: %s",
            affected,
        )
        datasets["majors_recent_graduates"] = df[df["Major"] != "FOOD SCIENCE"].copy()


def _fix_hierarchy_violations(datasets: dict[str, pd.DataFrame]) -> None:
    for (
        dataset_name,
        col_larger,
        col_smaller,
        affected_majors,
    ) in _HIERARCHY_CORRECTIONS:
        df = datasets[dataset_name]
        mask = df["Major"].isin(affected_majors)
        if not mask.any():
            continue
        pair_max = df.loc[mask, [col_larger, col_smaller]].max(axis=1)
        datasets[dataset_name].loc[mask, col_larger] = pair_max
        datasets[dataset_name].loc[mask, col_smaller] = pair_max
        log.info(
            "  Fixed hierarchy violation in %s: %s >= %s for %d row(s)",
            dataset_name,
            col_larger,
            col_smaller,
            mask.sum(),
        )


def _cast_categoricals(datasets: dict[str, pd.DataFrame]) -> None:
    for df in datasets.values():
        df["Major"] = df["Major"].astype("category")
        df["Major_category"] = df["Major_category"].astype("category")


def _save(datasets: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        out = processed_dir / f"{name}_clean.parquet"
        df.to_parquet(out, index=False)
        log.info("  Saved → %s  (%d rows)", out, len(df))


def run(raw_dir: Path, processed_dir: Path) -> None:
    log.info("Loading raw datasets...")
    datasets = _load_raw(raw_dir)

    log.info("Checking key consistency...")
    _check_key_consistency(datasets)

    log.info("Handling missing rows...")
    _drop_food_science(datasets)

    log.info("Fixing hierarchy violations...")
    _fix_hierarchy_violations(datasets)

    log.info("Casting categorical columns...")
    _cast_categoricals(datasets)

    log.info("Saving clean parquets...")
    _save(datasets, processed_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(PROJECT_DIR / "data" / "raw", PROJECT_DIR / "data" / "processed")
