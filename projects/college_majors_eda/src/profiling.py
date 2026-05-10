import logging
import warnings
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def run(raw_dir: Path, processed_dir: Path, profiles_dir: Path) -> None:
    profiles_dir.mkdir(parents=True, exist_ok=True)

    raw_files = {
        "majors_all_ages": raw_dir / "majors_all_ages.csv",
        "majors_degree_levels": raw_dir / "majors_degree_levels.csv",
        "majors_recent_graduates": raw_dir / "majors_recent_graduates.csv",
    }

    log.info("Profiling raw datasets (minimal=True)...")
    for name, path in raw_files.items():
        log.info(f"  {name} ...")
        df = pd.read_csv(path)
        profile = ProfileReport(df, minimal=True, title=name)
        out = profiles_dir / f"{name}.html"
        profile.to_file(out)
        log.info(f"  Saved → {out}")

    analytics_files = {
        "majors_all_ages_analytics": processed_dir
        / "majors_all_ages_analytics.parquet",
        "majors_degree_levels_analytics": processed_dir
        / "majors_degree_levels_analytics.parquet",
        "majors_recent_graduates_analytics": processed_dir
        / "majors_recent_graduates_analytics.parquet",
    }

    log.info("Profiling analytics datasets...")
    for name, path in analytics_files.items():
        log.info(f"  {name} ...")
        df = pd.read_parquet(path)
        profile = ProfileReport(df, title=name)
        out = profiles_dir / f"{name}.html"
        profile.to_file(out)
        log.info(f"  Saved → {out}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        PROJECT_DIR / "data" / "raw",
        PROJECT_DIR / "data" / "processed",
        PROJECT_DIR / "artifacts" / "profiles",
    )
