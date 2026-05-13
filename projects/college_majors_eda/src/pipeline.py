import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cleaning
import degree_levels_analysis
import feature_engineering
import gender_analysis
import profiling
import report
import salaries_analysis

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
PROFILES_DIR = PROJECT_DIR / "artifacts" / "profiles"
PLOTS_DIR = PROJECT_DIR / "artifacts" / "plots"
REPORTS_DIR = PROJECT_DIR / "reports"

_STAGE_ALIASES = {
    "cl": "cleaning",
    "fe": "feature_engineering",
    "sal": "salaries",
    "salaries": "salaries",
    "dl": "degree_levels",
    "degree_levels": "degree_levels",
    "gen": "gender",
    "gender": "gender",
    "profiling": "profiling",
    "cleaning": "cleaning",
    "feature_engineering": "feature_engineering",
}

_ORDERED_STAGES = [
    "profiling",
    "cleaning",
    "feature_engineering",
    "salaries",
    "degree_levels",
    "gender",
]


def _run_stage(name: str, log: logging.Logger) -> None:
    log.info("=== Stage: %s ===", name)
    if name == "profiling":
        profiling.run(RAW_DATA_DIR, PROCESSED_DATA_DIR, PROFILES_DIR)
    elif name == "cleaning":
        cleaning.run(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    elif name == "feature_engineering":
        feature_engineering.run(PROCESSED_DATA_DIR)
    elif name == "salaries":
        salaries_analysis.run(PROCESSED_DATA_DIR, PLOTS_DIR / "salaries")
    elif name == "degree_levels":
        degree_levels_analysis.run(PROCESSED_DATA_DIR, PLOTS_DIR / "degree_levels")
    elif name == "gender":
        gender_analysis.run(PROCESSED_DATA_DIR, PLOTS_DIR / "gender")
    else:
        raise ValueError(f"Unknown stage: {name!r}")
    log.info("=== Done: %s ===", name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pipeline")

    parser = argparse.ArgumentParser(
        description="College Majors EDA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        metavar="STAGE",
        help=(
            "Stages to run. Use 'all' for the full pipeline. "
            f"Available: {', '.join(_ORDERED_STAGES)}"
        ),
    )
    args = parser.parse_args()

    if args.stages == ["all"]:
        stages = _ORDERED_STAGES
    else:
        stages = []
        for s in args.stages:
            canonical = _STAGE_ALIASES.get(s)
            if canonical is None:
                parser.error(
                    f"Unknown stage '{s}'. "
                    f"Available: {', '.join(_ORDERED_STAGES)} (or aliases: cl, fe, sal, dl, gen)"
                )
            stages.append(canonical)
        seen = set()
        ordered = []
        for s in _ORDERED_STAGES:
            if s in stages and s not in seen:
                ordered.append(s)
                seen.add(s)
        stages = ordered

    run_all = args.stages == ["all"]

    log.info("Running stages: %s", stages)
    for stage in stages:
        _run_stage(stage, log)

    if run_all:
        log.info("=== Stage: report ===")
        report.run(PROCESSED_DATA_DIR, REPORTS_DIR)
        log.info("=== Done: report ===")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
