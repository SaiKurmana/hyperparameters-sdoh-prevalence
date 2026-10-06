"""Download CDC PLACES county data, then build the analytic file.

    python scripts/download_data.py                # outcome + all social-need measures
    python scripts/download_data.py --list         # show every available measure ID
    python scripts/download_data.py --age-adjusted # AgeAdjPrv instead of CrdPrv

No API key is required. CDC re-publishes PLACES annually and has changed measure
IDs between releases, so the requested IDs are validated against the live dataset
first and a mismatch prints what is actually available.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdoh_prevalence import config as cfg  # noqa: E402
from sdoh_prevalence.data import (  # noqa: E402
    available_measures,
    download_places,
    load_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the measure IDs in the live PLACES county dataset and exit",
    )
    parser.add_argument(
        "--age-adjusted",
        action="store_true",
        help="download age-adjusted prevalence instead of crude prevalence",
    )
    parser.add_argument(
        "--measures",
        nargs="*",
        default=None,
        help="override the measure IDs to download (default: outcome + social needs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.PLACES_RAW_CSV,
        help=f"raw CSV destination (default: {cfg.PLACES_RAW_CSV})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        print("requests is not installed. Run: pip install -r requirements.txt")
        return 2

    if args.list:
        measures = available_measures()
        print(f"{len(measures)} measures in dataset {cfg.PLACES_DATASET_ID}:\n")
        print(measures.to_string(index=False))
        return 0

    value_type = "AgeAdjPrv" if args.age_adjusted else cfg.DATA_VALUE_TYPE_ID

    try:
        raw_path = download_places(
            measures=args.measures,
            data_value_type=value_type,
            out_path=args.out,
        )
    except Exception as exc:  # noqa: BLE001 - the message is the useful part
        print(f"\nDownload failed: {exc}\n")
        print(f"Dataset landing page: {cfg.PLACES_LANDING_PAGE}")
        print("Run with --list to see the measure IDs this release actually has.")
        return 1

    print(f"\nRaw long-format file: {raw_path}")

    wide = load_dataset()
    cfg.ANALYTIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(cfg.ANALYTIC_CSV, index=False)

    print(f"Analytic wide file:   {cfg.ANALYTIC_CSV}")
    print(f"Counties:             {len(wide):,}")
    print(f"States:               {wide['stateabbr'].nunique()}")
    print(f"Value type:           {value_type}")
    print("\nNext: python scripts/03_regularization_sdoh.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
