"""Evaluate existing raw halide run UIDs with HalideEvaluation.

Examples
--------
Evaluate UIDs from the command line and print JSON::

    python scripts/ML_agent/evaluate_halide_uids.py UID1 UID2

Evaluate UIDs listed one-per-line in a text file and save CSV::

    python scripts/ML_agent/evaluate_halide_uids.py --uids-file uids.txt --output outcomes.csv

For quick beamline use, you can also edit the UIDS list below and run this
script with no positional UID arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tiled.client import from_profile, from_uri


# Edit this list for quick interactive use.
UIDS: list[str] = []


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation_halide import HalideEvaluation
from agent_halide import PLQY_PARAMS, SANDBOX_URI, TILED_PROFILE


def _load_uids(args: argparse.Namespace) -> list[str]:
    uids = list(args.uids or [])

    if args.uids_file:
        with open(args.uids_file, "r") as f:
            uids.extend(
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            )

    if not uids:
        uids = list(UIDS)

    if not uids:
        raise SystemExit(
            "No UIDs supplied. Pass UIDs as arguments, use --uids-file, "
            "or edit UIDS in this script."
        )

    return uids


def _raw_tiled_client(raw_uri: str | None, raw_profile: str):
    if raw_uri:
        return from_uri(raw_uri)
    return from_profile(raw_profile)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate existing raw halide run UIDs using HalideEvaluation."
    )
    parser.add_argument("uids", nargs="*", help="Raw catalog run UIDs to evaluate.")
    parser.add_argument(
        "--uids-file",
        help="Text file containing raw run UIDs, one per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--raw-profile",
        default=os.environ.get("TILED_PROFILE", TILED_PROFILE),
        help="Tiled profile for raw data. Ignored if --raw-uri is supplied.",
    )
    parser.add_argument(
        "--raw-uri",
        default=os.environ.get("TILED_URI", ""),
        help="Tiled URI for raw data. If omitted, --raw-profile is used.",
    )
    parser.add_argument(
        "--sandbox-uri",
        default=os.environ.get("TILED_SANDBOX_URI", SANDBOX_URI),
        help="Tiled sandbox URI containing pdfstream outputs.",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Results are always printed as JSON.",
    )
    args = parser.parse_args()

    uids = _load_uids(args)
    raw_client = _raw_tiled_client(args.raw_uri or None, args.raw_profile)
    sandbox_client = from_uri(args.sandbox_uri)

    evaluator = HalideEvaluation(
        tiled_client=raw_client,
        sandbox_client=sandbox_client,
        plqy_params=PLQY_PARAMS,
    )

    outcomes = []
    for i, uid in enumerate(uids):
        result = evaluator(uid, [{"_id": i}])[0]
        result["uid"] = uid
        outcomes.append(result)

    print(json.dumps(outcomes, indent=2, default=float))

    if args.output:
        df = pd.DataFrame(outcomes)
        df.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
