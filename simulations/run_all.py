"""Run every simulation study in order and write the result tables to ``results/``.

Every dataset used by these studies is generated from a known model. None of
the numbers they produce is an empirical finding about workplace injuries.

Usage::

    python3 simulations/run_all.py            # full run, a few minutes
    python3 simulations/run_all.py --quick    # reduced replications, ~30 seconds
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from simulations import (
    study_01_sample_size,
    study_02_rare_events,
    study_03_misspecification,
    study_04_coefficient_interpretation,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced replication counts; results are noisier but the structure is the same",
    )
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    silence_accelerate_matmul_warnings()
    os.makedirs(args.outdir, exist_ok=True)

    if args.quick:
        plans = [
            ("study 1: sample size", study_01_sample_size.main,
             ["--reps", "60", "--skip-bootstrap", "--outdir", args.outdir]),
            ("study 2: rare events", study_02_rare_events.main,
             ["--n", "60000", "--outdir", args.outdir]),
            ("study 3: misspecification", study_03_misspecification.main,
             ["--outdir", args.outdir]),
            ("study 4: coefficient interpretation",
             study_04_coefficient_interpretation.main,
             ["--reps", "150", "--outdir", args.outdir]),
        ]
    else:
        plans = [
            ("study 1: sample size", study_01_sample_size.main,
             ["--reps", "300", "--boot-reps", "40", "--n-boot", "199",
              "--outdir", args.outdir]),
            ("study 2: rare events", study_02_rare_events.main,
             ["--n", "200000", "--outdir", args.outdir]),
            ("study 3: misspecification", study_03_misspecification.main,
             ["--outdir", args.outdir]),
            ("study 4: coefficient interpretation",
             study_04_coefficient_interpretation.main,
             ["--reps", "1000", "--outdir", args.outdir]),
        ]

    started = time.time()
    for label, entry, argv_i in plans:
        print("\n" + "#" * 78)
        print(f"# {label}")
        print("#" * 78 + "\n")
        code = entry(argv_i)
        if code != 0:
            print(f"{label} exited with status {code}")
            return code

    print("\n" + "#" * 78)
    print(f"# all studies complete in {time.time() - started:.1f}s")
    print(f"# tables written to {os.path.abspath(args.outdir)}")
    print("#" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
