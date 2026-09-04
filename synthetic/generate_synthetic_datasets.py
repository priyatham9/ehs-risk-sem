"""Generate the fixed synthetic datasets checked in under ``synthetic/``.

These files exist so that the worked example in ``README.md`` and the
integration tests run on the same data every time. They are generated from
known models by :mod:`simulations.dgp` with fixed seeds, so they can be
regenerated exactly:

    python3 synthetic/generate_synthetic_datasets.py

Every file written here carries a header comment on its first lines declaring
that it is synthetic, naming the generating model, the seed, and the date it
was produced. Nothing in these files is a measurement of anything. No number
computed from them is an empirical claim, and none appears as a headline result
anywhere in this repository.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.dgp import (
    generate_binary_outcome,
    generate_dataset,
    generate_zero_inflated_count,
    make_four_factor_model,
)

HERE = os.path.dirname(os.path.abspath(__file__))

WARNING_BANNER = "SYNTHETIC DATA - GENERATED, NOT MEASURED - NOT AN EMPIRICAL FINDING"


def _write_with_header(frame: pd.DataFrame, path: str, header_lines: List[str]) -> None:
    """Write a CSV preceded by ``#``-prefixed header lines.

    ``pandas.read_csv(path, comment='#')`` reads the file back. The header is
    part of the artifact on purpose: a file that leaves this directory should
    still say what it is.
    """
    stamp = _dt.date.today().isoformat()
    lines = [f"# {WARNING_BANNER}", f"# generated: {stamp}"]
    lines.extend(f"# {line}" for line in header_lines)
    lines.append(
        "# regenerate with: python3 synthetic/generate_synthetic_datasets.py"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        frame.to_csv(fh, index=False, float_format="%.6g")


def build_survey_style(n: int = 1200, seed: int = 20260903) -> pd.DataFrame:
    """A survey-style dataset: five latents, three indicators each, continuous."""
    model = make_four_factor_model()
    data, latent = generate_dataset(model, n, seed=seed)
    data = data.copy()
    for j, name in enumerate(model.latent_names):
        data[f"latent_{name}"] = latent[:, j]
    return data


def build_rare_event(n: int = 25_000, seed: int = 424242) -> pd.DataFrame:
    """An operational dataset: latent predictors and a rare binary outcome.

    The base rate is set to 0.005 per row, which is roughly what a monthly
    crew-level recordable rate looks like. It is a modelling choice made for
    the example, not an estimate from any data source.
    """
    model = make_four_factor_model()
    rng = np.random.default_rng(seed)
    data, latent = generate_dataset(model, n, seed=seed)
    y, p_true = generate_binary_outcome(
        latent[:, :4], (0.6, 0.4, -0.3, -0.25), 0.005, rng
    )
    out = data.iloc[:, :12].copy()
    out["recordable"] = y.astype(int)
    out["true_probability"] = p_true
    return out


def build_zero_inflated(n: int = 8000, seed: int = 771) -> pd.DataFrame:
    """An establishment-style dataset with a zero-inflated count outcome."""
    model = make_four_factor_model()
    rng = np.random.default_rng(seed)
    data, latent = generate_dataset(model, n, seed=seed)
    counts, mu, p_zero = generate_zero_inflated_count(
        latent[:, :4],
        count_coefficients=(0.35, 0.20, -0.15, -0.10),
        zero_coefficients=(-0.30, 0.0, 0.0, 0.25),
        base_mean=1.5,
        zero_inflation=0.38,
        rng=rng,
    )
    out = data.iloc[:, :12].copy()
    out["recordable_count"] = counts.astype(int)
    out["poisson_mean"] = mu
    out["structural_zero_probability"] = p_zero
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=HERE)
    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    survey = build_survey_style()
    _write_with_header(
        survey,
        os.path.join(args.outdir, "survey_style.csv"),
        [
            "generating model: five standardized latent variables, three congeneric",
            "  indicators each, all loadings 0.75",
            "structural paths into 'Risk': UnsafeActs 0.45, OperationalStress 0.30,",
            "  SystemCondition -0.25, SafetyResponseCapability -0.20",
            "exogenous latent correlations: 0.35 between every pair",
            "n = 1200, seed = 20260903",
            "columns prefixed 'latent_' are the generating latent scores. They are",
            "  never observable in real data and are included only so that examples",
            "  can compare an estimate against the value that produced it.",
        ],
    )

    rare = build_rare_event()
    _write_with_header(
        rare,
        os.path.join(args.outdir, "rare_event.csv"),
        [
            "twelve indicators of four latent predictors (loadings 0.75) plus a rare",
            "  binary outcome generated from a logistic model on the latent values",
            "outcome coefficients: 0.6, 0.4, -0.3, -0.25; marginal rate 0.005",
            "n = 25000, seed = 424242",
            "'true_probability' is the generating probability for each row. It does",
            "  not exist in real data and is present only to make calibration error",
            "  separable from outcome noise.",
        ],
    )

    zi = build_zero_inflated()
    _write_with_header(
        zi,
        os.path.join(args.outdir, "zero_inflated_counts.csv"),
        [
            "twelve indicators of four latent predictors plus a zero-inflated Poisson",
            "  count outcome",
            "count coefficients 0.35, 0.20, -0.15, -0.10; base mean 1.5",
            "structural-zero coefficients -0.30, 0, 0, 0.25; structural-zero rate 0.38",
            "n = 8000, seed = 771",
            "observed zeros are a mixture of Poisson zeros and structural zeros; the",
            "  two are not separable from the counts alone",
        ],
    )

    for name in ("survey_style.csv", "rare_event.csv", "zero_inflated_counts.csv"):
        path = os.path.join(args.outdir, name)
        size = os.path.getsize(path)
        frame = pd.read_csv(path, comment="#")
        print(f"{name}: {frame.shape[0]} rows x {frame.shape[1]} cols, {size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
