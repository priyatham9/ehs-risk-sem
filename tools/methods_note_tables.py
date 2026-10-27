"""Build the tables and quoted numbers in ``docs/methods-note.md`` from ``results/``.

Every number in the methods note's Results section is read from a committed
CSV in ``results/`` by this script. Nothing is typed in by hand. The tables sit
in the note between marker comments::

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

Usage::

    python3 tools/methods_note_tables.py           # rewrite the blocks in place
    python3 tools/methods_note_tables.py --check   # exit 1 if the note is stale

``tests/test_methods_note.py`` runs the same check, and also checks that each
number quoted in the note's prose (``key_facts``) appears in the note.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
NOTE = os.path.join(ROOT, "docs", "methods-note.md")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def _int(x: float) -> str:
    return f"{int(round(float(x))):,}"


def _f(x: float, d: int = 3) -> str:
    return f"{float(x):.{d}f}"


def _md(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def table_requirements() -> str:
    df = _read("study01_requirements.csv")
    rows = [[r.requirement, r.basis, _int(r.n_required)] for r in df.itertuples()]
    return _md(["Requirement", "Basis", "N required"], rows)


def table_precision() -> str:
    df = _read("study01_recovery.csv")
    rows = []
    for n, g in df.groupby("n", sort=True):
        rows.append(
            [
                _int(n),
                _int(g["n_reps"].min()),
                _f(g["bias"].abs().max(), 4),
                f"{_f(g['empirical_sd'].min())} to {_f(g['empirical_sd'].max())}",
                _f(1.96 * g["empirical_sd"].max(), 2),
                f"{_f(g['se_ratio_analytic_over_empirical'].min(), 2)} to "
                f"{_f(g['se_ratio_analytic_over_empirical'].max(), 2)}",
                f"{_f(g['coverage_95_analytic'].min(), 2)} to "
                f"{_f(g['coverage_95_analytic'].max(), 2)}",
                _int(g["improper_solutions"].max() + g["failed_fits"].max()),
            ]
        )
    return _md(
        [
            "N",
            "Replications",
            "Largest abs. bias",
            "Empirical SD of path estimates",
            "Approx. 95% half-width (1.96 x largest SD)",
            "Analytic SE / empirical SD",
            "Analytic 95% CI coverage",
            "Improper or failed fits",
        ],
        rows,
    )


def table_distinguish() -> str:
    df = _read("study01_distinguishability.csv")
    rows = [
        [
            f"{r.coefficient_a} ({r.beta_a:+.2f}) vs {r.coefficient_b} ({r.beta_b:+.2f})",
            _f(r.absolute_difference, 2),
            _f(r.se_of_difference_at_n_250, 3),
            _int(r.n_for_80pct_power),
        ]
        for r in df.itertuples()
    ]
    return _md(
        ["Pair (generating values)", "Difference", "SE of difference at N = 250", "N for 80% power"],
        rows,
    )


def table_bootstrap() -> str:
    df = _read("study01_bootstrap.csv")
    rows = [
        [
            f"{r.predictor} ({r.true_beta:+.2f})",
            _f(r.coverage_analytic, 3),
            _f(r.coverage_bootstrap, 3),
            _f(r.mean_width_analytic, 3),
            _f(r.mean_width_bootstrap, 3),
        ]
        for r in df.itertuples()
    ]
    return _md(
        ["Path", "Analytic coverage", "Bootstrap coverage", "Analytic width", "Bootstrap width"],
        rows,
    )


def table_fit_vs_truth() -> str:
    eq = _read("study03_equivalent.csv")
    rev = _read("study03_reverse.csv").iloc[0]
    conf = _read("study03_confounder.csv")
    omitted = conf[conf["fitted_model"] == "confounder omitted"].iloc[0]
    reversed_ = eq[eq["fitted_model"].str.contains("reversed")].iloc[0]
    rows = [
        [
            "3A: Effect -> Cause fitted to Cause -> Effect data",
            _f(reversed_.cfi), _f(reversed_.rmsea), _f(reversed_.srmr),
            str(bool(reversed_.meets_conventional_cutoffs)),
            f"beta = {_f(reversed_.beta)}, direction wrong",
        ],
        [
            "3B: Climate -> Injuries fitted to Injuries -> Climate data",
            _f(rev.cfi), _f(rev.rmsea), _f(rev.srmr),
            str(bool(rev.meets_conventional_cutoffs)),
            f"beta = {_f(rev.estimated_beta)} (SE {_f(rev.se)}), direction wrong",
        ],
        [
            "3C: common cause omitted",
            _f(omitted.cfi), _f(omitted.rmsea), "not recorded",
            str(bool(omitted.meets_conventional_cutoffs)),
            f"beta = {_f(omitted.beta_predictor_on_outcome)} vs generating "
            f"{_f(omitted.true_direct_effect)} (bias {_f(omitted.bias)})",
        ],
    ]
    return _md(["Scenario", "CFI", "RMSEA", "SRMR", "Meets cutoffs", "What the estimate says"], rows)


TABLES = {
    "requirements": table_requirements,
    "precision": table_precision,
    "distinguish": table_distinguish,
    "bootstrap": table_bootstrap,
    "fit_vs_truth": table_fit_vs_truth,
}


def key_facts() -> Dict[str, str]:
    """Numbers quoted in the note's prose, each formatted as it appears there."""
    req = _read("study01_requirements.csv").set_index("requirement")["n_required"]
    rec = _read("study01_recovery.csv")
    dist = _read("study01_distinguishability.csv")
    boot = _read("study01_bootstrap.csv")
    rnd = _read("study04_roundness.csv").set_index("quantity")
    sd = rec.groupby("n")["empirical_sd"].max()

    def req_n(prefix: str) -> str:
        return _int(req[[k for k in req.index if k.startswith(prefix)][0]])

    width_ratio = boot["mean_width_bootstrap"] / boot["mean_width_analytic"]
    sum_abs = rnd.loc["sum of absolute coefficients"]
    return {
        "n_close_fit": req_n("global fit"),
        "n_se_05": req_n("path precision: SE = 0.05"),
        "n_se_025": req_n("path precision: SE = 0.025"),
        "n_045_vs_030": req_n("distinguish beta = 0.45"),
        "n_025_vs_020": req_n("distinguish beta = -0.25"),
        "half_width_n100": _f(1.96 * sd.loc[100], 2),
        "half_width_n200": _f(1.96 * sd.loc[200], 2),
        "half_width_n500": _f(1.96 * sd.loc[500], 2),
        "se_diff_n250": _f(dist["se_of_difference_at_n_250"].max(), 3),
        "max_n_large_gaps": _int(dist[dist["absolute_difference"] > 0.4]["n_for_80pct_power"].max()),
        "cov_analytic_min": _f(rec["coverage_95_analytic"].min(), 2),
        "cov_analytic_max": _f(rec["coverage_95_analytic"].max(), 2),
        "cov_boot_min": _f(boot["coverage_bootstrap"].min(), 3),
        "cov_boot_max": _f(boot["coverage_bootstrap"].max(), 3),
        "boot_reps": _int(boot["n_reps"].iloc[0]),
        "boot_analytic_min": _f(boot["coverage_analytic"].min(), 3),
        "boot_analytic_max": _f(boot["coverage_analytic"].max(), 3),
        "boot_wider_min_pct": _int(100 * (width_ratio.min() - 1)),
        "boot_wider_max_pct": _int(100 * (width_ratio.max() - 1)),
        "se_ratio_min_pct": _int(100 * rec["se_ratio_analytic_over_empirical"].min()),
        "se_ratio_max_pct": _int(100 * rec["se_ratio_analytic_over_empirical"].max()),
        "boot_mc_se_pts": _f(100 * boot["monte_carlo_se_of_coverage"].iloc[0], 1),
        "sum_abs_median": _f(sum_abs["median"], 2),
        "sum_abs_p05": _f(sum_abs["p05"], 2),
        "sum_abs_p95": _f(sum_abs["p95"], 2),
    }


_BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>\w+) -->\n)(?P<body>.*?)(\n<!-- END GENERATED: (?P=name) -->)",
    re.S,
)


def render(text: str) -> str:
    """Return ``text`` with every generated block rebuilt from ``results/``."""

    def sub(m: "re.Match[str]") -> str:
        name = m.group("name")
        if name not in TABLES:
            raise KeyError(f"unknown generated block: {name}")
        return m.group(1) + TABLES[name]() + m.group(4)

    return _BLOCK.sub(sub, text)


def blocks_in(text: str) -> List[str]:
    return [m.group("name") for m in _BLOCK.finditer(text)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    with open(NOTE, encoding="utf-8") as fh:
        text = fh.read()
    new = render(text)
    if args.check:
        if new != text:
            print("docs/methods-note.md is out of date with results/", file=sys.stderr)
            return 1
        return 0
    with open(NOTE, "w", encoding="utf-8") as fh:
        fh.write(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
