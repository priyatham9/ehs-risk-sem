"""Study 4. What the coefficients 0.45, 0.30, -0.25, -0.20 can and cannot mean.

SIMULATION. All data are generated from known models.

A formula of the shape

    Risk = 0.45*UnsafeActs + 0.30*OperationalStress
 - 0.25*SystemCondition - 0.20*SafetyResponseCapability

circulates in practitioner writing on safety analytics. This study takes it
seriously enough to ask four questions about it.

1. **Are these plausible as estimates at all?** Their absolute values sum to
   exactly 1.20 and each lands on a round twentieth. The study simulates what
   estimated coefficient vectors actually look like and reports how often four
   estimates land on round twentieths by chance.

2. **Do the coefficients identify a causal structure?** Several structurally
   different generating models are constructed, all of which produce the same
   fitted coefficient vector. A reader shown only the coefficients cannot tell
   which world they came from, and the four worlds imply different consequences
   of intervening.

3. **What happens if the formula is applied to raw data?** Standardized
   coefficients multiply z-scores. Feeding a raw count of unsafe acts or a 1-5
   stress rating into them produces a number with no defined range and no
   stable meaning. The study computes the same crew's score at two sites whose
   variables have different means and spreads, and shows the ranking reverse.

4. **Is the score comparable across units?** If the measurement model differs
   between two sites -- different loadings, which is what measurement
   non-invariance means -- then the same latent value produces different scores.
   The study measures the disagreement.

Run: ``python3 simulations/study_04_coefficient_interpretation.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.measurement import factor_scores, fit_congeneric_block
from ehs_risk_sem.model import ModelSpec, fit
from simulations.dgp import TrueModel, generate_dataset, make_four_factor_model

HEADLINE = np.array([0.45, 0.30, -0.25, -0.20])
PREDICTORS = [
    "UnsafeActs",
    "OperationalStress",
    "SystemCondition",
    "SafetyResponseCapability",
]


def _spec(model: TrueModel) -> ModelSpec:
    return ModelSpec(
        latents={n: model.indicator_names[n] for n in model.latent_names},
        paths=model.paths(),
    )


def roundness_check(n: int = 500, n_reps: int = 2000, seed: int = 8) -> pd.DataFrame:
    """How often do four estimated coefficients all land on round twentieths?

    A single estimate rounds to the nearest 0.05 about 100% of the time by
    construction, so the question has to be asked properly: how close are four
    estimates, jointly, to a grid of twentieths, and how often does their
    absolute sum land within a hair of a round number?

    The comparison quantity is the distance from the estimated vector to the
    nearest point on the 0.05 grid. Under any realistic sampling distribution
    that distance is on the order of the standard error, not zero.
    """
    model = make_four_factor_model(beta=tuple(HEADLINE))
    spec = _spec(model)
    rng = np.random.default_rng(seed)

    distances = np.zeros(n_reps)
    abs_sums = np.zeros(n_reps)
    exact = 0
    for r in range(n_reps):
        data, _ = generate_dataset(model, n, seed=int(rng.integers(0, 2**31 - 1)))
        est = fit(spec, data).paths["Risk"].beta
        grid = np.round(est / 0.05) * 0.05
        distances[r] = float(np.max(np.abs(est - grid)))
        abs_sums[r] = float(np.sum(np.abs(est)))
        if np.all(np.abs(est - grid) < 0.005):
            exact += 1

    # If each coefficient's position within its 0.05 cell were uniform, the
    # chance that all four land within 0.005 of a grid point is (0.01/0.05)^4.
    reference = (0.01 / 0.05) ** 4
    return pd.DataFrame(
        [
            {
                "quantity": "max distance from the nearest 0.05 grid point "
                "(bounded above by 0.025)",
                "median": float(np.median(distances)),
                "p05": float(np.quantile(distances, 0.05)),
                "p95": float(np.quantile(distances, 0.95)),
            },
            {
                "quantity": "sum of absolute coefficients",
                "median": float(np.median(abs_sums)),
                "p05": float(np.quantile(abs_sums, 0.05)),
                "p95": float(np.quantile(abs_sums, 0.95)),
            },
            {
                "quantity": f"observed fraction of {n_reps} replications with all "
                "four estimates within 0.005 of a twentieth",
                "median": exact / n_reps,
                "p05": np.nan,
                "p95": np.nan,
            },
            {
                "quantity": "expected fraction under a uniform position within cell",
                "median": reference,
                "p05": np.nan,
                "p95": np.nan,
            },
        ]
    )


def _solve_confounded_world(
    target: np.ndarray,
    hidden_to_predictor: float = 0.55,
    hidden_to_outcome: float = 0.35,
    exogenous_correlation: float = 0.30,
    n_iter: int = 200,
) -> TrueModel:
    """Build a confounded world whose omitted-variable regression returns ``target``.

    An unmeasured latent causes both ``UnsafeActs`` and ``Risk``. The direct
    effects of the four predictors on ``Risk`` are solved by fixed-point
    iteration so that the population regression of ``Risk`` on the four
    *measured* predictors reproduces ``target`` exactly. The generating direct
    effects then differ from the fitted coefficients by the amount of
    confounding, and an analyst who sees only the fitted coefficients has no
    way to detect it.
    """
    names = ["Hidden"] + PREDICTORS + ["Risk"]
    exo = np.full((4, 4), float(exogenous_correlation))
    np.fill_diagonal(exo, 1.0)

    def build(direct: np.ndarray) -> TrueModel:
        b = np.zeros((6, 6))
        b[1, 0] = hidden_to_predictor
        b[5, 0] = hidden_to_outcome
        b[5, 1:5] = direct
        return TrueModel(
            latent_names=names,
            loadings={k: np.full(3, 0.75) for k in names},
            beta=b,
            exogenous_corr=exo,
        )

    direct = np.array(target, dtype=float).copy()
    for _ in range(n_iter):
        phi = build(direct).implied_latent_correlation()
        obtained = np.linalg.solve(phi[1:5, 1:5], phi[1:5, 5])
        step = target - obtained
        direct = direct + step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return build(direct)


def structural_ambiguity(n: int = 60_000, seed: int = 12) -> pd.DataFrame:
    """Four generating worlds, one coefficient vector.

    Each world is constructed so that a regression of the outcome on the four
    predictors returns approximately the headline coefficients, yet the worlds
    differ in what would happen under intervention.
    """
    rows: List[dict] = []

    # World 1: the coefficients are the direct causal effects.
    m1 = make_four_factor_model(beta=tuple(HEADLINE), exogenous_correlation=0.35)
    data1, _ = generate_dataset(m1, n, seed=seed)
    est1 = fit(_spec(m1), data1).paths["Risk"].beta
    rows.append(
        {
            "world": "1. four direct causes, correlated exogenously",
            "fitted_coefficients": np.round(est1, 3).tolist(),
            "effect_of_intervening_on_UnsafeActs": 0.45,
            "note": "the only world in which the coefficient is an intervention effect",
        }
    )

    # World 2: the same regression coefficients arise with a chain, so
    # SystemCondition acts partly through UnsafeActs.
    names = PREDICTORS + ["Risk"]
    b2 = np.zeros((5, 5))
    b2[0, 2] = 0.40  # SystemCondition -> UnsafeActs
    b2[4, 0] = HEADLINE[0]
    b2[4, 1] = HEADLINE[1]
    b2[4, 2] = HEADLINE[2]
    b2[4, 3] = HEADLINE[3]
    exo_corr = np.full((3, 3), 0.35)
    np.fill_diagonal(exo_corr, 1.0)
    m2 = TrueModel(
        latent_names=names,
        loadings={k: np.full(3, 0.75) for k in names},
        beta=b2,
        exogenous_corr=exo_corr,
    )
    data2, _ = generate_dataset(m2, n, seed=seed + 1)
    spec2 = ModelSpec(
        latents={k: m2.indicator_names[k] for k in names},
        paths={"Risk": PREDICTORS},
    )
    est2 = fit(spec2, data2).paths["Risk"].beta
    total_sys = HEADLINE[2] + 0.40 * HEADLINE[0]
    rows.append(
        {
            "world": "2. SystemCondition also acts through UnsafeActs",
            "fitted_coefficients": np.round(est2, 3).tolist(),
            "effect_of_intervening_on_UnsafeActs": 0.45,
            "note": f"total effect of SystemCondition is {total_sys:+.3f}, not "
            f"{HEADLINE[2]:+.3f}; the reported coefficient is the direct effect only",
        }
    )

    # World 3: an unmeasured common cause of a predictor and the outcome,
    # constructed so the omitted-variable regression returns the headline vector.
    m3 = _solve_confounded_world(HEADLINE)
    true_direct = m3.beta[5, 1:5].copy()
    data3, _ = generate_dataset(m3, n, seed=seed + 2)
    measured = PREDICTORS + ["Risk"]
    spec3 = ModelSpec(
        latents={k: m3.indicator_names[k] for k in measured},
        paths={"Risk": PREDICTORS},
    )
    cols3 = [c for k in measured for c in m3.indicator_names[k]]
    est3 = fit(spec3, data3[cols3]).paths["Risk"].beta
    rows.append(
        {
            "world": "3. an unmeasured common cause of UnsafeActs and Risk",
            "fitted_coefficients": np.round(est3, 3).tolist(),
            "effect_of_intervening_on_UnsafeActs": round(float(true_direct[0]), 3),
            "note": "generating direct effects are "
            + np.array2string(np.round(true_direct, 3), separator=", ")
            + "; the population regression on the measured predictors returns the "
            "headline vector exactly",
        }
    )
    return pd.DataFrame(rows)


def raw_scale_demonstration(seed: int = 21) -> pd.DataFrame:
    """Apply the standardized formula to raw values at two sites.

    Standardized coefficients apply to z-scores computed in a reference
    population. Substituting raw values means the implicit reference population
    is whatever the units happen to be, so the same crew scores differently at
    two sites and the ranking between crews can invert.
    """
    # Two crews, identical in every physical respect.
    crews = pd.DataFrame(
        {
            "crew": ["A", "B"],
            "unsafe_acts_count": [6.0, 3.0],
            "overtime_hours": [4.0, 12.0],
            "system_condition_1to5": [3.0, 4.0],
            "response_capability_1to5": [3.0, 4.0],
        }
    )

    # Mean and standard deviation of each variable at each site. Only the
    # standard deviations affect the ranking between two crews, because the
    # means cancel in the difference of two z-scored composites.
    sites = {
        "Site 1 (overtime varies widely)": {
            "unsafe_acts_count": (5.0, 2.0),
            "overtime_hours": (10.0, 4.0),
            "system_condition_1to5": (3.5, 0.8),
            "response_capability_1to5": (3.5, 0.8),
        },
        "Site 2 (overtime tightly controlled)": {
            "unsafe_acts_count": (2.0, 2.0),
            "overtime_hours": (2.0, 0.5),
            "system_condition_1to5": (4.2, 0.4),
            "response_capability_1to5": (4.2, 0.4),
        },
    }
    cols = [
        "unsafe_acts_count",
        "overtime_hours",
        "system_condition_1to5",
        "response_capability_1to5",
    ]

    rows = []
    for site, stats in sites.items():
        for _, crew in crews.iterrows():
            raw = np.array([crew[c] for c in cols], dtype=float)
            z = np.array([(crew[c] - stats[c][0]) / stats[c][1] for c in cols])
            rows.append(
                {
                    "site": site,
                    "crew": crew["crew"],
                    "score_raw_inputs": float(HEADLINE @ raw),
                    "score_z_inputs": float(HEADLINE @ z),
                }
            )
    out = pd.DataFrame(rows)
    ranked = []
    for site in out["site"].unique():
        block = out[out["site"] == site]
        top = block.loc[block["score_z_inputs"].idxmax(), "crew"]
        ranked.extend([f"crew {top} scores higher"] * len(block))
    out["ranking_on_z_inputs"] = ranked
    return out


def invariance_demonstration(n: int = 5000, seed: int = 33) -> pd.DataFrame:
    """The same latent value scored under two different measurement models.

    Site A and Site B have identical latent distributions and identical
    structural coefficients. Only the loadings differ -- one indicator is a
    much weaker measure at Site B, which is what happens when a survey item or
    a leading-indicator definition is interpreted differently in two plants.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)

    def make_block(loadings: np.ndarray) -> pd.DataFrame:
        cols = {}
        for i, lam in enumerate(loadings):
            cols[f"i{i+1}"] = lam * latent + np.sqrt(1 - lam**2) * rng.standard_normal(n)
        return pd.DataFrame(cols)

    site_a = make_block(np.array([0.80, 0.78, 0.76]))
    site_b = make_block(np.array([0.80, 0.78, 0.30]))

    sol_a = fit_congeneric_block(site_a.to_numpy(), list(site_a.columns), "L", [0, 1, 2])
    sol_b = fit_congeneric_block(site_b.to_numpy(), list(site_b.columns), "L", [0, 1, 2])
    score_a = factor_scores(site_a.to_numpy(), sol_a.loadings, sol_a.uniquenesses)
    score_b_own = factor_scores(site_b.to_numpy(), sol_b.loadings, sol_b.uniquenesses)
    score_b_imposed = factor_scores(site_b.to_numpy(), sol_a.loadings, sol_a.uniquenesses)

    def pct_disagree(s1: np.ndarray, s2: np.ndarray, q: float = 0.90) -> float:
        t1 = np.quantile(s1, q)
        t2 = np.quantile(s2, q)
        return float(np.mean((s1 >= t1) != (s2 >= t2)))

    return pd.DataFrame(
        [
            {
                "quantity": "Site A loadings",
                "value": np.round(sol_a.loadings, 3).tolist(),
            },
            {
                "quantity": "Site B loadings",
                "value": np.round(sol_b.loadings, 3).tolist(),
            },
            {
                "quantity": "Site A composite reliability",
                "value": round(sol_a.omega, 3),
            },
            {
                "quantity": "Site B composite reliability",
                "value": round(sol_b.omega, 3),
            },
            {
                "quantity": "correlation of Site B scores under its own vs Site A's "
                "measurement model",
                "value": round(float(np.corrcoef(score_b_own, score_b_imposed)[0, 1]), 4),
            },
            {
                "quantity": "share of Site B units whose top-decile flag flips when "
                "Site A's measurement model is imposed",
                "value": round(pct_disagree(score_b_own, score_b_imposed), 4),
            },
            {
                "quantity": "correlation of Site A score with the generating latent",
                "value": round(float(np.corrcoef(score_a, latent)[0, 1]), 4),
            },
            {
                "quantity": "correlation of Site B score with the generating latent",
                "value": round(float(np.corrcoef(score_b_own, latent)[0, 1]), 4),
            },
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    silence_accelerate_matmul_warnings()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 90)

    print(__doc__)
    print("=" * 78)

    print("\n1. Do estimates land on round twentieths?\n")
    rc = roundness_check(n_reps=args.reps)
    print(rc.to_string(index=False))
    rc.to_csv(os.path.join(args.outdir, "study04_roundness.csv"), index=False)
    print(
        f"\n   |0.45| + |0.30| + |-0.25| + |-0.20| = {np.sum(np.abs(HEADLINE)):.2f}.\n"
        "   Nothing forces estimated standardized coefficients over correlated\n"
        "   predictors to sum to a round number, and the simulated distribution\n"
        "   above shows how wide that sum is in practice. A coefficient vector\n"
        "   with this shape is better described as a chosen weighting than as an\n"
        "   estimate -- which is a legitimate thing to publish, provided it is\n"
        "   labelled that way and not given standard errors."
    )

    print("\n2. Different worlds, the same coefficients\n")
    sa = structural_ambiguity()
    print(sa.to_string(index=False))
    sa.to_csv(os.path.join(args.outdir, "study04_ambiguity.csv"), index=False)
    print(
        "\n   The fitted coefficients are close across all three worlds. The effect\n"
        "   of actually changing UnsafeActs is not. A coefficient is an\n"
        "   intervention effect only under an assumed graph, and the graph is not\n"
        "   recoverable from the coefficients."
    )

    print("\n3. Standardized coefficients applied to raw inputs\n")
    rs = raw_scale_demonstration()
    print(rs.to_string(index=False))
    rs.to_csv(os.path.join(args.outdir, "study04_raw_scale.csv"), index=False)
    a_raw = rs[rs["crew"] == "A"]["score_raw_inputs"].iloc[0]
    b_raw = rs[rs["crew"] == "B"]["score_raw_inputs"].iloc[0]
    top_by_site = rs.groupby("site")["ranking_on_z_inputs"].first().to_dict()
    print(
        f"\n   With raw inputs the score ignores the site entirely (crew A: "
        f"{a_raw:.3f}, crew B: {b_raw:.3f}), because raw units carry no\n"
        "   reference population and the coefficients were defined for z-scores.\n"
        "   With z-scored inputs the ranking between the same two crews reverses:"
    )
    for site, who in top_by_site.items():
        print(f"     {site}: {who}")
    print(
        "   Only the standard deviations differ in the comparison, and only the\n"
        "   standard deviations are needed to flip it. Neither version has a\n"
        "   probability scale, an exposure denominator or a time window, so no\n"
        "   threshold on either one can be justified."
    )

    print("\n4. Measurement non-invariance between two sites\n")
    inv = invariance_demonstration()
    print(inv.to_string(index=False))
    inv.to_csv(os.path.join(args.outdir, "study04_invariance.csv"), index=False)
    print(
        "\n   A score of 3.2 does not mean the same thing at two sites unless the\n"
        "   measurement model is invariant across them, and invariance is a\n"
        "   testable claim that safety SEM papers rarely test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
