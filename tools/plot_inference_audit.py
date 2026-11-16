"""Export simulation audit figures from completed, checked result tables."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


COLORS = {"oracle": "#365e8d", "naive": "#c04432", "calibrated": "#007e73"}


def draw_grid(frame, stage, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True, sharey=True)
    subset = frame[frame.stage == stage]
    for row, pair in enumerate(("positive", "negative")):
        for column, loading in enumerate((.5, .75, .9)):
            axis = axes[row, column]
            block = subset[(subset.pair == pair) & (subset.loading == loading)].sort_values("multiplier")
            for label, key in (("Oracle OLS", "oracle"), ("Uncalibrated two-step", "naive"),
                               ("Null-calibrated diagnostic", "calibrated")):
                prefix = {"oracle": "oracle_rejection_rate", "naive": "observed_naive_rejection_rate",
                          "calibrated": "observed_calibrated_rejection_rate"}[key]
                low = block["calibrated_two_sample_bootstrap_low"] if key == "calibrated" else block[prefix + "_low"]
                high = block["calibrated_two_sample_bootstrap_high"] if key == "calibrated" else block[prefix + "_high"]
                axis.plot(block.multiplier, block[prefix], "o-", label=label, color=COLORS[key], linewidth=1.7, markersize=4)
                axis.fill_between(block.multiplier, low, high, color=COLORS[key], alpha=.12)
            axis.axhline(.8 if stage == "alternative" else .05, linestyle="--", color="#626b70", linewidth=1)
            axis.set_title(f"Loading {loading:.2f}", fontsize=11)
            axis.set_xticks([.5, 1, 2], ["½×", "1×", "2×"])
            axis.set_ylim(0, 1 if stage == "alternative" else .65)
            axis.grid(axis="y", alpha=.15)
            axis.spines[["top", "right"]].set_visible(False)
            if column == 0:
                contrast = ".15 contrast · N₀=770" if pair == "positive" else "−.05 contrast · N₀=6,884"
                axis.set_ylabel(contrast + "\nRejection probability")
            if row == 1:
                axis.set_xlabel("Sample size / advertised threshold")
    title = "Does the advertised sample size deliver 80% power?" if stage == "alternative" else "A high rejection rate can conceal false positives"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .055))
    fig.text(.5, .014, "Simulation only · 1,000 independent surveys per stage and cell · pointwise 95% Monte Carlo intervals\n"
             "Calibrated bands include uncertainty from a separate 1,000-draw null calibration; diagnostic is DGP-specific.",
             ha="center", fontsize=8, color="#454b50")
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main():
    destination = ROOT / "results/inference_audit"
    table = pd.read_csv(destination / "summary.csv")
    if len(table) != 36 or not (table.attempts == 1000).all():
        raise ValueError("Expected the complete preregistered 18-cell, 1,000-rep design")
    for stage in ("alternative", "null_validation"):
        draw_grid(table, stage, destination / f"figure_{stage}")
    # Machine-readable small display table; avoids dropping the type-I companion.
    rows = []
    for _, alt in table[(table.stage == "alternative") & (table.multiplier == 1)].iterrows():
        null = table[(table.cell == alt.cell) & (table.stage == "null_validation")].iloc[0]
        rows.append({"pair": alt.pair, "loading": alt.loading, "n": int(alt.n),
                     "oracle_power": alt.oracle_rejection_rate,
                     "naive_rejection": alt.observed_naive_rejection_rate,
                     "naive_false_positive": null.observed_naive_rejection_rate,
                     "calibrated_diagnostic_rejection": alt.observed_calibrated_rejection_rate,
                     "calibrated_diagnostic_low": alt.calibrated_two_sample_bootstrap_low,
                     "calibrated_diagnostic_high": alt.calibrated_two_sample_bootstrap_high,
                     "calibrated_heldout_false_positive": null.observed_calibrated_rejection_rate,
                     "conventional_fit_rate": alt.fit_good_rate,
                     "wrong_ranking_given_good_fit": alt.wrong_ranking_given_good_fit})
    pd.DataFrame(rows).to_csv(destination / "threshold_comparison.csv", index=False)


if __name__ == "__main__":
    main()
