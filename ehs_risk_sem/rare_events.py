"""Base-rate arithmetic for occupational injury prediction.

Everything in this module is elementary. It is here because the elementary
arithmetic is what determines whether a real-time safety risk score can work at
all, and it is routinely skipped.

The recurring result: at occupational injury base rates, discrimination is
nearly meaningless as a decision criterion. A classifier with 80% sensitivity
and 95% specificity, applied per worker-shift, produces roughly one true event
per several hundred alerts. Raising specificity to 99.9% still leaves more than
ten alerts per event.

Hopkins (2009) reached the same conclusion without any statistics, calling it
the "zoom effect": a rate is meaningful at the level of aggregation where
enough events occur to form one, and meaningless below it.

Note on the base rate constant below. ``BLS_PRIVATE_TRC_RATE_2024 = 2.3`` per
100 full-time-equivalent workers is the sole real-world quantity in this
package. It is verified against the primary BLS source, which states: "The 2024
incidence rate of total recordable cases in private industry was 2.3 cases per
100 full-time equivalent workers, down from 2.4 in 2023." (US Bureau of Labor
Statistics, *The Economics Daily*, 23 March 2026, drawn from the
Employer-Reported Workplace Injuries and Illnesses program.) Every function
here still takes the rate as an argument, so nothing in the package depends on
the constant and a different industry or year can be substituted freely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BLS_PRIVATE_TRC_RATE_2024",
    "OSHA_RATE_BASE_HOURS",
    "ExposureModel",
    "rate_per_hour",
    "probability_per_shift",
    "shifts_per_event",
    "ppv_from_sens_spec",
    "alerts_per_true_event",
    "n_for_relative_precision",
    "events_for_relative_precision",
    "worker_shifts_for_calibration",
    "rate_ratio_ci",
]

# See module docstring: verified against the primary BLS source.
BLS_PRIVATE_TRC_RATE_2024 = 2.3
BLS_SOURCE_NOTE = (
    "Total recordable case rate per 100 full-time-equivalent workers, "
    "US private industry, 2024, from the BLS Employer-Reported Workplace "
    "Injuries and Illnesses program (BLS, The Economics Daily, 23 March 2026: "
    "2.3 cases per 100 FTE, down from 2.4 in 2023). This is the only "
    "real-world quantity in the package; everything else is generated from a "
    "known model. Every function takes the rate as an argument, so another "
    "industry, year or establishment rate can be substituted."
)

# OSHA incidence rates are defined per 200,000 hours worked, which is 100
# full-time-equivalent workers at 2,000 hours per year.
OSHA_RATE_BASE_HOURS = 200_000.0


def rate_per_hour(trir: float, base_hours: float = OSHA_RATE_BASE_HOURS) -> float:
    """Convert an OSHA-style incidence rate to a rate per worker-hour.

    ``TRIR`` is cases per ``base_hours`` hours worked. At TRIR 2.3 this is
    1.15e-5 per worker-hour.
    """
    return float(trir) / float(base_hours)


def probability_per_shift(
    trir: float, shift_hours: float = 8.0, base_hours: float = OSHA_RATE_BASE_HOURS
) -> float:
    """Probability that a given worker-shift contains a recordable case.

    Uses the rate directly rather than a Poisson exceedance probability, since
    at these rates the two agree to seven decimal places and the linear form is
    the one a reader can check by hand.
    """
    return rate_per_hour(trir, base_hours) * float(shift_hours)


def shifts_per_event(
    trir: float, shift_hours: float = 8.0, base_hours: float = OSHA_RATE_BASE_HOURS
) -> float:
    """Expected number of worker-shifts between recordable cases."""
    p = probability_per_shift(trir, shift_hours, base_hours)
    if p <= 0:
        return float("inf")
    return 1.0 / p


def ppv_from_sens_spec(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Positive predictive value from sensitivity, specificity and prevalence.

    ``PPV = sens * prev / (sens * prev + (1 - spec) * (1 - prev))``
    """
    sens = float(sensitivity)
    spec = float(specificity)
    prev = float(prevalence)
    if not (0 <= sens <= 1 and 0 <= spec <= 1 and 0 <= prev <= 1):
        raise ValueError("sensitivity, specificity and prevalence must lie in [0, 1]")
    num = sens * prev
    den = num + (1.0 - spec) * (1.0 - prev)
    if den <= 0:
        return float("nan")
    return num / den


def alerts_per_true_event(
    sensitivity: float, specificity: float, prevalence: float
) -> float:
    """Number of alerts raised per true event detected, ``1 / PPV``.

    This is the quantity that determines whether an operational alerting system
    will be believed. It is reported instead of, not alongside, a bare PPV in
    most of this repository's output because the reciprocal is the form people
    can act on.
    """
    ppv = ppv_from_sens_spec(sensitivity, specificity, prevalence)
    if ppv <= 0:
        return float("inf")
    return 1.0 / ppv


def n_for_relative_precision(p: float, rel_precision: float = 0.10, z: float = 1.959963985) -> float:
    """Sample size needed to estimate a proportion to a relative precision.

    ``n = z^2 (1 - p) / (rel^2 * p)``

    At ``p = 9.2e-5`` (one recordable per 8-hour shift at TRIR 2.3) and 10%
    relative precision this is roughly 4.2 million worker-shifts. Below that
    sample size, calibration-in-the-large cannot be estimated, let alone a
    calibration curve.
    """
    p = float(p)
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    return float(z**2 * (1.0 - p) / (rel_precision**2 * p))


def events_for_relative_precision(rel_precision: float = 0.10, z: float = 1.959963985) -> float:
    """Number of events (not observations) needed for a given relative precision.

    ``events = z^2 / rel^2``, approximately, since ``n * p`` at small ``p``. At
    10% relative precision this is about 384 events, whatever the base rate.
    """
    return float(z**2 / rel_precision**2)


@dataclass
class ExposureModel:
    """Translate an establishment-level incidence rate into unit-level risk.

    Parameters
 ----------
    trir
        Total recordable incidence rate per ``base_hours`` hours.
    shift_hours
        Length of the exposure window that a score would be attached to.
    base_hours
        Denominator of the incidence rate; 200,000 for OSHA-style rates.

    The class exists to make one point explicit: a risk score has no meaning
    until the unit of analysis and the exposure window are named. "This crew is
    at elevated risk" is not a claim until it says elevated over what period,
    relative to what denominator.
    """

    trir: float = BLS_PRIVATE_TRC_RATE_2024
    shift_hours: float = 8.0
    base_hours: float = OSHA_RATE_BASE_HOURS

    def per_hour(self) -> float:
        return rate_per_hour(self.trir, self.base_hours)

    def per_shift(self) -> float:
        return probability_per_shift(self.trir, self.shift_hours, self.base_hours)

    def per_worker_year(self, hours_per_year: float = 2000.0) -> float:
        return self.per_hour() * hours_per_year

    def shifts_between_events(self) -> float:
        return shifts_per_event(self.trir, self.shift_hours, self.base_hours)

    def alert_burden(self, sensitivity: float, specificity: float) -> dict:
        """Alerting arithmetic at this exposure model's base rate."""
        prev = self.per_shift()
        return {
            "base_rate_per_shift": prev,
            "shifts_per_event": self.shifts_between_events(),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv_from_sens_spec(sensitivity, specificity, prev),
            "alerts_per_true_event": alerts_per_true_event(sensitivity, specificity, prev),
        }

    def summary(self) -> str:
        return "\n".join(
            [
                "Exposure model",
                f"  incidence rate             : {self.trir} per {self.base_hours:,.0f} hours",
                f"  per worker-hour            : {self.per_hour():.3e}",
                f"  per {self.shift_hours:g}-hour shift"
                + " " * max(1, 14 - len(f"{self.shift_hours:g}"))
                + f": {self.per_shift():.3e}",
                f"  worker-shifts per event    : {self.shifts_between_events():,.0f}",
                f"  per worker-year            : {self.per_worker_year():.4f}",
            ]
        )


def worker_shifts_for_calibration(
    trir: float = BLS_PRIVATE_TRC_RATE_2024,
    shift_hours: float = 8.0,
    rel_precision: float = 0.10,
) -> dict:
    """Sample size needed to calibrate a shift-level risk model.

    Returns both the required number of worker-shifts and the number of events
    those shifts are expected to contain.
    """
    p = probability_per_shift(trir, shift_hours)
    n = n_for_relative_precision(p, rel_precision)
    return {
        "base_rate_per_shift": p,
        "relative_precision": rel_precision,
        "worker_shifts_required": n,
        "expected_events": n * p,
        "events_required_approx": events_for_relative_precision(rel_precision),
    }


def rate_ratio_ci(
    events_a: int, exposure_a: float, events_b: int, exposure_b: float, z: float = 1.959963985
) -> dict:
    """Confidence interval for a ratio of two Poisson rates.

    Included because comparing an establishment or a shift against a benchmark
    is the operational use of these rates, and the interval is usually wide
    enough at realistic event counts to make the comparison uninformative --
    which is worth showing rather than asserting.
    """
    if exposure_a <= 0 or exposure_b <= 0:
        raise ValueError("exposures must be positive")
    if events_a <= 0 or events_b <= 0:
        return {
            "rate_a": events_a / exposure_a,
            "rate_b": events_b / exposure_b,
            "rate_ratio": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "note": "a zero event count makes the log-scale interval undefined",
        }
    rate_a = events_a / exposure_a
    rate_b = events_b / exposure_b
    rr = rate_a / rate_b
    se_log = np.sqrt(1.0 / events_a + 1.0 / events_b)
    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "rate_ratio": float(rr),
        "ci_low": float(rr * np.exp(-z * se_log)),
        "ci_high": float(rr * np.exp(z * se_log)),
        "note": "",
    }
