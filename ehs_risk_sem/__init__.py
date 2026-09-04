"""ehs-risk-sem: structural equation modelling for safety risk, stated honestly.

The package implements a limited-information SEM estimator on numpy alone, plus
the diagnostics that decide whether its output means anything: identification
checks, global fit indices, assumption tests, calibration measures for rare
binary outcomes, and sample-size calculations that separate global-fit power
from coefficient precision.

Its purpose is as much negative as positive. A model can be fitted, report
excellent fit, and support no causal claim whatever. The simulation studies
under ``simulations/`` demonstrate that on data whose generating process is
known.

Start with ``README.md``, then ``simulations/run_all.py``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (  # noqa: F401
    calibration,
    diagnostics,
    fit_indices,
    glm,
    identification,
    linalg,
    measurement,
    model,
    power,
    rare_events,
    special,
    structural,
)
from .model import ModelSpec, SEMResults, bootstrap_paths, fit  # noqa: F401

__all__ = [
    "__version__",
    "ModelSpec",
    "SEMResults",
    "fit",
    "bootstrap_paths",
    "calibration",
    "diagnostics",
    "fit_indices",
    "glm",
    "identification",
    "linalg",
    "measurement",
    "model",
    "power",
    "rare_events",
    "special",
    "structural",
]
