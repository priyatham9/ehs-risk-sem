"""Environment workarounds, isolated so they are visible rather than ambient.

Only one thing lives here. On macOS on Apple silicon, numpy 2.0 built against
the Accelerate BLAS emits spurious floating-point warnings from ``matmul``:

    RuntimeWarning: divide by zero encountered in matmul
    RuntimeWarning: overflow encountered in matmul
    RuntimeWarning: invalid value encountered in matmul

They fire on a plain product of two finite random matrices and the result is
finite and correct. Verified on this machine with::

    a = rng.standard_normal((20000, 4)); b = rng.standard_normal((4, 4))
    c = a @ b            # warns
    np.isfinite(c).all() # True

The library does not suppress these globally, because silencing
floating-point warnings across a numerical package would hide real problems.
:func:`silence_accelerate_matmul_warnings` is called explicitly by the
simulation runners and the test runner, where thousands of matrix products
would otherwise bury the output, and by nothing else.
"""

from __future__ import annotations

import warnings

__all__ = ["silence_accelerate_matmul_warnings"]

_MESSAGES = (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
    "underflow encountered in matmul",
)


def silence_accelerate_matmul_warnings() -> None:
    """Filter the known-spurious Accelerate ``matmul`` RuntimeWarnings.

    Narrow by construction: it matches only these four exact messages and only
    the ``RuntimeWarning`` category, so a genuine overflow anywhere other than
    inside ``matmul`` still surfaces.
    """
    for msg in _MESSAGES:
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
