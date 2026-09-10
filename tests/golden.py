# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Golden-array (`.npz`) verification helper for GeoSigma.

A *golden array* is a saved, trusted reference output that later runs are
compared against — a numerical regression guard. Outputs here are numeric grids
(covariance matrices, variance maps, kriging means), so we store them losslessly
as compressed ``.npz`` files under ``tests/golden/``.

Scope (Phase 0 decision):

* **Deterministic stages only.** Use golden arrays for outputs that have a single
  correct answer for a given input (covariance, Cholesky factor, kriging
  posterior mean/covariance, variance maps, masks, clustering reductions).
* **Self-regression baseline.** Goldens capture trusted *Python* output to guard
  refactors. MATLAB-parity goldens can be added later by dropping MATLAB-exported
  ``.npz`` files into ``tests/golden/`` under the same names.
* **Not for stochastic realizations.** Realization draws depend on the RNG stream
  (which will not match MATLAB), so they are verified by statistical invariants,
  not golden values. That harness is deferred until the kriging driver exists
  (Phase 6).

Workflow
--------
First run (or to re-bless after an intentional change), set the regen flag::

    GEOSIGMA_REGEN_GOLDEN=1 pytest

This writes/overwrites the ``.npz`` files, which are then committed. Normal runs
load the committed goldens and assert ``np.allclose`` within tolerance.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

#: When set to "1", missing/mismatching goldens are (re)written instead of
#: failing. Use deliberately to bless new or intentionally-changed outputs.
REGEN = os.environ.get("GEOSIGMA_REGEN_GOLDEN") == "1"


def assert_matches_golden(name: str, *, rtol: float = 1e-7, atol: float = 0.0,
                          **arrays: np.ndarray) -> None:
    """Assert that named arrays match the stored golden ``tests/golden/<name>.npz``.

    Parameters
    ----------
    name : str
        Golden file stem (``tests/golden/<name>.npz``).
    rtol, atol : float
        Tolerances forwarded to :func:`numpy.allclose`.
    **arrays
        Named arrays produced by the code under test. The same names must be
        present in the golden file.

    Behaviour
    ---------
    * ``REGEN`` set -> (over)write the golden file and return.
    * Golden missing (and not regenerating) -> fail with an instruction to run
      with ``GEOSIGMA_REGEN_GOLDEN=1``.
    * Otherwise -> compare keys, shapes and values; fail on any mismatch.
    """
    path = GOLDEN_DIR / f"{name}.npz"

    if REGEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return

    if not path.exists():
        raise AssertionError(
            f"No golden file for '{name}' at {path}. "
            f"Generate it once with:  GEOSIGMA_REGEN_GOLDEN=1 pytest"
        )

    with np.load(path) as golden:
        golden_keys = set(golden.files)
        got_keys = set(arrays)
        if golden_keys != got_keys:
            raise AssertionError(
                f"Golden '{name}' array names differ. "
                f"golden={sorted(golden_keys)} got={sorted(got_keys)}"
            )
        for key, got in arrays.items():
            ref = golden[key]
            got = np.asarray(got)
            if got.shape != ref.shape:
                raise AssertionError(
                    f"Golden '{name}'['{key}'] shape {got.shape} != {ref.shape}"
                )
            if not np.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.nanmax(np.abs(got - ref))
                raise AssertionError(
                    f"Golden '{name}'['{key}'] differs (max|Δ|={diff:.3e}, "
                    f"rtol={rtol}, atol={atol}). Re-bless with "
                    f"GEOSIGMA_REGEN_GOLDEN=1 if the change is intended."
                )
