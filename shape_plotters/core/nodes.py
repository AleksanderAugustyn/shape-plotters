"""Shared Gauss-Legendre node set — WMMM's dense grid (GL-2048 in x = cos θ).

Shape-independent, computed once at import, read-only: the Python mirror of
WMMM's program_run_time_constants_mod node-set metadata. Renders evaluate
R and dR/dθ at THETA; quadrature.py integrates with W (integrands with a
sin θ factor absorb it into dx = -sin θ dθ). GL is an open rule — no θ = 0/π
nodes — so pole values come from the libraries' analytic pole radii.
"""
from __future__ import annotations

import numpy as np

N_NODES = 2048

_x, _w = np.polynomial.legendre.leggauss(N_NODES)
X = _x[::-1].copy()           # cos(theta), descending +1 -> -1: exact GL abscissas
W = _w[::-1].copy()           # weights paired with THETA's ordering
THETA = np.arccos(X)          # ascending, strictly inside (0, pi)
SIN_THETA = np.sqrt(1.0 - X * X)

for _arr in (X, W, THETA, SIN_THETA):
    _arr.setflags(write=False)
del _x, _w, _arr
