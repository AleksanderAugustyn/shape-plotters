"""GL dot-product shape integrals on the shared node set (src/core/nodes.py).

Spectrally exact — the same scheme as WMMM's dense set: every integrand
carries a sin(theta) factor that the x = cos(theta) substitution absorbs.
Arrays must be sampled on nodes.THETA; theta is accepted for interface
stability and checked against the node set.
"""
from __future__ import annotations

import numpy as np

from src.core import nodes
from src.core.result import Array


def _require_node_set(theta: Array) -> None:
    if theta.shape != nodes.THETA.shape:
        raise ValueError(
            f"expected the shared GL node set ({nodes.N_NODES} nodes), got {theta.shape}")


def volume(theta: Array, radius: Array) -> float:
    """V = (2*pi/3) * sum w_i R_i^3 (star-convex body)."""
    _require_node_set(theta)
    return float((2.0 * np.pi / 3.0) * np.sum(nodes.W * radius**3))


def surface_area(theta: Array, radius: Array, dr_dtheta: Array) -> float:
    """S = 2*pi * sum w_i R_i sqrt(R_i^2 + R'_i^2), with lib-exact R'."""
    _require_node_set(theta)
    return float(2.0 * np.pi * np.sum(
        nodes.W * radius * np.sqrt(radius**2 + dr_dtheta**2)))


def z_cm(theta: Array, radius: Array) -> float:
    """z_cm = (pi/2) * sum w_i R_i^4 x_i / V; 0 if V <= 0."""
    _require_node_set(theta)
    v = volume(theta, radius)
    if v <= 0.0:
        return 0.0
    return float((np.pi / 2.0) * np.sum(nodes.W * radius**4 * nodes.X) / v)
