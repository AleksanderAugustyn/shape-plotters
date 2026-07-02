"""Python-quadrature shape integrals — labeled v1 stopgap until the libraries
expose volume/surface/z_cm natively (umbrella spec non-goal)."""
from __future__ import annotations

import numpy as np

from .result import Array


def volume(theta: Array, radius: Array) -> float:
    """V = (2*pi/3) * integral R^3 sin(theta) dtheta (star-convex body)."""
    return float((2.0 * np.pi / 3.0) * np.trapezoid(radius**3 * np.sin(theta), theta))


def surface_area(theta: Array, radius: Array) -> float:
    """S = 2*pi * integral R sin(theta) sqrt(R^2 + R'^2) dtheta.

    R' comes from np.gradient — display-grade precision, consistent with the
    plotted dR/dtheta.
    """
    dr = np.gradient(radius, theta)
    return float(2.0 * np.pi * np.trapezoid(
        radius * np.sin(theta) * np.sqrt(radius**2 + dr**2), theta))


def z_cm(theta: Array, radius: Array) -> float:
    """z_cm = (pi/2) * integral R^4 sin(theta) cos(theta) dtheta / V; 0 if V <= 0."""
    v = volume(theta, radius)
    if v <= 0.0:
        return 0.0
    return float((np.pi / 2.0) * np.trapezoid(
        radius**4 * np.sin(theta) * np.cos(theta), theta) / v)
