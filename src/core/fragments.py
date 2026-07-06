"""Cylindrical fragment volumes about the neck plane.

The neck (src/core/neck.py, or a library-native neck) splits a necked shape
into two pre-fragments. Their volumes are the disk integrals V = pi * ∫ rho(z)^2
dz on each side of z_neck — the cylindrical form, distinct from the whole-body
spherical integral in src/core/quadrature.py, which integrates the entire body
and cannot be split at a z-plane. Works for any rho(z) profile: FoS's lib-native
one and beta's parametric z = R cos θ, rho = R sin θ alike.
"""
from __future__ import annotations

import numpy as np

from src.core.result import Array


def fragment_volumes(z: Array, rho: Array, z_neck: float,
                     z_lo: float, z_hi: float) -> tuple[float, float]:
    """Cylindrical disk volumes below/above the neck plane.

    Parameters
    ----------
    z, rho : Array
        The rho(z) profile in a common length unit; order need not be sorted
        (beta's parametric z descends, FoS's ascends — both are handled).
    z_neck : float
        Axial split plane (same unit as z).
    z_lo, z_hi : float
        Pole positions (z_lo = -r_south, z_hi = +r_north) where the caps close
        at rho = 0. The profile endpoints rarely reach the exact poles (GL is an
        open rule for beta), so the caps are appended to include the end volume.

    Returns
    -------
    (V_below, V_above) : tuple[float, float]
        Disk volumes pi * ∫ rho(z)^2 dz below and above z_neck, in the cube of
        the length unit. (0.0, 0.0) for a degenerate profile (total <= 0).
    """
    # Cap the profile at the poles (rho = 0) and sort ascending in z so the
    # trapezoid is well defined regardless of the render's native ordering.
    zc = np.concatenate(([z_lo], np.asarray(z, dtype=float), [z_hi]))
    rc = np.concatenate(([0.0], np.asarray(rho, dtype=float), [0.0]))
    order = np.argsort(zc)
    zc, rc = zc[order], rc[order]

    z_neck = float(np.clip(z_neck, zc[0], zc[-1]))
    rho_neck = float(np.interp(z_neck, zc, rc))

    # Split at z_neck; the interpolated neck point closes both sides at the
    # shared plane so neither fragment loses the wedge next to the cut.
    below, above = zc < z_neck, zc > z_neck
    z_b = np.concatenate((zc[below], [z_neck]))
    r_b = np.concatenate((rc[below], [rho_neck]))
    z_a = np.concatenate(([z_neck], zc[above]))
    r_a = np.concatenate(([rho_neck], rc[above]))

    v_below = float(np.trapezoid(np.pi * r_b**2, z_b))
    v_above = float(np.trapezoid(np.pi * r_a**2, z_a))
    if not (np.isfinite(v_below) and np.isfinite(v_above)):
        return 0.0, 0.0
    if v_below + v_above <= 0.0:
        return 0.0, 0.0
    return v_below, v_above
