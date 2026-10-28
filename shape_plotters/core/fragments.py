"""Cylindrical fragment volumes about the neck plane.

The neck (shape_plotters/core/neck.py, or a library-native neck) splits a necked shape
into two pre-fragments. Their volumes are the disk integrals V = pi * ∫ rho^2 dz
on each side of z_neck — the cylindrical form, distinct from the whole-body
spherical integral in shape_plotters/core/quadrature.py, which integrates the entire body
and cannot be split at a z-plane.

The profile is integrated in its traversal order with signed dz, not sorted by
z: sorting assumes a single-valued rho(z), and a dimpled (folded) profile such
as Cassini's is not one. Along a simple closed curve the signed sum is exact —
where a z-plane crosses the curve three times it gives rho1^2 - rho2^2 +
rho3^2, the disk plus the annulus — and a separated shape's two tip nodes keep
the void at rho = 0. Single-valued profiles (beta's descending parametric z,
FoS's ascending one) give the same numbers as the old sorted integral.
"""
from __future__ import annotations

import numpy as np

from shape_plotters.core.result import Array


def fragment_volumes(z: Array, rho: Array, z_neck: float,
                     z_lo: float, z_hi: float) -> tuple[float, float]:
    """Cylindrical disk volumes below/above the neck plane.

    Parameters
    ----------
    z, rho : Array
        The profile curve in a common length unit, in traversal order from
        one pole to the other, either direction (beta's parametric z descends,
        FoS's ascends; a dimpled Cassini profile is not monotonic in z).
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
        the length unit. (0.0, 0.0) for a degenerate profile (zero total).
    """
    # Cap the profile at the poles (rho = 0) in its own traversal order, so the
    # capped curve runs pole to pole whichever way the render orders it.
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    south_first = z.size == 0 or abs(z[0] - z_lo) <= abs(z[-1] - z_lo)
    first, last = (z_lo, z_hi) if south_first else (z_hi, z_lo)
    zc = np.concatenate(([first], z, [last]))
    rc = np.concatenate(([0.0], rho, [0.0]))

    # Signed trapezoid of pi * rho^2 per segment, oriented so the total is positive.
    dv = 0.5 * np.pi * (rc[:-1] ** 2 + rc[1:] ** 2) * np.diff(zc)
    total = float(dv.sum())
    if not np.isfinite(total) or total == 0.0:
        return 0.0, 0.0
    sign = float(np.sign(total))
    dv *= sign

    # Cut the curve where it crosses the plane z = z_neck; of several crossings
    # (a fold reaching the plane) the neck is the one with the smallest rho.
    z_neck = float(np.clip(z_neck, zc.min(), zc.max()))
    d = zc - z_neck
    seg = np.flatnonzero(((d[:-1] <= 0.0) & (d[1:] >= 0.0))
                         | ((d[:-1] >= 0.0) & (d[1:] <= 0.0)))
    if seg.size == 0:
        return 0.0, 0.0
    dz = zc[seg + 1] - zc[seg]
    t = np.divide(-d[seg], dz, out=np.zeros_like(dz), where=dz != 0.0)
    rho_cut = rc[seg] + t * (rc[seg + 1] - rc[seg])
    k = int(np.argmin(rho_cut))
    i, r_cut = int(seg[k]), float(rho_cut[k])

    # The cut segment splits into (z_i -> z_neck) and (z_neck -> z_i+1); the
    # piece holding the first pole lies on that pole's side of the plane.
    w_in = 0.5 * np.pi * sign * (rc[i] ** 2 + r_cut**2) * (z_neck - zc[i])
    w_out = 0.5 * np.pi * sign * (r_cut**2 + rc[i + 1] ** 2) * (zc[i + 1] - z_neck)
    v_first = float(dv[:i].sum() + w_in)
    v_second = float(w_out + dv[i + 1:].sum())
    v_below, v_above = (v_first, v_second) if south_first else (v_second, v_first)
    if not (np.isfinite(v_below) and np.isfinite(v_above)):
        return 0.0, 0.0
    if v_below + v_above <= 0.0:
        return 0.0, 0.0
    return v_below, v_above
