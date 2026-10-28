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
the void at rho = 0. The split clips every segment at the plane rather than
cutting the curve at one chosen crossing: by Green's theorem the volume below
the plane is the signed integral over the part of the curve below it (the
chord on the plane has dz = 0, the axis rho = 0), whatever the number of
crossings. So a dimple reaching past the plane is split correctly, and a pole
cap that runs along the axis adds nothing on either side — FoS's poles are
R(θ)-frame values (zero-filled on failure) that can lie inside its profile.
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
    z0, z1, r0, r1 = zc[:-1], zc[1:], rc[:-1], rc[1:]
    full = 0.5 * np.pi * (r0**2 + r1**2) * (z1 - z0)
    total = float(full.sum())
    if not np.isfinite(total) or total == 0.0:
        return 0.0, 0.0

    # Clip each segment at the plane z = z_neck (rho interpolated linearly at
    # the cut) and keep its part below; a segment touching the plane at an
    # end lies on the side of its midpoint.
    d0, d1 = z0 - z_neck, z1 - z_neck
    cross = d0 * d1 < 0.0
    t = np.divide(d0, d0 - d1, out=np.zeros_like(d0), where=cross)
    r_cut2 = (r0 + t * (r1 - r0)) ** 2
    head = 0.5 * np.pi * (r0**2 + r_cut2) * (z_neck - z0)    # z0 -> plane
    tail = 0.5 * np.pi * (r_cut2 + r1**2) * (z1 - z_neck)    # plane -> z1
    below = np.where(cross, np.where(d0 < 0.0, head, tail),
                     np.where(d0 + d1 < 0.0, full, 0.0))
    v_below = float(np.sign(total) * below.sum())
    v_above = abs(total) - v_below
    if not (np.isfinite(v_below) and np.isfinite(v_above)):
        return 0.0, 0.0
    if v_below + v_above <= 0.0:
        return 0.0, 0.0
    return v_below, v_above
