"""Display-only neck heuristic for profiles without a lib-native neck.

Two highest interior local maxima of a 1-D profile, lowest minimum between
them. No fixed split point and no depth threshold — depth thresholds are WMMM
physics policy, not plotter geometry (spec 2026-07-03, section 3).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .result import Array


def find_neck_indices(rho: Array) -> tuple[int, int, int] | None:
    """Locate a neck in a 1-D profile.

    Returns
    -------
    (i_neck, i_lobe_a, i_lobe_b) with i_lobe_a < i_neck < i_lobe_b,
    or None when the profile has fewer than two interior maxima or the two
    highest maxima are adjacent.
    """
    peaks, _ = find_peaks(rho)
    if len(peaks) < 2:
        return None
    top_two = peaks[np.argsort(rho[peaks])[-2:]]
    i_a, i_b = int(top_two.min()), int(top_two.max())
    if i_b - i_a < 2:
        return None
    interior = np.arange(i_a + 1, i_b)
    i_neck = int(interior[np.argmin(rho[interior])])
    return i_neck, i_a, i_b


def neck_depth(rho: Array, i_neck: int, i_a: int, i_b: int) -> float:
    """1 - rho_neck / (lower of the two lobe maxima); 0 for degenerate input."""
    lower_max = min(rho[i_a], rho[i_b])
    if lower_max <= 0.0:
        return 0.0
    return float(1.0 - rho[i_neck] / lower_max)
