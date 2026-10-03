"""Cylindrical fragment volumes vs analytic sphere / spherical-cap values."""
import numpy as np
import pytest

from shape_plotters.core import fragments, nodes

# Unit sphere sampled on the shared GL nodes: z = cos θ, rho = sin θ.
Z = nodes.X.copy()            # descending +1 -> -1
RHO = nodes.SIN_THETA.copy()
HALF = 2.0 * np.pi / 3.0
WHOLE = 4.0 * np.pi / 3.0


def test_symmetric_split_halves_the_sphere() -> None:
    v_below, v_above = fragments.fragment_volumes(Z, RHO, 0.0, -1.0, 1.0)
    assert v_below == pytest.approx(HALF, rel=1e-3)
    assert v_above == pytest.approx(HALF, rel=1e-3)
    assert v_below + v_above == pytest.approx(WHOLE, rel=1e-4)


def test_offcenter_split_matches_spherical_cap() -> None:
    a = 0.5
    cap = np.pi * (2.0 / 3.0 - a + a**3 / 3.0)   # volume above plane z = a
    v_below, v_above = fragments.fragment_volumes(Z, RHO, a, -1.0, 1.0)
    assert v_above == pytest.approx(cap, rel=2e-3)
    assert v_below == pytest.approx(WHOLE - cap, rel=1e-3)


def test_ordering_independent_of_input_order() -> None:
    forward = fragments.fragment_volumes(Z, RHO, 0.2, -1.0, 1.0)
    reversed_ = fragments.fragment_volumes(Z[::-1], RHO[::-1], 0.2, -1.0, 1.0)
    assert forward == pytest.approx(reversed_, rel=1e-12)


def test_neck_beyond_range_is_clipped_to_a_pole() -> None:
    # z_neck past the north pole -> nothing above, whole volume below.
    v_below, v_above = fragments.fragment_volumes(Z, RHO, 5.0, -1.0, 1.0)
    assert v_above == pytest.approx(0.0, abs=1e-9)
    assert v_below == pytest.approx(WHOLE, rel=1e-4)


def test_degenerate_profile_returns_zero() -> None:
    z = np.linspace(-1.0, 1.0, 51)
    assert fragments.fragment_volumes(z, np.zeros_like(z), 0.0, -1.0, 1.0) == (0.0, 0.0)
