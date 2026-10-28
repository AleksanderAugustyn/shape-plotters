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


def test_folded_profile_volume() -> None:
    # A closed curve with a polar dimple: z(t) overshoots its pole value before
    # turning south, so rho(z) is multi-valued near the north pole (an overhang).
    def curve(n: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, np.pi, n)
        return np.cos(t) - 0.35 * np.cos(t) * np.exp(-6.0 * t**2), np.sin(t)

    z_fine, rho_fine = curve(400001)
    oracle = abs(np.pi * np.trapezoid(rho_fine**2, z_fine))
    z, rho = curve(4001)
    assert np.any(np.diff(z) > 0.0) and np.any(np.diff(z) < 0.0)   # really folded
    v_below, v_above = fragments.fragment_volumes(z, rho, 0.0, z_hi=z[0], z_lo=z[-1])
    assert v_below + v_above == pytest.approx(oracle, rel=1e-5)


def test_split_at_neck_crossing_not_by_sort() -> None:
    # FOLD_BOX: a dimpled Cassini shape inside the paper's box, split at the
    # library neck. 4001 profile nodes put the trapezoid error below 1e-6 (at
    # the render's 721 it is 1.5e-5); the old sort lost 0.22 of the south part.
    cp = pytest.importorskip("cassini_parameterization")
    fold_box = [0.125, 0.25, 0.0, 0.2, -0.175]
    prof = cp.rho_profile(fold_box, 4001)
    nk = cp.neck(fold_box)
    geo = cp.geometry(fold_box)
    assert prof.ok and nk.ok and nk.found and geo.ok and geo.is_folded
    assert nk.z_neck == pytest.approx(-0.154052, abs=1e-6)
    v_below, v_above = fragments.fragment_volumes(prof.z, prof.rho, nk.z_neck,
                                                  -geo.r_south, geo.r_north)
    assert v_above == pytest.approx(2.459268, abs=1e-5)
    assert v_below == pytest.approx(1.729521, abs=1e-5)
    assert v_below / (v_below + v_above) == pytest.approx(0.4129, abs=1e-4)


def test_tips_close_the_void() -> None:
    # Two unit spheres centred at z = ±1.5, drawn as one polyline whose
    # consecutive tip nodes (0.5, 0) and (-0.5, 0) run along the axis across
    # the void: the void contributes nothing and the midpoint split is even.
    phi = np.linspace(0.0, np.pi, 2001)
    z = np.concatenate((1.5 + np.cos(phi), -1.5 + np.cos(phi)))
    rho = np.concatenate((np.sin(phi), np.sin(phi)))
    v_below, v_above = fragments.fragment_volumes(z, rho, 0.0, -2.5, 2.5)
    assert v_below == pytest.approx(WHOLE, rel=1e-5)
    assert v_above == pytest.approx(WHOLE, rel=1e-5)
