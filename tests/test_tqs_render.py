"""TqsRender: contract wiring over the tqs_parameterization package."""
import math

import numpy as np
import pytest

pytest.importorskip("tqs_parameterization")

from shape_plotters.core import nodes  # noqa: E402
from shape_plotters.renders.tqs import TqsRender  # noqa: E402

SPHERE = dict(sigma1=0.0, sigma2=1.0, sigma3=1.0, alpha2=0.0, alpha3=0.0)
INITIAL = dict(sigma1=2.4, sigma2=-1.0, sigma3=1.0, alpha2=0.0, alpha3=0.0)
ASYM = dict(sigma1=1.6, sigma2=-0.6, sigma3=1.0, alpha2=0.3, alpha3=0.2)
PEAR = dict(sigma1=1.0, sigma2=-1.0, sigma3=1.0, alpha2=-1.4, alpha3=0.0)   # outside the slider box; library fixture
SEPARATED = dict(sigma1=3.0, sigma2=-1.0, sigma3=1.0, alpha2=0.0, alpha3=0.0)
PINCHED = {**SEPARATED, "sigma1": 2.0 * 2 ** 0.5}
STAR_FAIL = dict(sigma1=3.0, sigma2=-0.6, sigma3=1.0, alpha2=0.9, alpha3=0.0)
NO_GEOM = dict(sigma1=0.3, sigma2=1.5, sigma3=1.5, alpha2=0.0, alpha3=0.0)
SPHEROID_MID = dict(sigma1=0.8, sigma2=0.5, sigma3=1.0, alpha2=0.2, alpha3=0.1)


@pytest.fixture(scope="module")
def render() -> TqsRender:
    return TqsRender()


def test_sphere(render: TqsRender) -> None:
    res = render.compute(SPHERE, {})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-14)
    assert res.r_south == pytest.approx(1.0, abs=1e-14)
    assert res.z_cm == 0.0
    assert res.neck is None
    assert res.scalars["q2"] == pytest.approx(0.0, abs=1e-12)
    assert res.scalars["u"] == pytest.approx(1.0, abs=1e-14)
    assert res.notes == ("middle: spheroid",)
    assert res.drho_dz is not None and res.drho_dz.shape == res.rho.shape
    assert res.rho[0] == res.rho[-1] == 0.0


def test_initial_shape_is_necked_and_valid(render: TqsRender) -> None:
    res = render.compute(INITIAL, {})
    assert res.ok
    assert res.neck is not None
    assert res.neck.source == "lib"
    assert res.neck.rho > 0.0
    assert res.neck.depth > 0.3
    assert "sigma1_star" in res.scalars
    assert res.scalars["sigma1_star"] == pytest.approx(2 * 2 ** 0.5, abs=1e-12)
    assert res.scalars["sigma1_t"] == pytest.approx(2.0)
    assert res.notes == ("middle: one_sheet (waist inside)",)


def test_dr_dtheta_matches_gradient(render: TqsRender) -> None:
    res = render.compute(ASYM, {})
    assert res.ok
    fd = np.gradient(res.radius, res.theta)
    tol = 1e-3 * (1.0 + float(np.max(np.abs(res.dr_dtheta))))
    assert float(np.max(np.abs(res.dr_dtheta - fd))) < tol


def test_poles_come_from_geometry(render: TqsRender) -> None:
    res = render.compute(ASYM, {})
    assert res.r_north == pytest.approx(1.50029, abs=1e-5)
    assert res.r_south == pytest.approx(1.32879, abs=1e-5)


def test_pear_is_a_connected_two_sheet_flare(render: TqsRender) -> None:
    res = render.compute(PEAR, {})
    assert res.ok
    assert res.neck is None
    assert res.notes == ("middle: two_sheet (waist outside)",)


def test_separated_shape_draws_fragments_with_zero_neck(render: TqsRender) -> None:
    # shape() zero-fills its poles on 103; the fragment split needs the real
    # caps, so they come from geometry(), and the split plane from a rho = 0 neck.
    res = render.compute(SEPARATED, {})
    assert not res.ok
    assert res.status_name == "separated"
    assert res.neck is not None and res.neck.rho == 0.0
    assert res.rho.max() > 0.0                      # fragments drawn
    assert (res.rho[1:-1] == 0.0).any()             # the void between them
    assert res.r_north > 0.0 and res.r_south > 0.0
    assert res.notes[0].endswith("(waist inside, separated)")
    assert res.scalars["d"] == 0.0


def test_pinched_shape(render: TqsRender) -> None:
    res = render.compute(PINCHED, {})
    assert not res.ok
    assert res.status_name == "not_star_convex"
    assert res.neck is not None and res.neck.rho == 0.0
    assert res.r_north > 0.0
    assert res.notes[0].endswith("(waist inside, pinched)")
    assert res.rho.max() > 0.0


def test_star_fail_keeps_neck_and_profile(render: TqsRender) -> None:
    res = render.compute(STAR_FAIL, {})
    assert not res.ok
    assert res.status_name == "not_star_convex"
    assert res.neck is not None and res.neck.rho > 0.0
    assert res.rho.max() > 0.0


def test_invalid_returns_status_not_exception(render: TqsRender) -> None:
    res = render.compute(NO_GEOM, {})
    assert not res.ok
    assert res.status_name == "no_geometry"
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()
    assert not res.rho.any()
    assert res.scalars == {}
    assert res.notes == ()


def test_far_waist_row(render: TqsRender) -> None:
    # Near sigma2 = 0 with alpha2 != 0 the waist sits far outside the middle
    # (|l3| ~ alpha2/sigma2); only sigma2 = 0 exactly has no geometry.
    row = dict(sigma1=1.0, sigma3=1.0, alpha2=0.2, alpha3=0.0)
    res = render.compute({**row, "sigma2": 0.0}, {})
    assert not res.ok
    assert res.status_name == "no_geometry"
    for sigma2, note in ((0.01, "middle: spheroid"), (-0.01, "middle: two_sheet (waist outside)")):
        res = render.compute({**row, "sigma2": sigma2}, {})
        assert res.ok, sigma2
        assert all(math.isfinite(v) for v in res.scalars.values()), sigma2
        assert res.notes == (note,)


def test_stats_scalars_and_notes(render: TqsRender) -> None:
    res = render.compute(ASYM, {})
    assert list(res.scalars) == ["z1", "z2", "u", "alpha1", "sigma1_star", "sigma1_t", "d",
                                 "q2", "Q2_Pu240_b", "alpha_g", "eps_f1", "eps_f2"]
    assert res.length_keys == {"z1", "z2", "u", "d"}
    assert res.scalars["Q2_Pu240_b"] == pytest.approx(
        2 * 94 * (1.16 * 240 ** (1 / 3)) ** 2 * res.scalars["q2"] / 100)
    res = render.compute(SPHEROID_MID, {})
    assert "sigma1_star" not in res.scalars       # sigma2 >= 0: no scission locus
    assert "sigma1_t" in res.scalars
    assert res.notes == ("middle: spheroid",)


def test_slider_specs_and_filename(render: TqsRender) -> None:
    assert [s.key for s in render.slider_specs] == ["sigma1", "sigma2", "sigma3", "alpha2", "alpha3"]
    assert tuple(s.vinit for s in render.slider_specs) == (2.4, -1.0, 1.0, 0.0, 0.0)
    assert tuple(s.markers for s in render.slider_specs) == ((0.0,), (1.0,), (1.0,), (0.0,), (0.0,))
    assert render.has_extra_panel is True
    assert render.filename(240, INITIAL) == "tqs_A240_2.40_-1.00_1.00_0.00_0.00.png"


def test_energy_requests_empty(render: TqsRender) -> None:
    assert render.energy_requests(INITIAL, render.compute(INITIAL, {})) == []
