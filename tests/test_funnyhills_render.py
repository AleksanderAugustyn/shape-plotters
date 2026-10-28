"""FunnyhillsRender: contract wiring over the funnyhills_parameterization package."""
import numpy as np
import pytest

pytest.importorskip("funnyhills_parameterization")

from shape_plotters.core import nodes  # noqa: E402
from shape_plotters.renders.funnyhills import FunnyhillsRender  # noqa: E402

SPHERE = dict(c=1.0, h=0.0, alpha=0.0)
GS = dict(c=1.21, h=-0.15, alpha=0.0)
ASYM = dict(c=1.6, h=0.3, alpha=0.2)
SEPARATED = dict(c=2.2, h=0.0, alpha=0.0)
PINCHED = dict(c=2.0, h=0.0625, alpha=0.0)
STAR_FAIL = dict(c=2.13, h=-0.165, alpha=0.2)
POLE_NEG = dict(c=1.9, h=-0.3, alpha=0.2)
INVALID = dict(c=0.0, h=0.0, alpha=0.0)


@pytest.fixture(scope="module")
def render() -> FunnyhillsRender:
    return FunnyhillsRender()


def test_sphere(render: FunnyhillsRender) -> None:
    res = render.compute(SPHERE, {})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-14)
    assert res.r_south == pytest.approx(1.0, abs=1e-14)
    assert res.z_cm == 0.0
    assert res.infer_rtheta_overlay is False
    assert res.neck is None
    assert res.scalars["q2"] == pytest.approx(0.0, abs=1e-12)
    assert res.scalars["rho_cm"] == pytest.approx(0.375, abs=1e-14)
    assert res.scalars["a_coeff"] == 1.0
    assert res.notes == ("polynomial",)
    assert res.drho_dz is not None and res.drho_dz.shape == res.rho.shape
    assert res.rho[0] == res.rho[-1] == 0.0


def test_ground_state_is_exponential(render: FunnyhillsRender) -> None:
    res = render.compute(GS, {})
    assert res.ok
    assert res.notes == ("exponential",)
    assert res.neck is None
    assert res.scalars["b_coeff"] == pytest.approx(-0.195, abs=1e-12)


def test_asym_is_necked(render: FunnyhillsRender) -> None:
    res = render.compute(ASYM, {})
    assert res.ok
    assert res.neck is not None and res.neck.source == "lib"
    assert res.neck.rho > 0.0 and res.neck.depth > 0.0
    assert res.notes == ("polynomial, necked",)


def test_dr_dtheta_matches_central_differences(render: FunnyhillsRender) -> None:
    # np.gradient on the GL spacing (1.5e-3) misses by 0.12 here: R(theta)
    # turns within d_min = 0.057 of the centre of mass at the neck. Central
    # differences at +-1e-6 on the render's own nodes resolve it.
    import funnyhills_parameterization as fp
    res = render.compute(ASYM, {})
    assert res.ok
    h = 1e-6
    inner = (nodes.THETA > h) & (nodes.THETA < np.pi - h)
    arr = [ASYM[k] for k in ("c", "h", "alpha")]
    with fp.Cache(2049, nodes.THETA[inner]) as cache:
        up = cache.radius_and_derivative_at_thetas(arr, nodes.THETA[inner] + h)
        down = cache.radius_and_derivative_at_thetas(arr, nodes.THETA[inner] - h)
    fd = (up.radii - down.radii) / (2.0 * h)
    dr = res.dr_dtheta[inner]
    assert np.all(np.abs(fd - dr) <= 1e-6 * (1.0 + np.abs(dr)))


def test_poles_come_from_geometry(render: FunnyhillsRender) -> None:
    # shape() zero-fills its poles on 103; the fragment split needs the caps.
    res = render.compute(SEPARATED, {})
    assert res.r_north > 0.0 and res.r_south > 0.0


def test_separated_shape_draws_fragments_with_zero_neck(render: FunnyhillsRender) -> None:
    res = render.compute(SEPARATED, {})
    assert not res.ok
    assert res.status_name == "separated"
    assert res.neck is not None and res.neck.rho == 0.0
    assert res.rho.max() > 0.0                      # fragments drawn
    assert (res.rho[1:-1] == 0.0).any()             # the void between them
    assert "gap" in res.scalars
    assert res.scalars["lobe_volume_ratio"] == pytest.approx(1.0574, abs=1e-3)
    assert res.notes == ("polynomial, separated",)


def test_pinched_shape(render: FunnyhillsRender) -> None:
    res = render.compute(PINCHED, {})
    assert not res.ok
    assert res.status_name == "not_star_convex"
    assert res.neck is not None and res.neck.rho == 0.0
    assert res.notes[0].endswith("pinched")
    assert res.rho.max() > 0.0


def test_star_fail_keeps_neck_and_profile(render: FunnyhillsRender) -> None:
    res = render.compute(STAR_FAIL, {})
    assert not res.ok
    assert res.status_name == "not_star_convex"
    assert res.neck is not None and res.neck.rho > 0.0
    assert res.rho.max() > 0.0


def test_pole_negative_returns_status_not_exception(render: FunnyhillsRender) -> None:
    res = render.compute(POLE_NEG, {})
    assert not res.ok
    assert res.status_name == "pole_negative"
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()
    assert res.scalars == {}
    assert res.notes == ()


def test_invalid_c(render: FunnyhillsRender) -> None:
    res = render.compute(INVALID, {})
    assert not res.ok
    assert res.status_name == "invalid_c"


def test_stats_scalars_and_notes(render: FunnyhillsRender) -> None:
    res = render.compute(ASYM, {})
    assert list(res.scalars) == ["a_coeff", "b_coeff", "z_shift", "rho_cm", "q2", "d_min",
                                 "h_ellipsoid", "h_scission"]
    assert res.length_keys == {"z_shift", "gap", "rho_cm"}
    assert res.scalars["h_ellipsoid"] == pytest.approx(-0.15)
    assert res.scalars["h_scission"] == pytest.approx(5.0 / (2.0 * 1.6**3) - 0.15)


def test_slider_specs_and_filename(render: FunnyhillsRender) -> None:
    assert [s.key for s in render.slider_specs] == ["c", "h", "alpha"]
    assert tuple(s.vinit for s in render.slider_specs) == (1.0, 0.0, 0.0)
    assert tuple(s.markers for s in render.slider_specs) == ((1.0,), (0.0,), (0.0,))
    assert render.has_extra_panel is True
    assert render.filename(240, ASYM) == "funnyhills_A240_c1.60_h0.300_al0.20.png"


def test_energy_requests_empty(render: FunnyhillsRender) -> None:
    assert render.energy_requests(SPHERE, render.compute(SPHERE, {})) == []
