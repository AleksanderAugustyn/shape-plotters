"""CassiniRender: contract wiring over the cassini_parameterization package."""
import numpy as np
import pytest

pytest.importorskip("cassini_parameterization")

from shape_plotters.core import fragments, nodes  # noqa: E402
from shape_plotters.renders.cassini import PARAM_KEYS, CassiniRender  # noqa: E402


def _vec(*values: float) -> dict[str, float]:
    return dict(zip(PARAM_KEYS, values))


# Fixture vectors named as in the library plan's Fixtures table.
SPHERE = _vec(0.0, 0.0, 0.0, 0.0, 0.0)
GS = _vec(0.25, 0.0, 0.0, 0.0, 0.075)
SADDLE2 = _vec(0.65, 0.2, 0.0, 0.025, 0.05)
ASYM = _vec(0.8, 0.15, -0.05, 0.1, -0.05)
DIMPLED = _vec(0.5, 0.0, 0.0, 0.0, -0.2)
FOLD_BOX = _vec(0.125, 0.25, 0.0, 0.2, -0.175)
STAR_FAIL = _vec(0.95, -0.2, 0.0, -0.05, -0.05)
PINCHED = _vec(1.0, 0.0, 0.0, 0.0, 0.0)
SEPARATED = _vec(1.05, 0.0, 0.0, 0.0, 0.0)
R_NEG = _vec(0.0, -1.1, 0.0, 0.0, 0.0)
TOROIDAL = _vec(-1.1, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def render() -> CassiniRender:
    return CassiniRender()


def test_sphere(render: CassiniRender) -> None:
    res = render.compute(SPHERE, {})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-14)
    assert res.r_south == pytest.approx(1.0, abs=1e-14)
    assert res.z_cm == 0.0
    assert res.neck is None
    assert res.scalars["eps"] == pytest.approx(0.0, abs=1e-15)
    assert res.scalars["c"] == pytest.approx(1.0, abs=1e-14)
    assert res.scalars["q2"] == pytest.approx(0.0, abs=1e-12)
    assert res.scalars["d_min"] == pytest.approx(1.0, abs=1e-12)
    assert "gap" not in res.scalars
    assert res.notes == ("profile: single-valued",)
    assert res.drho_dz is not None and res.drho_dz.shape == res.rho.shape
    assert res.rho[0] == res.rho[-1] == 0.0


def test_ground_state_shape(render: CassiniRender) -> None:
    res = render.compute(GS, {})
    assert res.ok
    assert res.r_north == pytest.approx(1.227770, abs=1e-5)
    assert res.scalars["c"] == pytest.approx(0.957813, abs=1e-5)
    assert res.scalars["q2"] == pytest.approx(0.100001, abs=1e-5)
    assert res.neck is None


def test_second_saddle_is_asymmetric(render: CassiniRender) -> None:
    res = render.compute(SADDLE2, {})
    assert res.ok
    assert res.scalars["z_shift"] == pytest.approx(0.316153, abs=1e-5)
    assert res.r_north == pytest.approx(1.443267, abs=1e-5)
    assert res.r_south == pytest.approx(1.662467, abs=1e-5)
    assert res.neck is None


def test_dr_dtheta_matches_gradient(render: CassiniRender) -> None:
    # Interior nodes only: np.gradient's first-order endpoint differences miss
    # by 3.4e-3 on ASYM.
    res = render.compute(ASYM, {})
    assert res.ok
    fd = np.gradient(res.radius, res.theta)
    tol = 1e-3 * (1.0 + float(np.max(np.abs(res.dr_dtheta))))
    assert float(np.max(np.abs(res.dr_dtheta[1:-1] - fd[1:-1]))) < tol


def test_overlay_is_the_rtheta_outline(render: CassiniRender) -> None:
    res = render.compute(GS, {})
    assert res.overlay_z is not None and res.overlay_rho is not None
    assert res.overlay_z[0] == res.r_north and res.overlay_rho[0] == 0.0
    assert res.overlay_z[-1] == -res.r_south and res.overlay_rho[-1] == 0.0
    assert np.allclose(np.hypot(res.overlay_z[1:-1], res.overlay_rho[1:-1]), res.radius, atol=1e-12)
    assert res.overlay_z_cm == 0.0
    for vec in (STAR_FAIL, SEPARATED):
        assert render.compute(vec, {}).overlay_z is None


def test_dimpled_shape_is_valid(render: CassiniRender) -> None:
    res = render.compute(DIMPLED, {})
    assert res.ok
    assert res.notes == ("profile: folded (dimpled)",)
    assert res.overlay_z is not None
    assert res.neck is not None
    assert res.neck.rho == pytest.approx(0.687014, abs=1e-5)
    assert res.neck.z == pytest.approx(0.0, abs=1e-12)
    # The refined minimum of d (library d50bf6c), not the plan's 0.62747: that
    # was a 65-node sampled value from before the refinement.
    assert res.scalars["d_min"] == pytest.approx(0.62734, abs=1e-5)
    assert res.r_north == pytest.approx(1.189942, abs=1e-5)

    res = render.compute(FOLD_BOX, {})
    assert res.ok
    assert res.notes == ("profile: folded (dimpled)",)
    assert res.neck is not None
    v_below, v_above = fragments.fragment_volumes(res.z, res.rho, res.neck.z,
                                                  -res.r_south, res.r_north)
    assert v_below + v_above == pytest.approx(4.0 * np.pi / 3.0, rel=1e-4)
    assert v_below / (v_below + v_above) == pytest.approx(0.4129, abs=1e-3)


def test_asym_neck_at_equator(render: CassiniRender) -> None:
    res = render.compute(ASYM, {})
    assert res.neck is not None
    assert res.neck.z == pytest.approx(-res.scalars["z_shift"], abs=1e-12)
    assert res.neck.rho == pytest.approx(0.550778, abs=1e-5)


def test_star_fail_keeps_neck_and_profile(render: CassiniRender) -> None:
    res = render.compute(STAR_FAIL, {})
    assert not res.ok
    assert res.status_name == "not_star_convex"
    assert res.neck is not None and res.neck.rho > 0.0
    assert res.rho.max() > 0.0
    assert res.scalars["d_min"] < 0.0
    assert res.notes == ("profile: single-valued",)


def test_pinched_shape(render: CassiniRender) -> None:
    # alpha = 1 exactly is a slider position (step 0.025 from 0).
    res = render.compute(PINCHED, {})
    assert not res.ok
    assert res.status_name == "not_star_convex"
    assert res.neck is not None and res.neck.rho == 0.0
    assert res.r_north == pytest.approx(2.095290, abs=1e-5)
    assert res.rho.max() > 0.0
    assert (res.rho[1:-1] == 0.0).sum() >= 2           # both lobe tips
    assert res.notes[0].endswith(", pinched")


def test_separated_shape_draws_fragments_with_zero_neck(render: CassiniRender) -> None:
    # shape() zero-fills its poles on 103; the fragment split needs the real
    # caps, so they come from geometry(), and the split plane from a rho = 0 neck.
    res = render.compute(SEPARATED, {})
    assert not res.ok
    assert res.status_name == "separated"
    assert res.neck is not None and res.neck.rho == 0.0
    assert res.neck.z == pytest.approx(0.0, abs=1e-12)
    assert res.rho.max() > 0.0                          # fragments drawn
    inner_zero = res.rho[1:-1] == 0.0
    assert (inner_zero[:-1] & inner_zero[1:]).any()     # the two tips, consecutive
    assert res.r_north > 0.0 and res.r_south > 0.0
    assert res.scalars["gap"] == pytest.approx(0.687685, abs=1e-5)
    assert "gap" in res.length_keys
    assert res.notes[0].endswith(", separated")


def test_invalid_returns_status_not_exception(render: CassiniRender) -> None:
    res = render.compute(R_NEG, {})
    assert not res.ok
    assert res.status_name == "r_nonpositive"
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()
    assert not res.rho.any()
    assert res.scalars == {}
    assert res.notes == ()
    assert render.compute(TOROIDAL, {}).status_name == "toroidal"


def test_stats_scalars_and_keys(render: CassiniRender) -> None:
    res = render.compute(SADDLE2, {})
    assert list(res.scalars) == ["eps", "c", "z_shift", "q2", "d_min"]
    assert res.length_keys == {"z_shift", "gap"}
    res = render.compute(SEPARATED, {})
    assert list(res.scalars) == ["eps", "c", "z_shift", "gap", "q2", "d_min"]


def test_slider_specs_and_filename(render: CassiniRender) -> None:
    assert [s.key for s in render.slider_specs] == list(PARAM_KEYS)
    assert PARAM_KEYS == ("alpha", "alpha1", "alpha2", "alpha3", "alpha4")
    assert all(s.step == 0.025 for s in render.slider_specs)
    assert all(s.vinit == 0.0 for s in render.slider_specs)
    assert tuple(s.markers for s in render.slider_specs) == (
        (0.95, 1.0), (-0.25, 0.25), (), (-0.25, 0.25), (-0.25, 0.25))
    assert render.has_extra_panel is True
    assert render.filename(236, SADDLE2) == "cassini_A236_al0.650_a10.200_a20.000_a30.025_a40.050.png"


def test_energy_requests_empty(render: CassiniRender) -> None:
    assert render.energy_requests(GS, render.compute(GS, {})) == []
