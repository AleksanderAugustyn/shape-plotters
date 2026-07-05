"""FoSRender: contract wiring over the fos_parameterization package."""
import numpy as np
import pytest

from src.core import nodes
from src.renders.fos import FoSRender

SPHERE = {"c": 1.0, "a3": 0.0, "a4": 0.0, "a5": 0.0, "a6": 0.0, "a7": 0.0, "a8": 0.0}
# Probed lib-VALID necked shape (carried over from v0.1).
NECKED = {**SPHERE, "c": 2.0, "a4": 0.6}


@pytest.fixture(scope="module")
def render() -> FoSRender:
    return FoSRender()


def test_sphere(render: FoSRender) -> None:
    res = render.compute(SPHERE, {})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-14)
    assert res.r_south == pytest.approx(1.0, abs=1e-14)
    assert res.z_cm == 0.0                                 # FoS shape is COM-centered
    assert res.scalars["z_shift"] == pytest.approx(0.0, abs=1e-14)
    assert res.scalars["a2"] == pytest.approx(0.0, abs=1e-14)
    assert "z_shift" in res.length_keys
    assert res.neck is None
    assert res.drho_dz is not None and res.drho_dz.shape == res.rho.shape


def test_pole_radii_formula(render: FoSRender) -> None:
    # Spec (library design 3.2): r_north = c + z_shift, r_south = |z_shift - c|.
    res = render.compute(NECKED, {})
    assert res.ok
    zs, c = res.scalars["z_shift"], NECKED["c"]
    assert res.r_north == pytest.approx(c + zs, abs=1e-12)
    assert res.r_south == pytest.approx(abs(zs - c), abs=1e-12)
    assert res.z_cm == 0.0                                 # true shape COM-centered


def test_dr_dtheta_matches_gradient(render: FoSRender) -> None:
    res = render.compute(NECKED, {})
    assert res.ok
    fd = np.gradient(res.radius, res.theta)
    tol = 1e-3 * (1.0 + float(np.max(np.abs(res.dr_dtheta))))
    assert float(np.max(np.abs(res.dr_dtheta - fd))) < tol


def test_necked_shape(render: FoSRender) -> None:
    res = render.compute(NECKED, {})
    assert res.ok
    assert res.neck is not None
    assert res.neck.source == "lib"
    assert res.neck.rho > 0.0
    assert res.neck.depth > 0.0


def test_invalid_c_returns_status_not_exception(render: FoSRender) -> None:
    res = render.compute({**SPHERE, "c": 0.0}, {})
    assert not res.ok
    assert res.status_name in ("ERROR_INVALID_C", "ERROR_INVALID_ARGUMENTS")
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()


def test_invalid_not_star_convex_still_has_neck(render: FoSRender) -> None:
    # Strongly necked, left-right asymmetric shape the lib rejects for
    # star-convexity — the neck is still well-defined and must be plotted.
    res = render.compute({**SPHERE, "c": 2.0, "a3": 0.4, "a4": 0.67}, {})
    assert not res.ok
    assert res.status_name == "ERROR_NOT_STAR_CONVEX"
    assert res.neck is not None
    assert res.neck.source == "lib"
    assert res.neck.rho > 0.0


def test_separated_shape_has_no_neck(render: FoSRender) -> None:
    # Interior rho <= 0: the body has split, so the neck radius is 0 by
    # definition and no neck line should be drawn.
    res = render.compute({**SPHERE, "c": 2.5, "a4": 1.2}, {})
    assert res.status_name == "ERROR_RHO_NEGATIVE"
    assert res.neck is None


def test_slider_specs_and_filename(render: FoSRender) -> None:
    assert [s.key for s in render.slider_specs] == ["c", "a3", "a4", "a5", "a6", "a7", "a8"]
    assert render.slider_specs[0].markers == (1.0, 3.0)
    assert render.has_extra_panel is True
    name = render.filename(92, 144, {**SPHERE, "c": 1.2, "a3": 0.17})
    assert name == "fos_shape_Z92_N144_c1.20_a30.17_a40.00_a50.00_a60.00_a70.00_a80.00.png"


def test_energy_requests_always_single() -> None:
    from src.renders.fos import FoSRender
    render = FoSRender()
    p = dict(c=2.0, a3=0.2, a4=0.6, a5=0.0, a6=0.0, a7=0.0, a8=0.0)
    res = render.compute(p, {})
    assert res.ok
    (req,) = render.energy_requests(p, res)
    assert (req.label, req.param_type, req.com_correction) == ("FoS", "fos", True)
    assert req.shape == (2.0, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0)
