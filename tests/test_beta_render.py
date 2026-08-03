"""BetaRender: contract wiring over the beta_parameterization 3.0.0 cached tier."""
import numpy as np
import pytest
import beta_parameterization as bp

from src.core import nodes, quadrature
from src.renders.beta import BetaRender

# Probed lib-VALID and necked (carried over from v0.1).
NECKED_BETAS = [0.0, 1.5, 0.0, 0.8]


@pytest.fixture(scope="module")
def render() -> BetaRender:
    return BetaRender()


def _params(betas: list[float]) -> dict[str, float]:
    full = betas + [0.0] * (8 - len(betas))
    return {f"beta{i}": full[i - 1] for i in range(1, 9)}


def test_status_contract() -> None:
    # 3.0.0 renumbering: shared contract codes 0-6, library codes >= 100,
    # members lowercase. Guards the raw ints ShapeResult.status carries.
    assert bp.Status.valid == 0
    assert bp.Status.interior_negative == 102
    assert bp.CACHE_MAX_PARAMS == 8


def test_sphere(render: BetaRender) -> None:
    res = render.compute(_params([]), {"com": False})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-12)
    assert res.r_south == pytest.approx(1.0, abs=1e-12)
    assert res.z_cm == pytest.approx(0.0, abs=1e-9)        # sphere COM at origin
    assert res.scalars["vol_factor"] == pytest.approx(1.0, abs=1e-12)
    assert res.neck is None
    assert res.drho_dz is None


def test_volume_factor_from_library(render: BetaRender) -> None:
    # 3.0.0: the factor comes from the library (GL-512 over the raw shape),
    # replacing the Python GL-2048 replica. Old ShapePlotter PNG golden
    # 0.99598851 (trapezoid) still agrees to display tolerance, and the
    # returned radii integrate to the unit-sphere volume exactly.
    res = render.compute(_params([0.0, 0.20, 0.10]), {"com": False})
    assert res.ok
    assert res.scalars["vol_factor"] == pytest.approx(0.99598851, abs=1e-5)
    assert quadrature.volume(res.theta, res.radius) == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-12)


def test_beta2_equator_symmetry(render: BetaRender) -> None:
    # Pure beta2 is symmetric about the equator; GL nodes are symmetric in
    # x = cos(theta), so the radii must mirror exactly.
    res = render.compute(_params([0.0, 0.5]), {"com": False})
    assert res.ok
    assert np.allclose(res.radius, res.radius[::-1], atol=1e-12)
    assert res.r_north == pytest.approx(res.r_south, abs=1e-12)


def test_dr_dtheta_is_exact_not_gradient(render: BetaRender) -> None:
    # Wiring check only (analytic-derivative goldens live in the library):
    # np.gradient of the returned radii must agree to display tolerance.
    res = render.compute(_params(NECKED_BETAS), {"com": False})
    assert res.ok
    fd = np.gradient(res.radius, res.theta)
    # np.gradient's one-sided endpoint formula is inaccurate on the pole-adjacent
    # GL nodes (steep R, wide non-uniform spacing); the interior central
    # differences confirm the lib analytic derivative to ~1e-5.
    tol = 1e-3 * (1.0 + float(np.max(np.abs(res.dr_dtheta))))
    assert float(np.max(np.abs(res.dr_dtheta[1:-1] - fd[1:-1]))) < tol


def test_com_corrected_overlay(render: BetaRender) -> None:
    # Asymmetric shape proven lib-VALID in v0.1 (beta1 input is 0.0, so the COM
    # iteration moves corrected_beta10 away from 0). The default shape keeps its
    # off-axis COM; the COM-corrected overlay recenters it.
    p = _params([0.0, 0.85, 0.35, 0.18])
    res = render.compute(p, {})
    assert res.ok
    assert abs(res.z_cm) > 1e-3                                    # slider shape COM off axis
    assert res.scalars["corrected_beta10"] != 0.0                 # centering dipole
    # Overlay outline present, closed at rho = 0, and its own COM centered.
    assert res.overlay_z is not None and res.overlay_rho is not None
    assert res.overlay_rho[0] == 0.0 and res.overlay_rho[-1] == 0.0
    assert res.overlay_z_cm == pytest.approx(0.0, abs=1e-4)
    assert res.overlay_ok


def test_no_overlay_when_symmetric(render: BetaRender) -> None:
    # Even multipoles only: the COM is already centered, so corrected_beta10 ~ 0
    # and the overlay is suppressed.
    res = render.compute(_params([0.0, 0.30, 0.0, 0.10]), {})
    assert res.ok
    assert abs(res.scalars["corrected_beta10"]) <= 1e-3
    assert res.overlay_z is None


def test_pole_radii_scaled_with_volume_factor(render: BetaRender) -> None:
    res = render.compute(_params([0.0, 0.20, 0.10]), {"com": False})
    assert res.ok
    # Poles carry the same volume factor as the radii: R(theta->0) -> r_north.
    assert res.r_north == pytest.approx(float(res.radius[0]), abs=1e-4)
    assert res.r_south == pytest.approx(float(res.radius[-1]), abs=1e-4)


def test_necked_shape(render: BetaRender) -> None:
    res = render.compute(_params(NECKED_BETAS), {"com": False})
    assert res.ok
    assert res.neck is not None
    assert res.neck.source == "py heuristic"
    assert 0.0 < res.neck.depth < 1.0
    assert res.neck.rho > 0.0


def test_invalid_shape_returns_status_not_exception(render: BetaRender) -> None:
    res = render.compute(_params([0.0, 4.0]), {"com": False})  # interior negative
    assert not res.ok
    assert res.status == int(bp.Status.interior_negative)      # 102 in 3.0.0
    assert res.status_name == "interior_negative"              # lowercase in 3.0.0
    assert res.message
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()
    assert res.r_north == 0.0 and res.r_south == 0.0
    # Unchecked-path outline: drawn greyed, not collapsed to a point, with
    # negative radii preserved (rho = R sin(theta) < 0 where R < 0).
    assert res.rho.any() and res.z.any()
    assert (res.rho < 0.0).any()


def test_overlay_when_slider_beta1_differs(render: BetaRender) -> None:
    # Symmetric multipoles with a nonzero slider beta1: corrected_beta10 ~ 0,
    # yet the slider shape differs from the COM-corrected one — beta10 is a
    # shape parameter, not a translation. The overlay must show.
    res = render.compute(_params([0.5, 0.30, 0.0, 0.10]), {})
    assert res.ok
    assert abs(res.scalars["corrected_beta10"]) <= 1e-3
    assert res.overlay_z is not None


def test_no_overlay_when_slider_matches_corrected(render: BetaRender) -> None:
    # Slider beta1 set to the corrected dipole: the two shapes coincide.
    first = render.compute(_params([0.0, 0.85, 0.35, 0.18]), {})
    corrected = first.scalars["corrected_beta10"]
    res = render.compute(_params([corrected, 0.85, 0.35, 0.18]), {})
    assert res.ok
    assert res.overlay_z is None


def test_slider_specs_and_filename(render: BetaRender) -> None:
    assert [s.key for s in render.slider_specs] == [f"beta{i}" for i in range(1, 9)]
    assert (render.slider_specs[0].vmin, render.slider_specs[0].vmax) == (-1.6, 1.6)
    assert (render.slider_specs[1].vmin, render.slider_specs[1].vmax) == (0.0, 4.0)
    assert render.has_extra_panel is False
    name = render.filename(92, 144, _params([0.0, 1.25]))
    assert name == "92_144_0.00_1.25_0.00_0.00_0.00_0.00_0.00_0.00.png"


def test_energy_requests_single_without_overlay(render: BetaRender) -> None:
    p = _params([0.0, 0.30])
    res = render.compute(p, {})
    assert res.overlay_z is None
    (req,) = render.energy_requests(p, res)
    assert (req.label, req.param_type, req.com_correction) == ("slider", "legendre", False)
    assert len(req.shape) == 20                      # WMMM's legendre width
    assert req.shape[:8] == (0.0, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert req.shape[8:] == (0.0,) * 12              # zero-padded tail


def test_energy_requests_both_with_overlay(render: BetaRender) -> None:
    p = _params([0.0, 0.85, 0.35, 0.18])
    res = render.compute(p, {})
    assert res.overlay_z is not None
    reqs = render.energy_requests(p, res)
    assert [r.label for r in reqs] == ["slider", "COM corrected"]
    assert [r.com_correction for r in reqs] == [False, True]
    assert reqs[0].shape == reqs[1].shape            # WMMM recomputes beta10 itself
