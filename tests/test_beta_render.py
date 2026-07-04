"""BetaRender: contract wiring over the beta_parameterization node-set API."""
import numpy as np
import pytest

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


def test_volume_fix_matches_wmmm_and_old_plotter(render: BetaRender) -> None:
    # Old ShapePlotter PNG golden 0.99598851 (trapezoid); GL-exact differs only
    # in trailing digits — WMMM cross-check happens at the Task 8 gate.
    res = render.compute(_params([0.0, 0.20, 0.10]), {"com": False})
    assert res.ok
    assert res.scalars["vol_factor"] == pytest.approx(0.99598851, abs=1e-5)
    assert quadrature.volume(res.theta, res.radius) == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-12)   # GL volume factor makes this exact


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


def test_com_toggle(render: BetaRender) -> None:
    # Asymmetric shape proven lib-VALID in v0.1 (beta1 input is 0.0, so the
    # COM iteration must move corrected_beta10 away from 0).
    p = _params([0.0, 0.85, 0.35, 0.18])
    off = render.compute(p, {"com": False})
    on = render.compute(p, {"com": True})
    assert off.ok and on.ok
    assert off.scalars["corrected_beta10"] == 0.0                  # input beta1
    assert on.scalars["corrected_beta10"] != 0.0                   # COM moved it
    assert abs(off.z_cm) > 1e-3                                    # off-axis COM, COM off
    assert on.z_cm == pytest.approx(0.0, abs=1e-4)                 # COM correction centers it
    # Legacy-API parity (test-only usage; updated when the 2.3.0 cleanup lands):
    ref = render._cache.radius_grid_with_com_shift([0.0, 0.85, 0.35, 0.18])
    assert on.scalars["corrected_beta10"] == pytest.approx(ref.corrected_beta10, abs=1e-14)


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
    assert res.status_name == "ERROR_INTERIOR_NEGATIVE"
    assert res.message
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()
    assert res.r_north == 0.0 and res.r_south == 0.0


def test_slider_specs_and_filename(render: BetaRender) -> None:
    assert [s.key for s in render.slider_specs] == [f"beta{i}" for i in range(1, 9)]
    assert (render.slider_specs[0].vmin, render.slider_specs[0].vmax) == (-1.6, 1.6)
    assert (render.slider_specs[1].vmin, render.slider_specs[1].vmax) == (0.0, 4.0)
    assert render.has_extra_panel is False
    name = render.filename(92, 144, _params([0.0, 1.25]))
    assert name == "92_144_0.00_1.25_0.00_0.00_0.00_0.00_0.00_0.00.png"
