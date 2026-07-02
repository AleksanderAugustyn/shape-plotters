"""BetaRender: contract wiring over the beta_parameterization package."""
import numpy as np
import pytest

from src.core import quadrature
from src.renders.beta import BetaRender

# Probed in Task 3 Step 5: lib-VALID and necked (all five candidates passed;
# this is the plan default).
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
    assert np.allclose(res.radius, 1.0, atol=1e-6)
    assert res.scalars["vol_factor"] == pytest.approx(1.0, abs=1e-6)
    assert res.neck is None
    assert res.drho_dz is None


def test_volume_fix_matches_wmmm_and_old_plotter(render: BetaRender) -> None:
    # WMMM volume-fixes the beta grid (radius_grid_mod original_volume_factor);
    # the old ShapePlotter PNG for this shape shows Radius Fixing Factor
    # 0.99598851 — cross-plotter golden.
    res = render.compute(_params([0.0, 0.20, 0.10]), {"com": False})
    assert res.ok
    assert res.scalars["vol_factor"] == pytest.approx(0.99598851, abs=1e-5)
    assert quadrature.volume(res.theta, res.radius) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-6)


def test_com_toggle_reports_corrected_beta10(render: BetaRender) -> None:
    res = render.compute(_params([0.0, 0.85, 0.35, 0.18]), {"com": True})
    assert res.ok
    assert "corrected_beta10" in res.scalars


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


def test_slider_specs_and_filename(render: BetaRender) -> None:
    assert [s.key for s in render.slider_specs] == [f"beta{i}" for i in range(1, 9)]
    assert (render.slider_specs[0].vmin, render.slider_specs[0].vmax) == (-1.6, 1.6)
    assert (render.slider_specs[1].vmin, render.slider_specs[1].vmax) == (0.0, 4.0)
    assert render.has_extra_panel is False
    name = render.filename(92, 144, _params([0.0, 1.25]))
    assert name == "92_144_0.00_1.25_0.00_0.00_0.00_0.00_0.00_0.00.png"
