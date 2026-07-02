"""FoSRender: contract wiring over the fos_parameterization package."""
import numpy as np
import pytest

from src.renders.fos import FoSRender

SPHERE = {"c": 1.0, "a3": 0.0, "a4": 0.0, "a5": 0.0, "a6": 0.0, "a7": 0.0, "a8": 0.0}
# Probed lib-VALID necked shape (the archived FoSFitter c=2.25/a3=0.25/a4=0.72
# is ERROR_NOT_STAR_CONVEX in the library, anticipated by the plan).
NECKED = {**SPHERE, "c": 2.0, "a4": 0.6}


@pytest.fixture(scope="module")
def render() -> FoSRender:
    return FoSRender()


def test_sphere(render: FoSRender) -> None:
    res = render.compute(SPHERE, {})
    assert res.ok
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert res.scalars["z_shift"] == pytest.approx(0.0, abs=1e-14)
    # Lib convention: a2 is the volume-constraint correction a4/3 - a6/5 + a8/7
    # (compute_fos_a2_f) — 0 for the sphere, NOT FoSFitter's unrelated r0^2/c.
    assert res.scalars["a2"] == pytest.approx(0.0, abs=1e-14)
    assert "z_shift" in res.length_keys
    assert res.neck is None
    assert res.drho_dz is not None and res.drho_dz.shape == res.rho.shape


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


def test_slider_specs_and_filename(render: FoSRender) -> None:
    assert [s.key for s in render.slider_specs] == ["c", "a3", "a4", "a5", "a6", "a7", "a8"]
    assert render.slider_specs[0].markers == (1.0, 3.0)
    assert render.has_extra_panel is True
    name = render.filename(92, 144, {**SPHERE, "c": 1.2, "a3": 0.17})
    assert name == "fos_shape_Z92_N144_c1.20_a30.17_a40.00_a50.00_a60.00_a70.00_a80.00.png"
