"""GL quadrature vs analytic sphere values; scaling by length powers."""
import numpy as np
import pytest

from shape_plotters.core import nodes, quadrature

THETA = nodes.THETA
ONES = np.ones(nodes.N_NODES)
ZEROS = np.zeros(nodes.N_NODES)


def test_unit_sphere_volume() -> None:
    assert quadrature.volume(THETA, ONES) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-13)


def test_unit_sphere_surface() -> None:
    assert quadrature.surface_area(THETA, ONES, ZEROS) == pytest.approx(4.0 * np.pi, rel=1e-13)


def test_unit_sphere_z_cm_is_zero() -> None:
    assert quadrature.z_cm(THETA, ONES) == pytest.approx(0.0, abs=1e-13)


def test_shifted_sphere_z_cm() -> None:
    # Unit sphere centered at z = d: R(theta) = d*cos(theta) + sqrt(1 - d^2 sin^2 theta)
    d = 0.2
    r = d * nodes.X + np.sqrt(1.0 - d**2 * nodes.SIN_THETA**2)
    assert quadrature.volume(THETA, r) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-13)
    assert quadrature.z_cm(THETA, r) == pytest.approx(d, abs=1e-13)


def test_length_power_scaling() -> None:
    s = 7.13
    assert quadrature.volume(THETA, s * ONES) == pytest.approx(
        s**3 * quadrature.volume(THETA, ONES), rel=1e-12)
    assert quadrature.surface_area(THETA, s * ONES, ZEROS) == pytest.approx(
        s**2 * quadrature.surface_area(THETA, ONES, ZEROS), rel=1e-12)


def test_wrong_grid_rejected() -> None:
    with pytest.raises(ValueError):
        quadrature.volume(np.linspace(0.0, np.pi, 721), np.ones(721))
