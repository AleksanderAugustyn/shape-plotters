"""Quadrature vs analytic sphere values; scaling by length powers."""
import numpy as np
import pytest

from src.core import quadrature

N = 721
THETA = np.linspace(0.0, np.pi, N)


def test_unit_sphere_volume() -> None:
    r = np.ones(N)
    assert quadrature.volume(THETA, r) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-4)


def test_unit_sphere_surface() -> None:
    r = np.ones(N)
    assert quadrature.surface_area(THETA, r) == pytest.approx(4.0 * np.pi, rel=1e-4)


def test_unit_sphere_z_cm_is_zero() -> None:
    r = np.ones(N)
    assert quadrature.z_cm(THETA, r) == pytest.approx(0.0, abs=1e-10)


def test_shifted_sphere_z_cm() -> None:
    # Unit sphere centered at z = d: R(theta) = d*cos(theta) + sqrt(1 - d^2 sin^2(theta))
    d = 0.2
    r = d * np.cos(THETA) + np.sqrt(1.0 - d**2 * np.sin(THETA) ** 2)
    assert quadrature.volume(THETA, r) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-4)
    assert quadrature.z_cm(THETA, r) == pytest.approx(d, abs=1e-3)


def test_length_power_scaling() -> None:
    # fm display works by scaling results with R0-powers; verify the powers once.
    r = np.ones(N)
    s = 7.13
    assert quadrature.volume(THETA, s * r) == pytest.approx(s**3 * quadrature.volume(THETA, r), rel=1e-12)
    assert quadrature.surface_area(THETA, s * r) == pytest.approx(s**2 * quadrature.surface_area(THETA, r), rel=1e-9)
