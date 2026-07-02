"""Neck heuristic on synthetic profiles: two highest interior maxima, lowest
minimum between them, no fixed 90-degree split (the spec's asymmetric case)."""
import numpy as np
import pytest

from src.core.neck import find_neck_indices, neck_depth

X = np.linspace(0.0, np.pi, 721)


def _bumps(centers_heights_widths: list[tuple[float, float, float]]) -> np.ndarray:
    rho = np.zeros_like(X)
    for c, h, w in centers_heights_widths:
        rho += h * np.exp(-(((X - c) / w) ** 2))
    return rho


def test_single_hump_has_no_neck() -> None:
    rho = np.sin(X)  # sphere-like: one interior maximum
    assert find_neck_indices(rho) is None


def test_two_lobes_symmetric() -> None:
    rho = _bumps([(1.0, 1.0, 0.35), (2.1, 1.0, 0.35)])
    hit = find_neck_indices(rho)
    assert hit is not None
    i_neck, i_a, i_b = hit
    assert X[i_a] == pytest.approx(1.0, abs=0.02)
    assert X[i_b] == pytest.approx(2.1, abs=0.02)
    assert 1.0 < X[i_neck] < 2.1
    assert neck_depth(rho, i_neck, i_a, i_b) > 0.1


def test_both_lobes_on_one_side_of_90deg() -> None:
    # Both maxima below pi/2 (~1.571): the fixed-90-degree split would miss this.
    rho = _bumps([(0.7, 1.0, 0.2), (1.4, 0.9, 0.2)])
    hit = find_neck_indices(rho)
    assert hit is not None
    i_neck, i_a, i_b = hit
    assert X[i_b] < np.pi / 2
    assert 0.7 < X[i_neck] < 1.4


def test_ripple_bumps_pick_the_two_highest() -> None:
    # Third small ripple must not displace the true lobes.
    rho = _bumps([(0.8, 1.0, 0.25), (2.3, 0.95, 0.25), (1.55, 0.4, 0.1)])
    hit = find_neck_indices(rho)
    assert hit is not None
    _, i_a, i_b = hit
    assert X[i_a] == pytest.approx(0.8, abs=0.05)
    assert X[i_b] == pytest.approx(2.3, abs=0.05)
