"""Shared GL-2048 node set: shapes, ordering, exactness, immutability."""
import numpy as np
import pytest

from shape_plotters.core import nodes


def test_shapes_and_ordering() -> None:
    assert nodes.N_NODES == 2048
    for a in (nodes.THETA, nodes.X, nodes.W, nodes.SIN_THETA):
        assert a.shape == (nodes.N_NODES,) and a.dtype == np.float64
    assert np.all(np.diff(nodes.THETA) > 0)                     # ascending theta
    assert 0.0 < nodes.THETA[0] and nodes.THETA[-1] < np.pi     # open rule: no poles


def test_trig_consistency() -> None:
    np.testing.assert_allclose(nodes.X, np.cos(nodes.THETA), rtol=0.0, atol=1e-15)
    # sqrt(1 - x^2) loses ~1e-14 to cancellation at the pole-adjacent node (X -> 1);
    # display-only quantity, so machine-precision agreement is not required.
    np.testing.assert_allclose(nodes.SIN_THETA, np.sin(nodes.THETA), rtol=0.0, atol=1e-13)


def test_gl_exactness() -> None:
    # integral over x in [-1, 1]: dx = 1 -> 2;  x^2 -> 2/3 (GL exact for polynomials)
    assert np.sum(nodes.W) == pytest.approx(2.0, abs=1e-13)
    # GL-2048 is analytically exact for x^2; float64 weights cap agreement at ~2e-13.
    assert np.sum(nodes.W * nodes.X**2) == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_read_only() -> None:
    for a in (nodes.THETA, nodes.X, nodes.W, nodes.SIN_THETA):
        assert not a.flags.writeable
