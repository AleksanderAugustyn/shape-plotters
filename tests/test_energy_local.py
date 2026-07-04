"""Local-only WMMM smoke tests — skipped wherever wmmm is not installed.

Structural assertions only: no WMMM-derived numbers are committed. The
corrected_beta10 cross-check compares two live in-process values (render
scalar vs WMMM result) — two independent call paths through the same
library."""
from __future__ import annotations

import math

import pytest

from src.core import energy
from src.renders.beta import BetaRender
from src.renders.fos import FoSRender

pytestmark = pytest.mark.skipif(
    not energy.available(),
    reason="wmmm not installed (strictly local, out-of-band)")

_ENERGY_FIELDS = ("mass_excess", "total_energy", "macro_energy", "micro_energy",
                  "surface_energy", "coulomb_energy",
                  "proton_pairing_gap", "neutron_pairing_gap")


def test_beta_both_variants_and_com_crosscheck() -> None:
    render = BetaRender()
    values = [0.0, 0.85, 0.35, 0.18, 0.0, 0.0, 0.0, 0.0]
    params = {f"beta{i}": v for i, v in zip(range(1, 9), values)}
    res = render.compute(params, {})
    assert res.ok and res.overlay_z is not None
    requests = render.energy_requests(params, res)
    assert len(requests) == 2
    results = [energy.compute(r.param_type, 92, 144, r.shape,
                              com_correction=r.com_correction) for r in requests]
    for er in results:
        assert er.error is None
        assert er.is_valid
        assert all(math.isfinite(getattr(er, f)) for f in _ENERGY_FIELDS)
    # com off: beta10 evaluated as given (the slider value, here 0).
    assert results[0].corrected_beta10 == pytest.approx(0.0, abs=1e-12)
    # com on: WMMM's corrected dipole equals the render's — same library,
    # two independent call paths.
    assert results[1].corrected_beta10 == pytest.approx(
        res.scalars["corrected_beta10"], abs=1e-9)
    # The two variants are genuinely different shapes.
    assert results[0].total_energy != results[1].total_energy


def test_fos_single_request_smoke() -> None:
    render = FoSRender()
    params = dict(c=1.5, a3=0.1, a4=0.05, a5=0.0, a6=0.0, a7=0.0, a8=0.0)
    res = render.compute(params, {})
    assert res.ok
    (req,) = render.energy_requests(params, res)
    er = energy.compute(req.param_type, 92, 144, req.shape,
                        com_correction=req.com_correction)
    assert er.error is None and er.is_valid
    assert all(math.isfinite(getattr(er, f)) for f in _ENERGY_FIELDS)
