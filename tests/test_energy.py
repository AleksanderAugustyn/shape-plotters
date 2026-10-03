"""Energy adapter over a stub wmmm module — no real WMMM anywhere.

Committed tests carry no WMMM-derived numbers; the stub returns arbitrary
sentinels. The stub needs a real ModuleSpec: find_spec raises on modules
whose __spec__ is None."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

from shape_plotters.core import energy


def _install_stub(monkeypatch, log: list) -> types.ModuleType:
    stub = types.ModuleType("wmmm")
    stub.__spec__ = importlib.machinery.ModuleSpec("wmmm", loader=None)

    class Model:
        def __init__(self, param_type: str) -> None:
            log.append(("init", param_type))

        def set(self, name: str, value: float) -> None:
            log.append(("set", name, value))

        def compute(self, z: int, n: int, shape: list) -> SimpleNamespace:
            log.append(("compute", z, n, tuple(shape)))
            return SimpleNamespace(
                is_valid=True, mass_excess=1.0, total_energy=2.0,
                macro_energy=3.0, micro_energy=4.0, surface_energy=5.0,
                coulomb_energy=6.0, proton_pairing_gap=7.0,
                neutron_pairing_gap=8.0, proton_k=9, neutron_k=10,
                corrected_beta10=0.25, has_neck=False,
                neck_radius_fm=0.0, is_scissioning=False)

    stub.Model = Model
    monkeypatch.setitem(sys.modules, "wmmm", stub)
    return stub


@pytest.fixture(autouse=True)
def fresh_model_cache(monkeypatch):
    monkeypatch.setattr(energy, "_models", {})


def test_available_true_with_stub(monkeypatch) -> None:
    _install_stub(monkeypatch, [])
    assert energy.available() is True


def test_available_false_when_not_installed(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "wmmm", raising=False)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "wmmm" else real_find_spec(name, *a))
    assert energy.available() is False


def test_available_does_not_import(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "wmmm", raising=False)
    energy.available()
    assert "wmmm" not in sys.modules


def test_compute_maps_pointresult_fields(monkeypatch) -> None:
    log: list = []
    _install_stub(monkeypatch, log)
    res = energy.compute("fos", 92, 144, (2.0, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0))
    assert res.error is None and res.is_valid is True
    assert (res.mass_excess, res.total_energy, res.macro_energy, res.micro_energy,
            res.surface_energy, res.coulomb_energy) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert (res.proton_pairing_gap, res.neutron_pairing_gap) == (7.0, 8.0)
    assert (res.proton_k, res.neutron_k) == (9, 10)
    assert res.corrected_beta10 == 0.25
    assert ("compute", 92, 144, (2.0, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0)) in log


def test_model_cached_per_config(monkeypatch) -> None:
    log: list = []
    _install_stub(monkeypatch, log)
    energy.compute("legendre", 92, 144, [0.0] * 20, com_correction=False)
    energy.compute("legendre", 92, 144, [0.1] * 20, com_correction=False)
    energy.compute("legendre", 92, 144, [0.0] * 20, com_correction=True)
    assert len([e for e in log if e[0] == "init"]) == 2   # one per (type, com)
    assert [e for e in log if e[0] == "set"] == \
        [("set", "beta_10_com_shift", 0.0)]               # once, com-off model only


def test_compute_failure_returns_error_result(monkeypatch) -> None:
    stub = _install_stub(monkeypatch, [])

    class ExplodingModel:
        def __init__(self, param_type: str) -> None:
            raise RuntimeError("library not built")

    stub.Model = ExplodingModel
    res = energy.compute("fos", 92, 144, (1.0,) * 7)
    assert res.error is not None and "library not built" in res.error
    assert res.is_valid is False and res.total_energy == 0.0
