"""Optional WMMM energy adapter — the only module allowed to import wmmm.

The energy model is strictly local: wmmm is never a dependency of this
package (not even an optional extra) and is installed out-of-band as a local
editable install. This module carries the calling convention only — no model
code, data, or paths. The engine consults available() to decide whether the
Energy button exists; compute() returns EnergyResult, with failures in
EnergyResult.error rather than exceptions.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EnergyResult:
    """One WMMM point result (energies in MeV); zero-filled when error is set."""
    is_valid: bool
    mass_excess: float
    total_energy: float
    macro_energy: float
    micro_energy: float
    surface_energy: float
    coulomb_energy: float
    proton_pairing_gap: float
    neutron_pairing_gap: float
    proton_k: int
    neutron_k: int
    corrected_beta10: float     # cross-check channel (vs the render scalar), not displayed
    error: str | None = None


# Model handles are configuration-immutable: one per (param_type,
# com_correction) for the app lifetime (~20 ms per compute thereafter).
_models: dict[tuple[str, bool], object] = {}


def available() -> bool:
    """True when the wmmm package is importable; performs no import."""
    return importlib.util.find_spec("wmmm") is not None


def _error(message: str) -> EnergyResult:
    return EnergyResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0, 0, 0.0, error=message)


def _get_model(param_type: str, com_correction: bool):
    key = (param_type, com_correction)
    model = _models.get(key)
    if model is None:
        import wmmm  # deferred: loads the Fortran library on first use only
        model = wmmm.Model(param_type)
        if not com_correction:
            model.set("beta_10_com_shift", 0.0)
        _models[key] = model
    return model


def compute(param_type: str, z: int, n: int, shape: Sequence[float],
            com_correction: bool = True) -> EnergyResult:
    """One WMMM point computation; never raises.

    Parameters
    ----------
    param_type : str
        "legendre" (20 shape params) or "fos" (7).
    z, n : int
        Proton and neutron numbers.
    shape : Sequence[float]
        Shape parameters in WMMM's convention (raw slider values).
    com_correction : bool
        WMMM's beta_10_com_shift flag. False evaluates beta10 as given
        (legendre only — inert on the fos path).
    """
    try:
        model = _get_model(param_type, com_correction)
        r = model.compute(int(z), int(n), list(shape))
    except Exception as exc:  # any wmmm failure becomes a stats-box message
        return _error(f"{type(exc).__name__}: {exc}")
    return EnergyResult(
        is_valid=bool(r.is_valid), mass_excess=r.mass_excess,
        total_energy=r.total_energy, macro_energy=r.macro_energy,
        micro_energy=r.micro_energy, surface_energy=r.surface_energy,
        coulomb_energy=r.coulomb_energy,
        proton_pairing_gap=r.proton_pairing_gap,
        neutron_pairing_gap=r.neutron_pairing_gap,
        proton_k=int(r.proton_k), neutron_k=int(r.neutron_k),
        corrected_beta10=r.corrected_beta10)
