"""Render contract: what a parameterization render gives the engine.

Everything in ShapeResult is in R0 units (dimensionless). The engine owns
display units (fm toggle) and applies R0 = 1.16 * A^(1/3) only at draw time.

A render must expose:
    name: str
    slider_specs: list[SliderSpec]
    toggles: list[ToggleSpec]
    has_extra_panel: bool                # True -> engine draws rho(z) + drho_dz panel
    compute(params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult
    filename(z: int, n: int, params: dict[str, float]) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SliderSpec:
    key: str
    label: str
    vmin: float
    vmax: float
    vinit: float
    step: float
    markers: tuple[float, ...] = ()   # practical-limit lines (red dotted)


@dataclass(frozen=True)
class ToggleSpec:
    key: str
    label: str
    default: bool


@dataclass(frozen=True)
class NeckInfo:
    z: float        # R0 units
    rho: float      # R0 units
    depth: float    # 1 - rho_neck / (lower of the two lobe maxima)
    source: str     # "lib" (FoS) or "py heuristic" (beta)


@dataclass(frozen=True)
class ShapeResult:
    status: int                    # 0 = valid; engine greys the plot otherwise
    status_name: str               # symbolic name from the package Status IntEnum
    message: str
    theta: Array                   # shared GL node set (src/core/nodes.py)
    radius: Array                  # R(theta), R0 units
    z: Array                       # profile axis, R0 units
    rho: Array                     # rho(z), R0 units
    drho_dz: Array | None          # lib-native (FoS); None for beta
    neck: NeckInfo | None
    scalars: dict[str, float]      # lib-native, R0 units where dimensional
    length_keys: frozenset[str]    # which scalars scale with the fm toggle
    dr_dtheta: Array                 # lib-exact analytic dR/dθ, scaled like radius
    r_north: float                   # analytic R(0), R0 units, scaled like radius
    r_south: float                   # analytic R(pi), R0 units, scaled like radius
    z_cm: float                      # true-shape COM in the cross-section frame, R0 units

    @property
    def ok(self) -> bool:
        return self.status == 0
