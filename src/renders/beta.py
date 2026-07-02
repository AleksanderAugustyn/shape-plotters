"""Beta (Legendre) render over the beta_parameterization package.

R(theta)-native: the cylindrical profile is derived parametrically
(z = R cos(theta), rho = R sin(theta)). The neck is the Python display-only
heuristic (src/core/neck.py) — graduating it into the library is recorded
future work (spec section 7).
"""
from __future__ import annotations

import numpy as np
import beta_parameterization as bp

from src.core.neck import find_neck_indices, neck_depth
from src.core.result import NeckInfo, ShapeResult, SliderSpec, ToggleSpec

N_GRID = 721
N_BETAS = 8


class BetaRender:
    name = "beta"
    has_extra_panel = False
    # Ranges from the old ShapePlotter: beta1 (-1.6, 1.6), beta2 (0, 4), rest (-2, 2).
    slider_specs = [
        SliderSpec("beta1", "β1", -1.6, 1.6, 0.0, 0.01),
        SliderSpec("beta2", "β2", 0.0, 4.0, 0.0, 0.01),
    ] + [SliderSpec(f"beta{i}", f"β{i}", -2.0, 2.0, 0.0, 0.01) for i in range(3, N_BETAS + 1)]
    # Default off: parity with old ShapePlotter PNGs (no COM shift there).
    toggles = [ToggleSpec("com", "COM correction", False)]

    def __init__(self) -> None:
        self._cache = bp.Cache(max_beta_params=N_BETAS, n_grid=N_GRID)
        self._theta = bp.theta_grid(N_GRID)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        betas = [params[f"beta{i}"] for i in range(1, N_BETAS + 1)]
        if toggles.get("com", False):
            res = self._cache.radius_grid_with_com_shift(betas)
        else:
            res = self._cache.radius_grid(betas)
        z = res.radii * np.cos(self._theta)
        rho = res.radii * np.sin(self._theta)
        neck = None
        if res.ok:
            hit = find_neck_indices(rho)
            if hit is not None:
                i_neck, i_a, i_b = hit
                neck = NeckInfo(z=float(z[i_neck]), rho=float(rho[i_neck]),
                                depth=neck_depth(rho, i_neck, i_a, i_b),
                                source="py heuristic")
        scalars: dict[str, float] = {}
        if res.corrected_beta10 is not None:
            scalars["corrected_beta10"] = res.corrected_beta10
        return ShapeResult(
            status=int(res.status), status_name=res.status.name, message=res.message,
            theta=self._theta, radius=res.radii, z=z, rho=rho, drho_dz=None,
            neck=neck, scalars=scalars, length_keys=frozenset())

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        betas = "_".join(f"{params[f'beta{i}']:.2f}" for i in range(1, N_BETAS + 1))
        return f"{z}_{n}_{betas}.png"
