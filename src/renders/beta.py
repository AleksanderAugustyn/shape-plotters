"""Beta (Legendre) render over the beta_parameterization package.

GL-native: R(theta) and analytic dR/dtheta come from the library's node-set
API evaluated on the shared GL-2048 set (src/core/nodes.py) — in sync with
the energy model's dense grid. The neck is the Python display-only heuristic
(src/core/neck.py) — graduating it into the library is recorded future work.
"""
from __future__ import annotations

import numpy as np
import beta_parameterization as bp

from src.core import nodes
from src.core.neck import find_neck_indices, neck_depth
from src.core.result import NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# Cache constructor still requires a uniform-grid size; no uniform-grid
# feature is used (the argument goes away with the library's 2.3.0 cleanup).
N_GRID = 721
N_BETAS = 8
SPHERE_VOLUME = 4.0 * np.pi / 3.0  # unit sphere, R0 units


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
        self._node_set = self._cache.build_node_set(nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        betas = [params[f"beta{i}"] for i in range(1, N_BETAS + 1)]
        resolved = self._cache.resolve_shape(
            betas, apply_com_correction=toggles.get("com", False))
        rd = None
        if resolved.ok:
            rd = self._cache.radius_and_derivative(resolved.beta_con, self._node_set)
        ok = rd is not None and rd.ok

        vol_factor = 1.0
        if ok:
            # WMMM's exact GL volume factor (the library's original_volume_factor):
            # radii arrive pre-scaled everywhere — derivative and poles included.
            raw_volume = (2.0 * np.pi / 3.0) * float(np.sum(nodes.W * rd.radii**3))
            vol_factor = float((SPHERE_VOLUME / raw_volume) ** (1.0 / 3.0))
            radii = rd.radii * vol_factor
            dr_dtheta = rd.dr_dtheta * vol_factor
            r_north = resolved.r_north * vol_factor
            r_south = resolved.r_south * vol_factor
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)
            r_north = r_south = 0.0

        z = radii * nodes.X
        rho = radii * nodes.SIN_THETA
        neck = None
        if ok:
            hit = find_neck_indices(rho)
            if hit is not None:
                i_neck, i_a, i_b = hit
                neck = NeckInfo(z=float(z[i_neck]), rho=float(rho[i_neck]),
                                depth=neck_depth(rho, i_neck, i_a, i_b),
                                source="py heuristic")
        scalars: dict[str, float] = {"vol_factor": vol_factor}
        if resolved.ok:
            scalars["corrected_beta10"] = resolved.corrected_beta10
        primary = rd if resolved.ok else resolved
        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii, z=z, rho=rho, drho_dz=None,
            neck=neck, scalars=scalars, length_keys=frozenset(),
            dr_dtheta=dr_dtheta, r_north=r_north, r_south=r_south)

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        betas = "_".join(f"{params[f'beta{i}']:.2f}" for i in range(1, N_BETAS + 1))
        return f"{z}_{n}_{betas}.png"
