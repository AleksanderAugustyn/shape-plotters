"""FoS render over the fos_parameterization package.

rho(z)-native (lib-exact drho_dz, lib-native neck); R(theta) comes from the
library's own radius_grid, so no Python coordinate conversion is needed.
Neck position/radius are lib values; only the displayed depth reuses the
shared peak analysis on the profile.
"""
from __future__ import annotations

import fos_parameterization as fp

from src.core.neck import find_neck_indices, neck_depth
from src.core.result import NeckInfo, ShapeResult, SliderSpec, ToggleSpec

N_GRID = 721
PARAM_KEYS = ("c", "a3", "a4", "a5", "a6", "a7", "a8")


class FoSRender:
    name = "fos"
    has_extra_panel = True
    # Ranges and practical-limit markers from ShapePlotterFoSFitter.
    slider_specs = [
        SliderSpec("c", "c", 0.5, 3.5, 1.0, 0.01, markers=(1.0, 3.0)),
        SliderSpec("a3", "a3", -0.6, 0.6, 0.0, 0.01, markers=(0.0, 0.5)),
        SliderSpec("a4", "a4", -0.75, 0.75, 0.0, 0.01, markers=(-0.2, 0.72)),
    ] + [SliderSpec(f"a{i}", f"a{i}", -0.5, 0.5, 0.0, 0.01, markers=(-0.2, 0.2))
         for i in range(5, 9)]
    toggles: list[ToggleSpec] = []

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        arr = [params[k] for k in PARAM_KEYS]
        rad = fp.radius_grid(arr, N_GRID)
        prof = fp.rho_profile(arr, N_GRID)
        primary = rad if rad.status != 0 else prof
        neck_info = None
        if rad.ok and prof.ok:
            nk = fp.neck(arr)
            if nk.ok and nk.found:
                depth = 0.0
                hit = find_neck_indices(prof.rho)
                if hit is not None:
                    depth = neck_depth(prof.rho, *hit)
                neck_info = NeckInfo(z=nk.z_neck, rho=nk.rho_neck,
                                     depth=depth, source="lib")
        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=fp.theta_grid(N_GRID), radius=rad.radii,
            z=prof.z, rho=prof.rho, drho_dz=prof.drho_dz,
            neck=neck_info,
            scalars={"z_shift": fp.z_shift(arr), "a2": fp.a2(arr)},
            length_keys=frozenset({"z_shift"}))

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        return (f"fos_shape_Z{z}_N{n}_c{params['c']:.2f}"
                + "".join(f"_a{i}{params[f'a{i}']:.2f}" for i in range(3, 9))
                + ".png")
