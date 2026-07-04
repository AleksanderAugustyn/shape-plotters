"""FoS render over the fos_parameterization package.

rho(z)-native (lib-exact drho_dz, lib-native neck); R(theta) and analytic
dR/dtheta come from the library's arbitrary-node evaluator on the shared
GL-2048 set (src/core/nodes.py) — in sync with the energy model's dense
grid. Neck position/radius are lib values; only the displayed depth reuses
the shared peak analysis on the profile.
"""
from __future__ import annotations

import numpy as np
import fos_parameterization as fp

from src.core import nodes
from src.core.neck import find_neck_indices, neck_depth
from src.core.result import EnergyRequest, NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# rho(z) display-panel resolution (native COM-frame profile) — a display
# choice, not a calculation grid: the GL theta-nodes sample the star-convex
# frame and cannot replace it.
N_PROFILE_POINTS = 721
N_RHO_GRID = 7201   # shape() validity grid — WMMM's N_FOS_RHO_GRID_POINTS
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
        shp = fp.shape(arr, N_RHO_GRID)
        prof = fp.rho_profile(arr, N_PROFILE_POINTS)
        rd = fp.radius_and_derivative(arr, nodes.THETA, shp.z_shift) if shp.ok else None
        ok = shp.ok and rd is not None and rd.ok and prof.ok

        if ok:
            radii, dr_dtheta = rd.radii, rd.dr_dtheta
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)

        if not shp.ok:
            status, status_name, message = int(shp.status), shp.status.name, shp.message
        elif rd is not None and not rd.ok:
            status, status_name, message = int(rd.status), rd.status.name, ""
        else:
            status, status_name, message = int(prof.status), prof.status.name, prof.message

        neck_info = None
        if ok:
            nk = fp.neck(arr)
            if nk.ok and nk.found:
                depth = 0.0
                hit = find_neck_indices(prof.rho)
                if hit is not None:
                    depth = neck_depth(prof.rho, *hit)
                neck_info = NeckInfo(z=nk.z_neck, rho=nk.rho_neck,
                                     depth=depth, source="lib")
        # The FoS shape is COM-centered by definition (rho_profile is the COM
        # frame). The R(θ) representation carries the star-convexity shift, so its
        # own COM is offset — the engine draws that overlay separately.
        return ShapeResult(
            status=status, status_name=status_name, message=message,
            theta=nodes.THETA, radius=radii,
            z=prof.z, rho=prof.rho, drho_dz=prof.drho_dz,
            neck=neck_info,
            scalars={"z_shift": shp.z_shift, "a2": fp.a2(arr)},
            length_keys=frozenset({"z_shift"}),
            dr_dtheta=dr_dtheta, r_north=shp.r_north, r_south=shp.r_south, z_cm=0.0)

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        return (f"fos_shape_Z{z}_N{n}_c{params['c']:.2f}"
                + "".join(f"_a{i}{params[f'a{i}']:.2f}" for i in range(3, 9))
                + ".png")

    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """One request — the FoS dashed overlay is a reframing of the same
        shape, not a second shape; com_correction is inert on the FoS path."""
        return [EnergyRequest("FoS", "fos",
                              tuple(params[k] for k in PARAM_KEYS),
                              com_correction=True)]
