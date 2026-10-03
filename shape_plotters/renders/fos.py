"""FoS render over the fos_parameterization package.

rho(z)-native (lib-exact drho_dz, lib-native neck); R(theta) and analytic
dR/dtheta come from the library's cached tier on the shared GL-2048 set
(shape_plotters/core/nodes.py) — in sync with the energy model's dense grid. The cache
evaluates R(theta) in the total-shift frame internally, so no z_shift
plumbing. Neck position/radius are lib values on the cache's u-grid; only
the displayed depth reuses the shared peak analysis on the display profile.
The cache is read-only and every call resolves from params, so it may be
shared across threads. A separated shape draws its fragments from the
unchecked profile; the checked statuses still decide validity.

Fast-math precondition: params must be finite. Sliders only emit finite
values, so no screening happens here.
"""
from __future__ import annotations

import numpy as np
import fos_parameterization as fp

from shape_plotters.core import nodes
from shape_plotters.core.neck import find_neck_indices, neck_depth
from shape_plotters.core.result import EnergyRequest, NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# rho(z) display-panel resolution (native COM-frame profile) — a display
# choice, not a calculation grid: the GL theta-nodes sample the star-convex
# frame and cannot replace it. Tier-1 call: display density is a consumer
# choice the cache's u-grid must not dictate.
N_PROFILE_POINTS = 721
N_RHO_GRID = 7201   # cache u-grid — WMMM's N_FOS_RHO_GRID_POINTS
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

    def __init__(self) -> None:
        # One read-only cache: max_params 7, WMMM-parity u-grid, GL-2048 thetas.
        self._cache = fp.Cache(len(PARAM_KEYS), N_RHO_GRID, nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        arr = [params[k] for k in PARAM_KEYS]
        shp = self._cache.shape(arr)
        rd = self._cache.radius_and_derivative(arr) if shp.ok else None
        prof = fp.rho_z_grid(arr, N_PROFILE_POINTS)
        ok = shp.ok and rd is not None and rd.ok and prof.ok
        # A separated shape fails the checked profile with rho_negative; draw
        # its fragments from the unchecked profile (rho = 0 in the void). The
        # checked results keep deciding ok, status and title.
        drawn = (fp.rho_z_grid_unchecked(arr, N_PROFILE_POINTS)
                 if prof.status == fp.Status.rho_negative else prof)

        if ok:
            radii, dr_dtheta = rd.radii, rd.dr_dtheta
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)

        primary = shp if not shp.ok else (rd if not rd.ok else prof)

        # Cylindrical path carries its own rho-negative gate (2.0.0 gating
        # asymmetry): separated shapes fail inside neck(); non-star-convex
        # and beak-marginal shapes still yield a lib neck.
        neck_info = None
        nk = self._cache.neck(arr)
        if nk.ok and nk.found and nk.rho_neck > 0.0:
            depth = 0.0
            if drawn.ok:
                hit = find_neck_indices(drawn.rho)
                if hit is not None:
                    depth = neck_depth(drawn.rho, *hit)
            neck_info = NeckInfo(z=nk.z_neck, rho=nk.rho_neck,
                                 depth=depth, source="lib")
        # The FoS shape is COM-centered by definition (rho_z_grid is the COM
        # frame). The R(θ) representation carries the star-convexity shift, so its
        # own COM is offset — the engine draws that overlay separately.
        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii,
            z=drawn.z, rho=drawn.rho, drho_dz=drawn.drho_dz,
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
