"""Funny-Hills (Brack et al. 1972) render over the funnyhills_parameterization package.

The three sliders are the paper's {c, h, α}: c the half-length, h the neck
coordinate (h = 0 runs along the liquid-drop valley), α the asymmetry.
Everything shown comes from the library: R(θ) and analytic dR/dθ from one
read-only cache on the shared GL-2048 set (shape_plotters/core/nodes.py), the
uniform-z ρ(z) profile with its lib-exact dρ/dz, the exact neck, the resolved
geometry (A, B, the branch, the gap and the lobe volume), q2, ρ_cm and the
(c, h)-plane reference lines at the current c. Every call resolves from
params, so the cache may be shared across threads.

The checked outputs decide validity. A separated shape fails the checked
profile with `separated`; its fragments are drawn from the unchecked profile
(ρ = 0 in the void) while the checked status keeps the verdict and title.
Past scission the paper's normalization over-fills the fragments; the stats
show it as lobe_volume_ratio.

Poles and flags come from `geometry`, never from `shape`: `shape` zero-fills
its poles on not_star_convex and separated, and the engine passes the poles
to `fragment_volumes` as the cap positions, so a pinched or separated shape
would otherwise get zero-radius caps and a biased mass split.

A NeckInfo is built whenever the library neck is found, ρ_neck = 0 included:
for a pinched or separated shape that is the split plane between the
fragments (the void's midpoint).

R(θ) is about the true centre of mass, as tqs's is, so the engine never
infers an overlay frame.

Fast-math precondition: params must be finite. Sliders only emit finite
values, so no screening happens here.
"""
from __future__ import annotations

import numpy as np
import funnyhills_parameterization as fp

from shape_plotters.core import nodes
from shape_plotters.core.neck import find_neck_indices, neck_depth
from shape_plotters.core.result import EnergyRequest, NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# rho(z) display-panel resolution — a display choice, not a calculation grid.
N_PROFILE_POINTS = 721
N_U_GRID = 2049     # cache scan grid: star gate, d_min, R(θ) brackets
PARAM_KEYS = ("c", "h", "alpha")
FOUR_THIRDS_PI = 4.0 * np.pi / 3.0


class FunnyhillsRender:
    name = "funnyhills"
    has_extra_panel = True
    # The steps reach the paper's grid (Δc = 0.06, Δh = 0.075, α in steps of
    # 0.05) and the 240Pu stationary points exactly. Markers: the sphere
    # c = 1, the valley line h = 0, symmetry α = 0.
    slider_specs = [
        SliderSpec("c", "c", 0.7, 2.5, 1.0, 0.01, markers=(1.0,)),
        SliderSpec("h", "h", -0.4, 0.6, 0.0, 0.005, markers=(0.0,)),
        SliderSpec("alpha", "α", -0.5, 0.5, 0.0, 0.01, markers=(0.0,)),
    ]
    toggles: list[ToggleSpec] = []

    def __init__(self) -> None:
        self._cache = fp.Cache(N_U_GRID, nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        arr = [params[k] for k in PARAM_KEYS]
        cache = self._cache
        geo = cache.geometry(arr)
        shp = cache.shape(arr)
        rd = cache.radius_and_derivative(arr) if shp.ok else None
        prof = cache.rho_z_grid(arr, N_PROFILE_POINTS)
        drawn = (cache.rho_z_grid_unchecked(arr, N_PROFILE_POINTS)
                 if prof.status == fp.Status.separated else prof)

        ok = shp.ok and rd is not None and rd.ok and prof.ok
        primary = shp if not shp.ok else (rd if not rd.ok else prof)
        if ok:
            radii, dr_dtheta = rd.radii, rd.dr_dtheta
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)

        neck_info = None
        nk = cache.neck(arr)
        if nk.ok and nk.found:
            depth = 0.0
            if drawn.ok:
                hit = find_neck_indices(drawn.rho)
                if hit is not None:
                    depth = neck_depth(drawn.rho, *hit)
            neck_info = NeckInfo(z=nk.z_neck, rho=nk.rho_neck, depth=depth, source="lib")

        scalars: dict[str, float] = {}
        notes: tuple[str, ...] = ()
        if geo.ok:
            mo = cache.moments(arr)
            co = cache.coefficients(arr)
            scalars = {"a_coeff": geo.a_coeff, "b_coeff": geo.b_coeff, "z_shift": geo.z_shift}
            if geo.is_separated:
                scalars["gap"] = geo.gap
                scalars["lobe_volume_ratio"] = geo.lobe_volume / FOUR_THIRDS_PI
            scalars |= {"rho_cm": mo.rho_cm, "q2": mo.q2, "d_min": geo.d_min,
                        "h_ellipsoid": co.h_ellipsoid, "h_scission": co.h_scission}
            notes = (self._branch_note(geo, nk),)

        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii,
            z=drawn.z, rho=drawn.rho, drho_dz=drawn.drho_dz,
            neck=neck_info,
            scalars=scalars,
            length_keys=frozenset({"z_shift", "gap", "rho_cm"}),
            dr_dtheta=dr_dtheta,
            r_north=geo.r_north if geo.ok else 0.0,
            r_south=geo.r_south if geo.ok else 0.0,
            z_cm=0.0,
            notes=notes,
            infer_rtheta_overlay=False)

    @staticmethod
    def _branch_note(geo: fp.Geometry, nk: fp.NeckResult) -> str:
        note = geo.branch.name
        if nk.ok and nk.found and nk.rho_neck > 0.0:
            note += ", necked"
        if geo.is_pinched:
            note += ", pinched"
        if geo.is_separated:
            note += ", separated"
        return note

    def filename(self, a_mass: int, params: dict[str, float]) -> str:
        # h with three decimals: two cannot represent the 0.005 step.
        return (f"funnyhills_A{a_mass}_c{params['c']:.2f}_h{params['h']:.3f}"
                f"_al{params['alpha']:.2f}.png")

    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """None until WMMM gains a funnyhills parameterization."""
        return []
