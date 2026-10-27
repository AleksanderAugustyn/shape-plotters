"""Three-quadratic-surface (Nix) render over the tqs_parameterization package.

The five sliders are Nix's coordinates (σ1, σ2, σ3, α2, α3). Everything shown
comes from the library: R(θ) and analytic dR/dθ from one read-only cache on
the shared GL-2048 set (shape_plotters/core/nodes.py), the ρ(z) profile with
its lib-exact dρ/dz, the analytic neck, the resolved geometry, Möller's
coordinates and the scission locus. Every call resolves from params, so the
cache may be shared across threads.

The checked outputs decide validity. A separated shape fails the checked
profile with `separated`; its fragments are drawn from the unchecked profile
(ρ = 0 in the void) while the checked status keeps the verdict and title.

Poles and flags come from `geometry`, never from `shape`: `shape` zero-fills
its poles on not_star_convex and separated, and the engine passes the poles
to `fragment_volumes` as the cap positions, so a pinched or separated shape
would otherwise get zero-radius caps and a biased mass split.

A NeckInfo is built whenever the library neck is found, ρ_neck = 0 included:
for a pinched or separated shape that is the split plane between the
fragments (unlike fos, where a separated shape fails inside neck()).

Fast-math precondition: params must be finite. Sliders only emit finite
values, so no screening happens here.
"""
from __future__ import annotations

import numpy as np
import tqs_parameterization as tp

from shape_plotters.core import nodes
from shape_plotters.core.engine import R0_FM
from shape_plotters.core.neck import find_neck_indices, neck_depth
from shape_plotters.core.result import EnergyRequest, NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# rho(z) display-panel resolution — a display choice, not a calculation grid.
N_PROFILE_POINTS = 721
PARAM_KEYS = ("sigma1", "sigma2", "sigma3", "alpha2", "alpha3")
# Möller quotes Q2 for 240Pu whatever the shape's own A: Q2 = 2 Z R0² q2 (his eq. 14).
Z_PU = 94
A_PU = 240
_CONVEX_MIDDLES = (tp.MidKind.spheroid, tp.MidKind.cylinder)


class TqsRender:
    name = "tqs"
    has_extra_panel = True
    # Markers are the sphere point; the initial shape is a valid, clearly
    # necked one (on σ2 = σ3 the sphere is isolated: any σ1 > 0 greys out).
    slider_specs = [
        SliderSpec("sigma1", "σ1", 0.0, 3.5, 2.4, 0.01, markers=(0.0,)),
        SliderSpec("sigma2", "σ2", -2.0, 1.0, -1.0, 0.01, markers=(1.0,)),
        SliderSpec("sigma3", "σ3", 0.5, 2.0, 1.0, 0.01, markers=(1.0,)),
        SliderSpec("alpha2", "α2", -0.8, 0.8, 0.0, 0.01, markers=(0.0,)),
        SliderSpec("alpha3", "α3", -0.8, 0.8, 0.0, 0.01, markers=(0.0,)),
    ]
    toggles: list[ToggleSpec] = []

    def __init__(self) -> None:
        self._cache = tp.Cache(nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        arr = [params[k] for k in PARAM_KEYS]
        cache = self._cache
        geo = cache.geometry(arr)
        shp = cache.shape(arr)
        rd = cache.radius_and_derivative(arr) if shp.ok else None
        prof = cache.rho_z_grid(arr, N_PROFILE_POINTS)
        drawn = (cache.rho_z_grid_unchecked(arr, N_PROFILE_POINTS)
                 if prof.status == tp.Status.separated else prof)

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
            mo = cache.moller_coordinates(arr)
            sc = cache.scission_sigma1(arr)
            scalars = {"z1": geo.z_join[0], "z2": geo.z_join[1], "u": geo.u,
                       "alpha1": geo.alpha1}
            if sc.defined:
                scalars["sigma1_star"] = sc.sigma1_star
            scalars |= {"sigma1_t": sc.sigma1_t, "d": mo.d, "q2": mo.q2,
                        "Q2_Pu240_b": 2 * Z_PU * (R0_FM * A_PU ** (1 / 3)) ** 2 * mo.q2 / 100,
                        "alpha_g": mo.alpha_g, "eps_f1": mo.eps_f1, "eps_f2": mo.eps_f2}
            notes = (self._middle_note(geo),)

        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii,
            z=drawn.z, rho=drawn.rho, drho_dz=drawn.drho_dz,
            neck=neck_info,
            scalars=scalars,
            length_keys=frozenset({"z1", "z2", "u", "d"}),
            dr_dtheta=dr_dtheta,
            r_north=geo.r_north if geo.ok else 0.0,
            r_south=geo.r_south if geo.ok else 0.0,
            z_cm=0.0,
            notes=notes)

    @staticmethod
    def _middle_note(geo: tp.Geometry) -> str:
        # The library sets waist_inside only for σ2 < 0 middles; a convex
        # middle has no waist, so it gets no waist clause.
        name = geo.mid_kind.name
        if geo.mid_kind in _CONVEX_MIDDLES:
            return f"middle: {name}"
        flags = ("waist inside" if geo.waist_inside else "waist outside",
                 *(("separated",) if geo.is_separated else ()),
                 *(("pinched",) if geo.is_pinched else ()))
        return f"middle: {name} ({', '.join(flags)})"

    def filename(self, a_mass: int, params: dict[str, float]) -> str:
        return (f"tqs_A{a_mass}"
                + "".join(f"_{params[k]:.2f}" for k in PARAM_KEYS)
                + ".png")

    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """None until WMMM gains a tqs parameterization."""
        return []
