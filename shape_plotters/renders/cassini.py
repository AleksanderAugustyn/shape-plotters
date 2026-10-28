"""Cassini-oval (Pashkevich) render over the cassini_parameterization package.

The five sliders are α and the Legendre coefficients α1 … α4. Everything
shown comes from the library: R(θ) and analytic dR/dθ from one read-only
cache on the shared GL-2048 set (shape_plotters/core/nodes.py), the ρ(z)
profile with its lib-exact dρ/dz, the neck, the resolved geometry and q2.
Every call resolves from params, so the cache may be shared across threads.

The profile is the parametric curve, not a function of z: a dimpled (folded)
shape overhangs at a pole and is drawn as it is, and core/fragments.py
integrates it in traversal order. The R(θ) outline is supplied as the overlay
whenever the R(θ) path is valid, so the R(θ) representation of a dimpled
shape can be checked against the profile by eye; R(θ) is about the true
centre of mass, so the engine never infers one.

The checked outputs decide validity. A separated shape fails the checked
profile with `separated`; its fragments are drawn from the unchecked profile,
whose two tip nodes keep the void at ρ = 0, while the checked status keeps
the verdict and title.

Poles and flags come from `geometry`, never from `shape`: `shape` zero-fills
its poles on not_star_convex and separated, and the engine passes the poles
to `fragment_volumes` as the cap positions, so a pinched or separated shape
would otherwise get zero-radius caps and a biased mass split.

A NeckInfo is built whenever the library neck is found, ρ_neck = 0 included:
for a pinched or separated shape that is the split plane between the
fragments (the gap midpoint).

Fast-math precondition: params must be finite. Sliders only emit finite
values, so no screening happens here.
"""
from __future__ import annotations

import numpy as np
import cassini_parameterization as cp

from shape_plotters.core import nodes
from shape_plotters.core.neck import find_neck_indices, neck_depth
from shape_plotters.core.result import EnergyRequest, NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# rho(z) display-panel resolution — a display choice, not a calculation grid.
N_PROFILE_POINTS = 721
N_X_GRID = 2049     # cache scan grid: star gate, d_min, neck search, R(θ) brackets
PARAM_KEYS = ("alpha", "alpha1", "alpha2", "alpha3", "alpha4")
MAX_PARAMS = len(PARAM_KEYS)


class CassiniRender:
    name = "cassini"
    has_extra_panel = True
    # Step 0.025 reaches the paper's lattice exactly. Markers: the grid edge
    # α = 0.95 and scission α = 1; ±0.25 is the paper's box (α2 is not in it).
    slider_specs = [
        SliderSpec("alpha", "α", -0.25, 1.1, 0.0, 0.025, markers=(0.95, 1.0)),
        SliderSpec("alpha1", "α1", -0.5, 0.5, 0.0, 0.025, markers=(-0.25, 0.25)),
        SliderSpec("alpha2", "α2", -0.5, 0.5, 0.0, 0.025),
        SliderSpec("alpha3", "α3", -0.5, 0.5, 0.0, 0.025, markers=(-0.25, 0.25)),
        SliderSpec("alpha4", "α4", -0.5, 0.5, 0.0, 0.025, markers=(-0.25, 0.25)),
    ]
    toggles: list[ToggleSpec] = []

    def __init__(self) -> None:
        self._cache = cp.Cache(MAX_PARAMS, N_X_GRID, nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        arr = [params[k] for k in PARAM_KEYS]
        cache = self._cache
        geo = cache.geometry(arr)
        shp = cache.shape(arr)
        rd = cache.radius_and_derivative(arr) if shp.ok else None
        prof = cache.rho_profile(arr, N_PROFILE_POINTS)
        drawn = (cache.rho_profile_unchecked(arr, N_PROFILE_POINTS)
                 if prof.status == cp.Status.separated else prof)

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

        r_north = geo.r_north if geo.ok else 0.0
        r_south = geo.r_south if geo.ok else 0.0
        overlay_z = overlay_rho = None
        if ok:
            # Pre-closed at the exact poles, as beta's outline is.
            overlay_z = np.concatenate(([r_north], radii * nodes.X, [-r_south]))
            overlay_rho = np.concatenate(([0.0], radii * nodes.SIN_THETA, [0.0]))

        scalars: dict[str, float] = {}
        notes: tuple[str, ...] = ()
        if geo.ok:
            scalars = {"eps": geo.eps, "c": geo.c, "z_shift": geo.z_shift}
            if geo.is_separated:
                scalars["gap"] = geo.gap
            scalars |= {"q2": cache.quadrupole(arr).q2, "d_min": geo.d_min}
            notes = (self._profile_note(geo),)

        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii,
            z=drawn.z, rho=drawn.rho, drho_dz=drawn.drho_dz,
            neck=neck_info,
            scalars=scalars,
            length_keys=frozenset({"z_shift", "gap"}),
            dr_dtheta=dr_dtheta,
            r_north=r_north,
            r_south=r_south,
            z_cm=0.0,
            overlay_z=overlay_z,
            overlay_rho=overlay_rho,
            overlay_z_cm=0.0,
            overlay_ok=True,
            notes=notes)

    @staticmethod
    def _profile_note(geo: cp.Geometry) -> str:
        note = "profile: folded (dimpled)" if geo.is_folded else "profile: single-valued"
        if geo.is_pinched:
            note += ", pinched"
        if geo.is_separated:
            note += ", separated"
        return note

    def filename(self, a_mass: int, params: dict[str, float]) -> str:
        # Three decimals: two cannot represent the 0.025 step.
        return (f"cassini_A{a_mass}_al{params['alpha']:.3f}"
                + "".join(f"_a{i}{params[f'alpha{i}']:.3f}" for i in range(1, 5))
                + ".png")

    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """None until WMMM gains a cassini parameterization."""
        return []
