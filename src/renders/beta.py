"""Beta (Legendre) render over the beta_parameterization package.

GL-native: R(theta) and analytic dR/dtheta come from the library's cached
tier evaluated on the shared GL-2048 primary theta set (src/core/nodes.py) —
in sync with the energy model's dense grid. One read-only cache serves every
call; each call resolves from its betas, so the cache may be shared across
threads. Volume conservation and the COM correction are per-call library
options: radii, dR/dtheta and the polar radii arrive pre-scaled, and
resolve_shape reports the applied volume_factor. The neck is
the Python display-only heuristic (src/core/neck.py) — graduating it into the
library is recorded future work.

Two shapes are drawn. The default (blue) shape uses beta10 (= the beta1 slider,
the l=1 dipole term) as set. The COM-corrected (orange) overlay ignores that
slider value and uses corrected_beta10 — the dipole the library computes from
beta2..beta8 to place the center of mass at the origin —
shown only when corrected_beta10 differs from the slider beta1 by more than
the overlay threshold (the two shapes then genuinely differ).

Fast-math precondition: params must be finite. Sliders only emit finite
values, so no screening happens here.
"""
from __future__ import annotations

import numpy as np
import beta_parameterization as bp

from src.core import nodes, quadrature
from src.core.neck import find_neck_indices, neck_depth
from src.core.result import EnergyRequest, NeckInfo, ShapeResult, SliderSpec

N_BETAS = 8
# WMMM's legendre parameterization takes 8 betas, all slider-driven.
WMMM_N_LEGENDRE_PARAMS = 8
# The COM-corrected shape coincides with the slider shape when the corrected
# dipole equals the slider beta1 (beta10 is a shape parameter, not a
# translation knob); below this |corrected_beta10 - beta1| the orange overlay
# is suppressed (slider units).
OVERLAY_BETA10_THRESHOLD = 0.001


class BetaRender:
    name = "beta"
    has_extra_panel = False
    # Legend labels for the engine's orange overlay artists (the COM-corrected
    # shape reuses the same drawing channel as the FoS R(θ) overlay).
    overlay_label = "COM corrected"
    overlay_zcm_label = "z_cm (corrected)"
    # Ranges from the old ShapePlotter: beta1 (-1.6, 1.6), beta2 (0, 4), rest (-2, 2).
    slider_specs = [
        SliderSpec("beta1", "β1", -1.6, 1.6, 0.0, 0.01),
        SliderSpec("beta2", "β2", 0.0, 4.0, 0.0, 0.01),
    ] + [SliderSpec(f"beta{i}", f"β{i}", -2.0, 2.0, 0.0, 0.01) for i in range(3, N_BETAS + 1)]
    toggles = []

    def __init__(self) -> None:
        # The blue (raw) and orange (COM-corrected) shapes share the cache;
        # they differ only in the per-call apply_com option.
        self._cache = bp.Cache(N_BETAS, nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        betas = [params[f"beta{i}"] for i in range(1, N_BETAS + 1)]
        # Default (blue) shape: the slider betas as-is, including the slider beta10.
        resolved = self._cache.resolve_shape(betas, conserve_volume=True)
        rd = (self._cache.radius_and_derivative(betas, conserve_volume=True)
              if resolved.ok else None)
        ok = rd is not None and rd.ok

        vol_factor = 1.0
        if ok:
            # Radii, derivative and poles arrive pre-scaled by the library's
            # volume factor (conserve_volume=True); no Python rescaling.
            radii = rd.radii
            dr_dtheta = rd.dr_dtheta
            r_north = resolved.r_north
            r_south = resolved.r_south
            vol_factor = resolved.volume_factor
            # The slider shape's COM sits on the z axis at z_cm (nonzero for
            # asymmetric betas — the red marker shows the offset).
            z_cm = quadrature.z_cm(nodes.THETA, radii)
            z = radii * nodes.X
            rho = radii * nodes.SIN_THETA
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)
            r_north = r_south = z_cm = 0.0
            # Unchecked path: keeps R(theta) even where it goes negative, so
            # the broken (self-crossing) outline still draws — the engine
            # greys it — instead of the shape collapsing to a point. Unscaled
            # by design: invalid shapes get no volume conservation.
            grid = self._cache.radius_grid_unchecked(betas)
            z = grid.radii * nodes.X
            rho = grid.radii * nodes.SIN_THETA

        neck = None
        if ok:
            hit = find_neck_indices(rho)
            if hit is not None:
                i_neck, i_a, i_b = hit
                neck = NeckInfo(z=float(z[i_neck]), rho=float(rho[i_neck]),
                                depth=neck_depth(rho, i_neck, i_a, i_b),
                                source="py heuristic")

        scalars: dict[str, float] = {"vol_factor": vol_factor}
        # COM-corrected (orange) overlay: beta10 recomputed from beta2..beta8 to
        # center the COM. Built only when it differs from the slider shape
        # (|corrected_beta10 - beta1| > threshold); below that the two coincide.
        overlay_z = overlay_rho = None
        overlay_z_cm = 0.0
        overlay_ok = True
        corrected = self._cache.resolve_shape(betas, conserve_volume=True, apply_com=True)
        if corrected.ok:
            scalars["corrected_beta10"] = corrected.corrected_beta10
            if ok and abs(corrected.corrected_beta10 - betas[0]) > OVERLAY_BETA10_THRESHOLD:
                rd_c = self._cache.radius_and_derivative(
                    betas, conserve_volume=True, apply_com=True)
                if rd_c.ok:
                    radii_c = rd_c.radii
                    z_c = radii_c * nodes.X
                    rho_c = radii_c * nodes.SIN_THETA
                    # Close at the analytic poles (the engine's convention);
                    # corrected.r_north/r_south are already volume-scaled.
                    rn_c, rs_c = corrected.r_north, corrected.r_south
                    first, last = (-rs_c, rn_c) if z_c[0] < z_c[-1] else (rn_c, -rs_c)
                    overlay_z = np.concatenate(([first], z_c, [last]))
                    overlay_rho = np.concatenate(([0.0], rho_c, [0.0]))
                    overlay_z_cm = quadrature.z_cm(nodes.THETA, radii_c)
        elif ok and corrected.status == bp.Status.interior_negative:
            # The COM-centering beta10 exists (poles fine, COM converged) but
            # the corrected shape is interior-negative — the library catches
            # this at resolve_shape, because the volume factor is computed
            # after validation. Draw the near-miss greyed via the unchecked
            # path so it stays visible instead of the overlay silently
            # vanishing. apply_com is explicit: without it this is the
            # uncorrected shape.
            grid_c = self._cache.radius_grid_unchecked(betas, apply_com=True)
            overlay_z = grid_c.radii * nodes.X
            overlay_rho = grid_c.radii * nodes.SIN_THETA
            overlay_ok = False

        primary = rd if resolved.ok else resolved
        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii, z=z, rho=rho, drho_dz=None,
            neck=neck, scalars=scalars, length_keys=frozenset(),
            dr_dtheta=dr_dtheta, r_north=r_north, r_south=r_south, z_cm=z_cm,
            overlay_z=overlay_z, overlay_rho=overlay_rho, overlay_z_cm=overlay_z_cm,
            overlay_ok=overlay_ok)

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        betas = "_".join(f"{params[f'beta{i}']:.2f}" for i in range(1, N_BETAS + 1))
        return f"{z}_{n}_{betas}.png"

    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """WMMM requests: the slider (blue) shape, plus the COM-corrected
        (orange) shape when the overlay is on screen. WMMM recomputes beta10
        itself under com_correction=True, so both carry the same betas."""
        shape = tuple(params[f"beta{i}"] for i in range(1, N_BETAS + 1)) \
            + (0.0,) * (WMMM_N_LEGENDRE_PARAMS - N_BETAS)
        requests = [EnergyRequest("slider", "legendre", shape, com_correction=False)]
        if result.overlay_z is not None:
            requests.append(
                EnergyRequest("COM corrected", "legendre", shape, com_correction=True))
        return requests
