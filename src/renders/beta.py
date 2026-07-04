"""Beta (Legendre) render over the beta_parameterization package.

GL-native: R(theta) and analytic dR/dtheta come from the library's node-set
API evaluated on the shared GL-2048 set (src/core/nodes.py) — in sync with
the energy model's dense grid. The neck is the Python display-only heuristic
(src/core/neck.py) — graduating it into the library is recorded future work.

Two shapes are drawn. The default (blue) shape uses beta10 (= the beta1 slider,
the l=1 dipole term) as set. The COM-corrected (orange) overlay ignores that
slider value and uses corrected_beta10 — the dipole the library computes from
beta2..beta8 to place the center of mass at the origin —
shown only when corrected_beta10 differs from the slider beta1 by more than
the overlay threshold (the two shapes then genuinely differ).
"""
from __future__ import annotations

import numpy as np
import beta_parameterization as bp

from src.core import nodes, quadrature
from src.core.neck import find_neck_indices, neck_depth
from src.core.result import NeckInfo, ShapeResult, SliderSpec

N_BETAS = 8
SPHERE_VOLUME = 4.0 * np.pi / 3.0  # unit sphere, R0 units
# The COM-corrected shape coincides with the slider shape when the corrected
# dipole equals the slider beta1 (beta10 is a shape parameter, not a
# translation knob); below this |corrected_beta10 - beta1| the orange overlay
# is suppressed (slider units).
OVERLAY_BETA10_THRESHOLD = 0.001


def _unit_volume_factor(radii: np.ndarray) -> float:
    """Return the WMMM GL volume factor that renormalizes ``radii`` to a unit sphere.

    Parameters
    ----------
    radii : np.ndarray
        R(theta) on the shared GL node set, before volume normalization.

    Returns
    -------
    float
        Scale factor s such that s * radii encloses the unit-sphere volume.
    """
    raw_volume = (2.0 * np.pi / 3.0) * float(np.sum(nodes.W * radii**3))
    return float((SPHERE_VOLUME / raw_volume) ** (1.0 / 3.0))


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
        self._cache = bp.Cache(max_beta_params=N_BETAS)
        self._node_set = self._cache.build_node_set(nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        betas = [params[f"beta{i}"] for i in range(1, N_BETAS + 1)]
        # Default (blue) shape: the slider betas as-is, including the slider beta10.
        resolved = self._cache.resolve_shape(betas, apply_com_correction=False)
        rd = None
        if resolved.ok:
            rd = self._cache.radius_and_derivative(resolved.beta_con, self._node_set)
        ok = rd is not None and rd.ok

        vol_factor = 1.0
        if ok:
            # WMMM's exact GL volume factor (the library's original_volume_factor):
            # radii arrive pre-scaled everywhere — derivative and poles included.
            vol_factor = _unit_volume_factor(rd.radii)
            radii = rd.radii * vol_factor
            dr_dtheta = rd.dr_dtheta * vol_factor
            r_north = resolved.r_north * vol_factor
            r_south = resolved.r_south * vol_factor
            # The slider shape's COM sits on the z axis at z_cm (nonzero for
            # asymmetric betas — the red marker shows the offset).
            z_cm = quadrature.z_cm(nodes.THETA, radii)
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)
            r_north = r_south = z_cm = 0.0

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
        # COM-corrected (orange) overlay: beta10 recomputed from beta2..beta8 to
        # center the COM. Built only when it differs from the slider shape
        # (|corrected_beta10 - beta1| > threshold); below that the two coincide.
        overlay_z = overlay_rho = None
        overlay_z_cm = 0.0
        corrected = self._cache.resolve_shape(betas, apply_com_correction=True)
        if corrected.ok:
            scalars["corrected_beta10"] = corrected.corrected_beta10
            if ok and abs(corrected.corrected_beta10 - betas[0]) > OVERLAY_BETA10_THRESHOLD:
                rd_c = self._cache.radius_and_derivative(corrected.beta_con, self._node_set)
                if rd_c.ok:
                    vf_c = _unit_volume_factor(rd_c.radii)
                    radii_c = rd_c.radii * vf_c
                    rn_c = corrected.r_north * vf_c
                    rs_c = corrected.r_south * vf_c
                    z_c = radii_c * nodes.X
                    rho_c = radii_c * nodes.SIN_THETA
                    # Close at the analytic poles (the engine's convention).
                    first, last = (-rs_c, rn_c) if z_c[0] < z_c[-1] else (rn_c, -rs_c)
                    overlay_z = np.concatenate(([first], z_c, [last]))
                    overlay_rho = np.concatenate(([0.0], rho_c, [0.0]))
                    overlay_z_cm = quadrature.z_cm(nodes.THETA, radii_c)

        primary = rd if resolved.ok else resolved
        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii, z=z, rho=rho, drho_dz=None,
            neck=neck, scalars=scalars, length_keys=frozenset(),
            dr_dtheta=dr_dtheta, r_north=r_north, r_south=r_south, z_cm=z_cm,
            overlay_z=overlay_z, overlay_rho=overlay_rho, overlay_z_cm=overlay_z_cm)

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        betas = "_".join(f"{params[f'beta{i}']:.2f}" for i in range(1, N_BETAS + 1))
        return f"{z}_{n}_{betas}.png"
