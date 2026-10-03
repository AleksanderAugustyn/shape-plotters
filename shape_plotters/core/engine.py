"""ShapePlotterApp: parameterization-independent interactive figure.

Owns display units (fm/R0 toggle; ShapeResult is R0 units), layout, widgets,
invalid-shape greying, stats, and save. Everything shape-specific comes from
the render (contract in shape_plotters/core/result.py).

Artists are created once at build; update() only mutates data, text, and
visibility. v0.1 cleared and replotted every axes per slider event — legend
and text layout made that the frame-time bottleneck (see
docs/superpowers/specs/2026-07-03-engine-v0.2-performance-design.md).

A full-figure redraw is still ~100 ms, almost all of it ticks, legends and
widget axes that do not change. While a slider is dragged, update() therefore
blits: the static layer comes from a cached background (blit.py) and only the
dynamic artists and the dragged slider are drawn. Everything outside a drag
takes the ordinary full-draw path, so a settled figure is pixel-identical to
one that was never dragged (see
docs/superpowers/specs/2026-10-03-engine-v0.3-blitting-design.md).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

from shape_plotters.core import energy, fragments, quadrature
from shape_plotters.core.blit import DragBlitter
from shape_plotters.core.result import ShapeResult
from shape_plotters.core.widgets import IntTextBox, SliderRow

# Display convention (both old plotters used 1.16); distinct from WMMM physics constants.
R0_FM = 1.16

VALID_COLOR = "tab:blue"
DERIV_COLOR = "tab:red"
NECK_COLOR = "tab:green"
RTHETA_COLOR = "tab:orange"   # R(θ) star-convex representation (FoS shifted shapes)
INVALID_COLOR = "0.55"
UNITS_LABEL = "fm units"

# WMMM scission bands: neck radius 1.2-1.5 fm (comparable because display
# R0 = 1.16 fm equals WMMM's geometric R0; see FORD.md / v0.2 handoff).
SCISSION_BAND_FM = (1.2, 1.5)


class ShapePlotterApp:
    def __init__(self, render) -> None:
        self.render = render
        self.fm_units = True
        self.toggle_state = {t.key: t.default for t in render.toggles}
        self.last_result: ShapeResult | None = None
        self._last_ok: bool | None = None
        self._last_overlay_ok: bool | None = None
        self._stats_base = ""
        self._energy_lines: list[str] = []
        self._build_figure()
        self._build_artists()
        self._build_widgets()
        # Blitted slider drags; None where the canvas cannot blit.
        self._blitter = (DragBlitter(self.fig, self._dynamic)
                         if self.fig.canvas.supports_blit else None)
        self._drag_pending = None   # slider whose press was drawn the ordinary way
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.update()

    # ---------- construction ----------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 9))
        try:
            self.fig.canvas.manager.set_window_title(f"shape-plotters — {self.render.name}")
        except AttributeError:
            pass  # headless backends may lack a window manager
        ncols = 3 if self.render.has_extra_panel else 2
        gs = self.fig.add_gridspec(
            1, ncols + 1, left=0.05, right=0.98, top=0.96, bottom=0.52,
            wspace=0.30, width_ratios=[1.0] * ncols + [0.55])
        self.ax_radius = self.fig.add_subplot(gs[0])
        self.ax_shape = self.fig.add_subplot(gs[1])
        self.ax_extra = self.fig.add_subplot(gs[2]) if self.render.has_extra_panel else None
        self.ax_stats = self.fig.add_subplot(gs[ncols])
        self.ax_stats.axis("off")
        self._data_axes = tuple(
            ax for ax in (self.ax_radius, self.ax_shape, self.ax_extra) if ax is not None)

    def _build_artists(self) -> None:
        # fm_units defaults to True; the unit toggle re-texts labels in place.
        unit = "fm"
        ax = self.ax_radius
        (self.r_line,) = ax.plot([], [], color=VALID_COLOR, lw=2, label=f"R(θ) [{unit}]")
        (self.dr_line,) = ax.plot([], [], color=DERIV_COLOR, lw=2, ls=":",
                                  label="dR/dθ (lib)")
        ax.set_xlim(0.0, np.pi)  # also disables x-autoscale on this axes
        ax.set_xlabel("θ [rad]")
        self.radius_legend = ax.legend(loc="upper center", fontsize=12)
        ax.grid(alpha=0.3)

        ax = self.ax_shape
        (self.shape_upper,) = ax.plot([], [], color=VALID_COLOR, lw=2, label="shape")
        (self.shape_lower,) = ax.plot([], [], color=VALID_COLOR, lw=2)
        (self.neck_line,) = ax.plot([], [], color=NECK_COLOR, ls="--", lw=1.5)
        # z_cm marker: red point on the z axis (COM of an axially symmetric shape).
        (self.zcm_point,) = ax.plot([], [], marker="o", ms=6, ls="",
                                    color="tab:red", zorder=5, label="z_cm")
        # Orange dashed overlay. Two sources share these artists (a render only
        # ever drives one): the FoS R(θ) star-convex representation drawn where
        # it actually sits, or a render-supplied outline (beta's COM-corrected
        # shape). Labels come from the render so each names its own overlay.
        ov_label = getattr(self.render, "overlay_label", "R(θ) star-convex")
        ov_zcm_label = getattr(self.render, "overlay_zcm_label", "z_cm (R(θ))")
        (self.rtheta_upper,) = ax.plot([], [], color=RTHETA_COLOR, lw=1.2, ls="--",
                                       alpha=0.9, visible=False, label=ov_label)
        (self.rtheta_lower,) = ax.plot([], [], color=RTHETA_COLOR, lw=1.2, ls="--",
                                       alpha=0.9, visible=False)
        (self.rtheta_zcm,) = ax.plot([], [], marker="o", ms=6, ls="",
                                     color=RTHETA_COLOR, zorder=5, visible=False,
                                     label=ov_zcm_label)
        self.scission_bands = [
            ax.axhspan(SCISSION_BAND_FM[0], SCISSION_BAND_FM[1],
                       color=NECK_COLOR, alpha=0.15),
            ax.axhspan(-SCISSION_BAND_FM[1], -SCISSION_BAND_FM[0],
                       color=NECK_COLOR, alpha=0.15),
        ]
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"z [{unit}]")
        ax.set_ylabel(f"ρ [{unit}]")
        ax.grid(alpha=0.3)
        # Distinguishes the true shape from the R(θ) overlay; built once and only
        # shown while the overlay is active (update() toggles visibility).
        self.shape_legend = ax.legend(loc="upper right", fontsize=12)
        self.shape_legend.set_visible(False)

        self.extra_legend = None
        self.zcm_extra = None
        if self.ax_extra is not None:
            ax = self.ax_extra
            (self.extra_rho,) = ax.plot([], [], color=VALID_COLOR, lw=2,
                                        label=f"ρ(z) [{unit}]")
            (self.extra_drho,) = ax.plot([], [], color=DERIV_COLOR, lw=1.5, ls=":",
                                         label="dρ/dz (lib)")
            (self.zcm_extra,) = ax.plot([], [], marker="o", ms=6, ls="",
                                        color="tab:red", zorder=5)
            ax.set_xlabel(f"z [{unit}]")
            self.extra_legend = ax.legend(loc="upper center", fontsize=12)
            ax.grid(alpha=0.3)

        self.stats_text = self.ax_stats.text(
            0.0, 1.0, "", va="top", family="monospace", fontsize=12,
            transform=self.ax_stats.transAxes)

        # Everything update() mutates per frame. The rest of the figure is the
        # static layer that a blitted drag frame reuses.
        dynamic = [self.r_line, self.dr_line, self.shape_upper, self.shape_lower,
                   self.neck_line, self.zcm_point, self.rtheta_upper,
                   self.rtheta_lower, self.rtheta_zcm, self.stats_text]
        if self.ax_extra is not None:
            dynamic += [self.extra_rho, self.extra_drho, self.zcm_extra]
        self._dynamic = tuple(dynamic)

    def _build_widgets(self) -> None:
        self.rows: dict[str, SliderRow] = {}
        y = 0.06
        for spec in self.render.slider_specs:
            row = SliderRow(self.fig, y, spec.label, spec.vmin, spec.vmax,
                            spec.vinit, spec.step, spec.markers)
            row.slider.on_changed(self.update)
            self.rows[spec.key] = row
            y += 0.031
        self.z_box = IntTextBox(self.fig, (0.88, 0.40, 0.06, 0.030), "Z ", 92, self._on_zn)
        self.n_box = IntTextBox(self.fig, (0.88, 0.36, 0.06, 0.030), "N ", 144, self._on_zn)
        self.btn_reset = Button(self.fig.add_axes((0.86, 0.31, 0.10, 0.030)), "Reset")
        self.btn_reset.on_clicked(self._reset)
        self.btn_save = Button(self.fig.add_axes((0.86, 0.27, 0.10, 0.030)), "Save")
        self.btn_save.on_clicked(self._save)
        labels = [UNITS_LABEL] + [t.label for t in self.render.toggles]
        actives = [self.fm_units] + [t.default for t in self.render.toggles]
        n = len(labels)
        self.checks = CheckButtons(
            self.fig.add_axes((0.86, 0.26 - 0.04 * n, 0.11, 0.04 * n)), labels, actives)
        self.checks.on_clicked(self._on_check)
        self.btn_energy = None
        if energy.available():
            # Right column, below the toggle block; absent (layout untouched)
            # when the local WMMM install is missing.
            self.btn_energy = Button(
                self.fig.add_axes((0.86, 0.26 - 0.04 * n - 0.045, 0.10, 0.030)),
                "Energy")
            self.btn_energy.on_clicked(self._on_energy)
        self._suppress_widget_draws()

    def _suppress_widget_draws(self) -> None:
        # update() issues the single authoritative draw_idle(); without this,
        # Slider.set_val adds a second full-figure draw per event. IntTextBox
        # is excluded: TextBox needs its own draws for typing echo.
        widgets: list = [self.btn_reset, self.btn_save, self.checks]
        if self.btn_energy is not None:
            widgets.append(self.btn_energy)
        for row in self.rows.values():
            widgets += [row.slider, row.btn_dec, row.btn_inc]
        for w in widgets:
            if hasattr(w, "drawon"):  # guard against future mpl API changes
                w.drawon = False

    # ---------- callbacks ----------

    def _on_check(self, label: str) -> None:
        if label == UNITS_LABEL:
            self.fm_units = not self.fm_units
            self._relabel_units()
        else:
            for t in self.render.toggles:
                if t.label == label:
                    self.toggle_state[t.key] = not self.toggle_state[t.key]
        self.update()

    def _relabel_units(self) -> None:
        # Rare event: re-text legends/labels in place instead of rebuilding them.
        unit = "fm" if self.fm_units else "R0"
        self.radius_legend.get_texts()[0].set_text(f"R(θ) [{unit}]")
        self.ax_shape.set_xlabel(f"z [{unit}]")
        self.ax_shape.set_ylabel(f"ρ [{unit}]")
        if self.ax_extra is not None:
            self.ax_extra.set_xlabel(f"z [{unit}]")
            self.extra_legend.get_texts()[0].set_text(f"ρ(z) [{unit}]")
        self._reposition_scission()

    def _on_zn(self) -> None:
        # Z/N changes the fm<->R0 scale, so fm-fixed band edges move in R0 mode.
        self._reposition_scission()
        self.update()

    def _fm_to_display(self, v_fm: float) -> float:
        if self.fm_units:
            return v_fm
        return v_fm / (R0_FM * float(self.z_box.value + self.n_box.value) ** (1.0 / 3.0))

    def _reposition_scission(self) -> None:
        lo, hi = (self._fm_to_display(v) for v in SCISSION_BAND_FM)
        upper, lower = self.scission_bands
        self._set_band(upper, lo, hi)
        self._set_band(lower, -hi, -lo)

    @staticmethod
    def _set_band(band, ylo: float, yhi: float) -> None:
        if hasattr(band, "set_height"):  # Rectangle (mpl >= 3.8, incl. 3.11)
            band.set_y(ylo)
            band.set_height(yhi - ylo)
        else:                            # Polygon fallback for older mpl
            band.set_xy([[0.0, ylo], [0.0, yhi], [1.0, yhi], [1.0, ylo]])

    def _reset(self, _event=None) -> None:
        # Each set_val fires update(); fine at 8 sliders.
        for spec in self.render.slider_specs:
            self.rows[spec.key].slider.set_val(spec.vinit)

    def _save(self, _event=None) -> None:
        # A drag whose release never arrived must not leak its held view into
        # the file: leave drag mode first.
        self._on_release(None)
        params = {k: row.slider.val for k, row in self.rows.items()}
        fname = self.render.filename(self.z_box.value, self.n_box.value, params)
        self.fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved {fname}")

    def _refresh_stats(self) -> None:
        self.stats_text.set_text("\n".join([self._stats_base, *self._energy_lines]))

    def _on_energy(self, _event=None) -> None:
        result = self.last_result
        if result is None or not result.ok:
            self._energy_lines = ["", "WMMM: shape invalid (not computed)"]
        else:
            params = {k: row.slider.val for k, row in self.rows.items()}
            requests = self.render.energy_requests(params, result)
            lines: list[str] = []
            for req in requests:
                res = energy.compute(
                    req.param_type, self.z_box.value, self.n_box.value,
                    req.shape, com_correction=req.com_correction)
                lines += self._energy_block(
                    req.label if len(requests) > 1 else None, res)
            self._energy_lines = lines
        self._refresh_stats()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _energy_block(label: str | None, res: energy.EnergyResult) -> list[str]:
        # Energies are MeV — deliberately outside the fm/R0 unit toggle.
        head = "WMMM" if label is None else f"WMMM ({label})"
        if res.error is not None:
            return ["", f"{head}: error", f"  {res.error}"]
        if not res.is_valid:
            return ["", f"{head}: invalid shape"]
        return ["", f"{head} [MeV]:",
                f"  E_total = {res.total_energy:.4f}",
                f"  E_macro = {res.macro_energy:.4f}",
                f"  E_micro = {res.micro_energy:.4f}",
                f"  mass_ex = {res.mass_excess:.4f}",
                f"  E_surf  = {res.surface_energy:.4f}",
                f"  E_coul  = {res.coulomb_energy:.4f}",
                f"  gap_p = {res.proton_pairing_gap:.4f}  k_p = {res.proton_k}",
                f"  gap_n = {res.neutron_pairing_gap:.4f}  k_n = {res.neutron_k}"]

    # ---------- drawing ----------

    def _scale(self) -> float:
        if not self.fm_units:
            return 1.0
        return R0_FM * float(self.z_box.value + self.n_box.value) ** (1.0 / 3.0)

    def update(self, _val=None) -> None:
        dragging = self._sync_drag_mode()
        params = {k: row.slider.val for k, row in self.rows.items()}
        result = self.render.compute(params, self.toggle_state)
        self.last_result = result
        scale = self._scale()
        unit = "fm" if self.fm_units else "R0"

        r = result.radius * scale
        self.r_line.set_data(result.theta, r)
        self.dr_line.set_data(result.theta, result.dr_dtheta * scale)

        z, rho = result.z * scale, result.rho * scale
        rn, rs = result.r_north * scale, result.r_south * scale
        # Beta's R(θ) parametric outline is open at the GL poles → append the
        # analytic poles; the FoS ρ(z) profile already ends at ρ=0 → draw as-is.
        if rho.size > 1 and max(abs(float(rho[0])), abs(float(rho[-1]))) > 1e-9:
            first, last = (-rs, rn) if z[0] < z[-1] else (rn, -rs)
            z_c = np.concatenate(([first], z, [last]))
            rho_c = np.concatenate(([0.0], rho, [0.0]))
        else:
            z_c, rho_c = z, rho
        self.shape_upper.set_data(z_c, rho_c)
        self.shape_lower.set_data(z_c, -rho_c)
        if result.neck is not None:
            zn, rn_neck = result.neck.z * scale, result.neck.rho * scale
            self.neck_line.set_data([zn, zn], [-rn_neck, rn_neck])
            self.neck_line.set_visible(True)
        else:
            self.neck_line.set_visible(False)

        if self.ax_extra is not None:
            self.extra_rho.set_data(z, rho)
            if result.drho_dz is not None:
                # drho/dz is a unit-free slope: both lengths scale identically.
                self.extra_drho.set_data(z, result.drho_dz)
                self.extra_drho.set_visible(True)
            else:
                self.extra_drho.set_visible(False)

        self._apply_validity(result.ok)
        title = "" if result.ok else f"{result.status_name}: {result.message}"
        if self.ax_shape.get_title() != title:
            self.ax_shape.set_title(title, color="tab:red", fontsize=12)

        v = quadrature.volume(result.theta, result.radius) * scale**3
        s = quadrature.surface_area(result.theta, result.radius, result.dr_dtheta) * scale**2
        zc = result.z_cm * scale                      # true-shape COM (display frame)
        self.zcm_point.set_data([zc], [0.0])
        self.zcm_point.set_visible(result.ok)
        if self.zcm_extra is not None:
            self.zcm_extra.set_data([zc], [0.0])
            self.zcm_extra.set_visible(result.ok)

        # Orange overlay on the cross-section. A render may supply the outline
        # directly (beta's COM-corrected shape, pre-closed at the poles); else
        # the engine infers the FoS R(θ) star-convex representation and draws it
        # where it actually sits, shown only when its COM differs from the true
        # shape's (FoS with a3/a5/a7 ≠ 0). Beta's R(θ) IS the true shape, so that
        # inference never fires for beta.
        if result.overlay_z is not None:
            show_rtheta = bool(result.ok)
            if show_rtheta:
                self.rtheta_upper.set_data(result.overlay_z * scale, result.overlay_rho * scale)
                self.rtheta_lower.set_data(result.overlay_z * scale, -result.overlay_rho * scale)
                self.rtheta_zcm.set_data([result.overlay_z_cm * scale], [0.0])
        else:
            zc_r = quadrature.z_cm(result.theta, result.radius) * scale
            show_rtheta = bool(result.ok and abs(zc_r - zc) > 1e-6 * (1.0 + abs(zc)))
            if show_rtheta:
                zr = result.radius * np.cos(result.theta) * scale
                rr = result.radius * np.sin(result.theta) * scale
                first, last = (-rs, rn) if zr[0] < zr[-1] else (rn, -rs)
                zr = np.concatenate(([first], zr, [last]))
                rr = np.concatenate(([0.0], rr, [0.0]))
                self.rtheta_upper.set_data(zr, rr)
                self.rtheta_lower.set_data(zr, -rr)
                self.rtheta_zcm.set_data([zc_r], [0.0])
        self._apply_overlay_validity(result.overlay_ok)
        for art in (self.rtheta_upper, self.rtheta_lower):
            art.set_visible(show_rtheta)
        # No COM marker for an invalid corrected shape — it isn't centered anywhere.
        self.rtheta_zcm.set_visible(show_rtheta and result.overlay_ok)
        self.shape_legend.set_visible(show_rtheta)

        # Fragment split at the neck (R0 units; the fm scale cancels in a ratio,
        # so only the mass split f*(Z+N) needs the display Z/N).
        frag = None
        if result.neck is not None:
            v_lo, v_hi = fragments.fragment_volumes(
                result.z, result.rho, result.neck.z,
                -result.r_south, result.r_north)
            total = v_lo + v_hi
            if total > 0.0 and v_lo > 0.0 and v_hi > 0.0:
                a_mass = self.z_box.value + self.n_box.value
                fracs = sorted((v_lo / total, v_hi / total), reverse=True)
                frag = (fracs[0], fracs[1],
                        fracs[0] * a_mass, fracs[1] * a_mass, a_mass)

        self._stats_base = self._stats_block(result, scale, unit, v, s, zc, frag)
        self._energy_lines = []   # any shape/Z/N/unit change invalidates energies
        self._refresh_stats()

        self._present(dragging)

    # ---------- presenting a frame ----------

    def _sync_drag_mode(self) -> bool:
        """Match drag mode to the sliders; True while a drag is being blitted.

        A click is not a drag: the first update of a press is drawn the
        ordinary way and drag mode starts with the next one. A plain click
        then costs one full draw, not a drag-mode draw plus the release refit.
        """
        if self._blitter is None:
            return False
        slider = next((row.slider for row in self.rows.values()
                       if row.slider.drag_active), None)
        if slider is None:
            self._drag_pending = None
            if self._blitter.active:      # the release event never arrived
                self._end_drag()
            return False
        if self._blitter.active:
            return True
        if self._drag_pending is not slider:
            self._drag_pending = slider
            return False
        self._blitter.begin(slider.ax)
        return True

    def _present(self, dragging: bool) -> None:
        """Put the mutated artists on screen: a full draw, or a blitted drag frame."""
        if not dragging:
            # visible_only: the hidden neck line keeps stale data by design.
            for ax in self._data_axes:
                ax.relim(visible_only=True)
                ax.autoscale_view()
            self.fig.canvas.draw_idle()
            return
        # Dragging: the view holds still, because rescaling changes the tick
        # labels and with them the cached background.
        if self._blitter.ready:
            self._blitter.blit()
        else:
            self.fig.canvas.draw_idle()

    def _end_drag(self) -> None:
        """Leave drag mode and restore the tight view the non-drag path draws."""
        self._blitter.end()
        for ax in self._data_axes:
            ax.relim(visible_only=True)
            ax.autoscale_view()

    def _on_release(self, _event) -> None:
        self._drag_pending = None
        if self._blitter is not None and self._blitter.active:
            self._end_drag()
            self.fig.canvas.draw_idle()

    def _apply_validity(self, ok: bool) -> None:
        # Color/alpha churn only on the valid<->invalid flip, not per frame.
        if ok == self._last_ok:
            return
        self._last_ok = ok
        color = VALID_COLOR if ok else INVALID_COLOR
        deriv = DERIV_COLOR if ok else INVALID_COLOR
        alpha = 1.0 if ok else 0.45
        for line in (self.r_line, self.shape_upper, self.shape_lower):
            line.set_color(color)
            line.set_alpha(alpha)
        self.dr_line.set_color(deriv)
        self.dr_line.set_alpha(alpha)
        if self.ax_extra is not None:
            self.extra_rho.set_color(color)
            self.extra_rho.set_alpha(alpha)
            self.extra_drho.set_color(deriv)
            self.extra_drho.set_alpha(alpha)

    def _apply_overlay_validity(self, ok: bool) -> None:
        # The COM-corrected overlay greys when its own shape is invalid (a
        # centering beta10 exists but self-intersects). Flip-cached like the
        # main-shape greying to avoid per-frame color churn.
        if ok == self._last_overlay_ok:
            return
        self._last_overlay_ok = ok
        color = RTHETA_COLOR if ok else INVALID_COLOR
        alpha = 1.0 if ok else 0.45
        for line in (self.rtheta_upper, self.rtheta_lower):
            line.set_color(color)
            line.set_alpha(alpha)

    def _stats_block(self, result: ShapeResult, scale: float, unit: str,
                     v: float, s: float, zc: float,
                     frag: tuple[float, float, float, float, int] | None = None) -> str:
        lines = [f"[{self.render.name}]  units: {unit}"]
        if not result.ok:
            lines += [f"INVALID: {result.status_name} ({result.status})", ""]
        for key, val in result.scalars.items():
            if key in result.length_keys:
                lines.append(f"{key} = {val * scale:.4f} {unit}")
            else:
                lines.append(f"{key} = {val:.4f}")
        if result.neck is not None:
            lines += [f"neck ({result.neck.source}):",
                      f"  z = {result.neck.z * scale:.4f} {unit}",
                      f"  ρ = {result.neck.rho * scale:.4f} {unit}",
                      f"  depth = {result.neck.depth:.3f}"]
        lines += ["", f"volume  = {v:.4f} {unit}³ (GL)",
                  f"surface = {s:.4f} {unit}² (GL)",
                  f"z_cm    = {zc:.4f} {unit} (GL)"]
        if frag is not None:
            f_hi, f_lo, a_hi, a_lo, a_tot = frag
            lines += ["", "fragments (@ z_neck):",
                      f"  vol frac = {f_hi:.2f} : {f_lo:.2f}",
                      f"  mass A   = {a_hi:.0f} : {a_lo:.0f}  (A={a_tot})"]
        return "\n".join(lines)

    def run(self) -> None:
        plt.show()
