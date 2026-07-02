"""ShapePlotterApp: parameterization-independent interactive figure.

Owns display units (fm/R0 toggle; ShapeResult is R0 units), layout, widgets,
invalid-shape greying, stats, and save. Everything shape-specific comes from
the render (contract in src/core/result.py).

Axes are cleared and replotted on every update — at 721 points this is fast
enough for slider drags and removes all artist-state bookkeeping.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

from src.core import quadrature
from src.core.result import ShapeResult
from src.core.widgets import IntTextBox, SliderRow

# Display convention (both old plotters used 1.16); distinct from WMMM physics constants.
R0_FM = 1.16

VALID_COLOR = "tab:blue"
DERIV_COLOR = "tab:red"
NECK_COLOR = "tab:green"
INVALID_COLOR = "0.55"
UNITS_LABEL = "fm units"


class ShapePlotterApp:
    def __init__(self, render) -> None:
        self.render = render
        self.fm_units = True
        self.toggle_state = {t.key: t.default for t in render.toggles}
        self.last_result: ShapeResult | None = None
        self._build_figure()
        self._build_widgets()
        self.update()

    # ---------- construction ----------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 9))
        try:
            self.fig.canvas.manager.set_window_title(f"ShapePlotters — {self.render.name}")
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

    def _build_widgets(self) -> None:
        self.rows: dict[str, SliderRow] = {}
        y = 0.06
        for spec in self.render.slider_specs:
            row = SliderRow(self.fig, y, spec.label, spec.vmin, spec.vmax,
                            spec.vinit, spec.step, spec.markers)
            row.slider.on_changed(self.update)
            self.rows[spec.key] = row
            y += 0.031
        self.z_box = IntTextBox(self.fig, (0.88, 0.40, 0.06, 0.030), "Z ", 92, self.update)
        self.n_box = IntTextBox(self.fig, (0.88, 0.36, 0.06, 0.030), "N ", 144, self.update)
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

    # ---------- callbacks ----------

    def _on_check(self, label: str) -> None:
        if label == UNITS_LABEL:
            self.fm_units = not self.fm_units
        else:
            for t in self.render.toggles:
                if t.label == label:
                    self.toggle_state[t.key] = not self.toggle_state[t.key]
        self.update()

    def _reset(self, _event=None) -> None:
        # Each set_val fires update(); fine at 8 sliders.
        for spec in self.render.slider_specs:
            self.rows[spec.key].slider.set_val(spec.vinit)

    def _save(self, _event=None) -> None:
        params = {k: row.slider.val for k, row in self.rows.items()}
        fname = self.render.filename(self.z_box.value, self.n_box.value, params)
        self.fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved {fname}")

    # ---------- drawing ----------

    def _scale(self) -> float:
        if not self.fm_units:
            return 1.0
        return R0_FM * float(self.z_box.value + self.n_box.value) ** (1.0 / 3.0)

    def update(self, _val=None) -> None:
        params = {k: row.slider.val for k, row in self.rows.items()}
        result = self.render.compute(params, self.toggle_state)
        self.last_result = result
        scale = self._scale()
        unit = "fm" if self.fm_units else "R0"
        color = VALID_COLOR if result.ok else INVALID_COLOR
        deriv = DERIV_COLOR if result.ok else INVALID_COLOR
        alpha = 1.0 if result.ok else 0.45

        ax = self.ax_radius
        ax.clear()
        r = result.radius * scale
        ax.plot(result.theta, r, color=color, alpha=alpha, lw=2, label=f"R(θ) [{unit}]")
        ax.plot(result.theta, np.gradient(r, result.theta), color=deriv, alpha=alpha,
                lw=2, ls=":", label="dR/dθ (display)")
        ax.set_xlim(0.0, np.pi)
        ax.set_xlabel("θ [rad]")
        ax.legend(loc="upper center", fontsize=8)
        ax.grid(alpha=0.3)

        ax = self.ax_shape
        ax.clear()
        z, rho = result.z * scale, result.rho * scale
        ax.plot(z, rho, color=color, alpha=alpha, lw=2)
        ax.plot(z, -rho, color=color, alpha=alpha, lw=2)
        if result.neck is not None:
            zn, rn = result.neck.z * scale, result.neck.rho * scale
            ax.vlines(zn, -rn, rn, color=NECK_COLOR, ls="--", lw=1.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"z [{unit}]")
        ax.set_ylabel(f"ρ [{unit}]")
        ax.grid(alpha=0.3)
        ax.set_title(f"{result.status_name}: {result.message}" if not result.ok else "",
                     color="tab:red", fontsize=9)

        if self.ax_extra is not None:
            ax = self.ax_extra
            ax.clear()
            ax.plot(z, rho, color=color, alpha=alpha, lw=2, label=f"ρ(z) [{unit}]")
            if result.drho_dz is not None:
                # drho/dz is a unit-free slope: both lengths scale identically.
                ax.plot(z, result.drho_dz, color=deriv, alpha=alpha, lw=1.5, ls=":",
                        label="dρ/dz (lib)")
            ax.set_xlabel(f"z [{unit}]")
            ax.legend(loc="upper center", fontsize=8)
            ax.grid(alpha=0.3)

        self._update_stats(result, scale, unit)
        self.fig.canvas.draw_idle()

    def _update_stats(self, result: ShapeResult, scale: float, unit: str) -> None:
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
        v = quadrature.volume(result.theta, result.radius) * scale**3
        s = quadrature.surface_area(result.theta, result.radius) * scale**2
        zc = quadrature.z_cm(result.theta, result.radius) * scale
        lines += ["", f"volume  = {v:.4f} {unit}³ (py quad)",
                  f"surface = {s:.4f} {unit}² (py quad)",
                  f"z_cm    = {zc:.4f} {unit} (py quad)"]
        self.ax_stats.clear()
        self.ax_stats.axis("off")
        self.ax_stats.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace",
                           fontsize=9, transform=self.ax_stats.transAxes)

    def run(self) -> None:
        plt.show()
