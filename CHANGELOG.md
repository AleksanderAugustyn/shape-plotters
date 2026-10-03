# Changelog

Notable changes to shape-plotters. Versions follow semantic versioning.

## 0.2.0

Interactive performance. Measured in a real window on the development PC
(matplotlib 3.11): a drag step drops from about 110 ms to 33-46 ms, a click
on a slider from about 300 ms to about 105 ms.

### Changed

- **Slider drags are blitted.** While a slider is dragged, only the curves,
  the stats text and that slider are redrawn; ticks, legends and the other
  widgets come from a cached background. The axes hold still during the drag
  and refit on release; if the shape outgrows the view, the view jumps out
  once to a looser fit. During a drag the curves draw on top of legends.
  Everything outside a drag, including saved PNGs, is drawn exactly as before.
- **A click costs one redraw instead of three.** Every mouse press used to
  trigger two extra full redraws, one per Z/N text box.

## 0.1.0

First release.

### Added

- **Installable package.** `pip install shape-plotters` provides the
  `shape-plotters` command; the parameterization libraries are pinned
  dependencies.
- **Shared engine, one render per parameterization.** `shape-plotters beta`
  and `shape-plotters fos` open the same interactive figure; everything
  shape-specific comes from the render (contract in
  `shape_plotters/core/result.py`).
- **Beta render** over beta-parameterization 4.0.1: sliders β1–β8, volume
  conserved by the library. An orange overlay shows the centre-of-mass
  corrected shape whenever its β1 differs from the slider value; a corrected
  shape that turns out invalid is drawn greyed instead of vanishing.
- **FoS render** over fos-parameterization 3.0.0: sliders c and a3–a8 with
  practical-limit markers, an extra ρ(z) panel with the library's dρ/dz, the
  library's neck, and an orange overlay of the R(θ) star-convex representation
  when its centre of mass is shifted. Separated shapes draw both fragments.
- **Energy-model grid.** R(θ) and the analytic dR/dθ are evaluated by the
  libraries on a 2048-node Gauss-Legendre grid in cos θ, the dense grid of
  the WMMM energy model, so the plot shows the shape the model evaluates.
- **Stats panel:** volume, surface and centre of mass by Gauss-Legendre
  quadrature on that grid, neck position, radius and depth, and the fragment
  volume and mass split at the neck.
- **Invalid shapes** stay visible: greyed, with the library's status name and
  message as the panel title.
- **Display units:** fm / R₀ toggle with Z and N boxes (R₀ = 1.16 A^(1/3) fm).
- **Scission bands:** the neck-radius range 1.2–1.5 fm is shaded on the
  cross-section panel.
- **Controls:** ± nudge buttons on every slider, Reset, and Save (300 dpi PNG
  named after the parameters).
- **Optional Energy button.** Shown only when a local `wmmm` package is
  importable; prints WMMM energies for the shape on screen. `wmmm` is not a
  dependency and no model code, data or output is part of this repository.
- **Performance guards:** pytest budgets for artist mutation and frame time
  (`tests/test_perf.py`).

## Before 0.1.0

No changelog was kept during initial development (from 2026-07-03). See the
git history.
