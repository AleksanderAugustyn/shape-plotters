# Changelog

Notable changes to shape-plotters. Versions follow semantic versioning.

## Unreleased

Nothing has been released yet. The first tagged release will be 0.1.0 and
will carry everything below.

### Added

- **Shared engine, one render per parameterization.** `python main.py beta`
  and `python main.py fos` open the same interactive figure; everything
  shape-specific comes from the render (contract in `src/core/result.py`).
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
