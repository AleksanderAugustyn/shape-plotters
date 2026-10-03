# shape-plotters

Interactive matplotlib plotters for axially symmetric nuclear shapes. Move a
slider and the shape, its derivative and its integral properties update. One
shared engine draws the figure; each shape parameterization is a render module.

| render | parameterization | sliders | library |
|---|---|---|---|
| `beta` | spherical-harmonic (Legendre) expansion | β1–β8 | [beta-parameterization](https://github.com/AleksanderAugustyn/beta-parameterization) |
| `fos` | Fourier-over-Spheroid | c, a3–a8 | [fos-parameterization](https://github.com/AleksanderAugustyn/fos-parameterization) |

The shapes are not re-implemented here. R(θ) and the analytic dR/dθ come from
the compiled libraries, evaluated on a 2048-node Gauss-Legendre grid in cos θ.
That is the dense grid of the WMMM macroscopic-microscopic energy model, so
the plot shows exactly the shape the model evaluates.

## What the figure shows

- **R(θ) and dR/dθ**, and the **cross-section** ρ(z) with equal axes. The FoS
  render adds a ρ(z) panel with the library's dρ/dz.
- **Stats:** volume, surface and centre of mass by Gauss-Legendre quadrature,
  the neck position, radius and depth, and the fragment volume and mass split
  at the neck.
- **Overlays** (orange, dashed): for beta, the centre-of-mass corrected shape
  when its β1 differs from the slider value; for FoS, the R(θ) star-convex
  representation when its centre of mass is shifted.
- **Scission bands:** the neck-radius range 1.2–1.5 fm, shaded.
- **Invalid shapes** stay on screen, greyed, with the library's status as the
  panel title.

Controls: ± nudge buttons on every slider, an fm / R₀ unit toggle with Z and N
boxes (R₀ = 1.16 A^(1/3) fm), Reset, and Save (300 dpi PNG named after the
parameters, written to the current directory). Red dotted marks on the FoS
sliders are practical limits.

While a slider is dragged the axes hold still, so the shape visibly changes
instead of the axes rescaling around it; they refit when the mouse is
released. If the shape outgrows the view mid-drag, the view jumps out once.

## Requirements

- Linux x86-64. The parameterization libraries ship manylinux wheels only;
  WSL2 with WSLg works.
- Python 3.10 or newer with a matplotlib GUI backend. Tk is the default
  (`python3-tk` on Debian/Ubuntu).

## Install

    pip install shape-plotters

or `pipx install shape-plotters` to get the command without touching another
environment. The parameterization libraries come along as prebuilt wheels
(Fortran shared library and libgfortran bundled), pinned to the exact versions
each render is written against.

Under WSL, keep the virtual environment on the Linux filesystem, not under
`/mnt/c`. The Windows drive is a 9P mount; a venv there makes matplotlib
rendering sluggish (~5× slower).

## Run

    shape-plotters beta
    shape-plotters fos

`python -m shape_plotters beta` does the same.

## Energy button

An Energy button appears when a Python package named `wmmm` is importable. It
prints the WMMM energies of the shape on screen. WMMM is not public yet and is
not a dependency: without it the button is absent and nothing else changes.
This repository contains the calling convention only, no model code, data or
output.

## Development

    git clone https://github.com/AleksanderAugustyn/shape-plotters.git
    cd shape-plotters
    python3 -m venv ~/.venvs/shape-plotters
    source ~/.venvs/shape-plotters/bin/activate
    pip install -e ".[test]"
    python -m pytest

The WMMM smoke tests skip when `wmmm` is not installed. The frame-time budgets
in `tests/test_perf.py` assume a venv on the Linux filesystem.

    shape_plotters/cli.py          entry point and render registry
    shape_plotters/core/           engine, widgets, quadrature, neck and fragment helpers
    shape_plotters/core/result.py  the render contract (what a render gives the engine)
    shape_plotters/renders/        one module per parameterization
    tests/                         pytest suite, including frame-time budgets

A new parameterization is a new module in `shape_plotters/renders/` that
implements the render contract, plus one registry entry in `cli.py` and a
pinned library in `pyproject.toml`. The engine needs no change.

Releases are cut by pushing a version tag such as `0.1.0`; the `wheels`
workflow builds, tests and publishes to PyPI.

## License

MIT, see [LICENSE](https://github.com/AleksanderAugustyn/shape-plotters/blob/master/LICENSE).
Changes are recorded in
[CHANGELOG.md](https://github.com/AleksanderAugustyn/shape-plotters/blob/master/CHANGELOG.md).
