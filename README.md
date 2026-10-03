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

## Requirements

- Linux x86-64. The parameterization libraries ship manylinux wheels only;
  WSL2 with WSLg works.
- Python 3.10 or newer with a matplotlib GUI backend. Tk is the default
  (`python3-tk` on Debian/Ubuntu).

## Setup

    git clone https://github.com/AleksanderAugustyn/shape-plotters.git
    cd shape-plotters
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy scipy matplotlib pytest
    pip install -r requirements-libs.txt

The parameterization libraries install from PyPI as prebuilt wheels (Fortran
shared library and libgfortran bundled), pinned in `requirements-libs.txt`.

### WSL

Keep the virtual environment on the Linux filesystem, not under `/mnt/c`. The
Windows drive is a 9P mount; a venv there makes matplotlib rendering sluggish
(~5× slower). If the repo lives under `/mnt/c`, create the venv in your Linux
home instead:

    python3 -m venv ~/.venvs/shape-plotters
    source ~/.venvs/shape-plotters/bin/activate

## Run

    python main.py beta
    python main.py fos

## Energy button

An Energy button appears when a Python package named `wmmm` is importable. It
prints the WMMM energies of the shape on screen. WMMM is not public yet and is
not a dependency: without it the button is absent and nothing else changes.
This repository contains the calling convention only, no model code, data or
output.

## Layout

    main.py              entry point: python main.py beta|fos
    src/core/            engine, widgets, quadrature, neck and fragment helpers
    src/core/result.py   the render contract (what a render gives the engine)
    src/renders/         one module per parameterization
    tests/               pytest suite, including frame-time budgets

A new parameterization is a new module in `src/renders/` that implements the
contract in `src/core/result.py`, plus one entry in `main.py`. The engine
needs no change.

## Tests

    python -m pytest

The WMMM smoke tests skip when `wmmm` is not installed.

## License

MIT, see [LICENSE](LICENSE). Changes are recorded in
[CHANGELOG.md](CHANGELOG.md).
