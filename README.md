# ShapePlotters

Interactive matplotlib plotters for nuclear shape parameterizations.
One shared engine; each parameterization is a render module.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy scipy matplotlib pytest
    pip install -r requirements-libs.txt \
      --find-links https://github.com/AleksanderAugustyn/beta-parameterization/releases/expanded_assets/2.3.2 \
      --find-links https://github.com/AleksanderAugustyn/fos-parameterization/releases/expanded_assets/1.3.0

The parameterization libraries install as prebuilt Linux wheels from each
repo's GitHub Release (Fortran shared library and libgfortran bundled). To
pin new versions, bump `requirements-libs.txt` and the `--find-links` tags
together. When these libraries reach PyPI, drop the `--find-links` flags.

### WSL

Keep the virtual environment on the Linux filesystem, not under `/mnt/c`. The
Windows drive is a 9P mount; a venv there makes matplotlib rendering sluggish
(~5× slower). If the repo lives under `/mnt/c`, create the venv in your Linux
home instead:

    python3 -m venv ~/.venvs/shapeplotters
    source ~/.venvs/shapeplotters/bin/activate

## Run

    python main.py beta
    python main.py fos

## Tests

    python -m pytest
