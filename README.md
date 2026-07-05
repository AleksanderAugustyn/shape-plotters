# ShapePlotters

Interactive matplotlib plotters for nuclear shape parameterizations.
One shared engine; each parameterization is a render module.

## Run

    .venv/bin/python main.py beta
    .venv/bin/python main.py fos

## Setup (WSL)

    python3 -m venv .venv
    .venv/bin/pip install numpy scipy matplotlib pytest
    .venv/bin/pip install -r requirements-libs.txt \
      --find-links https://github.com/AleksanderAugustyn/beta-parameterization/releases/expanded_assets/2.3.2 \
      --find-links https://github.com/AleksanderAugustyn/fos-parameterization/releases/expanded_assets/1.2.1

The parameterization libraries install as prebuilt Linux wheels from each
repo's GitHub Release (Fortran shared library and libgfortran bundled). To
pin new versions, bump `requirements-libs.txt` and the `--find-links` tags
together. When these libraries reach PyPI, drop the `--find-links` flags.

## Tests

    .venv/bin/python -m pytest
