# shape-plotters

Interactive matplotlib plotters for nuclear shape parameterizations.
One shared engine; each parameterization is a render module.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy scipy matplotlib pytest
    pip install -r requirements-libs.txt

The parameterization libraries install from PyPI as prebuilt manylinux wheels
(Fortran shared library and libgfortran bundled).

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

## Tests

    python -m pytest
