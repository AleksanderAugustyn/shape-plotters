# ShapePlotters

Interactive matplotlib plotters for nuclear shape parameterizations.
One shared engine; each parameterization is a render module.

## Run

    .venv/bin/python main.py beta
    .venv/bin/python main.py fos

## Setup (WSL)

    python3 -m venv .venv
    .venv/bin/pip install numpy scipy matplotlib pytest
    .venv/bin/pip install -e /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/python
    .venv/bin/pip install -e /mnt/c/Users/aleks/CLionProjects/Fortran/fos-parameterization/python

The ctypes packages locate the shared libraries in each library repo's
`build/release/` (build with `cmake --build build/release` there), or via
`BETA_PARAM_LIB` / `FOS_PARAM_LIB` env vars.

## Tests

    .venv/bin/python -m pytest
