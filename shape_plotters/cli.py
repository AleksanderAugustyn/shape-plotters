"""Command-line entry point: shape-plotters beta|fos|tqs (or python -m shape_plotters)."""
from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence

from shape_plotters.core.engine import ShapePlotterApp
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender

RENDERS = {"beta": BetaRender, "fos": FoSRender}

# tqs-parameterization is not on PyPI yet (a locally built wheel), so it is
# not a dependency: register the render only where the package is installed.
if importlib.util.find_spec("tqs_parameterization") is not None:
    from shape_plotters.renders.tqs import TqsRender
    RENDERS["tqs"] = TqsRender


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="shape-plotters", description="Interactive nuclear-shape plotter")
    parser.add_argument("render", choices=sorted(RENDERS))
    args = parser.parse_args(argv)
    ShapePlotterApp(RENDERS[args.render]()).run()
