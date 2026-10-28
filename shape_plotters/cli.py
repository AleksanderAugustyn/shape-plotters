"""Command-line entry point: shape-plotters beta|fos|tqs|cassini (or python -m shape_plotters)."""
from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence

from shape_plotters.core.engine import ShapePlotterApp
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender

RENDERS = {"beta": BetaRender, "fos": FoSRender}

# tqs- and cassini-parameterization are not on PyPI yet (locally built
# wheels), so they are not dependencies: register each render only where its
# package is installed.
if importlib.util.find_spec("tqs_parameterization") is not None:
    from shape_plotters.renders.tqs import TqsRender
    RENDERS["tqs"] = TqsRender
if importlib.util.find_spec("cassini_parameterization") is not None:
    from shape_plotters.renders.cassini import CassiniRender
    RENDERS["cassini"] = CassiniRender


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="shape-plotters", description="Interactive nuclear-shape plotter")
    parser.add_argument("render", choices=sorted(RENDERS))
    args = parser.parse_args(argv)
    ShapePlotterApp(RENDERS[args.render]()).run()
