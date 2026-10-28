"""Command-line entry point: shape-plotters beta|fos|tqs|cassini|funnyhills (or python -m shape_plotters)."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from shape_plotters.core.engine import ShapePlotterApp
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.cassini import CassiniRender
from shape_plotters.renders.fos import FoSRender
from shape_plotters.renders.funnyhills import FunnyhillsRender
from shape_plotters.renders.tqs import TqsRender

RENDERS = {
    "beta": BetaRender,
    "fos": FoSRender,
    "tqs": TqsRender,
    "cassini": CassiniRender,
    "funnyhills": FunnyhillsRender,
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="shape-plotters", description="Interactive nuclear-shape plotter")
    parser.add_argument("render", choices=sorted(RENDERS))
    args = parser.parse_args(argv)
    ShapePlotterApp(RENDERS[args.render]()).run()
