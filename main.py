"""shape-plotters entry point: python main.py beta|fos"""
import argparse

from src.core.engine import ShapePlotterApp
from src.renders.beta import BetaRender
from src.renders.fos import FoSRender

RENDERS = {"beta": BetaRender, "fos": FoSRender}


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive nuclear-shape plotter")
    parser.add_argument("render", choices=sorted(RENDERS))
    args = parser.parse_args()
    ShapePlotterApp(RENDERS[args.render]()).run()


if __name__ == "__main__":
    main()
