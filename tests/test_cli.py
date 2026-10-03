"""Command-line entry: the render argument picks the render the app is built on."""
import subprocess
import sys

import pytest

from shape_plotters import cli
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender


class _SpyApp:
    """Stands in for ShapePlotterApp so no figure is built; records the wiring."""

    instances: list["_SpyApp"] = []

    def __init__(self, render: object) -> None:
        self.render = render
        self.ran = False
        _SpyApp.instances.append(self)

    def run(self) -> None:
        self.ran = True


@pytest.mark.parametrize("name, render_cls", [("beta", BetaRender), ("fos", FoSRender)])
def test_render_argument_selects_render(monkeypatch, name: str, render_cls: type) -> None:
    _SpyApp.instances = []
    monkeypatch.setattr(cli, "ShapePlotterApp", _SpyApp)
    cli.main([name])
    (app,) = _SpyApp.instances
    assert type(app.render) is render_cls
    assert app.ran


def test_module_is_runnable() -> None:
    # `python -m shape_plotters` must reach the same parser as the console script.
    out = subprocess.run([sys.executable, "-m", "shape_plotters", "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "beta" in out.stdout and "fos" in out.stdout
