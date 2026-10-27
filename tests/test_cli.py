"""Command-line entry: the render argument picks the render the app is built on."""
import importlib
import importlib.util
import subprocess
import sys

import pytest

from shape_plotters import cli
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender

_real_find_spec = importlib.util.find_spec
HAS_TQS = importlib.util.find_spec("tqs_parameterization") is not None


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
    assert ("tqs" in out.stdout) == HAS_TQS


@pytest.mark.skipif(not HAS_TQS, reason="tqs_parameterization not installed")
def test_tqs_registered_when_importable(monkeypatch) -> None:
    from shape_plotters.renders.tqs import TqsRender
    _SpyApp.instances = []
    monkeypatch.setattr(cli, "ShapePlotterApp", _SpyApp)
    cli.main(["tqs"])
    (app,) = _SpyApp.instances
    assert type(app.render) is TqsRender and app.ran


def test_registry_without_tqs(monkeypatch) -> None:
    # Re-import cli with find_spec answering None for the package: the
    # registry has exactly beta and fos (CI and any machine without the wheel).
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "tqs_parameterization"
                        else _real_find_spec(name, *a, **k))
    try:
        fresh = importlib.reload(cli)
        assert sorted(fresh.RENDERS) == ["beta", "fos"]
    finally:
        monkeypatch.undo()
        importlib.reload(cli)   # restore with the real find_spec
