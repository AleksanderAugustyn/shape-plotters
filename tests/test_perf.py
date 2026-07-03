"""Engine update-loop budgets under Agg (default ext4 venv — see venv caveat).

Two tiers:

- Artist mutation (draw neutered): the clear-and-replot regression guard.
  v0.1 rebuilt every artist per frame: ~15/20 ms (beta/fos). v0.2 persistent
  artists: ~1 ms. Budget 5 ms.
- One-draw frame (update() alone; under Agg its draw_idle draws immediately):
  loose interactivity backstop. Full-figure rasterization (~35 axes) costs
  ~73-95 ms and dominates any engine improvement short of blitting; measured
  v0.1 ~104/135 ms, v0.2 ~77/95 ms. Budget 250 ms.

Venv caveat (measured 2026-07-03): identical code is ~5x slower under the
project-local .venv on /mnt/c (WSL2 9P filesystem). The default ext4 venv
(/home/alex/.virtualenvs/default) is authoritative for these budgets.
"""
import time

import pytest

from src.core.engine import ShapePlotterApp
from src.renders.beta import BetaRender
from src.renders.fos import FoSRender

MUTATION_BUDGET_MS = 5.0
FRAME_BUDGET_MS = 250.0


def _mean_ms(fn, n: int = 10) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


@pytest.mark.parametrize("render_cls", [BetaRender, FoSRender], ids=["beta", "fos"])
def test_artist_mutation_under_budget(render_cls) -> None:
    import matplotlib.pyplot as plt

    app = ShapePlotterApp(render_cls())
    try:
        app.fig.canvas.draw()                    # warm-up: font cache + first layout
        app.fig.canvas.draw_idle = lambda: None  # isolate mutation from rasterization
        ms = _mean_ms(app.update)
        assert ms < MUTATION_BUDGET_MS, f"{ms:.1f} ms/update >= {MUTATION_BUDGET_MS} ms budget"
    finally:
        plt.close(app.fig)


@pytest.mark.parametrize("render_cls", [BetaRender, FoSRender], ids=["beta", "fos"])
def test_frame_time_under_budget(render_cls) -> None:
    import matplotlib.pyplot as plt

    app = ShapePlotterApp(render_cls())
    try:
        app.update()
        app.fig.canvas.draw()      # warm-up: font cache + first layout
        ms = _mean_ms(app.update)  # one full draw per call (immediate under Agg)
        assert ms < FRAME_BUDGET_MS, f"{ms:.0f} ms/frame >= {FRAME_BUDGET_MS:.0f} ms budget"
    finally:
        plt.close(app.fig)
