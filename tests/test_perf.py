"""Engine update-loop budgets under Agg (default ext4 venv — see venv caveat).

Each test prints its figure; `python -m pytest tests/test_perf.py -s` is the
one-command measurement on any machine. The budgets are calibrated for the
development PC and may fail on a slower machine; the printed figures are
still the measurement.

Three tiers:

- Artist mutation (draw neutered): the clear-and-replot regression guard.
  v0.1 rebuilt every artist per frame: ~15/20 ms (beta/fos). v0.2 persistent
  artists: ~1 ms. Budget 5 ms.
  Two-tier wheels (beta 4.0.1, fos 3.0.0; 2026-10-03): 1.4 / 2.2 ms —
  every call resolves from params; the repeated vector is no longer served
  from stored state.
- One-draw frame (update() alone; under Agg its draw_idle draws immediately):
  loose interactivity backstop for everything that is not a drag. Full-figure
  rasterization (~35 axes) costs ~73-95 ms; measured v0.1 ~104/135 ms,
  v0.2 ~77/95 ms. Budget 250 ms.
- Blitted drag step (one slider motion event during a drag): the static layer
  is restored from a cached background and only the data lines, the stats
  text and the dragged slider are drawn. Measured 2026-10-03: ~20-35 ms on
  matplotlib 3.11 (its text rendering dominates), ~10 ms on 3.10. Budget 60 ms.

Venv caveat (measured 2026-07-03): identical code is ~5x slower under the
project-local .venv on /mnt/c (WSL2 9P filesystem). The default ext4 venv
(/home/alex/.virtualenvs/default) is authoritative for these budgets.
"""
import statistics
import time

import pytest

from shape_plotters.core.engine import ShapePlotterApp
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender
from tests.dragging import begin_drag, move, release, steps

MUTATION_BUDGET_MS = 5.0
FRAME_BUDGET_MS = 250.0
DRAG_STEP_BUDGET_MS = 60.0


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
        print(f"\nPERF artist mutation [{app.render.name}]: {ms:.2f} ms")
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
        print(f"\nPERF full frame [{app.render.name}]: {ms:.1f} ms")
        assert ms < FRAME_BUDGET_MS, f"{ms:.0f} ms/frame >= {FRAME_BUDGET_MS:.0f} ms budget"
    finally:
        plt.close(app.fig)


@pytest.mark.parametrize("render_cls, presets, key, span", [
    # Shrinking drags: no refits, so every timed step is a blitted frame.
    pytest.param(BetaRender, {"beta2": 1.0}, "beta2", (1.0, 0.6, -0.01), id="beta"),
    pytest.param(FoSRender, {"c": 1.6, "a4": 0.3}, "a4", (0.3, -0.1, -0.01), id="fos"),
])
def test_drag_step_under_budget(render_cls, presets, key, span) -> None:
    import matplotlib.pyplot as plt

    app = ShapePlotterApp(render_cls())
    try:
        for name, value in presets.items():
            app.rows[name].slider.set_val(value)
        app.fig.canvas.draw()      # warm-up: font cache + first layout
        times_ms = []
        for value in begin_drag(app, key, steps(*span)):
            t0 = time.perf_counter()
            move(app, key, value)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        release(app, key)
        ms = statistics.median(times_ms)
        print(f"\nPERF blitted drag step [{app.render.name}]: {ms:.1f} ms "
              f"(median of {len(times_ms)})")
        assert ms < DRAG_STEP_BUDGET_MS, (
            f"{ms:.1f} ms/step >= {DRAG_STEP_BUDGET_MS:.0f} ms budget")
    finally:
        plt.close(app.fig)
