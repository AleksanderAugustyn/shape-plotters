"""Simulated slider drags for tests: real MouseEvents on the slider axes.

Under Agg, draw_idle() draws immediately, so counting draw_event callbacks
counts full-figure redraws.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backend_bases import MouseEvent

from shape_plotters.core.engine import ShapePlotterApp


def steps(start: float, stop: float, step: float) -> list[float]:
    """Inclusive slider values from start to stop, on the 0.01 slider grid."""
    count = round((stop - start) / step)
    return [round(start + i * step, 2) for i in range(count + 1)]


def _mouse(app: ShapePlotterApp, name: str, key: str, value: float) -> None:
    slider = app.rows[key].slider
    x, y = slider.ax.transData.transform((value, 0.5))
    canvas = app.fig.canvas
    canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))


def press(app: ShapePlotterApp, key: str, value: float) -> None:
    _mouse(app, "button_press_event", key, value)


def move(app: ShapePlotterApp, key: str, value: float) -> None:
    _mouse(app, "motion_notify_event", key, value)


def release(app: ShapePlotterApp, key: str) -> None:
    _mouse(app, "button_release_event", key, app.rows[key].slider.val)


def begin_drag(app: ShapePlotterApp, key: str, values: list[float]) -> list[float]:
    """Press and move until frames are being blitted; returns the unused values.

    The press and the first motion are either drawn the ordinary way or start
    drag mode, depending on whether the press itself moved the slider. After
    two motions the background is captured either way.
    """
    press(app, key, values[0])
    move(app, key, values[1])
    move(app, key, values[2])
    assert app._blitter.active and app._blitter.ready
    return values[3:]


def pixels(app: ShapePlotterApp) -> np.ndarray:
    return np.asarray(app.fig.canvas.buffer_rgba()).copy()


def count_draws(app: ShapePlotterApp) -> list[int]:
    """A list that grows by one on every full-figure redraw from now on."""
    draws: list[int] = []
    app.fig.canvas.mpl_connect("draw_event", lambda _event: draws.append(1))
    return draws


def animated(app: ShapePlotterApp) -> list:
    """Artists still flagged animated: must be empty outside a drag."""
    candidates = (*app._dynamic, *(row.slider.ax for row in app.rows.values()))
    return [artist for artist in candidates if artist.get_animated()]
