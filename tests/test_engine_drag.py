"""Slider drags: blitted frames, the held view, and the return to ordinary drawing."""
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from shape_plotters.core.engine import ShapePlotterApp
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender
from tests.dragging import (animated, begin_drag, count_draws, move, pixels, press,
                            release, steps)

# Drags that shrink the shape: the data never leaves the view and neither the
# title nor the overlay legend changes, so every frame after the start can blit.
SHRINKING = [
    pytest.param(BetaRender, {"beta2": 1.0}, "beta2", (1.0, 0.85, -0.01), id="beta"),
    pytest.param(FoSRender, {"c": 1.6, "a4": 0.3}, "a4", (0.3, 0.15, -0.01), id="fos"),
]


@pytest.fixture
def make_app():
    apps: list[ShapePlotterApp] = []

    def make(render_cls, **presets: float) -> ShapePlotterApp:
        app = ShapePlotterApp(render_cls())
        for key, value in presets.items():
            app.rows[key].slider.set_val(value)
        app.fig.canvas.draw()
        apps.append(app)
        return app

    yield make
    for app in apps:
        plt.close(app.fig)


# ---------- a click is not a drag ----------

def test_click_costs_one_full_draw_and_is_not_a_drag(make_app) -> None:
    app = make_app(BetaRender)
    draws = count_draws(app)
    press(app, "beta2", 0.3)
    release(app, "beta2")
    assert len(draws) == 1
    assert not app._blitter.active and animated(app) == []
    reference = make_app(BetaRender, beta2=0.3)
    assert np.array_equal(pixels(app), pixels(reference))


# ---------- blitted frames ----------

@pytest.mark.parametrize("render_cls, presets, key, span", SHRINKING)
def test_drag_frames_are_blitted_not_redrawn(make_app, render_cls, presets, key, span) -> None:
    app = make_app(render_cls, **presets)
    rest = begin_drag(app, key, steps(*span))
    draws = count_draws(app)
    for value in rest:
        move(app, key, value)
    assert draws == []                                   # no full-figure redraw
    assert app.rows[key].slider.val == pytest.approx(span[1])
    assert np.array_equal(app.r_line.get_ydata(),
                          app.last_result.radius * app._scale())


@pytest.mark.parametrize("render_cls, presets, key, span", SHRINKING + [
    # The neck line disappears on the way: a dynamic artist turning invisible.
    pytest.param(BetaRender, {"beta2": 1.2}, "beta2", (1.2, 0.4, -0.04),
                 id="beta-neck-vanishes"),
])
def test_every_blitted_frame_equals_a_full_redraw(
        make_app, render_cls, presets, key, span) -> None:
    app = make_app(render_cls, **presets)
    for value in begin_drag(app, key, steps(*span)):
        move(app, key, value)
        blitted = pixels(app)
        app.fig.canvas.draw()
        assert np.array_equal(blitted, pixels(app)), f"frame at {key}={value}"


def test_widgets_look_the_same_in_blitted_frames(make_app) -> None:
    # Everything below the panels: the dragged slider (handle, fill, value
    # text), the other sliders, buttons and the check marks, which CheckButtons
    # draws through its own blit machinery. A blitted frame must show them
    # exactly as the full redraw after the release does.
    app = make_app(BetaRender, beta2=1.0)
    for value in begin_drag(app, "beta2", steps(1.0, 0.9, -0.01)):
        move(app, "beta2", value)
    panels = (*app._data_axes, app.ax_stats)
    widgets_top = max(ax.bbox.y1 for ax in app.fig.axes if ax not in panels)
    height = pixels(app).shape[0]
    widget_rows = slice(height - int(widgets_top), height)   # image rows run top to bottom
    during = pixels(app)[widget_rows]
    release(app, "beta2")
    assert np.array_equal(during, pixels(app)[widget_rows])


# ---------- leaving drag mode ----------

@pytest.mark.parametrize("render_cls, presets, key, span", SHRINKING)
def test_release_restores_ordinary_drawing(make_app, render_cls, presets, key, span) -> None:
    app = make_app(render_cls, **presets)
    for value in begin_drag(app, key, steps(*span)):
        move(app, key, value)
    release(app, key)
    assert not app._blitter.active and animated(app) == []
    reference = make_app(render_cls, **{**presets, key: span[1]})
    assert np.array_equal(pixels(app), pixels(reference))


def test_updates_outside_a_drag_never_enter_drag_mode(make_app) -> None:
    app = make_app(BetaRender)
    app.rows["beta2"].slider.set_val(0.4)       # programmatic
    app.rows["beta2"]._nudge(+0.01)             # nudge button
    app._on_check("fm units")                   # unit toggle
    app._reset()                                # Reset button
    assert not app._blitter.active and animated(app) == []


# ---------- robustness ----------

def test_lost_release_ends_drag_mode_on_the_next_update(make_app) -> None:
    app = make_app(BetaRender, beta2=1.0)
    for value in begin_drag(app, "beta2", steps(1.0, 0.9, -0.01)):
        move(app, "beta2", value)
    app.rows["beta2"].slider.drag_active = False    # the release event never arrived
    app.rows["beta3"].slider.set_val(0.1)           # any later update
    assert not app._blitter.active and animated(app) == []
    reference = make_app(BetaRender, beta2=0.9, beta3=0.1)
    assert np.array_equal(pixels(app), pixels(reference))


def test_export_mid_drag_does_not_poison_later_frames(make_app) -> None:
    app = make_app(BetaRender, beta2=1.0)
    rest = begin_drag(app, "beta2", steps(1.0, 0.9, -0.01))
    app.fig.savefig(io.BytesIO(), format="rgba", dpi=100)    # e.g. the toolbar's save key
    for value in rest:
        move(app, "beta2", value)
        frame = pixels(app)
        app.fig.canvas.draw()
        assert np.array_equal(frame, pixels(app)), f"frame at beta2={value}"


def test_save_button_in_a_stale_drag_saves_the_ordinary_figure(
        make_app, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)                      # _save() writes into the cwd
    app = make_app(BetaRender, beta2=1.0)
    for value in begin_drag(app, "beta2", steps(1.0, 0.9, -0.01)):
        move(app, "beta2", value)
    app.rows["beta2"].slider.drag_active = False     # the release event never arrived
    app._save()
    assert not app._blitter.active and animated(app) == []
    (png,) = tmp_path.glob("*.png")
    after_drag = png.read_bytes()
    reference = make_app(BetaRender, beta2=0.9)
    reference._save()                                # same parameters: same file name
    assert after_drag == png.read_bytes()


def test_canvas_without_blitting_redraws_every_frame(make_app, monkeypatch) -> None:
    monkeypatch.setattr(FigureCanvasAgg, "supports_blit", False)
    app = make_app(BetaRender, beta2=1.0)
    assert app._blitter is None
    draws = count_draws(app)
    values = steps(1.0, 0.9, -0.01)
    press(app, "beta2", values[0])                  # on the handle: no value change
    for value in values[1:]:
        move(app, "beta2", value)
    assert len(draws) == len(values) - 1            # one full redraw per motion
    release(app, "beta2")
    assert animated(app) == []


def test_dragging_past_the_slider_end_stops_at_the_limit(make_app) -> None:
    app = make_app(BetaRender, beta2=3.9)
    press(app, "beta2", 3.9)
    for value in (3.95, 4.0, 4.3, 4.8):             # the last two are off the slider
        move(app, "beta2", value)
    assert app.rows["beta2"].slider.val == pytest.approx(4.0)
    release(app, "beta2")
    assert not app._blitter.active and animated(app) == []
    reference = make_app(BetaRender, beta2=4.0)
    assert np.array_equal(pixels(app), pixels(reference))
