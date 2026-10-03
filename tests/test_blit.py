"""DragBlitter on a bare figure: background capture, frame composition, cleanup."""
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shape_plotters.core.blit import DragBlitter


def _pixels(fig) -> np.ndarray:
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.fixture
def scene():
    """A plot axes with one dynamic line, and a small axes standing in for a slider."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes((0.1, 0.3, 0.8, 0.6))
    (line,) = ax.plot([0.0, 1.0], [0.0, 1.0], lw=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.5, 1.5)
    knob_ax = fig.add_axes((0.1, 0.1, 0.8, 0.05))
    (knob,) = knob_ax.plot([0.2], [0.5], marker="o")
    knob_ax.set_xlim(0.0, 1.0)
    knob_ax.set_ylim(0.0, 1.0)
    blitter = DragBlitter(fig, [line])
    fig.canvas.draw()
    yield fig, ax, line, knob_ax, knob, blitter
    plt.close(fig)


def test_blitted_frame_equals_full_redraw(scene) -> None:
    fig, _ax, line, knob_ax, knob, blitter = scene
    blitter.begin(knob_ax)
    fig.canvas.draw()                        # captures the static background
    before = _pixels(fig)
    line.set_ydata([1.0, -1.0])              # the dynamic layer changes
    knob.set_xdata([0.8])                    # and so does the dragged control
    blitter.blit()
    blitted = _pixels(fig)
    fig.canvas.draw()
    assert (blitted != before).any()         # the frame really changed
    assert np.array_equal(blitted, _pixels(fig))


def test_background_exists_only_after_a_draw_in_drag_mode(scene) -> None:
    fig, _ax, _line, knob_ax, _knob, blitter = scene
    assert not blitter.active and not blitter.ready   # the fixture's draw captured nothing
    blitter.begin(knob_ax)
    assert blitter.active and not blitter.ready
    fig.canvas.draw()
    assert blitter.ready
    blitter.invalidate()
    assert not blitter.ready
    fig.canvas.draw()
    assert blitter.ready


def test_foreign_draw_mid_drag_recaptures_background(scene) -> None:
    fig, ax, line, knob_ax, _knob, blitter = scene
    blitter.begin(knob_ax)
    fig.canvas.draw()
    ax.set_title("static change")            # stands in for a resize or expose
    fig.canvas.draw()                        # a full draw the blitter did not ask for
    line.set_ydata([1.0, -1.0])
    blitter.blit()
    blitted = _pixels(fig)
    fig.canvas.draw()
    assert np.array_equal(blitted, _pixels(fig))   # the title is in the blitted frame


def test_end_restores_ordinary_drawing(scene) -> None:
    fig, _ax, line, knob_ax, _knob, blitter = scene
    reference = _pixels(fig)                 # drawn before any drag
    blitter.begin(knob_ax)
    fig.canvas.draw()
    blitter.end()
    fig.canvas.draw()
    assert not blitter.active and not blitter.ready
    assert not line.get_animated() and not knob_ax.get_animated()
    assert np.array_equal(_pixels(fig), reference)


def test_export_mid_drag_is_complete_and_drops_the_background(scene) -> None:
    # savefig can run while a drag is active (the toolbar's save key, or a
    # Save click after a lost release). It is not a screen draw: it must not
    # become the background, and the exported file must not miss anything.
    fig, _ax, line, knob_ax, _knob, blitter = scene

    def export(fmt: str) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=100)
        return buffer.getvalue()

    reference = export("rgba")               # saved before any drag
    blitter.begin(knob_ax)
    fig.canvas.draw()
    assert export("rgba") == reference       # line and slider axes present, each drawn once
    export("pdf")                            # a canvas that cannot copy regions
    assert not blitter.ready                 # the export replaced the canvas contents
    fig.canvas.draw()
    line.set_ydata([1.0, -1.0])
    blitter.blit()
    blitted = _pixels(fig)
    fig.canvas.draw()
    assert np.array_equal(blitted, _pixels(fig))
