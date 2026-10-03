"""Widget helpers under the Agg backend (conftest sets MPLBACKEND)."""
import matplotlib.pyplot as plt
import pytest
from matplotlib.backend_bases import MouseEvent

from shape_plotters.core.widgets import IntTextBox, SliderRow


@pytest.fixture()
def fig():
    f = plt.figure()
    yield f
    plt.close(f)


def _press(fig, x_frac: float, y_frac: float) -> None:
    """Deliver a left-button press at a position given in figure fractions."""
    width, height = fig.canvas.get_width_height()
    event = MouseEvent("button_press_event", fig.canvas,
                       x_frac * width, y_frac * height, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


def test_slider_row_nudges_within_range(fig) -> None:
    row = SliderRow(fig, 0.5, "b2", 0.0, 4.0, 0.0, 0.01)
    row._nudge(+0.01)
    assert row.slider.val == pytest.approx(0.01)
    row._nudge(-0.01)
    row._nudge(-0.01)  # would go below vmin -> clamped (no change)
    assert row.slider.val == pytest.approx(0.0)


def test_slider_row_fires_on_changed(fig) -> None:
    row = SliderRow(fig, 0.5, "c", 0.5, 3.5, 1.0, 0.01)
    seen: list[float] = []
    row.slider.on_changed(seen.append)
    row._nudge(+0.01)
    assert seen and seen[-1] == pytest.approx(1.01)


def test_int_textbox_survives_window_resize(fig) -> None:
    # mpl 3.11.0 regression: TextBox's resize handler crashes on
    # ResizeEvent.inaxes; IntTextBox replaces the connection.
    from matplotlib.backend_bases import ResizeEvent

    IntTextBox(fig, (0.1, 0.1, 0.1, 0.05), "Z", 92, lambda: None)
    fig.canvas.callbacks.exception_handler = None  # re-raise instead of logging
    ResizeEvent("resize_event", fig.canvas)._process()


def test_int_textbox_accepts_and_reverts(fig) -> None:
    calls: list[int] = []
    box = IntTextBox(fig, (0.1, 0.1, 0.1, 0.05), "Z", 92, lambda: calls.append(1))
    box._submit("96")
    assert box.value == 96 and calls
    box._submit("abc")
    assert box.value == 96          # reverted
    box._submit("-3")
    assert box.value == 96          # positive ints only


def test_press_elsewhere_does_not_redraw_when_not_typing(fig) -> None:
    # TextBox.stop_typing() does a full canvas.draw() on every press outside
    # the box. With the Z and N boxes that was two full redraws per click
    # anywhere in the figure.
    IntTextBox(fig, (0.1, 0.1, 0.1, 0.05), "Z", 92, lambda: None)
    fig.canvas.draw()
    draws: list[int] = []
    fig.canvas.mpl_connect("draw_event", lambda _event: draws.append(1))
    _press(fig, 0.8, 0.8)                    # far from the box
    assert draws == []


def test_press_elsewhere_still_ends_typing(fig) -> None:
    calls: list[int] = []
    box = IntTextBox(fig, (0.1, 0.1, 0.1, 0.05), "Z", 92, lambda: calls.append(1))
    fig.canvas.draw()
    box.box.begin_typing()
    _press(fig, 0.8, 0.8)
    assert not box.box.capturekeystrokes     # typing ended
    assert calls                             # and the text was submitted
