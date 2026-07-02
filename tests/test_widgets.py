"""Widget helpers under the Agg backend (conftest sets MPLBACKEND)."""
import matplotlib.pyplot as plt
import pytest

from src.core.widgets import IntTextBox, SliderRow


@pytest.fixture()
def fig():
    f = plt.figure()
    yield f
    plt.close(f)


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
