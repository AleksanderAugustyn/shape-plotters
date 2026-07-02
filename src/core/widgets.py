"""Parameterization-independent matplotlib widget helpers."""
from __future__ import annotations

from typing import Callable

import matplotlib
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider, TextBox


class SliderRow:
    """Slider with -/+ nudge buttons and optional practical-limit markers.

    Extracted from ShapePlotterFoSFitter's create_slider. The grey line marks
    the reset value; red dotted lines mark practical limits.
    """

    def __init__(self, fig: Figure, y_pos: float, label: str, vmin: float,
                 vmax: float, vinit: float, step: float,
                 markers: tuple[float, ...] = ()) -> None:
        ax_dec = fig.add_axes((0.20, y_pos, 0.016, 0.024))
        ax_sl = fig.add_axes((0.25, y_pos, 0.45, 0.024))
        ax_inc = fig.add_axes((0.73, y_pos, 0.016, 0.024))
        self.slider = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=step)
        self.btn_dec = Button(ax_dec, "-")
        self.btn_inc = Button(ax_inc, "+")
        self.slider.ax.vlines(vinit, 0, 1, color="k", alpha=0.3, linewidth=1)
        if markers:
            self.slider.ax.vlines(list(markers), 0, 1, color="r", linestyle=":",
                                  alpha=0.7, linewidth=1.5)
        self._step = step
        self.btn_dec.on_clicked(lambda _event: self._nudge(-self._step))
        self.btn_inc.on_clicked(lambda _event: self._nudge(+self._step))

    def _nudge(self, delta: float) -> None:
        new = self.slider.val + delta
        if self.slider.valmin <= new <= self.slider.valmax:
            self.slider.set_val(new)


class IntTextBox:
    """TextBox accepting positive ints; reverts to the last good value otherwise."""

    def __init__(self, fig: Figure, rect: tuple[float, float, float, float],
                 label: str, initial: int, on_change: Callable[[], None]) -> None:
        self.box = TextBox(fig.add_axes(rect), label, initial=str(initial))
        if matplotlib.__version__.startswith("3.11"):
            # mpl 3.11.0 regression: TextBox._resize is wrapped by
            # _call_with_reparented_event, which reads event.inaxes — absent on
            # ResizeEvent — so every window resize logs an AttributeError.
            # Replace the connection (last one made in TextBox.__init__),
            # keeping the intended stop-typing-on-resize behavior.
            fig.canvas.mpl_disconnect(self.box._cids[-1])
            fig.canvas.mpl_connect("resize_event", lambda _e: self.box.stop_typing())
        self.value = int(initial)
        self._on_change = on_change
        self.box.on_submit(self._submit)

    def _submit(self, text: str) -> None:
        try:
            v = int(text)
            if v <= 0:
                raise ValueError(text)
        except ValueError:
            self.box.set_val(str(self.value))  # re-fires _submit with the old value
            return
        self.value = v
        self._on_change()
