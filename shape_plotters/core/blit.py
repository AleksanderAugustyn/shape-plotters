"""Background cache for blitted slider drags.

During a drag only a few artists change: the data lines, the stats text and
the dragged slider. DragBlitter keeps a copy of everything else (the static
layer) and composes each frame from that copy plus the dynamic artists,
instead of redrawing the whole figure.

It knows pixels, not shapes. What counts as a change to the static layer
(view limits, titles, legends) is the engine's policy; the engine reports it
through invalidate().
"""
from __future__ import annotations

from collections.abc import Sequence

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class DragBlitter:
    """Owns the static background and the ``animated`` flags for one drag.

    Parameters
    ----------
    fig : Figure
        The figure to compose frames on. Its canvas must support blitting.
    dynamic : Sequence[Artist]
        Artists that change on every frame of a drag.
    """

    def __init__(self, fig: Figure, dynamic: Sequence[Artist]) -> None:
        self._fig = fig
        self._dynamic = tuple(dynamic)
        self._slider_ax: Axes | None = None
        self._background = None
        fig.canvas.mpl_connect("draw_event", self._on_draw)
        # A window resize only schedules its redraw; a motion event handled
        # before it must not blit the old-size background.
        fig.canvas.mpl_connect("resize_event", lambda _event: self.invalidate())

    @property
    def active(self) -> bool:
        """True between begin() and end()."""
        return self._slider_ax is not None

    @property
    def ready(self) -> bool:
        """True when a background matching the current static layer is held."""
        return self._background is not None

    def begin(self, slider_ax: Axes) -> None:
        """Enter drag mode; full draws now leave the dynamic layer to us.

        The whole slider axes is animated, not its handle and value text
        individually: no private Slider attributes, and the value text cannot
        overprint the copy baked into the background.
        """
        self._slider_ax = slider_ax
        self._background = None
        for artist in (*self._dynamic, slider_ax):
            artist.set_animated(True)

    def end(self) -> None:
        """Leave drag mode; the next full draw is an ordinary one."""
        if self._slider_ax is None:
            return
        for artist in (*self._dynamic, self._slider_ax):
            artist.set_animated(False)
        self._slider_ax = None
        self._background = None

    def invalidate(self) -> None:
        """The static layer changed; the next full draw recaptures it."""
        self._background = None

    def blit(self) -> None:
        """Compose one frame: the saved background plus the dynamic layer."""
        canvas = self._fig.canvas
        canvas.restore_region(self._background)
        self._draw_dynamic()
        canvas.blit(self._fig.bbox)

    def _on_draw(self, event) -> None:
        # Runs at the end of every full draw. In drag mode that draw skipped
        # the animated artists, so the canvas holds exactly the static layer.
        if self._slider_ax is None:
            return
        if self._fig.canvas.is_saving():
            # An export (savefig), not a screen draw. It already drew the
            # animated artists, except whole animated axes: add the slider.
            # It may also have re-rendered the canvas at another size or on
            # another canvas class, so the held background is no longer valid.
            self._slider_ax.draw(event.renderer)
            self._background = None
            return
        self._background = self._fig.canvas.copy_from_bbox(self._fig.bbox)
        self._draw_dynamic()

    def _draw_dynamic(self) -> None:
        for artist in self._dynamic:
            self._fig.draw_artist(artist)
        self._fig.draw_artist(self._slider_ax)
