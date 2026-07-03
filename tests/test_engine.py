"""Engine smoke tests under Agg: full draw path, unit toggle, invalid greying."""
import numpy as np
import pytest

from src.core.engine import R0_FM, ShapePlotterApp
from src.renders.beta import BetaRender
from src.renders.fos import FoSRender


@pytest.fixture(params=[BetaRender, FoSRender], ids=["beta", "fos"])
def app(request):
    a = ShapePlotterApp(request.param())
    yield a
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_smoke_draw_and_save(app, tmp_path) -> None:
    key = "beta2" if app.render.name == "beta" else "c"
    app.rows[key].slider.set_val(1.5)   # triggers update()
    out = tmp_path / "smoke.png"
    app.fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_unit_toggle_scales_and_relabels(app) -> None:
    a = app.z_box.value + app.n_box.value
    assert app._scale() == pytest.approx(R0_FM * a ** (1.0 / 3.0))
    assert "[fm]" in app.ax_shape.get_xlabel()
    app._on_check("fm units")           # -> R0 units
    assert app._scale() == 1.0
    assert "[R0]" in app.ax_shape.get_xlabel()


def test_invalid_shape_greys_and_banners() -> None:
    a = ShapePlotterApp(BetaRender())
    a.rows["beta2"].slider.set_val(4.0)  # interior negative -> invalid
    assert not a.last_result.ok
    assert "ERROR_INTERIOR_NEGATIVE" in a.ax_shape.get_title()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_neck_line_visibility_tracks_result() -> None:
    a = ShapePlotterApp(BetaRender())
    assert a.last_result.neck is None          # defaults: sphere, no neck
    assert not a.neck_line.get_visible()
    a.rows["beta2"].slider.set_val(1.5)        # necked shape (verified)
    assert a.last_result.neck is not None
    assert a.neck_line.get_visible()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_widget_draws_suppressed(app) -> None:
    widgets = [app.btn_reset, app.btn_save, app.checks]
    for row in app.rows.values():
        widgets += [row.slider, row.btn_dec, row.btn_inc]
    on = [w for w in widgets if getattr(w, "drawon", False)]
    assert not on, f"widgets still self-drawing: {on}"


def test_save_uses_render_filename(app, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app._save()
    expected = app.render.filename(
        app.z_box.value, app.n_box.value,
        {k: r.slider.val for k, r in app.rows.items()})
    assert (tmp_path / expected).exists()
