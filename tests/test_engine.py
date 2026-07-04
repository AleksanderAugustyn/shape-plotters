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


def _band_top(band) -> float:
    if hasattr(band, "get_height"):            # Rectangle (mpl >= 3.8)
        return band.get_y() + band.get_height()
    return max(xy[1] for xy in band.get_xy())  # Polygon fallback


def test_scission_bands_default_hidden(app) -> None:
    assert len(app.scission_bands) == 2
    assert not any(b.get_visible() for b in app.scission_bands)


def test_scission_toggle_flips_visibility(app) -> None:
    app._on_check("scission bands")
    assert all(b.get_visible() for b in app.scission_bands)
    app._on_check("scission bands")
    assert not any(b.get_visible() for b in app.scission_bands)


def test_scission_bands_rescale_on_unit_toggle(app) -> None:
    assert _band_top(app.scission_bands[0]) == pytest.approx(1.5)  # fm mode
    app._on_check("fm units")                                      # -> R0 units
    a = app.z_box.value + app.n_box.value
    expected = 1.5 / (R0_FM * a ** (1.0 / 3.0))
    assert _band_top(app.scission_bands[0]) == pytest.approx(expected)


def test_save_uses_render_filename(app, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app._save()
    expected = app.render.filename(
        app.z_box.value, app.n_box.value,
        {k: r.slider.val for k, r in app.rows.items()})
    assert (tmp_path / expected).exists()


def test_dr_overlay_uses_lib_derivative(app) -> None:
    expected = app.last_result.dr_dtheta * app._scale()
    np.testing.assert_array_equal(app.dr_line.get_ydata(), expected)


def test_cross_section_closes_at_poles() -> None:
    a = ShapePlotterApp(BetaRender())
    a.rows["beta2"].slider.set_val(1.5)
    xy = a.shape_upper.get_xydata()
    scale = a._scale()
    assert xy[0][1] == 0.0 and xy[-1][1] == 0.0          # rho = 0 at both ends
    # Beta z runs from +north to -south (theta ascending -> cos descending).
    assert xy[0][0] == pytest.approx(a.last_result.r_north * scale)
    assert xy[-1][0] == pytest.approx(-a.last_result.r_south * scale)
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_zcm_marker_tracks_com() -> None:
    a = ShapePlotterApp(BetaRender())
    assert a.zcm_point.get_visible()
    assert a.zcm_point.get_ydata()[0] == 0.0             # point sits on the z axis
    assert a.zcm_point.get_xdata()[0] == pytest.approx(0.0, abs=1e-9)   # sphere
    a.rows["beta3"].slider.set_val(0.4)                  # octupole asymmetry, COM off
    assert abs(a.zcm_point.get_xdata()[0]) > 0.01        # clear offset (fm), COM off
    a._on_check("COM correction")                        # COM on -> back to ~0
    assert a.zcm_point.get_xdata()[0] == pytest.approx(0.0, abs=1e-4)
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_zcm_marker_on_fos_profile_panel() -> None:
    a = ShapePlotterApp(FoSRender())
    a.rows["a3"].slider.set_val(0.3)                     # asymmetric; z-shift centers COM
    assert a.zcm_point.get_visible()
    assert a.zcm_point.get_xdata()[0] == pytest.approx(0.0, abs=1e-3)
    assert a.zcm_extra is not None and a.zcm_extra.get_visible()
    assert a.zcm_extra.get_xdata()[0] == a.zcm_point.get_xdata()[0]
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_zcm_marker_hidden_on_invalid() -> None:
    a = ShapePlotterApp(BetaRender())
    a.rows["beta2"].slider.set_val(4.0)                  # invalid shape
    assert not a.zcm_point.get_visible()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_stats_labeled_gl(app) -> None:
    text = app.stats_text.get_text()
    assert "(GL)" in text
    assert "(py quad)" not in text
