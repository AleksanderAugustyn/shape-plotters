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


def test_scission_bands_visible_by_default(app) -> None:
    assert len(app.scission_bands) == 2
    assert all(b.get_visible() for b in app.scission_bands)


def test_scission_has_no_checkbox(app) -> None:
    labels = [t.get_text() for t in app.checks.labels]
    assert "scission bands" not in labels


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
    # Slider shape keeps its off-axis COM (red marker); the orange COM-corrected
    # overlay appears with its own marker recentered on the origin.
    assert abs(a.zcm_point.get_xdata()[0]) > 0.01        # slider shape offset (fm)
    assert a.rtheta_upper.get_visible() and a.rtheta_zcm.get_visible()
    assert a.shape_legend.get_visible()
    assert a.rtheta_zcm.get_xdata()[0] == pytest.approx(0.0, abs=1e-4)   # corrected COM centered
    ry = a.rtheta_upper.get_ydata()
    assert ry[0] == 0.0 and ry[-1] == 0.0                # overlay closes at rho = 0
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


def test_fos_rtheta_overlay_when_star_convex_shift() -> None:
    a = ShapePlotterApp(FoSRender())
    a.rows["c"].slider.set_val(2.0)
    a.rows["a3"].slider.set_val(0.2)
    a.rows["a4"].slider.set_val(0.6)                     # asymmetric -> shift != 0
    assert a.last_result.ok
    # True shape is COM-centered; its red marker sits at the origin.
    assert a.zcm_point.get_xdata()[0] == pytest.approx(0.0, abs=1e-9)
    # R(θ) representation drawn where it actually sits, with its own marker.
    assert a.rtheta_upper.get_visible() and a.rtheta_zcm.get_visible()
    assert a.shape_legend.get_visible()                  # legend labels both outlines
    assert abs(a.rtheta_zcm.get_xdata()[0]) > 0.1        # clearly offset (fm)
    ry = a.rtheta_upper.get_ydata()
    assert ry[0] == 0.0 and ry[-1] == 0.0                # overlay closes at rho = 0
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_no_rtheta_overlay_for_symmetric_fos() -> None:
    a = ShapePlotterApp(FoSRender())
    a.rows["c"].slider.set_val(2.0)
    a.rows["a4"].slider.set_val(0.6)                     # symmetric -> shift == 0
    assert a.last_result.ok
    assert not a.rtheta_upper.get_visible()
    assert not a.rtheta_zcm.get_visible()
    assert not a.shape_legend.get_visible()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_no_beta_overlay_when_symmetric() -> None:
    a = ShapePlotterApp(BetaRender())
    a.rows["beta2"].slider.set_val(1.5)                  # even multipole only -> COM centered
    assert a.last_result.ok
    # Symmetric betas: corrected_beta10 ~ 0, so the COM-corrected overlay is
    # suppressed (it would coincide with the slider shape).
    assert a.last_result.overlay_z is None
    assert not a.rtheta_upper.get_visible()
    assert not a.rtheta_zcm.get_visible()
    assert not a.shape_legend.get_visible()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_beta_com_overlay_when_asymmetric() -> None:
    a = ShapePlotterApp(BetaRender())
    a.rows["beta2"].slider.set_val(1.5)
    a.rows["beta3"].slider.set_val(0.4)                  # asymmetric beta -> COM off axis
    assert a.last_result.ok
    # Slider shape's COM is off axis; the COM-corrected overlay is drawn orange.
    assert a.last_result.overlay_z is not None
    assert a.rtheta_upper.get_visible() and a.rtheta_zcm.get_visible()
    assert a.shape_legend.get_visible()
    assert a.rtheta_zcm.get_xdata()[0] == pytest.approx(0.0, abs=1e-4)
    import matplotlib.pyplot as plt
    plt.close(a.fig)


# --- WMMM energy button (stub adapter — no real WMMM anywhere) ---

def _fake_energy_result(**overrides):
    from src.core.energy import EnergyResult
    base = dict(is_valid=True, mass_excess=1.0, total_energy=2.0,
                macro_energy=3.0, micro_energy=4.0, surface_energy=5.0,
                coulomb_energy=6.0, proton_pairing_gap=7.0,
                neutron_pairing_gap=8.0, proton_k=9, neutron_k=10,
                corrected_beta10=0.0, error=None)
    base.update(overrides)
    return EnergyResult(**base)


def test_energy_button_absent_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("src.core.engine.energy.available", lambda: False)
    a = ShapePlotterApp(BetaRender())
    assert a.btn_energy is None
    assert "WMMM" not in a.stats_text.get_text()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_energy_click_appends_block_and_any_change_clears(monkeypatch) -> None:
    monkeypatch.setattr("src.core.engine.energy.available", lambda: True)
    calls = []

    def fake_compute(param_type, z, n, shape, com_correction=True):
        calls.append((param_type, z, n, tuple(shape), com_correction))
        return _fake_energy_result()

    monkeypatch.setattr("src.core.engine.energy.compute", fake_compute)
    a = ShapePlotterApp(BetaRender())
    assert a.btn_energy is not None
    a._on_energy()
    text = a.stats_text.get_text()
    assert "WMMM [MeV]:" in text and "E_total = 2.0000" in text
    assert calls == [("legendre", 92, 144, (0.0,) * 20, False)]  # sphere: 1 request
    a.rows["beta2"].slider.set_val(0.3)          # slider change clears
    assert "WMMM" not in a.stats_text.get_text()
    a._on_energy()
    assert "WMMM" in a.stats_text.get_text()
    a.z_box._submit("94")                        # Z/N change clears too
    assert "WMMM" not in a.stats_text.get_text()
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_energy_two_labeled_blocks_when_overlay(monkeypatch) -> None:
    monkeypatch.setattr("src.core.engine.energy.available", lambda: True)
    monkeypatch.setattr("src.core.engine.energy.compute",
                        lambda *a, **k: _fake_energy_result())
    a = ShapePlotterApp(BetaRender())
    a.rows["beta3"].slider.set_val(0.4)          # asymmetric -> overlay present
    assert a.last_result.overlay_z is not None
    a._on_energy()
    text = a.stats_text.get_text()
    assert "WMMM (slider) [MeV]:" in text
    assert "WMMM (COM corrected) [MeV]:" in text
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_energy_invalid_shape_not_computed(monkeypatch) -> None:
    monkeypatch.setattr("src.core.engine.energy.available", lambda: True)
    calls = []
    monkeypatch.setattr("src.core.engine.energy.compute",
                        lambda *a, **k: calls.append(a) or _fake_energy_result())
    a = ShapePlotterApp(BetaRender())
    a.rows["beta2"].slider.set_val(4.0)          # interior negative -> invalid
    a._on_energy()
    assert "WMMM: shape invalid (not computed)" in a.stats_text.get_text()
    assert calls == []
    import matplotlib.pyplot as plt
    plt.close(a.fig)


def test_energy_error_and_invalid_results_render(monkeypatch) -> None:
    monkeypatch.setattr("src.core.engine.energy.available", lambda: True)
    monkeypatch.setattr("src.core.engine.energy.compute",
                        lambda *a, **k: _fake_energy_result(is_valid=False))
    a = ShapePlotterApp(BetaRender())
    a._on_energy()
    assert "WMMM: invalid shape" in a.stats_text.get_text()
    monkeypatch.setattr("src.core.engine.energy.compute",
                        lambda *a, **k: _fake_energy_result(error="boom"))
    a._on_energy()
    text = a.stats_text.get_text()
    assert "WMMM: error" in text and "boom" in text
    import matplotlib.pyplot as plt
    plt.close(a.fig)
