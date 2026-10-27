"""The A slider supplies the mass number; Z and N exist only for the WMMM overlay."""
import re

import pytest

from shape_plotters.core.engine import R0_FM, ShapePlotterApp
from shape_plotters.core.widgets import SliderRow
from shape_plotters.renders.beta import BetaRender
from shape_plotters.renders.fos import FoSRender
from tests.test_engine import _fake_energy_result

ENGINE_ENERGY = "shape_plotters.core.engine.energy"


def _close(app) -> None:
    import matplotlib.pyplot as plt
    plt.close(app.fig)


def _band_top(band) -> float:
    if hasattr(band, "get_height"):            # Rectangle (mpl >= 3.8)
        return band.get_y() + band.get_height()
    return max(xy[1] for xy in band.get_xy())  # Polygon fallback


def _with_wmmm(monkeypatch, calls: list | None = None) -> None:
    monkeypatch.setattr(f"{ENGINE_ENERGY}.available", lambda: True)

    def fake_compute(param_type, z, n, shape, com_correction=True):
        if calls is not None:
            calls.append((z, n))
        return _fake_energy_result()

    monkeypatch.setattr(f"{ENGINE_ENERGY}.compute", fake_compute)


@pytest.fixture(params=[BetaRender, FoSRender], ids=["beta", "fos"])
def app(request, monkeypatch):
    monkeypatch.setattr(f"{ENGINE_ENERGY}.available", lambda: False)
    a = ShapePlotterApp(request.param())
    yield a
    _close(a)


def test_a_slider_is_an_integer_slider_row(app) -> None:
    assert isinstance(app.a_row, SliderRow)
    assert app.mass_number == 236
    assert app.a_row.slider.valstep == 1
    assert (app.a_row.slider.valmin, app.a_row.slider.valmax) == (16, 300)


def test_a_slider_drives_fm_scale(app) -> None:
    assert app._scale() == pytest.approx(R0_FM * 236 ** (1.0 / 3.0))
    app.a_row.slider.set_val(240)
    assert app.mass_number == 240
    assert app._scale() == pytest.approx(R0_FM * 240 ** (1.0 / 3.0))


def test_a_slider_moves_scission_bands_in_r0_mode(app) -> None:
    app._on_check("fm units")                                      # -> R0 units
    before = _band_top(app.scission_bands[0])
    app.a_row.slider.set_val(240)
    after = _band_top(app.scission_bands[0])
    assert after == pytest.approx(1.5 / (R0_FM * 240 ** (1.0 / 3.0)))
    assert after != before


def test_fragment_mass_split_uses_slider_a(monkeypatch) -> None:
    monkeypatch.setattr(f"{ENGINE_ENERGY}.available", lambda: False)
    a = ShapePlotterApp(BetaRender())
    a.a_row.slider.set_val(240)
    a.rows["beta2"].slider.set_val(1.5)                  # necked shape (verified)
    mass = re.search(r"mass A   = (\d+) : (\d+)  \(A=(\d+)\)", a.stats_text.get_text())
    assert mass is not None and int(mass.group(3)) == 240
    _close(a)


def test_save_filename_carries_a(app, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app.a_row.slider.set_val(240)
    app._save()
    expected = app.render.filename(240, {k: r.slider.val for k, r in app.rows.items()})
    assert "A240" in expected
    assert (tmp_path / expected).exists()


def test_a_row_draws_are_suppressed(app) -> None:
    widgets = [app.a_row.slider, app.a_row.btn_dec, app.a_row.btn_inc]
    assert not [w for w in widgets if getattr(w, "drawon", False)]


def test_z_box_absent_without_wmmm(app) -> None:
    assert app.z_box is None
    assert not hasattr(app, "n_box")                      # N is A - Z, never an input


def test_z_box_present_with_wmmm_and_n_follows_from_a(monkeypatch) -> None:
    calls: list = []
    _with_wmmm(monkeypatch, calls)
    a = ShapePlotterApp(BetaRender())
    assert a.z_box is not None
    assert a.z_box.value == 92
    a.z_box._submit("94")
    a.a_row.slider.set_val(240)
    a._on_energy()
    assert calls == [(94, 146)]                           # N = A - Z
    assert a.mass_number == 240                           # Z never moves the slider
    _close(a)


def test_z_box_sits_in_the_right_column_beside_the_energy_button(monkeypatch) -> None:
    _with_wmmm(monkeypatch)
    a = ShapePlotterApp(BetaRender())
    energy_box = a.btn_energy.ax.get_position()
    pos = a.z_box.box.ax.get_position()
    assert pos.x0 >= 0.8                                  # right column, not the slider stack
    assert abs(pos.y0 - energy_box.y0) < 0.12             # next to the button, not up at the old slot
    _close(a)


def test_energy_block_names_the_nucleus(monkeypatch) -> None:
    _with_wmmm(monkeypatch)
    a = ShapePlotterApp(BetaRender())
    a._on_energy()
    assert "Z = 92  N = 144" in a.stats_text.get_text()
    _close(a)


def test_energy_refuses_z_not_below_a(monkeypatch) -> None:
    calls: list = []
    _with_wmmm(monkeypatch, calls)
    a = ShapePlotterApp(BetaRender())
    a.a_row.slider.set_val(92)                            # A = Z -> N = 0
    a._on_energy()
    assert calls == []
    assert "WMMM: Z = 92 is not below A = 92" in a.stats_text.get_text()
    _close(a)


def test_a_change_clears_energy_lines(monkeypatch) -> None:
    _with_wmmm(monkeypatch)
    a = ShapePlotterApp(BetaRender())
    a._on_energy()
    assert "WMMM" in a.stats_text.get_text()
    a.a_row.slider.set_val(240)
    assert "WMMM" not in a.stats_text.get_text()
    _close(a)
