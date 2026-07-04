# N_GRID Retirement + WMMM Energy Button — Design

**Status:** Approved (design)
**Date:** 2026-07-04
**Builds on:** `2026-07-04-gl-native-grids-design.md` — realizes its §10 "WMMM energy button" future work (isolation contract carried over verbatim) and closes the "constructor's `n_grid` argument retained until the library's 2.3.0 cleanup" note in its §6. Also amends the COM-corrected overlay trigger added after that spec (commit bb37419; see §4 here).

---

## 1. Goal

Two follow-ups to the GL-native migration:

1. **N_GRID retirement.** ShapePlotters no longer passes any uniform-grid size. The beta `Cache` is constructed node-set-only — the same init path WMMM's GL-native usage needs. The library keeps its uniform-grid features; they just become optional.
2. **WMMM energy button.** An optional, strictly-local Energy button shows WMMM energies for the current shape(s). Present only when `wmmm` is importable in the venv; the repo carries calling convention only.

Plus one correctness fix folded in because the energy feature exposes it: the beta COM-overlay trigger compares against the slider β₁ (§4).

## 2. Prerequisite: beta-parameterization 2.3.0 — optional `n_grid` (blocking, part of this project)

The cache's uniform θ-table (`legendre_theta_grid`) feeds only the legacy `radius_grid`-family entry points; the GL-native path (`resolve_shape` → `compute_radius_and_derivative` over caller `NodeSet`s) never touches it. Making the table optional is additive — no math changes, no golden changes, nothing deleted:

- Fortran: `cache_init_s(max_beta_params, [n_grid], error_code, message)` — `n_grid` becomes `optional`. Absent ⇒ the uniform table is not allocated. Legacy uniform entry points on a table-less cache return a clean error code (new `LEGENDRE_ERROR_NO_UNIFORM_GRID` or equivalent), never touch unallocated storage.
- C API: `beta_param_cache_create` keeps its signature; `n_grid <= 0` is the documented "no uniform table" sentinel.
- Python: `Cache(max_beta_params, n_grid=None)` — `None` maps to the sentinel.
- Tests: node-set results from a table-less cache equal those from an `n_grid` cache; legacy call on a table-less cache errors cleanly; existing goldens byte-identical.
- Release: tag **2.3.0**; rebuild **both** `build/` and `build/release` before ShapePlotters work starts (the Python loader prefers `build/release`; a stale ABI there segfaults).

fos-parameterization needs nothing — its `N_GRID` in the plotter is not a library argument. WMMM's own init (legacy 7201 table) is out of scope (§9).

## 3. ShapePlotters grid cleanup

- `beta.py`: delete `N_GRID` and its "goes away with 2.3.0" comment; construct `bp.Cache(max_beta_params=N_BETAS)`.
- `fos.py`: rename `N_GRID` → `N_PROFILE_POINTS` (value stays 721). It is the ρ(z) display-panel resolution — the native COM-frame profile — not a calculation grid, and GL θ-nodes cannot replace it (they sample the star-convex-shifted frame). Comment says so.
- Gate: `grep -rn "n_grid\|N_GRID" src/` returns nothing. (`N_RHO_GRID = 7201` stays — WMMM-bit-compatible `shape()` validity grid.)

## 4. Beta overlay trigger: compare against the slider β₁

β₁₀ is a genuine shape parameter, not a COM-translation knob: (β₂ = 0.7, β₁₀ = 0) and (β₂ = 0.7, β₁₀ = 1) are two different shapes, not z-translates. The current trigger `|corrected_beta10| > 0.001` therefore hides the orange COM-corrected overlay whenever the *corrected* dipole is small, even if the slider shape carries a large β₁ — e.g. a reflection-symmetric shape with slider β₁ = 0.5 shows no overlay although the corrected shape differs substantially.

Fix: the overlay (and everything keyed to it, including the second energy request in §6) triggers on

```
|corrected_beta10 − slider β₁| > OVERLAY_BETA10_THRESHOLD   (0.001, unchanged)
```

The threshold constant, its docstring, and the module docstring's overlay description are updated to match. Overlay tests re-baselined: symmetric shape + slider β₁ = 0.5 now shows the overlay; asymmetric shape with slider β₁ set to its corrected value hides it.

## 5. Energy adapter: `src/core/energy.py` — the only file that may import `wmmm`

- `available() -> bool` via `importlib.util.find_spec("wmmm")` — no import, no library load at startup.
- `compute(param_type, z, n, shape, com_correction=True) -> EnergyResult` — first call imports `wmmm` inside `try/except`; `Model` instances cached per `(param_type, com_correction)` for the app lifetime (~17–19 ms per point thereafter). The COM-off variant is configured once at creation via `Model.set("beta_10_com_shift", 0.0)`.
- `EnergyResult`: local frozen dataclass — `is_valid`, `total_energy`, `macro_energy`, `micro_energy`, `mass_excess`, `surface_energy`, `coulomb_energy`, `proton_pairing_gap`, `neutron_pairing_gap`, `proton_k`, `neutron_k`, `corrected_beta10`, plus `error: str | None` for import/runtime failures (all-zero fields then). `corrected_beta10` is carried for the §8 cross-check, not displayed.
- Isolation contract (from the GL spec §10, unchanged): `wmmm` never appears in `pyproject.toml` — not even as an optional extra; installed only out-of-band as a local editable install; the repo carries calling convention only — no model code, data, or paths.

## 6. Render contract: `energy_requests`

```python
@dataclass(frozen=True)
class EnergyRequest:          # src/core/result.py — no wmmm dependency
    label: str                # stats-block header; mirrors the plot legend
    param_type: str           # "legendre" | "fos"
    shape: tuple[float, ...]
    com_correction: bool

def energy_requests(self, params, result) -> list[EnergyRequest]
```

- **Beta:** always one request for the blue slider shape — `("slider", "legendre", betas zero-padded to 20, com_correction=False)`. When the orange overlay is present (`result.overlay_z is not None`, i.e. the §4 trigger fired), a second request `("COM corrected", "legendre", same padded betas, com_correction=True)` — WMMM ignores the slider β₁ and recomputes β₁₀ from β₂..β₈ exactly as the render's `resolve_shape(apply_com_correction=True)` does, so no second shape vector exists.
- **FoS:** always exactly one request — `("FoS", "fos", (c, a3..a8), com_correction=True)`; the flag is inert on the FoS path. The FoS dashed overlay is the same shape in a different frame — one energy. This is why the 1-vs-2 decision lives in the render, not in the engine's overlay state.
- Energies are frame- and unit-toggle-independent (MeV); requests carry raw slider values, never display-scaled ones.

## 7. Engine

- Button "Energy" under Save, created only when `energy.available()` — absent otherwise, layout unchanged.
- Click → for each `render.energy_requests(params, result)`: `energy.compute(...)` → one labeled block appended to the stats box: total, macro, micro, mass excess, surface, Coulomb energies (MeV) + proton/neutron pairing gaps and k-values. Single-request case drops the label decoration. Worst case two computes ≈ 40 ms per click.
- **Clear-on-change:** any slider, toggle, or Z/N change wipes the energy block (one flag checked in `update()`). The stats box never pairs an energy with a shape it doesn't describe — which makes the Save caveat from the GL spec §10 a feature: Energy then Save yields a PNG with a *consistent* energy annotation.
- Errors, in order of checks: plotter shape invalid → "WMMM: shape invalid (not computed)", no compute call; `EnergyResult.error` set (import/runtime failure) → the error text; `is_valid == False` → "WMMM: invalid shape". The app never crashes from the energy path.

## 8. Testing and gate

1. beta-parameterization 2.3.0: table-less cache node-set results equal `n_grid`-cache results; legacy entry point on table-less cache → clean error; C sentinel `n_grid = 0`; Python `Cache(max_beta_params=8)` round-trip. Existing goldens byte-identical.
2. Grid gate: `grep -rn "n_grid\|N_GRID" src/` empty; both renders unchanged on the GL goldens (radii, vol factor, scalars).
3. Overlay trigger (§4): symmetric shape + slider β₁ = 0.5 ⇒ overlay present; asymmetric shape with slider β₁ = corrected β₁₀ ⇒ overlay absent; boundary at the 0.001 difference.
4. `energy_requests`: beta ⇒ 1 request (com off) without overlay, 2 with (labels "slider" / "COM corrected", both shapes zero-padded to length 20); FoS ⇒ always 1 request of length 7. Raw slider values regardless of unit toggle.
5. Adapter with a stub `wmmm` injected via `sys.modules`: `available()` truth table; lazy import (no import before first `compute`); model caching per `(param_type, com_correction)`; `set("beta_10_com_shift", 0.0)` called exactly once for the COM-off model; import failure ⇒ `EnergyResult.error`, no exception.
6. Engine with stub: button present/absent; click appends blocks; every param/toggle/Z/N change clears them; error-path messages of §7. Committed tests carry **no WMMM-derived numbers** (policy: the public repo stays free of model outputs).
7. Local-only smoke test, `@pytest.mark.skipif(not energy.available(), ...)`: real compute for one asymmetric beta shape, both variants — asserts `is_valid`, finite fields, and WMMM's `corrected_beta10` equals the render's `scalars["corrected_beta10"]` (two independent paths through the same library); one FoS compute `is_valid`. Assertions structural — no numeric goldens committed.
8. Perf: energy computes happen only on click; the drag path is untouched — existing perf baselines re-run unchanged.

Gate: `main.py beta` and `main.py fos` under WSLg in the default venv (wmmm present ⇒ button visible; energies plausible for e.g. Z=92, N=144; two blocks when the orange overlay shows); headless stub check covers the button-absent case; Energy → Save PNG spot-checked.

## 9. Non-goals

- Removing uniform-grid features from the libraries — they stay, opt-in via `n_grid`.
- WMMM repo changes: its legacy 7201 cache init stays (H-project's own business); `beta_10_com_shift` already exists end-to-end.
- fos-parameterization changes (none needed).
- Live/auto energy recomputation (17–19 ms per point would roughly double drag latency); batch/PES features; energy display in unit-toggle units (always MeV).
- Publishing anything WMMM: `pyproject.toml` untouched; no optional extra; no model numbers in committed tests.
