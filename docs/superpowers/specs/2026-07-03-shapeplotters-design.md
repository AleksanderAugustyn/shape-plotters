# ShapePlotters — G5 Design

**Status:** Approved (design)
**Date:** 2026-07-03
**Supersedes:** the two-repo G5 (ShapePlotterBeta + ShapePlotterFoS) in `2026-07-02-parameterization-libraries-design.md` §6. That doc's §4/§6/§9/§10 are amended to match.
**Home:** RewriteProject `docs/` is gitignored — this file stays uncommitted. The ShapePlotters repo carries a committed copy under its own `docs/superpowers/specs/`.

---

## 1. Decision

One repo, `ShapePlotters` (~/PycharmProjects, GitHub AleksanderAugustyn/ShapePlotters), replaces the planned ShapePlotterBeta + ShapePlotterFoS pair. A shared matplotlib engine owns everything parameterization-independent; each parameterization is a render module. `python main.py beta|fos` selects the render per run — no in-GUI switching, no side-by-side view.

Rationale: the plotter family already sprawls (ShapePlotter, ShapePlotterFoS, Cassini, 3D, two fitters, mostly copy-paste descendants); the display contract (arrays to draw + scalars to print) is naturally uniform even though the libraries' math APIs are not. Abstracting at the display boundary is cheap; future parameterizations (Cassini, 3QS) become render modules, not repos.

The two empty placeholder dirs `~/PycharmProjects/ShapePlotterBeta` and `~/PycharmProjects/ShapePlotterFoS` are deleted. The old ShapePlotter and ShapePlotterFoSFitter repos stay untouched as references (layout template, archived PNGs for the gate).

## 2. Repo layout

```
ShapePlotters/
  main.py                # argparse: `python main.py beta|fos` → render → ShapePlotterApp.run()
  pyproject.toml         # numpy, scipy, matplotlib
  README.md              # setup: venv, editable installs, lib discovery
  src/
    core/
      engine.py          # ShapePlotterApp — all parameterization-independent UI
      widgets.py         # slider factory with ± nudge buttons, Z/N textboxes
      quadrature.py      # v1 stopgap: volume, surface, z_cm from profile arrays
      result.py          # ShapeResult dataclass + render-contract documentation
    renders/
      beta.py            # BetaRender — wraps beta_parameterization (Cache)
      fos.py             # FoSRender — wraps fos_parameterization (standalone fns)
  tests/                 # pytest, headless (Agg backend)
```

Structure follows ShapePlotterFoSFitter's `src/` layout minus the math (its `parameterizations/*.py` pure-Python math is replaced by the library packages; its `utilities/converter.py` has no successor — see §3).

## 3. Render contract

Duck-typed (no ABC), documented in `src/core/result.py`. Each render exposes:

- `name: str` — "beta" / "fos".
- `slider_specs: list[SliderSpec]` — (key, label, min, max, default, step). Beta: β₁..β₈ (ranges from old ShapePlotter; its β₉..β₁₂ dropped). FoS: c, a₃..a₈ matching ShapePlotterFoSFitter ranges.
- `toggles: list[ToggleSpec]` — beta: "COM correction" checkbox (selects `Cache.compute_radius_grid` vs `Cache.compute_radius_grid_with_com_shift`; BetaRender holds one `Cache` so slider drags reuse the precomputed Legendre tables); FoS: none.
- `compute(params: dict, toggles: dict) -> ShapeResult` — no Z/N: geometry is R0-unit dimensionless (§ units model).
- `has_extra_panel: bool` — True for FoS (ρ(z) + dρ/dz panel); the engine draws it from ShapeResult arrays, no render draw callback.
- `filename(z: int, n: int, params: dict) -> str` — reproduces each old plotter's PNG naming convention (beta: `Z_N_β1.._β8.png`; FoS: `fos_shape_Z92_N144_c…_a3….png`) so new saves sort next to archived PNGs.

```python
@dataclass
class ShapeResult:
    status: int                        # 0 = valid (packages' Status IntEnum values)
    message: str
    theta: np.ndarray                  # θ grid
    radius: np.ndarray                 # R(θ), R0 units (dimensionless)
    z: np.ndarray                      # profile axis, R0 units
    rho: np.ndarray                    # ρ(z), R0 units
    drho_dz: np.ndarray | None         # lib-native (FoS); None for beta
    neck: tuple[float, float] | None   # (z_neck, rho_neck), R0 units; FoS lib-native,
                                       # beta Python heuristic (see below)
    scalars: dict[str, float]          # lib-native: z_shift, a2 / corrected_beta10, ...
    length_keys: frozenset[str]        # which scalars are lengths (unit toggle applies)
```

**Units model:** everything in `ShapeResult` is in R0 units — renders never scale. The engine owns display units via a fm/R0 toggle: in fm mode it multiplies arrays, neck coordinates, and `length_keys` scalars by R0 = 1.16·A^(1/3) at draw time (r0 = 1.16 fm, both old plotters' value — display convention only, distinct from WMMM's physics constants); quadrature volume/surface/z_cm scale by the matching powers. Axis labels and stats units switch with the toggle. Default is fm (gate compares against the old plotters' archived PNGs, which are in fm). Z/N affect only the fm scale and the save filename — neither library's geometry depends on them.

**Beta neck heuristic** (Python, in BetaRender — display-only, same standing as dR/dθ; graduates to the Fortran lib later if it proves useful): over ρ(θ) = R(θ)·sin(θ) on the lib's θ grid, find the two highest interior local maxima (wherever they fall — no fixed 90° split, so asymmetric shapes with both lobe maxima on one side of 90° still resolve); the neck is the lowest local minimum between them, reported as (z, ρ) with its depth (1 − ρ_neck/ρ_lower-max) shown in stats. No local minimum between the two maxima, or fewer than two interior maxima → no neck. No depth threshold — shallow ripple dips are reported and judged by eye (thresholds are WMMM physics policy, not plotter geometry). FoS keeps its lib-native neck.

Both renders fill both representations — no Python-side coordinate conversion:

- BetaRender is R(θ)-native; derives the profile parametrically (z = R cosθ, ρ = R sinθ).
- FoSRender is ρ(z)-native (`rho_profile`, lib-native dρ/dz, `neck`); gets R(θ) from the lib's own `radius_grid`.

dR/dθ is computed engine-side via `np.gradient` — display only.

**Beta volume fixing** (execution amendment, 2026-07-03): the lib returns unfixed R(θ), but WMMM's grid layer applies a volume-conservation radius multiplier (`radius_grid_mod` `original_volume_factor`) and the old ShapePlotter drew the fixed shape. BetaRender applies the same factor, ((4π/3)/V_shape)^(1/3), before returning and reports it as the `vol_factor` scalar (cross-checked against the old plotter's Radius Fixing Factor golden 0.99598851 for β₂=0.2, β₃=0.1). FoS conserves volume via the a₂ constraint — no fix needed (verified to 5 digits at the gate).

## 4. Engine behavior

**Layout** (old ShapePlotter's gridspec, generalized): top-left R(θ) + dR/dθ (dotted derivative overlay, ShapePlotter style); top-right cross-section (mirrored ρ vs z, equal aspect; vertical neck line at z_neck when `neck` present — FoSFitter style); right side stats textbox; bottom strip sliders + buttons. One optional third plot slot: FoS fills it with ρ(z) + lib-native dρ/dz; beta leaves it empty and the gridspec collapses to two panels. No other layout branching.

**Widgets:** FoSFitter's `create_slider` pattern (Slider + ∓/± nudge buttons), Z/N TextBoxes (defaults 92/144), Reset, Save, fm/R0-units toggle (engine-owned, see §3), per-render CheckButtons from `toggles`. All callbacks funnel into one `update()`.

**Update loop:** widget event → params dict from sliders → `render.compute(...)` → redraw panels, rewrite stats. Stats = `ShapeResult.scalars` (lib-native; beta's neck values marked as the Python heuristic) + neck depth when a neck is present + quadrature volume / surface / z_cm labeled `(py quad)` — v1 stopgap per umbrella §2. All values in the current display unit.

**Invalid shapes:** `status != 0` → draw returned arrays in grey/reduced alpha, banner the Status name + message on the cross-section panel. No hot-path exceptions (package contract, umbrella §5).

**Save:** writes `render.filename(...)` to CWD.

## 5. Environment

- Repo `.venv` (PyCharm WSL interpreter), deps numpy/scipy/matplotlib.
- Editable installs of both sibling packages:
  `pip install -e /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/python`
  `pip install -e /mnt/c/Users/aleks/CLionProjects/Fortran/fos-parameterization/python`
- Shared-library discovery: the packages' `_libloader` (env vars `BETA_PARAM_LIB` / `FOS_PARAM_LIB`, then build-dir fallbacks). README documents both routes.
- First implementation task: WSLg check — open a bare matplotlib window before any engine code (umbrella risk §8).

## 6. Testing and gate

Tests (pytest, Agg, no display):
1. Each render computes a sphere (beta: all β=0; FoS: c=1, a₃..a₈=0) → status 0, expected scalars, no neck reported.
2. `quadrature.py` vs analytic unit-sphere volume/surface/z_cm (R0 units) to tolerance; fm scaling by R0-powers checked once.
3. Beta neck heuristic: a two-lobed shape (e.g. large β₂+β₄) → neck found between the lobe maxima; an asymmetric shape with both lobe maxima on one side of 90° → neck still found (the case the fixed-90° split misses).
4. An invalid shape returns nonzero status without raising; ShapeResult still populated enough to draw.
5. Headless smoke: instantiate `ShapePlotterApp` per render, save a figure to a temp dir (full draw path, no display).

Gate (substance unchanged from umbrella): `main.py beta` and `main.py fos` both launch under WSLg; save a known shape per render and spot-check against archived PNGs in the old ShapePlotter / ShapePlotterFoSFitter repos.

## 7. Non-goals (unchanged from umbrella §2)

No fitting/high-precision ports from ShapePlotterFoSFitter; no Fortran-side volume/surface/z_cm (quadrature is the labeled v1 stopgap); no lib-native beta neck (the Python heuristic is v0.1; graduating it into beta-parameterization → 2.2.0 is recorded future work); no Cassini/3D migration in v0.1; no PyPI packaging; no in-GUI parameterization switching.

## 8. Umbrella-spec amendments (applied to 2026-07-02 doc)

- §4 end-state: `ShapePlotterBeta, ShapePlotterFoS | 0.1` row → `ShapePlotters | 0.1 | shared matplotlib engine + per-parameterization renders over the libraries' Python packages`.
- §6 G5: rewritten to this design (pointer to this doc).
- §9: "G5's two repos are independent — optional parallel agents" → one repo, inline execution.
- §10 playbook step 5: plotter graduates to "a render module in ShapePlotters", not a dedicated repo.
