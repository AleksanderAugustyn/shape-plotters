# ShapePlotters on GL-Native Grids — Design

**Status:** Approved (design)
**Date:** 2026-07-04
**Builds on:** `2026-07-03-shapeplotters-design.md` (v0.1 design; its §3 render contract and §6 stopgap labels are amended here) and RewriteProject's `2026-07-03-gl-native-grid-libraries-design.md` (the H1/H2 library APIs this consumes — both already implemented, tagged, and installed).
**Closes out:** the `numpy.gradient` dR/dθ stopgap and the "(py quad)" trapezoid quadrature — both recorded as future work in the v0.1 spec and as a non-goal in the GL-native library spec (§6).

---

## 1. Goal

ShapePlotters adopts the same GL-native shape evaluation as the energy model (WMMM): library-exact analytic dR/dθ, GL-2048-in-cos(θ) dense node set, spectrally exact quadrature, analytic pole radii, and the WMMM volume-factor scheme. After this migration, ShapePlotters has zero callers of the legacy uniform-grid entry points, unblocking the H-project's final library cleanup tags (beta 2.3.0 / fos 1.2.0).

What changes, concretely:

| Today (v0.1) | After |
|---|---|
| Uniform 721-point θ grid (`bp.theta_grid`/`fp.theta_grid`), poles included | GL-2048 nodes in x = cos(θ) — WMMM's dense set exactly; poles closed with analytic radii |
| dR/dθ via `np.gradient` (engine display + surface integrand) | Lib-exact analytic dR/dθ (`RadiusDerivativeResult.dr_dtheta`) |
| `quadrature.py` trapezoid, labeled "(py quad)" stopgap | GL weighted dot products, spectrally exact — same scheme as WMMM |
| Beta volume factor from trapezoid volume (approximate) | GL-sum volume factor, applied to R, dR/dθ, and pole radii (WMMM §4.3 step 3) |
| Beta via `Cache.radius_grid` / `radius_grid_with_com_shift` | `Cache.resolve_shape` + `Cache.radius_and_derivative` over a `NodeSet` |
| FoS via `fp.radius_grid` | `fp.shape` + `fp.radius_and_derivative` |

GL-in-x nodes are near-uniform in θ (arcsine node density in x cancels the arccos mapping), so the display curve and the index-based neck heuristic behave like today's uniform grid. Every quadrature integrand here (R³sinθ, R sinθ√(R²+R′²), R⁴sinθcosθ) carries a sinθ factor, which the x = cos(θ) substitution absorbs — every integral is a plain weighted dot product.

## 2. Prerequisite: beta-parameterization 2.2.x patch (blocking, part of this project)

The Fortran `cache_resolve_shape_s` already has `apply_com_correction` (optional, default `.true.`), but the C API wrapper drops it, so Python can't reach it. Patch release:

- `beta_param_cache_resolve_shape` gains an `apply_com_correction` int flag (nonzero = true).
- Python `Cache.resolve_shape(params, apply_com_correction=True)` kwarg.
- No Fortran math changes. Existing goldens untouched; one new binding test (flag off ⇒ `corrected_beta10` equals input beta10 and matches `radius_grid`'s no-COM radii at the uniform grid's own thetas).

fos-parameterization needs nothing — `fp.shape` + `fp.radius_and_derivative` (1.1.0) already cover the contract.

## 3. Shared node module: `src/core/nodes.py`

Mirrors WMMM's `program_run_time_constants_mod`: shape-independent node data computed once at import, module-level, read-only.

```python
x, w = numpy.polynomial.legendre.leggauss(2048)   # GL-2048 in x = cos(theta)
theta = np.arccos(x)[::-1]                        # ascending theta; w reversed to match
```

Ascending-θ ordering keeps the arrays drop-in for every existing consumer (plots, neck indices). Both renders and `quadrature.py` consume this module; `bp.theta_grid`/`fp.theta_grid` are no longer called anywhere.

## 4. `quadrature.py`: trapezoid → GL dot products

`volume`, `surface_area`, `z_cm` become weighted sums over the node module's weights:

- `volume = (2π/3) · Σ wᵢ Rᵢ³`
- `surface_area = 2π · Σ wᵢ Rᵢ √(Rᵢ² + R′ᵢ²)` — `surface_area(theta, radius, dr_dtheta)` takes the lib-exact derivative as an argument; the internal `np.gradient` call is deleted.
- `z_cm = (π/2) · Σ wᵢ Rᵢ⁴ cosθᵢ / V`

The module docstring's "v1 stopgap" label and the engine's "(py quad)" stats label go away — the stats label becomes "(GL)". Signatures keep the `theta` argument for interface stability, but the weights come from `nodes.py` (functions assume the shared node set; that's the only grid in the app).

## 5. Render contract: `ShapeResult` gains three fields

```python
dr_dtheta: np.ndarray   # lib-exact, volume-factor-scaled where applicable (beta)
r_north: float          # analytic R(0), same scaling
r_south: float          # analytic R(pi), same scaling
```

GL nodes are an open rule — θ = 0/π are never nodes — so pole values must arrive separately, exactly as WMMM's pole-value consumers work (GL-native library spec §4.4). All three follow the existing units model: R0 units in the result, engine scales for fm display.

On invalid shapes (`status != 0`) the libraries zero-fill outputs; `dr_dtheta` and the pole radii are zero-filled alongside `radius`, and the engine's existing greyed-draw path is unchanged.

## 6. BetaRender

- `__init__`: one `Cache` (constructor's `n_grid` argument retained until the library's 2.3.0 cleanup removes the uniform path; no uniform-grid feature is used) + one `NodeSet` built from the shared θ via `cache.build_node_set`.
- `compute`: `resolve_shape(betas, apply_com_correction=toggles["com"])` → on success `radius_and_derivative(res.beta_con, node_set)`.
- Volume factor: `((4π/3) / ((2π/3)·Σ wᵢRᵢ³))^(1/3)` from the GL sum — exact, same role and placement as WMMM §4.3 step 3 — applied uniformly to `radius`, `dr_dtheta`, `r_north`, `r_south`. Radii arrive pre-scaled at every consumer, preserving the existing contract.
- Scalars unchanged: `vol_factor`, `corrected_beta10`. COM toggle survives with default off (old-PNG parity).
- Neck heuristic unchanged: index-based over ρ(θ) = R sinθ on the GL nodes.

## 7. FoSRender

- `compute`: `fp.shape(params, 7201)` — 7201 is WMMM's `N_FOS_RHO_GRID_POINTS`, so `z_shift` is bit-comparable — then `fp.radius_and_derivative(params, theta, z_shift)`.
- `r_north`/`r_south` come from `fp.shape` directly. No volume factor (a₂ constraint conserves volume).
- `rho_profile(params, 721)` stays for the ρ(z) + dρ/dz panel — native representation, not part of the radius-grid migration. Lib-native neck (`fp.neck`) unchanged.

## 8. Engine

- dR/dθ overlay uses `result.dr_dtheta` directly; the `np.gradient` call is deleted. fm mode scales it by R0 like other lengths.
- Cross-section closure is per-render, decided by whether the outline already reaches ρ = 0. Beta's R(θ) parametric outline is open at the GL poles (θ = 0/π are never nodes), so the engine appends (z = +r_north, ρ = 0) and (z = −r_south, ρ = 0) at draw time. The FoS ρ(z) profile already terminates at ρ = 0 at the tips and is drawn as-is — appending the star-convex-frame poles there would draw a spurious ρ = 0 segment (the profile is COM-centered, the poles are not; see the overlay bullet).
- Stats quadrature calls pass `dr_dtheta` through to `surface_area`.
- z_cm marker: red point at (z_cm, 0) on the cross-section panel (and the FoS ρ(z) panel, which shares the z axis) — the shapes are axially symmetric, so the COM lies on the z axis. Uses `result.z_cm`, the true-shape COM in the display frame: for beta that is the GL-quadrature z_cm of R(θ); for FoS it is 0, because the FoS shape is COM-centered by definition. Beta with COM correction off shows the actual offset; COM on and FoS show z ≈ 0. Persistent artist, updated per draw.
- R(θ) star-convex overlay (FoS, non-zero star-convexity shift). Converting a FoS shape to R(θ) requires shifting it to a star-convex origin (`shape().z_shift`), which breaks COM-centering — z_cm(R) ≠ 0 while the true shape (the ρ(z) profile) stays COM-centered. When z_cm(R) ≠ `result.z_cm`, the engine overlays the R(θ) reconstruction *where it actually sits* (not recentered) in a distinct dashed style, with its own z_cm marker at z_cm(R), plus a legend labelling both outlines. Beta's R(θ) is itself the true shape, so the frames coincide and the overlay stays hidden. Left-panel R(θ)/dR/dθ always show the true star-convex curves.

## 9. Testing and gate

1. Sphere (both renders): volume/surface/z_cm vs analytic values at ~1e-13 (GL exact; tightened from trapezoid tolerance), `dr_dtheta` ≈ 0, `r_north = r_south = 1`.
2. Deformed beta shape: lib `dr_dtheta` agrees with `np.gradient` of the returned radii to display tolerance (~1e-3 relative) — wiring check only; analytic-derivative goldens live in the libraries.
3. Beta vol-factor golden re-baselined (trapezoid → exact GL shifts trailing digits of 0.99598851) and cross-checked against WMMM's exact GL value for the same shape.
4. COM toggle: off ⇒ `corrected_beta10 == beta1` input; on ⇒ matches the old `radius_grid_with_com_shift` value.
5. FoS: `z_shift`/`a2` scalars unchanged vs v0.1 (same library math, same 7201 grid); pole radii match `c + z_shift` / `|z_shift − c|`.
6. Invalid shape: zero-filled arrays including `dr_dtheta`/poles, greyed draw path intact, no exceptions.
7. Cross-section closure: beta's drawn outline begins/ends at ρ = 0 with the analytic pole z-values; the FoS profile is drawn as-is (already ρ = 0 at the tips — no appended poles, no spurious tip segment).
8. z_cm marker: point present at (`result.z_cm`, 0) on the cross-section panel; at z ≈ 0 for beta with COM on and for FoS (COM-centered by definition); at the quadrature z_cm value for beta with COM off (headless draw-path check).
10. FoS R(θ) overlay: for a non-zero star-convexity shift (asymmetric FoS), the R(θ) reconstruction is drawn where it actually sits with its own z_cm marker at z_cm(R) and a visible legend; hidden for symmetric FoS and for beta (frames coincide).
9. Existing perf tests re-run: per-drag work is dot products; 2048-point lines vs 721 is negligible for matplotlib. No perf regression vs the engine v0.2 baselines.

Gate: `main.py beta` and `main.py fos` under WSLg; saved PNGs for known shapes spot-checked against v0.1 output (shapes visually identical; vol factor differs only in trailing digits).

## 10. Non-goals

- Fortran-side volume/surface/z_cm (quadrature stays Python — now exact, no longer a stopgap).
- Graduating the beta neck heuristic into the library (unchanged future work).
- Cassini/3D renders, PyPI packaging, in-GUI parameterization switching (unchanged from v0.1).
- Multiple node sets (WMMM's folding/coulomb sets serve energy integrals the plotter doesn't compute — one dense set suffices).
- Runtime-selectable node count; 2048 is a compile-time-style constant in `nodes.py`.
- WMMM energy button (recorded future work, feasibility confirmed 2026-07-04). `wmmm.Model(param_type).compute(z, n, shape)` maps 1:1 onto both renders' params plus the engine's Z/N boxes; a button-triggered call would show `PointResult` energies in the stats box. Isolation contract if built: `wmmm` imported behind an `importlib.util.find_spec` guard (button absent when not installed), never listed in `pyproject.toml` — not even as an optional extra — and installed only out-of-band as a local editable install. The repo carries calling convention only: no model code, data, or paths. Caveat to decide then: Save PNGs include whatever is in the stats box.
