# Compiled-library candidates

Living list of Python-side work that could move into the compiled
parameterization libraries. None are performance problems today; entries
exist for accuracy/parity and so future sessions keep this list current.
Append here whenever new per-frame Python work appears in the plotters or
the energy model.

| Candidate | Current Python cost | Fortran API needed |
|---|---|---|
| Volume / surface / z_cm (stats) | 3 quadratures over 721 pts per frame (µs; `src/core/quadrature.py`, display-grid accuracy) | exact integrals returned by the shape library |
| dR/dθ display curve | `np.gradient` over 721 pts (µs; finite-difference accuracy) | analytic derivative on the display grid |
| (reference) `render.compute` | already compiled: 0.1 ms beta / 0.4 ms fos | — |

## Incremental single-parameter updates (library-level, future)

Access patterns are one-parameter-at-a-time: slider drags in the plotters,
innermost loops over deformation grids in energy calculations. Both
parameterizations are linear in their amplitude parameters (R(θ) in the β_l;
FoS ρ² in the a_n), so per-term contributions on a fixed grid are cacheable.
Belongs in the compiled libraries so plotters and the energy model both benefit.

- Prefer caching the basis functions (Y_l0 on the θ grid, FoS f_n on the u
  grid) and recomputing the full sum (~n_terms × n_grid flops, trivial) over
  delta-updating the sum: same win, and no floating-point drift. Repeated
  delta updates accumulate rounding and break run-to-run reproducibility.
- Grid-shaping parameters (FoS c/elongation) and grid changes invalidate the
  cache; volume-fixing renormalization still needs a full pass (cheap once
  the basis is cached).
- Not a plotter bottleneck (compute is 0.1–0.4 ms); the payoff is
  energy-model grid scans. The cache must be behaviorally invisible:
  identical results with or without it.
