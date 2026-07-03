# GL-Native Grids Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate ShapePlotters to the libraries' GL-native APIs — GL-2048-in-cos(θ) node set, lib-exact analytic dR/dθ, spectrally exact quadrature, analytic pole radii, WMMM-style volume factor — plus a red z_cm marker on the plots.

**Architecture:** A shared read-only node module (`src/core/nodes.py`, the Python mirror of WMMM's `program_run_time_constants_mod`) supplies GL-2048 θ/x/weights. Renders evaluate R and dR/dθ there via `bp.Cache.resolve_shape`/`radius_and_derivative` and `fp.shape`/`fp.radius_and_derivative`; `quadrature.py` becomes GL dot products; the engine consumes lib-exact `dr_dtheta`, closes cross-sections at the analytic poles, and draws the z_cm point. One small prerequisite in the beta-parameterization repo: plumb the existing Fortran `apply_com_correction` flag through the C API and Python binding (v2.2.2).

**Tech Stack:** Python 3 (numpy, matplotlib, pytest, Agg backend), Fortran 2018 + C API + ctypes for the beta-parameterization patch.

**Spec:** `docs/superpowers/specs/2026-07-04-gl-native-grids-design.md`

## Global Constraints

- Activate the venv before any Python/pytest command: `source /home/alex/.virtualenvs/default/bin/activate` (has editable installs of both parameterization packages).
- ShapePlotters repo root: `/mnt/c/Users/aleks/PycharmProjects/ShapePlotters`. Run pytest from there.
- beta-parameterization repo root: `/mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization` (Task 1 only).
- Everything in `ShapeResult` stays in R0 units; the engine scales at draw time (unchanged units model).
- Node set: GL-2048 in x = cos(θ) (`N_NODES = 2048`), ascending θ. FoS validity grid: `N_RHO_GRID = 7201` (WMMM's `N_FOS_RHO_GRID_POINTS`).
- After Task 5, no file under `src/` may call `bp.Cache.radius_grid*`, `fp.radius_grid`, `bp.theta_grid`, or `fp.theta_grid` (Task 8 enforces).
- The test suite must be green at the end of every task.
- Fortran: 2018, `implicit none`, intent on all arguments (Task 1 follows existing file style).

## File Structure

**beta-parameterization (Task 1, tag 2.2.2):**
- `include/beta_parameterization.h` — add `apply_com_correction` param to `beta_param_cache_resolve_shape`
- `include/beta_parameterization.hpp` — C++ wrapper: trailing `bool apply_com_correction = true`
- `src/beta_parameterization_c_api_mod.f08` — pass the flag through to `cache_resolve_shape_s`
- `tests/c_api_smoke_test.cpp` — updated call + no-COM check
- `python/beta_parameterization/_cdefs.py`, `python/beta_parameterization/__init__.py` — ctypes prototype + kwarg
- `python/tests/test_api.py` — new test
- `python/pyproject.toml` (2.2.0 → 2.2.2), `CMakeLists.txt` (2.2.1 → 2.2.2)

**ShapePlotters:**
- Create: `src/core/nodes.py` (GL node data), `tests/test_nodes.py`
- Modify: `src/core/result.py` (3 new fields), `src/renders/beta.py`, `src/renders/fos.py`, `src/core/quadrature.py`, `src/core/engine.py`
- Tests: `tests/test_quadrature.py` (rewrite), `tests/test_beta_render.py`, `tests/test_fos_render.py`, `tests/test_engine.py` (additions)

---

### Task 1: beta-parameterization 2.2.2 — plumb `apply_com_correction` through C API and Python

The Fortran `cache_resolve_shape_s` already has the optional flag (added in v2.2.1, Fortran-only). This task exposes it to C and Python. All work in `/mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization`.

**Files:**
- Modify: `include/beta_parameterization.h:148-151` (+ docblock above)
- Modify: `include/beta_parameterization.hpp:132-156`
- Modify: `src/beta_parameterization_c_api_mod.f08:249-298`
- Modify: `tests/c_api_smoke_test.cpp:136-138` (+ new checks)
- Modify: `python/beta_parameterization/_cdefs.py:48-51`
- Modify: `python/beta_parameterization/__init__.py` (`Cache.resolve_shape`)
- Modify: `python/pyproject.toml:7`, `CMakeLists.txt:6`
- Test: `python/tests/test_api.py`

**Interfaces:**
- Consumes: existing Fortran `cache_resolve_shape_s(..., apply_com_correction)` (optional logical, default `.true.`).
- Produces: Python `Cache.resolve_shape(params, apply_com_correction: bool = True) -> ResolvedShape` — Task 4 relies on this exact kwarg. C signature gains `int apply_com_correction` between `r_south` and `message_buf_len`.

- [ ] **Step 1: Create a feature branch**

```bash
cd /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization
git checkout master && git checkout -b feature/resolve-com-flag-2.2.2
```

- [ ] **Step 2: Write the failing Python test**

Append to `python/tests/test_api.py`:

```python
def test_resolve_shape_no_com_flag() -> None:
    params = [0.1, 0.2, 0.05, 0.1]
    with bp.Cache(8, 181) as cache:
        com = cache.resolve_shape(params)
        no_com = cache.resolve_shape(params, apply_com_correction=False)
        assert com.ok and no_com.ok
        assert no_com.corrected_beta10 == 0.1        # input beta10 untouched
        assert com.corrected_beta10 != 0.1           # COM iteration moved it
        # no-COM parity with the legacy uniform-grid API (compute_radius_grid)
        ref = cache.radius_grid(params)
        interior = bp.theta_grid(181)[1:-1]
        with cache.build_node_set(interior) as ns:
            res = cache.radius_and_derivative(no_com.beta_con, ns)
            np.testing.assert_allclose(res.radii, ref.radii[1:-1], rtol=0.0, atol=1e-15)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
source /home/alex/.virtualenvs/default/bin/activate
python -m pytest python/tests/test_api.py::test_resolve_shape_no_com_flag -v
```

Expected: FAIL with `TypeError: resolve_shape() got an unexpected keyword argument 'apply_com_correction'`

- [ ] **Step 4: Update the C header**

In `include/beta_parameterization.h`, change the declaration (line 148) and add one docblock line above it (after the `@param r_south` line):

```c
 * @param apply_com_correction  Nonzero (default behavior): run the COM
 *                              iteration. Zero: skip it; corrected_beta10
 *                              receives the input beta10 (no-COM semantics
 *                              of beta_param_compute_radius_grid).
```

```c
int beta_param_cache_resolve_shape(
        const beta_param_cache_t* cache, const double* params, int n_params,
        double* beta_con, double* corrected_beta10, double* r_north, double* r_south,
        int apply_com_correction, int message_buf_len, char* message_buf);
```

- [ ] **Step 5: Update the Fortran C API wrapper**

In `src/beta_parameterization_c_api_mod.f08`, function `beta_param_cache_resolve_shape` (lines 249-298): add the parameter to the signature and dummy list, and pass it through:

```fortran
    function beta_param_cache_resolve_shape( &
            handle, params, n_params, beta_con, corrected_beta10, &
            r_north, r_south, apply_com_correction, message_buf_len, message_buf) &
            result(status) bind(c, name='beta_param_cache_resolve_shape')
```

Add with the other dummies:

```fortran
        integer(kind = ik_c),     intent(in), value :: apply_com_correction
```

Change the internal call (lines 288-290) to:

```fortran
        call cache_ptr%resolve_shape( &
                f_params, f_beta_con, f_corrected, f_r_north, f_r_south, &
                error_code, f_message, &
                apply_com_correction = (apply_com_correction /= 0_ik_c))
```

- [ ] **Step 6: Update the C++ header wrapper**

In `include/beta_parameterization.hpp` (lines 132-156): add a trailing defaulted parameter and forward it. New signature and call:

```cpp
    [[nodiscard]] Status resolve_shape(
            std::span<const double> params,
            std::span<double>       beta_con,
            double&                 corrected_beta10,
            double&                 r_north,
            double&                 r_south,
            std::string&            message,
            bool                    apply_com_correction = true) const
```

```cpp
        const int s = beta_param_cache_resolve_shape(
                handle_,
                params.data(), static_cast<int>(params.size()),
                beta_con.data(), &corrected_beta10, &r_north, &r_south,
                apply_com_correction ? 1 : 0,
                static_cast<int>(buf.size()), buf.data());
```

- [ ] **Step 7: Update the C smoke test**

In `tests/c_api_smoke_test.cpp` (lines 136-138): the existing call gains `1,` before the buffer args:

```cpp
        int st = beta_param_cache_resolve_shape(
                ns_cache, ns_params, 4, beta_con, &corrected_beta10, &r_north, &r_south,
                1, static_cast<int>(nbuf.size()), nbuf.data());
```

After the `"C: resolve_shape pole radii positive"` check (line 140), add:

```cpp
        double corrected_no_com = -1.0;
        st = beta_param_cache_resolve_shape(
                ns_cache, ns_params, 4, beta_con, &corrected_no_com, &r_north, &r_south,
                0, static_cast<int>(nbuf.size()), nbuf.data());
        check(st == BETA_PARAM_VALID, "C: resolve_shape no-COM VALID");
        check(corrected_no_com == 0.1, "C: no-COM corrected_beta10 == input beta10");
```

(Exact `== 0.1` is intentional: the no-COM branch copies the input verbatim. The hpp call at line 172 needs no edit — the default argument covers it.)

- [ ] **Step 8: Update the ctypes prototype and Python binding**

`python/beta_parameterization/_cdefs.py` (lines 48-51) — insert one `c_int`:

```python
    lib.beta_param_cache_resolve_shape.argtypes = [
        ctypes.c_void_p, c_dbl_p, ctypes.c_int,
        c_dbl_p, c_dbl_p, c_dbl_p, c_dbl_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
```

`python/beta_parameterization/__init__.py` — `Cache.resolve_shape` gains the kwarg; docstring gets one extra paragraph; the lib call gains the flag argument:

```python
    def resolve_shape(self, params: npt.ArrayLike,
                      apply_com_correction: bool = True) -> ResolvedShape:
        """Resolve a shape once (COM iteration + polar pre-check).

        Node-set-independent: feed the returned beta_con to
        radius_and_derivative for any number of node sets.

        apply_com_correction=False skips the COM iteration
        (corrected_beta10 = input beta10), matching radius_grid's
        no-COM semantics.
        """
        handle = self._require_handle()
        arr = _as_params(params)
        beta_con = np.zeros(self.max_beta_params, dtype=np.float64)
        corrected = ctypes.c_double(0.0)
        r_north = ctypes.c_double(0.0)
        r_south = ctypes.c_double(0.0)
        buf = ctypes.create_string_buffer(MESSAGE_BUFFER_SIZE)
        status = _get_lib().beta_param_cache_resolve_shape(
            handle,
            arr.ctypes.data_as(c_dbl_p), arr.size,
            beta_con.ctypes.data_as(c_dbl_p),
            ctypes.byref(corrected), ctypes.byref(r_north), ctypes.byref(r_south),
            1 if apply_com_correction else 0,
            MESSAGE_BUFFER_SIZE, buf)
        return ResolvedShape(
            beta_con=beta_con, corrected_beta10=corrected.value,
            r_north=r_north.value, r_south=r_south.value,
            status=Status(status), message=buf.value.decode(errors="replace"))
```

- [ ] **Step 9: Rebuild and run the full library test suite**

```bash
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including the updated `c_api_smoke_test`.

- [ ] **Step 10: Run the Python tests to verify the new test passes**

```bash
python -m pytest python/tests -q
```

Expected: all pass (the editable install picks up the source changes; `_libloader` finds the rebuilt `build/` library).

- [ ] **Step 11: Bump versions**

- `python/pyproject.toml:7`: `version = "2.2.0"` → `version = "2.2.2"`
- `CMakeLists.txt:6`: `VERSION 2.2.1` → `VERSION 2.2.2`

- [ ] **Step 12: Commit, merge, tag**

```bash
git add -A
git commit -m "feat: expose apply_com_correction through C API and Python bindings, v2.2.2"
git checkout master
git merge --no-ff feature/resolve-com-flag-2.2.2 -m "Merge feature/resolve-com-flag-2.2.2: apply_com_correction in C API + Python, v2.2.2"
git tag 2.2.2
```

---

### Task 2: Shared GL node module (`src/core/nodes.py`)

All remaining tasks run in `/mnt/c/Users/aleks/PycharmProjects/ShapePlotters`.

**Files:**
- Create: `src/core/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Produces: module constants `nodes.N_NODES: int = 2048`, `nodes.THETA: ndarray` (ascending, open — no 0/π), `nodes.X: ndarray` (= cos THETA, exact GL abscissas), `nodes.W: ndarray` (GL weights paired with THETA), `nodes.SIN_THETA: ndarray`. All read-only float64 arrays of shape (2048,). Tasks 3-8 import `from src.core import nodes`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_nodes.py`:

```python
"""Shared GL-2048 node set: shapes, ordering, exactness, immutability."""
import numpy as np
import pytest

from src.core import nodes


def test_shapes_and_ordering() -> None:
    assert nodes.N_NODES == 2048
    for a in (nodes.THETA, nodes.X, nodes.W, nodes.SIN_THETA):
        assert a.shape == (nodes.N_NODES,) and a.dtype == np.float64
    assert np.all(np.diff(nodes.THETA) > 0)                     # ascending theta
    assert 0.0 < nodes.THETA[0] and nodes.THETA[-1] < np.pi     # open rule: no poles


def test_trig_consistency() -> None:
    np.testing.assert_allclose(nodes.X, np.cos(nodes.THETA), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(nodes.SIN_THETA, np.sin(nodes.THETA), rtol=0.0, atol=1e-15)


def test_gl_exactness() -> None:
    # integral over x in [-1, 1]: dx = 1 -> 2;  x^2 -> 2/3 (GL exact for polynomials)
    assert np.sum(nodes.W) == pytest.approx(2.0, abs=1e-13)
    assert np.sum(nodes.W * nodes.X**2) == pytest.approx(2.0 / 3.0, abs=1e-13)


def test_read_only() -> None:
    for a in (nodes.THETA, nodes.X, nodes.W, nodes.SIN_THETA):
        assert not a.flags.writeable
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_nodes.py -v
```

Expected: FAIL with `ImportError: cannot import name 'nodes'` (module does not exist).

- [ ] **Step 3: Write the module**

Create `src/core/nodes.py`:

```python
"""Shared Gauss-Legendre node set — WMMM's dense grid (GL-2048 in x = cos θ).

Shape-independent, computed once at import, read-only: the Python mirror of
WMMM's program_run_time_constants_mod node-set metadata. Renders evaluate
R and dR/dθ at THETA; quadrature.py integrates with W (integrands with a
sin θ factor absorb it into dx = -sin θ dθ). GL is an open rule — no θ = 0/π
nodes — so pole values come from the libraries' analytic pole radii.
"""
from __future__ import annotations

import numpy as np

N_NODES = 2048

_x, _w = np.polynomial.legendre.leggauss(N_NODES)
X = _x[::-1].copy()           # cos(theta), descending +1 -> -1: exact GL abscissas
W = _w[::-1].copy()           # weights paired with THETA's ordering
THETA = np.arccos(X)          # ascending, strictly inside (0, pi)
SIN_THETA = np.sqrt(1.0 - X * X)

for _arr in (X, W, THETA, SIN_THETA):
    _arr.setflags(write=False)
del _x, _w, _arr
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_nodes.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/core/nodes.py tests/test_nodes.py
git commit -m "feat: shared GL-2048 node module (WMMM dense set)"
```

---

### Task 3: `ShapeResult` gains `dr_dtheta`, `r_north`, `r_south`

Additive with temporary defaults so both renders keep working until Tasks 4-5 fill the fields; Task 8 makes them required.

**Files:**
- Modify: `src/core/result.py:50-66`

**Interfaces:**
- Produces: `ShapeResult.dr_dtheta: Array | None = None`, `ShapeResult.r_north: float = 0.0`, `ShapeResult.r_south: float = 0.0`. Tasks 4-5 set them; Tasks 6-7 consume them; Task 8 removes the defaults.

- [ ] **Step 1: Add the fields**

In `src/core/result.py`, append to the `ShapeResult` dataclass after `length_keys`:

```python
    # GL-native additions (defaults dropped once both renders fill them):
    dr_dtheta: Array | None = None   # lib-exact analytic dR/dθ, scaled like radius
    r_north: float = 0.0             # analytic R(0), R0 units, scaled like radius
    r_south: float = 0.0             # analytic R(pi), R0 units, scaled like radius
```

- [ ] **Step 2: Verify the suite is still green (additive change)**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/core/result.py
git commit -m "feat: ShapeResult carries lib-exact dr_dtheta and analytic pole radii"
```

---

### Task 4: BetaRender on the GL-native API

**Files:**
- Modify: `src/renders/beta.py` (full rewrite of `__init__`/`compute`)
- Test: `tests/test_beta_render.py`

**Interfaces:**
- Consumes: `bp.Cache.resolve_shape(params, apply_com_correction=bool) -> ResolvedShape` (Task 1), `bp.Cache.build_node_set(thetas) -> NodeSet`, `bp.Cache.radius_and_derivative(beta_con, node_set) -> RadiusDerivativeResult`, `nodes.THETA/X/SIN_THETA/W/N_NODES` (Task 2), `ShapeResult` new fields (Task 3).
- Produces: `ShapeResult` with `theta = nodes.THETA` (2048), volume-factor-scaled `radius`/`dr_dtheta`/`r_north`/`r_south`; scalars `vol_factor` always, `corrected_beta10` whenever resolve succeeded (also with COM off, where it equals input β₁ — behavior change vs v0.1, per spec §9 test 4).

- [ ] **Step 1: Update the tests**

Replace the body of `tests/test_beta_render.py` with:

```python
"""BetaRender: contract wiring over the beta_parameterization node-set API."""
import numpy as np
import pytest

from src.core import nodes, quadrature
from src.renders.beta import BetaRender

# Probed lib-VALID and necked (carried over from v0.1).
NECKED_BETAS = [0.0, 1.5, 0.0, 0.8]


@pytest.fixture(scope="module")
def render() -> BetaRender:
    return BetaRender()


def _params(betas: list[float]) -> dict[str, float]:
    full = betas + [0.0] * (8 - len(betas))
    return {f"beta{i}": full[i - 1] for i in range(1, 9)}


def test_sphere(render: BetaRender) -> None:
    res = render.compute(_params([]), {"com": False})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-12)
    assert res.r_south == pytest.approx(1.0, abs=1e-12)
    assert res.scalars["vol_factor"] == pytest.approx(1.0, abs=1e-12)
    assert res.neck is None
    assert res.drho_dz is None


def test_volume_fix_matches_wmmm_and_old_plotter(render: BetaRender) -> None:
    # Old ShapePlotter PNG golden 0.99598851 (trapezoid); GL-exact differs only
    # in trailing digits — WMMM cross-check happens at the Task 8 gate.
    res = render.compute(_params([0.0, 0.20, 0.10]), {"com": False})
    assert res.ok
    assert res.scalars["vol_factor"] == pytest.approx(0.99598851, abs=1e-5)
    assert quadrature.volume(res.theta, res.radius) == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-5)   # trapezoid until Task 6 tightens to 1e-12


def test_dr_dtheta_is_exact_not_gradient(render: BetaRender) -> None:
    # Wiring check only (analytic-derivative goldens live in the library):
    # np.gradient of the returned radii must agree to display tolerance.
    res = render.compute(_params(NECKED_BETAS), {"com": False})
    assert res.ok
    fd = np.gradient(res.radius, res.theta)
    tol = 1e-3 * (1.0 + float(np.max(np.abs(res.dr_dtheta))))
    assert float(np.max(np.abs(res.dr_dtheta - fd))) < tol


def test_com_toggle(render: BetaRender) -> None:
    # Asymmetric shape proven lib-VALID in v0.1 (beta1 input is 0.0, so the
    # COM iteration must move corrected_beta10 away from 0).
    p = _params([0.0, 0.85, 0.35, 0.18])
    off = render.compute(p, {"com": False})
    on = render.compute(p, {"com": True})
    assert off.ok and on.ok
    assert off.scalars["corrected_beta10"] == 0.0                  # input beta1
    assert on.scalars["corrected_beta10"] != 0.0                   # COM moved it
    # Legacy-API parity (test-only usage; updated when the 2.3.0 cleanup lands):
    ref = render._cache.radius_grid_with_com_shift([0.0, 0.85, 0.35, 0.18])
    assert on.scalars["corrected_beta10"] == pytest.approx(ref.corrected_beta10, abs=1e-14)


def test_pole_radii_scaled_with_volume_factor(render: BetaRender) -> None:
    res = render.compute(_params([0.0, 0.20, 0.10]), {"com": False})
    assert res.ok
    # Poles carry the same volume factor as the radii: R(theta->0) -> r_north.
    assert res.r_north == pytest.approx(float(res.radius[0]), abs=1e-4)
    assert res.r_south == pytest.approx(float(res.radius[-1]), abs=1e-4)


def test_necked_shape(render: BetaRender) -> None:
    res = render.compute(_params(NECKED_BETAS), {"com": False})
    assert res.ok
    assert res.neck is not None
    assert res.neck.source == "py heuristic"
    assert 0.0 < res.neck.depth < 1.0
    assert res.neck.rho > 0.0


def test_invalid_shape_returns_status_not_exception(render: BetaRender) -> None:
    res = render.compute(_params([0.0, 4.0]), {"com": False})  # interior negative
    assert not res.ok
    assert res.status_name == "ERROR_INTERIOR_NEGATIVE"
    assert res.message
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()
    assert res.r_north == 0.0 and res.r_south == 0.0


def test_slider_specs_and_filename(render: BetaRender) -> None:
    assert [s.key for s in render.slider_specs] == [f"beta{i}" for i in range(1, 9)]
    assert (render.slider_specs[0].vmin, render.slider_specs[0].vmax) == (-1.6, 1.6)
    assert (render.slider_specs[1].vmin, render.slider_specs[1].vmax) == (0.0, 4.0)
    assert render.has_extra_panel is False
    name = render.filename(92, 144, _params([0.0, 1.25]))
    assert name == "92_144_0.00_1.25_0.00_0.00_0.00_0.00_0.00_0.00.png"
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
python -m pytest tests/test_beta_render.py -v
```

Expected: FAIL — `test_sphere` (theta is not `nodes.THETA`), `test_dr_dtheta_is_exact_not_gradient` (`dr_dtheta` is None), etc.

- [ ] **Step 3: Rewrite the render**

Replace `src/renders/beta.py` with:

```python
"""Beta (Legendre) render over the beta_parameterization package.

GL-native: R(theta) and analytic dR/dtheta come from the library's node-set
API evaluated on the shared GL-2048 set (src/core/nodes.py) — in sync with
the energy model's dense grid. The neck is the Python display-only heuristic
(src/core/neck.py) — graduating it into the library is recorded future work.
"""
from __future__ import annotations

import numpy as np
import beta_parameterization as bp

from src.core import nodes
from src.core.neck import find_neck_indices, neck_depth
from src.core.result import NeckInfo, ShapeResult, SliderSpec, ToggleSpec

# Cache constructor still requires a uniform-grid size; no uniform-grid
# feature is used (the argument goes away with the library's 2.3.0 cleanup).
N_GRID = 721
N_BETAS = 8
SPHERE_VOLUME = 4.0 * np.pi / 3.0  # unit sphere, R0 units


class BetaRender:
    name = "beta"
    has_extra_panel = False
    # Ranges from the old ShapePlotter: beta1 (-1.6, 1.6), beta2 (0, 4), rest (-2, 2).
    slider_specs = [
        SliderSpec("beta1", "β1", -1.6, 1.6, 0.0, 0.01),
        SliderSpec("beta2", "β2", 0.0, 4.0, 0.0, 0.01),
    ] + [SliderSpec(f"beta{i}", f"β{i}", -2.0, 2.0, 0.0, 0.01) for i in range(3, N_BETAS + 1)]
    # Default off: parity with old ShapePlotter PNGs (no COM shift there).
    toggles = [ToggleSpec("com", "COM correction", False)]

    def __init__(self) -> None:
        self._cache = bp.Cache(max_beta_params=N_BETAS, n_grid=N_GRID)
        self._node_set = self._cache.build_node_set(nodes.THETA)

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        betas = [params[f"beta{i}"] for i in range(1, N_BETAS + 1)]
        resolved = self._cache.resolve_shape(
            betas, apply_com_correction=toggles.get("com", False))
        rd = None
        if resolved.ok:
            rd = self._cache.radius_and_derivative(resolved.beta_con, self._node_set)
        ok = rd is not None and rd.ok

        vol_factor = 1.0
        if ok:
            # WMMM's exact GL volume factor (radius_grid_mod original_volume_factor):
            # radii arrive pre-scaled everywhere — derivative and poles included.
            raw_volume = (2.0 * np.pi / 3.0) * float(np.sum(nodes.W * rd.radii**3))
            vol_factor = float((SPHERE_VOLUME / raw_volume) ** (1.0 / 3.0))
            radii = rd.radii * vol_factor
            dr_dtheta = rd.dr_dtheta * vol_factor
            r_north = resolved.r_north * vol_factor
            r_south = resolved.r_south * vol_factor
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)
            r_north = r_south = 0.0

        z = radii * nodes.X
        rho = radii * nodes.SIN_THETA
        neck = None
        if ok:
            hit = find_neck_indices(rho)
            if hit is not None:
                i_neck, i_a, i_b = hit
                neck = NeckInfo(z=float(z[i_neck]), rho=float(rho[i_neck]),
                                depth=neck_depth(rho, i_neck, i_a, i_b),
                                source="py heuristic")
        scalars: dict[str, float] = {"vol_factor": vol_factor}
        if resolved.ok:
            scalars["corrected_beta10"] = resolved.corrected_beta10
        primary = rd if resolved.ok else resolved
        return ShapeResult(
            status=int(primary.status), status_name=primary.status.name,
            message=primary.message,
            theta=nodes.THETA, radius=radii, z=z, rho=rho, drho_dz=None,
            neck=neck, scalars=scalars, length_keys=frozenset(),
            dr_dtheta=dr_dtheta, r_north=r_north, r_south=r_south)

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        betas = "_".join(f"{params[f'beta{i}']:.2f}" for i in range(1, N_BETAS + 1))
        return f"{z}_{n}_{betas}.png"
```

- [ ] **Step 4: Run the render tests, then the full suite**

```bash
python -m pytest tests/test_beta_render.py -v
python -m pytest tests/ -q
```

Expected: all pass. (Engine/neck/perf tests keep working: `np.gradient` and `np.trapezoid` both handle the non-uniform GL θ.) If `test_volume_fix_matches_wmmm_and_old_plotter` fails on the golden, record the exact GL `vol_factor` in the test comment and verify it differs from 0.99598851 only past the 5th decimal before adjusting the tolerance — a larger shift means a weights bug, not a re-baseline.

- [ ] **Step 5: Commit**

```bash
git add src/renders/beta.py tests/test_beta_render.py
git commit -m "feat: BetaRender on GL-native node-set API with exact volume factor"
```

---

### Task 5: FoSRender on the GL-native API

**Files:**
- Modify: `src/renders/fos.py` (rewrite of `compute`)
- Test: `tests/test_fos_render.py`

**Interfaces:**
- Consumes: `fp.shape(params, n_rho_grid) -> ShapeResult(z_shift, r_north, r_south, status, message)`, `fp.radius_and_derivative(params, thetas, z_shift) -> RadiusDerivativeResult(radii, dr_dtheta, status)` (no message field), `fp.rho_profile`/`fp.neck`/`fp.a2` (unchanged), `nodes` (Task 2).
- Produces: `ShapeResult` with `theta = nodes.THETA`, lib pole radii, `z_shift` scalar from `fp.shape` (the standalone `fp.z_shift` call is dropped). `z`/`rho`/`drho_dz` stay 721-point profile arrays — the engine treats them independently of `theta`/`radius`.

- [ ] **Step 1: Update the tests**

Replace the body of `tests/test_fos_render.py` with:

```python
"""FoSRender: contract wiring over the fos_parameterization package."""
import numpy as np
import pytest

from src.core import nodes
from src.renders.fos import FoSRender

SPHERE = {"c": 1.0, "a3": 0.0, "a4": 0.0, "a5": 0.0, "a6": 0.0, "a7": 0.0, "a8": 0.0}
# Probed lib-VALID necked shape (carried over from v0.1).
NECKED = {**SPHERE, "c": 2.0, "a4": 0.6}


@pytest.fixture(scope="module")
def render() -> FoSRender:
    return FoSRender()


def test_sphere(render: FoSRender) -> None:
    res = render.compute(SPHERE, {})
    assert res.ok
    assert res.theta is nodes.THETA
    assert np.allclose(res.radius, 1.0, atol=1e-12)
    assert np.allclose(res.dr_dtheta, 0.0, atol=1e-9)
    assert res.r_north == pytest.approx(1.0, abs=1e-14)
    assert res.r_south == pytest.approx(1.0, abs=1e-14)
    assert res.scalars["z_shift"] == pytest.approx(0.0, abs=1e-14)
    assert res.scalars["a2"] == pytest.approx(0.0, abs=1e-14)
    assert "z_shift" in res.length_keys
    assert res.neck is None
    assert res.drho_dz is not None and res.drho_dz.shape == res.rho.shape


def test_pole_radii_formula(render: FoSRender) -> None:
    # Spec (library design 3.2): r_north = c + z_shift, r_south = |z_shift - c|.
    res = render.compute(NECKED, {})
    assert res.ok
    zs, c = res.scalars["z_shift"], NECKED["c"]
    assert res.r_north == pytest.approx(c + zs, abs=1e-12)
    assert res.r_south == pytest.approx(abs(zs - c), abs=1e-12)


def test_dr_dtheta_matches_gradient(render: FoSRender) -> None:
    res = render.compute(NECKED, {})
    assert res.ok
    fd = np.gradient(res.radius, res.theta)
    tol = 1e-3 * (1.0 + float(np.max(np.abs(res.dr_dtheta))))
    assert float(np.max(np.abs(res.dr_dtheta - fd))) < tol


def test_necked_shape(render: FoSRender) -> None:
    res = render.compute(NECKED, {})
    assert res.ok
    assert res.neck is not None
    assert res.neck.source == "lib"
    assert res.neck.rho > 0.0
    assert res.neck.depth > 0.0


def test_invalid_c_returns_status_not_exception(render: FoSRender) -> None:
    res = render.compute({**SPHERE, "c": 0.0}, {})
    assert not res.ok
    assert res.status_name in ("ERROR_INVALID_C", "ERROR_INVALID_ARGUMENTS")
    assert res.neck is None
    assert res.radius.shape == (nodes.N_NODES,) and not res.radius.any()
    assert res.dr_dtheta.shape == (nodes.N_NODES,) and not res.dr_dtheta.any()


def test_slider_specs_and_filename(render: FoSRender) -> None:
    assert [s.key for s in render.slider_specs] == ["c", "a3", "a4", "a5", "a6", "a7", "a8"]
    assert render.slider_specs[0].markers == (1.0, 3.0)
    assert render.has_extra_panel is True
    name = render.filename(92, 144, {**SPHERE, "c": 1.2, "a3": 0.17})
    assert name == "fos_shape_Z92_N144_c1.20_a30.17_a40.00_a50.00_a60.00_a70.00_a80.00.png"
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
python -m pytest tests/test_fos_render.py -v
```

Expected: FAIL — `test_sphere` (theta not `nodes.THETA`), `test_pole_radii_formula` (`r_north == 0.0` default), etc.

- [ ] **Step 3: Rewrite the render**

Replace `src/renders/fos.py` with:

```python
"""FoS render over the fos_parameterization package.

rho(z)-native (lib-exact drho_dz, lib-native neck); R(theta) and analytic
dR/dtheta come from the library's arbitrary-node evaluator on the shared
GL-2048 set (src/core/nodes.py) — in sync with the energy model's dense
grid. Neck position/radius are lib values; only the displayed depth reuses
the shared peak analysis on the profile.
"""
from __future__ import annotations

import numpy as np
import fos_parameterization as fp

from src.core import nodes
from src.core.neck import find_neck_indices, neck_depth
from src.core.result import NeckInfo, ShapeResult, SliderSpec, ToggleSpec

N_GRID = 721        # rho(z) display panel only
N_RHO_GRID = 7201   # shape() validity grid — WMMM's N_FOS_RHO_GRID_POINTS
PARAM_KEYS = ("c", "a3", "a4", "a5", "a6", "a7", "a8")


class FoSRender:
    name = "fos"
    has_extra_panel = True
    # Ranges and practical-limit markers from ShapePlotterFoSFitter.
    slider_specs = [
        SliderSpec("c", "c", 0.5, 3.5, 1.0, 0.01, markers=(1.0, 3.0)),
        SliderSpec("a3", "a3", -0.6, 0.6, 0.0, 0.01, markers=(0.0, 0.5)),
        SliderSpec("a4", "a4", -0.75, 0.75, 0.0, 0.01, markers=(-0.2, 0.72)),
    ] + [SliderSpec(f"a{i}", f"a{i}", -0.5, 0.5, 0.0, 0.01, markers=(-0.2, 0.2))
         for i in range(5, 9)]
    toggles: list[ToggleSpec] = []

    def compute(self, params: dict[str, float], toggles: dict[str, bool]) -> ShapeResult:
        arr = [params[k] for k in PARAM_KEYS]
        shp = fp.shape(arr, N_RHO_GRID)
        prof = fp.rho_profile(arr, N_GRID)
        rd = fp.radius_and_derivative(arr, nodes.THETA, shp.z_shift) if shp.ok else None
        ok = shp.ok and rd is not None and rd.ok and prof.ok

        if ok:
            radii, dr_dtheta = rd.radii, rd.dr_dtheta
        else:
            radii = np.zeros(nodes.N_NODES)
            dr_dtheta = np.zeros(nodes.N_NODES)

        if not shp.ok:
            status, status_name, message = int(shp.status), shp.status.name, shp.message
        elif rd is not None and not rd.ok:
            status, status_name, message = int(rd.status), rd.status.name, ""
        else:
            status, status_name, message = int(prof.status), prof.status.name, prof.message

        neck_info = None
        if ok:
            nk = fp.neck(arr)
            if nk.ok and nk.found:
                depth = 0.0
                hit = find_neck_indices(prof.rho)
                if hit is not None:
                    depth = neck_depth(prof.rho, *hit)
                neck_info = NeckInfo(z=nk.z_neck, rho=nk.rho_neck,
                                     depth=depth, source="lib")
        return ShapeResult(
            status=status, status_name=status_name, message=message,
            theta=nodes.THETA, radius=radii,
            z=prof.z, rho=prof.rho, drho_dz=prof.drho_dz,
            neck=neck_info,
            scalars={"z_shift": shp.z_shift, "a2": fp.a2(arr)},
            length_keys=frozenset({"z_shift"}),
            dr_dtheta=dr_dtheta, r_north=shp.r_north, r_south=shp.r_south)

    def filename(self, z: int, n: int, params: dict[str, float]) -> str:
        return (f"fos_shape_Z{z}_N{n}_c{params['c']:.2f}"
                + "".join(f"_a{i}{params[f'a{i}']:.2f}" for i in range(3, 9))
                + ".png")
```

(`shp.r_north`/`shp.r_south` are passed verbatim — the library zero-fills them on failure, matching the invalid-shape contract.)

- [ ] **Step 4: Run the render tests, then the full suite**

```bash
python -m pytest tests/test_fos_render.py -v
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/renders/fos.py tests/test_fos_render.py
git commit -m "feat: FoSRender on GL-native shape + radius_and_derivative API"
```

---

### Task 6: `quadrature.py` → exact GL dot products

**Files:**
- Modify: `src/core/quadrature.py` (full rewrite)
- Modify: `src/core/engine.py:291` (`surface_area` call gains `dr_dtheta`)
- Test: `tests/test_quadrature.py` (rewrite)
- Modify: `tests/test_beta_render.py` (tighten one tolerance)

**Interfaces:**
- Consumes: `nodes.W/X/N_NODES/THETA` (Task 2); `result.dr_dtheta` (Tasks 4-5).
- Produces: `volume(theta, radius) -> float`, `surface_area(theta, radius, dr_dtheta) -> float` (**signature change**: third argument now required), `z_cm(theta, radius) -> float`. All raise `ValueError` if `theta` is not the shared node set's shape. Task 7 relies on these exact signatures.

- [ ] **Step 1: Rewrite the tests**

Replace `tests/test_quadrature.py` with:

```python
"""GL quadrature vs analytic sphere values; scaling by length powers."""
import numpy as np
import pytest

from src.core import nodes, quadrature

THETA = nodes.THETA
ONES = np.ones(nodes.N_NODES)
ZEROS = np.zeros(nodes.N_NODES)


def test_unit_sphere_volume() -> None:
    assert quadrature.volume(THETA, ONES) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-13)


def test_unit_sphere_surface() -> None:
    assert quadrature.surface_area(THETA, ONES, ZEROS) == pytest.approx(4.0 * np.pi, rel=1e-13)


def test_unit_sphere_z_cm_is_zero() -> None:
    assert quadrature.z_cm(THETA, ONES) == pytest.approx(0.0, abs=1e-13)


def test_shifted_sphere_z_cm() -> None:
    # Unit sphere centered at z = d: R(theta) = d*cos(theta) + sqrt(1 - d^2 sin^2 theta)
    d = 0.2
    r = d * nodes.X + np.sqrt(1.0 - d**2 * nodes.SIN_THETA**2)
    assert quadrature.volume(THETA, r) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-13)
    assert quadrature.z_cm(THETA, r) == pytest.approx(d, abs=1e-13)


def test_length_power_scaling() -> None:
    s = 7.13
    assert quadrature.volume(THETA, s * ONES) == pytest.approx(
        s**3 * quadrature.volume(THETA, ONES), rel=1e-12)
    assert quadrature.surface_area(THETA, s * ONES, ZEROS) == pytest.approx(
        s**2 * quadrature.surface_area(THETA, ONES, ZEROS), rel=1e-12)


def test_wrong_grid_rejected() -> None:
    with pytest.raises(ValueError):
        quadrature.volume(np.linspace(0.0, np.pi, 721), np.ones(721))
```

- [ ] **Step 2: Run tests to verify failures**

```bash
python -m pytest tests/test_quadrature.py -v
```

Expected: FAIL — `surface_area` rejects the third argument (`TypeError`), `test_wrong_grid_rejected` gets no `ValueError`, sphere tolerances too tight for trapezoid.

- [ ] **Step 3: Rewrite the module**

Replace `src/core/quadrature.py` with:

```python
"""GL dot-product shape integrals on the shared node set (src/core/nodes.py).

Spectrally exact — the same scheme as WMMM's dense set: every integrand
carries a sin(theta) factor that the x = cos(theta) substitution absorbs.
Arrays must be sampled on nodes.THETA; theta is accepted for interface
stability and checked against the node set.
"""
from __future__ import annotations

import numpy as np

from src.core import nodes
from src.core.result import Array


def _require_node_set(theta: Array) -> None:
    if theta.shape != nodes.THETA.shape:
        raise ValueError(
            f"expected the shared GL node set ({nodes.N_NODES} nodes), got {theta.shape}")


def volume(theta: Array, radius: Array) -> float:
    """V = (2*pi/3) * sum w_i R_i^3 (star-convex body)."""
    _require_node_set(theta)
    return float((2.0 * np.pi / 3.0) * np.sum(nodes.W * radius**3))


def surface_area(theta: Array, radius: Array, dr_dtheta: Array) -> float:
    """S = 2*pi * sum w_i R_i sqrt(R_i^2 + R'_i^2), with lib-exact R'."""
    _require_node_set(theta)
    return float(2.0 * np.pi * np.sum(
        nodes.W * radius * np.sqrt(radius**2 + dr_dtheta**2)))


def z_cm(theta: Array, radius: Array) -> float:
    """z_cm = (pi/2) * sum w_i R_i^4 x_i / V; 0 if V <= 0."""
    _require_node_set(theta)
    v = volume(theta, radius)
    if v <= 0.0:
        return 0.0
    return float((np.pi / 2.0) * np.sum(nodes.W * radius**4 * nodes.X) / v)
```

- [ ] **Step 4: Update the engine call site**

In `src/core/engine.py`, `_stats_block` (line 291):

```python
        s = quadrature.surface_area(result.theta, result.radius, result.dr_dtheta) * scale**2
```

- [ ] **Step 5: Tighten the beta-render volume assertion**

In `tests/test_beta_render.py::test_volume_fix_matches_wmmm_and_old_plotter`, the trapezoid-era check becomes exact:

```python
    assert quadrature.volume(res.theta, res.radius) == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-12)   # GL volume factor makes this exact
```

(Delete the "trapezoid until Task 6" comment.)

- [ ] **Step 6: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/quadrature.py src/core/engine.py tests/test_quadrature.py tests/test_beta_render.py
git commit -m "feat: quadrature as exact GL dot products; surface uses lib dR/dtheta"
```

---

### Task 7: Engine — lib dR/dθ, pole closure, z_cm marker, (GL) labels

**Files:**
- Modify: `src/core/engine.py` (`_build_artists`, `update`, `_stats_block`)
- Test: `tests/test_engine.py` (additions)

**Interfaces:**
- Consumes: `result.dr_dtheta`/`r_north`/`r_south` (Tasks 4-5), `quadrature.surface_area(theta, radius, dr_dtheta)` (Task 6).
- Produces: new engine attributes `self.zcm_point` (Line2D on `ax_shape`) and `self.zcm_extra` (Line2D on `ax_extra`, or `None`); `_stats_block(self, result, scale, unit, v, s, zc)` (volume/surface/z_cm now computed in `update()` and passed in).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py`:

```python
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
    a.rows["beta3"].slider.set_val(0.3)                  # asymmetric, COM off
    assert abs(a.zcm_point.get_xdata()[0]) > 0.01        # visible offset (fm)
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
```

- [ ] **Step 2: Run tests to verify failures**

```bash
python -m pytest tests/test_engine.py -v
```

Expected: new tests FAIL (`AttributeError: 'ShapePlotterApp' object has no attribute 'zcm_point'`, gradient-vs-lib mismatch, "(py quad)" still present).

- [ ] **Step 3: Update `_build_artists`**

In `src/core/engine.py`:

The dR/dθ legend label (line 73-74) becomes lib-sourced:

```python
        (self.dr_line,) = ax.plot([], [], color=DERIV_COLOR, lw=2, ls=":",
                                  label="dR/dθ (lib)")
```

After the `neck_line` creation (line 83), add the z_cm point (red per spec §8 — the COM of an axially symmetric shape lies on the z axis):

```python
        (self.zcm_point,) = ax.plot([], [], marker="o", ms=6, ls="",
                                    color="tab:red", zorder=5)
```

In the extra-panel block (inside `if self.ax_extra is not None:`), after `self.extra_drho`, add — and set the fallback before the `if`:

```python
        self.zcm_extra = None
        if self.ax_extra is not None:
            ...
            (self.zcm_extra,) = ax.plot([], [], marker="o", ms=6, ls="",
                                        color="tab:red", zorder=5)
```

(Note: `self.extra_legend = None` currently precedes the block; keep both fallbacks together.)

- [ ] **Step 4: Update `update()`**

Replace the radius/derivative/cross-section block (lines 220-232) with:

```python
        r = result.radius * scale
        self.r_line.set_data(result.theta, r)
        self.dr_line.set_data(result.theta, result.dr_dtheta * scale)

        z, rho = result.z * scale, result.rho * scale
        # GL nodes exclude theta = 0/pi; close the outline with the analytic poles.
        rn, rs = result.r_north * scale, result.r_south * scale
        first, last = (-rs, rn) if (z.size > 1 and z[0] < z[-1]) else (rn, -rs)
        z_c = np.concatenate(([first], z, [last]))
        rho_c = np.concatenate(([0.0], rho, [0.0]))
        self.shape_upper.set_data(z_c, rho_c)
        self.shape_lower.set_data(z_c, -rho_c)
        if result.neck is not None:
            zn, rn_neck = result.neck.z * scale, result.neck.rho * scale
            self.neck_line.set_data([zn, zn], [-rn_neck, rn_neck])
            self.neck_line.set_visible(True)
        else:
            self.neck_line.set_visible(False)
```

(The neck-local variable is renamed `rn_neck` to avoid shadowing the pole radius `rn`. The FoS profile arrays already touch ρ = 0 at their ends; the appended pole points are duplicates there, which is harmless. The extra-panel `set_data(z, rho)` keeps the unclosed arrays.)

Then, before `self.stats_text.set_text(...)` (line 248), insert the quadrature + marker block and change the stats call:

```python
        v = quadrature.volume(result.theta, result.radius) * scale**3
        s = quadrature.surface_area(result.theta, result.radius, result.dr_dtheta) * scale**2
        zc = quadrature.z_cm(result.theta, result.radius) * scale
        self.zcm_point.set_data([zc], [0.0])
        self.zcm_point.set_visible(result.ok)
        if self.zcm_extra is not None:
            self.zcm_extra.set_data([zc], [0.0])
            self.zcm_extra.set_visible(result.ok)

        self.stats_text.set_text(self._stats_block(result, scale, unit, v, s, zc))
```

- [ ] **Step 5: Update `_stats_block`**

New signature and quadrature lines (the computation moved to `update()`; labels become "(GL)"):

```python
    def _stats_block(self, result: ShapeResult, scale: float, unit: str,
                     v: float, s: float, zc: float) -> str:
```

and the closing lines:

```python
        lines += ["", f"volume  = {v:.4f} {unit}³ (GL)",
                  f"surface = {s:.4f} {unit}² (GL)",
                  f"z_cm    = {zc:.4f} {unit} (GL)"]
        return "\n".join(lines)
```

(Delete the three `quadrature.` lines that were inside `_stats_block`.)

- [ ] **Step 6: Run the engine tests, then the full suite**

```bash
python -m pytest tests/test_engine.py -v
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/engine.py tests/test_engine.py
git commit -m "feat: engine draws lib dR/dtheta, pole-closed outline, red z_cm point"
```

---

### Task 8: Contract tightening, legacy-API sweep, perf, gate

**Files:**
- Modify: `src/core/result.py` (defaults removed, docstring updated)

**Interfaces:**
- Produces: `ShapeResult.dr_dtheta: Array`, `r_north: float`, `r_south: float` — required fields; final render contract.

- [ ] **Step 1: Make the new fields required**

In `src/core/result.py`, the Task 3 block becomes (no defaults, no "temporary" comment):

```python
    dr_dtheta: Array                 # lib-exact analytic dR/dθ, scaled like radius
    r_north: float                   # analytic R(0), R0 units, scaled like radius
    r_south: float                   # analytic R(pi), R0 units, scaled like radius
```

Also update the `theta` field comment (line 55) to reflect the grid:

```python
    theta: Array                   # shared GL node set (src/core/nodes.py)
```

- [ ] **Step 2: Sweep for legacy uniform-grid callers**

```bash
grep -rn "radius_grid\|theta_grid" src/
```

Expected: no matches. (This is the payoff gate: ShapePlotters no longer blocks the libraries' 2.3.0/1.2.0 cleanup tags. `tests/test_beta_render.py` retains one deliberate legacy call for COM parity — updated when the cleanup lands.)

- [ ] **Step 3: Full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 4: Perf check**

```bash
python -m pytest tests/test_perf.py -v
```

Expected: pass within the recorded v0.2 budgets. Perf numbers are venv-dependent (see `docs/handoffs/2026-07-04-engine-v0.2-performance-handoff.md`) — run in the same venv that produced the baseline before judging a regression.

- [ ] **Step 5: Commit**

```bash
git add src/core/result.py
git commit -m "feat: dr_dtheta and pole radii are required ShapeResult fields"
```

- [ ] **Step 6: Manual gate (user-assisted, not automatable)**

1. `python main.py beta` and `python main.py fos` under WSLg: drag sliders, toggle COM/units/scission, confirm the red z_cm point sits at z ≈ 0 with COM on (beta) and always for FoS, and moves with COM off.
2. Save a known shape per render; spot-check against archived v0.1 PNGs (shapes visually identical; vol factor differs only in trailing digits; no visible pole gap on the outline — if the R(θ) curve itself shows endpoint gaps, spec §8 allows adding pole closure there too).
3. Vol-factor cross-check vs WMMM: for β₂=0.2, β₃=0.1, compare the displayed `vol_factor` against WMMM's `original_volume_factor` for the same shape (spec §9 test 3).
4. Report results; if all good, the migration is complete.
