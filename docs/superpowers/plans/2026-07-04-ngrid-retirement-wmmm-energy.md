# N_GRID Retirement + WMMM Energy Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ShapePlotters constructs its beta cache node-set-only (no uniform-grid size anywhere), fixes the COM-overlay trigger to compare against the slider β₁, and gains an optional, strictly-local WMMM Energy button.

**Architecture:** beta-parameterization 2.3.0 makes the cache's uniform θ-table optional (Fortran `optional :: n_grid`, C-API `n_grid <= 0` sentinel, Python `n_grid=None`); nothing is deleted. In ShapePlotters, a new `src/core/energy.py` adapter is the only module allowed to import `wmmm` (find_spec guard, lazy import, one cached `Model` per `(param_type, com_correction)`); renders declare `energy_requests()` returning 1–2 `EnergyRequest`s; the engine adds a click-only Energy button whose stats-box blocks are wiped by any parameter change.

**Tech Stack:** Fortran 2018 + C API + ctypes (beta-parameterization repo), Python 3.13 / numpy / matplotlib (ShapePlotters), pytest with a stub `wmmm` module.

**Spec:** `docs/superpowers/specs/2026-07-04-ngrid-retirement-wmmm-energy-design.md`

## Global Constraints

- WMMM is strictly local: `wmmm` never appears in `pyproject.toml` (not even as an optional extra); only `src/core/energy.py` may import it; committed tests carry **no WMMM-derived numbers**.
- Library work happens in `/mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization` (Fortran 2018, `implicit none` every scope, intent on all arguments, `only:` imports). Nothing uniform-grid gets deleted — the table becomes opt-in.
- Python commands run in the default venv: `source /home/alex/.virtualenvs/default/bin/activate`.
- After the library change: rebuild **both** `build/` and `build/release` (the Python loader prefers `build/release`; a stale ABI there segfaults).
- `OVERLAY_BETA10_THRESHOLD = 0.001` keeps its value; the trigger becomes `|corrected_beta10 − slider β₁| > 0.001`.
- Energies are always MeV — never scaled by the fm/R0 unit toggle.
- ShapePlotters test/gate commands: `python -m pytest` from `/mnt/c/Users/aleks/PycharmProjects/ShapePlotters`.

---

### Task 1: beta-parameterization Fortran — optional `n_grid` + `LEGENDRE_ERROR_NO_UNIFORM_GRID`

**Repo:** `/mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization` (all paths in this task relative to it)

**Files:**
- Modify: `src/beta_parameterization_mod.f08` (error constants ~line 70; `cache_init_s` ~line 161; `cache_compute_radius_grid_s` ~line 553; `cache_compute_radius_grid_with_com_shift_s` ~line 651)
- Test: `tests/beta_param_node_set_test.f08`

**Interfaces:**
- Consumes: existing `cache_t` API (`init`, `build_node_set`, `resolve_shape`, `compute_radius_and_derivative`, `compute_radius_grid`, `compute_radius_grid_with_com_shift`, `n_grid_get`).
- Produces: `cache_init_s(self, max_beta_params, [n_grid], error_code, message)` with `optional :: n_grid` (absent ⇒ node-set-only cache, `n_grid_get() == 0`); public constant `LEGENDRE_ERROR_NO_UNIFORM_GRID = 10_ik`; both legacy uniform entry points return it on a table-less cache. Tasks 2–3 rely on these exact names.

- [ ] **Step 1: Check callers of `cache_init_s` for positional compatibility**

Run:
```bash
grep -rn "%init(" /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/src /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/tests
```
Expected: all callers pass `n_grid` positionally as the 2nd argument — they keep working unchanged because an `optional` dummy in place still binds positionally. (WMMM's `beta_cache%init(..., BETA_CACHE_LEGACY_GRID_POINTS, ...)` in the RewriteProject is likewise unaffected.)

- [ ] **Step 2: Write the failing Fortran test**

In `tests/beta_param_node_set_test.f08`:

1. Extend the `use beta_parameterization_mod, only:` list with `LEGENDRE_ERROR_NO_UNIFORM_GRID`.
2. Add `call test_node_set_only_cache()` to the call list before `call test_summary()`.
3. Add the subroutine inside `contains`:

```fortran
    !> A cache built without n_grid serves the node-set API identically to a
    !> uniform-grid cache; the legacy uniform entry points refuse cleanly.
    subroutine test_node_set_only_cache()
        type(cache_t)        :: full, lean
        type(node_set_t)     :: ns_full, ns_lean
        real(kind = rk)      :: thetas(5), params(4)
        real(kind = rk)      :: bc_full(8), bc_lean(8)
        real(kind = rk)      :: r_full(5), r_lean(5), dr_full(5), dr_lean(5)
        real(kind = rk)      :: radii_buf(181)
        real(kind = rk)      :: corr_full, corr_lean, rn_f, rs_f, rn_l, rs_l
        integer(kind = ik)   :: code, i
        character(len = 256) :: message

        call lean%init(8_ik, error_code = code, message = message)
        call assert_int_eq(code, LEGENDRE_VALID, 'lean init: VALID')
        call assert_int_eq(lean%n_grid_get(), 0_ik, 'lean init: n_grid == 0')

        call full%init(8_ik, 181_ik, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'full init: VALID')

        thetas = [0.3_rk, 0.9_rk, 1.5_rk, 2.1_rk, 2.7_rk]
        params = [0.0_rk, 0.85_rk, 0.35_rk, 0.18_rk]

        call lean%build_node_set(thetas, ns_lean, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'lean node set: built')
        call full%build_node_set(thetas, ns_full, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'full node set: built')

        call lean%resolve_shape(params, bc_lean, corr_lean, rn_l, rs_l, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'lean resolve: VALID')
        call full%resolve_shape(params, bc_full, corr_full, rn_f, rs_f, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'full resolve: VALID')
        call assert_close(corr_lean, corr_full, 0.0_rk, 'resolve: corrected_beta10 identical')
        call assert_close(rn_l, rn_f, 0.0_rk, 'resolve: r_north identical')
        call assert_close(rs_l, rs_f, 0.0_rk, 'resolve: r_south identical')

        call lean%compute_radius_and_derivative(bc_lean, ns_lean, r_lean, dr_lean, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'lean evaluate: VALID')
        call full%compute_radius_and_derivative(bc_full, ns_full, r_full, dr_full, code, message)
        call assert_int_eq(code, LEGENDRE_VALID, 'full evaluate: VALID')
        do i = 1_ik, 5_ik
            call assert_close(r_lean(i), r_full(i), 0.0_rk, 'evaluate: radii identical')
            call assert_close(dr_lean(i), dr_full(i), 0.0_rk, 'evaluate: dr identical')
        end do

        call lean%compute_radius_grid(params, radii_buf, code, message)
        call assert_int_eq(code, LEGENDRE_ERROR_NO_UNIFORM_GRID, &
                'lean radius_grid: NO_UNIFORM_GRID')
        call assert_true(len_trim(message) > 0, 'lean radius_grid: message non-empty')
        call lean%compute_radius_grid_with_com_shift(params, radii_buf, corr_lean, code, message)
        call assert_int_eq(code, LEGENDRE_ERROR_NO_UNIFORM_GRID, &
                'lean com radius_grid: NO_UNIFORM_GRID')
    end subroutine test_node_set_only_cache
```

- [ ] **Step 3: Build to verify it fails**

Run:
```bash
cmake --build /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build -j 8 2>&1 | tail -5
```
Expected: compile error — `LEGENDRE_ERROR_NO_UNIFORM_GRID` not found in `beta_parameterization_mod`.

- [ ] **Step 4: Implement in `src/beta_parameterization_mod.f08`**

4a. After `LEGENDRE_ERROR_POLE_NODE = 9_ik` (~line 79) add:

```fortran
    integer(kind = ik), parameter, public :: LEGENDRE_ERROR_NO_UNIFORM_GRID  = 10_ik
```

4b. Rewrite `cache_init_s` — `n_grid` becomes optional; the uniform table is built only when it is present. Replace the signature, argument declarations, the `n_grid < 2` check, the `self%n_grid = n_grid` assignment, and the θ-table block; keep everything else (max_beta_params check, norm constants, GL nodes, `legendre_gl`) exactly as is:

```fortran
    !! @param[in]  n_grid          Optional number of θ grid points (≥ 2) for
    !!                             the legacy uniform-grid entry points. Absent
    !!                             ⇒ node-set-only cache: the uniform table is
    !!                             not allocated and compute_radius_grid[_with_
    !!                             com_shift] return LEGENDRE_ERROR_NO_UNIFORM_GRID.
    subroutine cache_init_s(self, max_beta_params, n_grid, error_code, message)

        class(cache_t),     intent(out) :: self
        integer(kind = ik), intent(in)  :: max_beta_params
        integer(kind = ik), intent(in), optional :: n_grid
        integer(kind = ik), intent(out) :: error_code
        character(len = *), intent(out) :: message
```

with the body edits:

```fortran
        if (present(n_grid)) then
            if (n_grid < 2_ik) then
                error_code = LEGENDRE_ERROR_INVALID_MAX_PARAMS  ! reuse — bad init parameter
                write(message, '(A,I0)') 'n_grid must be >= 2, got ', n_grid
                return
            end if
            self%n_grid = n_grid
        else
            self%n_grid = 0_ik    ! node-set-only cache: no uniform table
        end if
```

and wrap the θ-table block:

```fortran
        ! Legendre P_k at the θ grid — only when a uniform grid was requested.
        if (present(n_grid)) then
            allocate(self%legendre_theta_grid(n_grid, max_beta_params + 1_ik))
            allocate(theta_x(n_grid))
            h = PI_C / real(n_grid - 1_ik, rk)
            do i = 1_ik, n_grid
                theta_x(i) = cos(real(i - 1_ik, rk) * h)
            end do
            call precompute_legendre_table_s(theta_x, max_beta_params, self%legendre_theta_grid)
            deallocate(theta_x)
        end if
```

4c. In **both** `cache_compute_radius_grid_s` and `cache_compute_radius_grid_with_com_shift_s`, insert a guard immediately after the `radii(:) = 0.0_rk` initialization (before the param-count check):

```fortran
        if (self%n_grid == 0_ik) then
            error_code = LEGENDRE_ERROR_NO_UNIFORM_GRID
            message    = 'cache built without a uniform grid (init n_grid absent); use the node-set API'
            return
        end if
```

(`cache_compute_radius_grid_with_com_shift_s` also zero-inits `corrected_beta10`; the guard goes after all the intent(out) initializations.)

4d. Confirm no other consumer of the table:
```bash
grep -n "legendre_theta_grid" /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/src/*.f08
```
Expected hits only in the type declaration, `cache_init_s`, `cache_destroy_s`, and the two legacy compute subroutines.

- [ ] **Step 5: Build and run ctest**

Run:
```bash
cmake --build /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build -j 8 && \
ctest --test-dir /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build --output-on-failure
```
Expected: all suites pass, including the extended `node_set` (goldens byte-identical — no math changed).

- [ ] **Step 6: Commit (beta-parameterization repo)**

```bash
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization add src/beta_parameterization_mod.f08 tests/beta_param_node_set_test.f08
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization commit -m "feat: optional n_grid — node-set-only cache, NO_UNIFORM_GRID error for legacy entry points"
```

---

### Task 2: beta-parameterization C API + headers — `n_grid <= 0` sentinel

**Repo:** `/mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization`

**Files:**
- Modify: `src/beta_parameterization_c_api_mod.f08` (`beta_param_cache_create`, ~line 65)
- Modify: `include/beta_parameterization.h` (error defines ~line 52; `beta_param_cache_create` doc ~line 70)
- Modify: `include/beta_parameterization.hpp` (`Cache` constructors, ~line 50)
- Test: `tests/c_api_smoke_test.cpp`

**Interfaces:**
- Consumes: Task 1's optional-`n_grid` `cache_init_s` and `LEGENDRE_ERROR_NO_UNIFORM_GRID = 10`.
- Produces: `beta_param_cache_create(max_beta_params, n_grid, …)` where `n_grid <= 0` builds a node-set-only cache; `#define BETA_PARAM_ERROR_NO_UNIFORM_GRID 10`; C++ `explicit Cache(int max_beta_params)` delegating constructor. Task 3's ctypes binding passes `0` for "no grid".

- [ ] **Step 1: Write the failing smoke-test additions**

In `tests/c_api_smoke_test.cpp`, after the existing raw-C-API section (before the C++ wrapper section), add:

```cpp
    // --- Node-set-only cache: n_grid <= 0 sentinel ---
    beta_param_cache_t* lean = beta_param_cache_create(
            8, 0, static_cast<int>(buf.size()), buf.data());
    check(lean != nullptr, "C: create(8, 0) node-set-only cache succeeds");
    s = beta_param_cache_compute_radius_grid(
            lean, params.data(), static_cast<int>(params.size()),
            radii.data(), static_cast<int>(buf.size()), buf.data());
    check(s == BETA_PARAM_ERROR_NO_UNIFORM_GRID,
          "C: uniform entry point on node-set-only cache -> code 10");
    check(buf[0] != '\0', "C: NO_UNIFORM_GRID message is non-empty");
    beta_param_cache_destroy(lean);
```

and in the C++ wrapper section:

```cpp
    beta_param::Cache lean_cxx{8};
    check(lean_cxx.n_grid() == 0, "C++: single-arg Cache reports n_grid 0");
```

- [ ] **Step 2: Build to verify it fails**

Run:
```bash
cmake --build /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build -j 8 2>&1 | tail -5
```
Expected: compile error — `BETA_PARAM_ERROR_NO_UNIFORM_GRID` undeclared and no single-arg `Cache` constructor.

- [ ] **Step 3: Implement**

3a. `include/beta_parameterization.h` — after `#define BETA_PARAM_ERROR_POLE_NODE 9`:

```c
#define BETA_PARAM_ERROR_NO_UNIFORM_GRID     10
```

and extend the `beta_param_cache_create` doc comment:

```c
/**
 * ...
 * n_grid <= 0 builds a node-set-only cache: the uniform-grid entry points
 * (beta_param_cache_compute_radius_grid[_with_com_shift]) then return
 * BETA_PARAM_ERROR_NO_UNIFORM_GRID; the node-set API is unaffected.
 */
```

3b. `src/beta_parameterization_c_api_mod.f08` — in `beta_param_cache_create`, replace the single `call cache_ptr%init(...)` with:

```fortran
        if (n_grid <= 0_ik_c) then
            call cache_ptr%init(int(max_beta_params, ik), &
                    error_code = error_code, message = f_message)
        else
            call cache_ptr%init(int(max_beta_params, ik), int(n_grid, ik), &
                    error_code, f_message)
        end if
```

3c. `include/beta_parameterization.hpp` — add a delegating constructor right after the existing two-argument constructor:

```cpp
    /** Node-set-only cache: uniform-grid entry points return
     *  BETA_PARAM_ERROR_NO_UNIFORM_GRID (the wrapper's own size check
     *  rejects non-empty buffers first, since n_grid() == 0). */
    explicit Cache(int max_beta_params) : Cache(max_beta_params, 0) {}
```

- [ ] **Step 4: Build and run ctest**

Run:
```bash
cmake --build /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build -j 8 && \
ctest --test-dir /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build --output-on-failure
```
Expected: all pass, including `c_api_smoke`.

- [ ] **Step 5: Commit**

```bash
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization add src/beta_parameterization_c_api_mod.f08 include/beta_parameterization.h include/beta_parameterization.hpp tests/c_api_smoke_test.cpp
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization commit -m "feat: C API n_grid<=0 sentinel + C++ single-arg Cache for node-set-only caches"
```

---

### Task 3: beta-parameterization Python binding, version 2.3.0, tag, rebuild both trees

**Repo:** `/mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization`

**Files:**
- Modify: `python/beta_parameterization/api.py` (`Status` enum ~line 29; `Cache.__init__` ~line 131)
- Modify: `CMakeLists.txt` (VERSION, line 6), `python/pyproject.toml` (version, line 7)
- Test: `python/tests/test_api.py`

**Interfaces:**
- Consumes: Task 2's C sentinel.
- Produces: `bp.Cache(max_beta_params, n_grid=None)` (None ⇒ node-set-only, `cache.n_grid == 0`); `bp.Status.ERROR_NO_UNIFORM_GRID == 10`. Task 4 constructs `bp.Cache(max_beta_params=8)`.

- [ ] **Step 1: Write the failing Python tests**

Append to `python/tests/test_api.py`:

```python
def test_node_set_only_cache_matches_full() -> None:
    thetas = np.linspace(0.2, np.pi - 0.2, 9)
    with bp.Cache(max_beta_params=8) as lean, \
            bp.Cache(max_beta_params=8, n_grid=181) as full:
        assert lean.n_grid == 0
        rs_lean = lean.resolve_shape(G2_PARAMS)
        rs_full = full.resolve_shape(G2_PARAMS)
        assert rs_lean.ok and rs_full.ok
        assert rs_lean.corrected_beta10 == rs_full.corrected_beta10
        rd_lean = lean.radius_and_derivative(rs_lean.beta_con, lean.build_node_set(thetas))
        rd_full = full.radius_and_derivative(rs_full.beta_con, full.build_node_set(thetas))
        assert rd_lean.ok and rd_full.ok
        np.testing.assert_array_equal(rd_lean.radii, rd_full.radii)
        np.testing.assert_array_equal(rd_lean.dr_dtheta, rd_full.dr_dtheta)


def test_node_set_only_cache_rejects_uniform_entry_points() -> None:
    with bp.Cache(max_beta_params=8) as lean:
        res = lean.radius_grid(G1_PARAMS)
        assert res.status == bp.Status.ERROR_NO_UNIFORM_GRID
        assert not res.ok
        res_com = lean.radius_grid_with_com_shift(G1_PARAMS)
        assert res_com.status == bp.Status.ERROR_NO_UNIFORM_GRID
```

- [ ] **Step 2: Run to verify failure**

```bash
source /home/alex/.virtualenvs/default/bin/activate && \
python -m pytest /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/python/tests/test_api.py -v -k node_set_only
```
Expected: FAIL — `Cache.__init__` requires `n_grid`; `Status` has no `ERROR_NO_UNIFORM_GRID`.

- [ ] **Step 3: Implement in `python/beta_parameterization/api.py`**

3a. `Status` enum — after `POLE_NODE = 9`:

```python
    ERROR_NO_UNIFORM_GRID = 10
```

3b. `Cache.__init__` — `n_grid` optional (class docstring gains the note "n_grid=None builds a node-set-only cache; the uniform-grid entry points then return Status.ERROR_NO_UNIFORM_GRID."):

```python
    def __init__(self, max_beta_params: int, n_grid: int | None = None) -> None:
        lib = _get_lib()
        buf = ctypes.create_string_buffer(MESSAGE_BUFFER_SIZE)
        c_n_grid = 0 if n_grid is None else int(n_grid)
        handle = lib.beta_param_cache_create(
            int(max_beta_params), c_n_grid, MESSAGE_BUFFER_SIZE, buf)
        if not handle:
            raise BetaParamError(
                f"Cache init failed: {buf.value.decode(errors='replace')}")
        self._handle: Optional[ctypes.c_void_p] = ctypes.c_void_p(handle)
        self.max_beta_params = int(max_beta_params)
        self.n_grid = c_n_grid
```

(`radius_grid` on a node-set-only cache then allocates a zero-length buffer and returns the library's status 10 — no special-casing needed.)

- [ ] **Step 4: Run the full library Python suite**

```bash
python -m pytest /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/python/tests/ -v
```
Expected: all pass (existing goldens untouched).

- [ ] **Step 5: Bump version to 2.3.0**

- `CMakeLists.txt` line 6: `VERSION 2.2.2` → `VERSION 2.3.0`
- `python/pyproject.toml` line 7: `version = "2.2.2"` → `version = "2.3.0"`

- [ ] **Step 6: Rebuild BOTH trees and re-run everything**

```bash
cmake --build /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build -j 8 && \
cmake --build /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build/release -j 8 && \
ctest --test-dir /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build --output-on-failure && \
python -m pytest /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/python/tests/ -q
```
Expected: both builds succeed, all tests pass. If `build/release` is not a configured CMake tree, configure it first: `cmake -S /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization -B /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization/build/release -DCMAKE_BUILD_TYPE=Release`.

- [ ] **Step 7: Commit and tag**

```bash
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization add python/beta_parameterization/api.py python/tests/test_api.py CMakeLists.txt python/pyproject.toml
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization commit -m "feat(python): Cache(n_grid=None) node-set-only caches; version 2.3.0"
git -C /mnt/c/Users/aleks/CLionProjects/Fortran/beta-parameterization tag 2.3.0
```

---

### Task 4: ShapePlotters — drop `N_GRID` (beta), rename it (fos), grep gate

**Repo:** `/mnt/c/Users/aleks/PycharmProjects/ShapePlotters` (all remaining tasks)

**Files:**
- Modify: `src/renders/beta.py:23-25,65`
- Modify: `src/renders/fos.py:18,38`
- Modify: `tests/test_beta_render.py:72-74`

**Interfaces:**
- Consumes: `bp.Cache(max_beta_params=8)` from Task 3 (verify first — Step 1).
- Produces: `beta.py` with no `N_GRID`; `fos.py` with `N_PROFILE_POINTS = 721`. No signature changes.

- [ ] **Step 1: Verify 2.3.0 is what the venv imports**

```bash
source /home/alex/.virtualenvs/default/bin/activate && python -c "
import inspect
import beta_parameterization as bp
print(inspect.signature(bp.Cache.__init__))
print(bp.Status.ERROR_NO_UNIFORM_GRID)
"
```
Expected: `(self, max_beta_params: 'int', n_grid: 'int | None' = None)` and `Status.ERROR_NO_UNIFORM_GRID`. If this fails, Task 3 was not completed (or the build trees are stale).

- [ ] **Step 2: Update the legacy-parity test to use its own uniform-grid cache**

In `tests/test_beta_render.py`, add the import at the top (after `import pytest`):

```python
import beta_parameterization as bp
```

and replace the last two statements of `test_com_corrected_overlay` (the `ref = render._cache.radius_grid_with_com_shift(...)` block) with:

```python
    # Legacy-API parity against a separate uniform-grid cache — the render's
    # own cache is node-set-only as of beta-parameterization 2.3.0.
    with bp.Cache(max_beta_params=8, n_grid=181) as legacy:
        ref = legacy.radius_grid_with_com_shift([0.0, 0.85, 0.35, 0.18])
    assert res.scalars["corrected_beta10"] == pytest.approx(ref.corrected_beta10, abs=1e-14)
```

- [ ] **Step 3: Edit the renders**

`src/renders/beta.py` — delete lines 23–25 (the "Cache constructor still requires…" comment and `N_GRID = 721`) and change the constructor call:

```python
    def __init__(self) -> None:
        self._cache = bp.Cache(max_beta_params=N_BETAS)
        self._node_set = self._cache.build_node_set(nodes.THETA)
```

`src/renders/fos.py` — rename the constant (line 18) and its use (line 38):

```python
# rho(z) display-panel resolution (native COM-frame profile) — a display
# choice, not a calculation grid: the GL theta-nodes sample the star-convex
# frame and cannot replace it.
N_PROFILE_POINTS = 721
```

```python
        prof = fp.rho_profile(arr, N_PROFILE_POINTS)
```

- [ ] **Step 4: Grep gate + full suite**

```bash
grep -rn "n_grid\|N_GRID" /mnt/c/Users/aleks/PycharmProjects/ShapePlotters/src/ ; echo "exit=$?"
python -m pytest -q
```
Expected: grep exits 1 (no matches in `src/`); full suite passes (GL goldens unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/renders/beta.py src/renders/fos.py tests/test_beta_render.py
git commit -m "feat: node-set-only beta cache (lib 2.3.0); N_GRID retired, fos display constant renamed"
```

---

### Task 5: Beta overlay trigger — compare against the slider β₁

**Files:**
- Modify: `src/renders/beta.py:8-12` (module docstring), `:28-30` (threshold comment), `:114` (trigger)
- Test: `tests/test_beta_render.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: overlay shown iff `abs(corrected.corrected_beta10 - betas[0]) > OVERLAY_BETA10_THRESHOLD`. Task 6's beta `energy_requests` keys off `result.overlay_z is not None`, so this trigger decides the 1-vs-2 energy split.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_beta_render.py`:

```python
def test_overlay_when_slider_beta1_differs(render: BetaRender) -> None:
    # Symmetric multipoles with a nonzero slider beta1: corrected_beta10 ~ 0,
    # yet the slider shape differs from the COM-corrected one — beta10 is a
    # shape parameter, not a translation. The overlay must show.
    res = render.compute(_params([0.5, 0.30, 0.0, 0.10]), {})
    assert res.ok
    assert abs(res.scalars["corrected_beta10"]) <= 1e-3
    assert res.overlay_z is not None


def test_no_overlay_when_slider_matches_corrected(render: BetaRender) -> None:
    # Slider beta1 set to the corrected dipole: the two shapes coincide.
    first = render.compute(_params([0.0, 0.85, 0.35, 0.18]), {})
    corrected = first.scalars["corrected_beta10"]
    res = render.compute(_params([corrected, 0.85, 0.35, 0.18]), {})
    assert res.ok
    assert res.overlay_z is None
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_beta_render.py -v -k "slider_beta1_differs or slider_matches_corrected"
```
Expected: `test_overlay_when_slider_beta1_differs` FAILS (`overlay_z is None` under the old trigger); `test_no_overlay_when_slider_matches_corrected` may pass or fail — the fix must turn both green.

- [ ] **Step 3: Implement in `src/renders/beta.py`**

3a. Threshold comment (replace lines 28–30):

```python
# The COM-corrected shape coincides with the slider shape when the corrected
# dipole equals the slider beta1 (beta10 is a shape parameter, not a
# translation knob); below this |corrected_beta10 - beta1| the orange overlay
# is suppressed (slider units).
OVERLAY_BETA10_THRESHOLD = 0.001
```

3b. Trigger (line 114):

```python
            if ok and abs(corrected.corrected_beta10 - betas[0]) > OVERLAY_BETA10_THRESHOLD:
```

3c. Module docstring — replace the last sentence of the second paragraph ("— shown only when it differs meaningfully from the slider shape.") with:

```
shown only when corrected_beta10 differs from the slider beta1 by more than
the overlay threshold (the two shapes then genuinely differ).
```

- [ ] **Step 4: Full suite**

```bash
python -m pytest -q
```
Expected: all pass — the engine overlay tests (`test_no_beta_overlay_when_symmetric`, `test_beta_com_overlay_when_asymmetric`, `test_zcm_marker_tracks_com`) use slider β₁ = 0 and are unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/renders/beta.py tests/test_beta_render.py
git commit -m "fix: beta COM overlay triggers on |corrected_beta10 - slider beta1|"
```

---

### Task 6: `EnergyRequest` + render `energy_requests()`

**Files:**
- Modify: `src/core/result.py` (new dataclass + contract docstring)
- Modify: `src/renders/beta.py`, `src/renders/fos.py`
- Test: `tests/test_beta_render.py`, `tests/test_fos_render.py`

**Interfaces:**
- Consumes: `result.overlay_z` (Task 5 semantics).
- Produces (Tasks 7–9 rely on these exactly):
  - `src.core.result.EnergyRequest(label: str, param_type: str, shape: tuple[float, ...], com_correction: bool)` — frozen dataclass.
  - `BetaRender.energy_requests(params, result) -> list[EnergyRequest]` — always `("slider", "legendre", 20-tuple, False)`; plus `("COM corrected", "legendre", same 20-tuple, True)` when `result.overlay_z is not None`.
  - `FoSRender.energy_requests(params, result) -> list[EnergyRequest]` — always exactly `("FoS", "fos", 7-tuple, True)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_beta_render.py`:

```python
def test_energy_requests_single_without_overlay(render: BetaRender) -> None:
    p = _params([0.0, 0.30])
    res = render.compute(p, {})
    assert res.overlay_z is None
    (req,) = render.energy_requests(p, res)
    assert (req.label, req.param_type, req.com_correction) == ("slider", "legendre", False)
    assert len(req.shape) == 20                      # WMMM's legendre width
    assert req.shape[:8] == (0.0, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert req.shape[8:] == (0.0,) * 12              # zero-padded tail


def test_energy_requests_both_with_overlay(render: BetaRender) -> None:
    p = _params([0.0, 0.85, 0.35, 0.18])
    res = render.compute(p, {})
    assert res.overlay_z is not None
    reqs = render.energy_requests(p, res)
    assert [r.label for r in reqs] == ["slider", "COM corrected"]
    assert [r.com_correction for r in reqs] == [False, True]
    assert reqs[0].shape == reqs[1].shape            # WMMM recomputes beta10 itself
```

Append to `tests/test_fos_render.py` (construct the render inline; keep the test self-contained):

```python
def test_energy_requests_always_single() -> None:
    from src.renders.fos import FoSRender
    render = FoSRender()
    p = dict(c=2.0, a3=0.2, a4=0.6, a5=0.0, a6=0.0, a7=0.0, a8=0.0)
    res = render.compute(p, {})
    assert res.ok
    (req,) = render.energy_requests(p, res)
    assert (req.label, req.param_type, req.com_correction) == ("FoS", "fos", True)
    assert req.shape == (2.0, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_beta_render.py tests/test_fos_render.py -v -k energy_requests
```
Expected: FAIL — `energy_requests` not defined.

- [ ] **Step 3: Implement**

3a. `src/core/result.py` — add after `ToggleSpec`:

```python
@dataclass(frozen=True)
class EnergyRequest:
    """One WMMM point computation the engine should run on an Energy click.

    Renders own the physics semantics (how many shapes are on screen and in
    which parameter convention); the engine just iterates requests. Carries
    no wmmm dependency — src/core/energy.py resolves it.
    """
    label: str                 # stats-block header; mirrors the plot legend
    param_type: str            # "legendre" | "fos"
    shape: tuple[float, ...]   # raw slider values in WMMM's convention
    com_correction: bool       # WMMM's beta_10_com_shift flag
```

and extend the module docstring contract list with:

```
    energy_requests(params: dict[str, float], result: ShapeResult)
        -> list[EnergyRequest]           # WMMM computations for this shape(s)
```

3b. `src/renders/beta.py` — extend the result import to include `EnergyRequest`, add the constant next to `N_BETAS`:

```python
# WMMM's legendre parameterization takes 20 betas; sliders drive the first 8.
WMMM_N_LEGENDRE_PARAMS = 20
```

and add the method to `BetaRender`:

```python
    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """WMMM requests: the slider (blue) shape, plus the COM-corrected
        (orange) shape when the overlay is on screen. WMMM recomputes beta10
        itself under com_correction=True, so both carry the same betas."""
        shape = tuple(params[f"beta{i}"] for i in range(1, N_BETAS + 1)) \
            + (0.0,) * (WMMM_N_LEGENDRE_PARAMS - N_BETAS)
        requests = [EnergyRequest("slider", "legendre", shape, com_correction=False)]
        if result.overlay_z is not None:
            requests.append(
                EnergyRequest("COM corrected", "legendre", shape, com_correction=True))
        return requests
```

3c. `src/renders/fos.py` — extend the result import with `EnergyRequest`, add to `FoSRender`:

```python
    def energy_requests(self, params: dict[str, float],
                        result: ShapeResult) -> list[EnergyRequest]:
        """One request — the FoS dashed overlay is a reframing of the same
        shape, not a second shape; com_correction is inert on the FoS path."""
        return [EnergyRequest("FoS", "fos",
                              tuple(params[k] for k in PARAM_KEYS),
                              com_correction=True)]
```

- [ ] **Step 4: Run and commit**

```bash
python -m pytest -q
git add src/core/result.py src/renders/beta.py src/renders/fos.py tests/test_beta_render.py tests/test_fos_render.py
git commit -m "feat: EnergyRequest render contract — beta 1-2 requests keyed to the overlay, fos always one"
```

---

### Task 7: `src/core/energy.py` — the WMMM adapter

**Files:**
- Create: `src/core/energy.py`
- Create: `tests/test_energy.py`

**Interfaces:**
- Consumes: nothing from this repo (deliberately standalone; only stdlib).
- Produces (Task 8 relies on these exactly):
  - `energy.available() -> bool`
  - `energy.compute(param_type: str, z: int, n: int, shape: Sequence[float], com_correction: bool = True) -> EnergyResult`
  - `energy.EnergyResult` — frozen dataclass, fields: `is_valid: bool`, `mass_excess`, `total_energy`, `macro_energy`, `micro_energy`, `surface_energy`, `coulomb_energy`, `proton_pairing_gap`, `neutron_pairing_gap` (floats), `proton_k`, `neutron_k` (ints), `corrected_beta10: float`, `error: str | None = None`.
  - Module-level `_models: dict[tuple[str, bool], object]` cache (tests reset it).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_energy.py`:

```python
"""Energy adapter over a stub wmmm module — no real WMMM anywhere.

Committed tests carry no WMMM-derived numbers; the stub returns arbitrary
sentinels. The stub needs a real ModuleSpec: find_spec raises on modules
whose __spec__ is None."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

from src.core import energy


def _install_stub(monkeypatch, log: list) -> types.ModuleType:
    stub = types.ModuleType("wmmm")
    stub.__spec__ = importlib.machinery.ModuleSpec("wmmm", loader=None)

    class Model:
        def __init__(self, param_type: str) -> None:
            log.append(("init", param_type))

        def set(self, name: str, value: float) -> None:
            log.append(("set", name, value))

        def compute(self, z: int, n: int, shape: list) -> SimpleNamespace:
            log.append(("compute", z, n, tuple(shape)))
            return SimpleNamespace(
                is_valid=True, mass_excess=1.0, total_energy=2.0,
                macro_energy=3.0, micro_energy=4.0, surface_energy=5.0,
                coulomb_energy=6.0, proton_pairing_gap=7.0,
                neutron_pairing_gap=8.0, proton_k=9, neutron_k=10,
                corrected_beta10=0.25, has_neck=False,
                neck_radius_fm=0.0, is_scissioning=False)

    stub.Model = Model
    monkeypatch.setitem(sys.modules, "wmmm", stub)
    return stub


@pytest.fixture(autouse=True)
def fresh_model_cache(monkeypatch):
    monkeypatch.setattr(energy, "_models", {})


def test_available_true_with_stub(monkeypatch) -> None:
    _install_stub(monkeypatch, [])
    assert energy.available() is True


def test_available_false_when_not_installed(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "wmmm", raising=False)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "wmmm" else real_find_spec(name, *a))
    assert energy.available() is False


def test_available_does_not_import(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "wmmm", raising=False)
    energy.available()
    assert "wmmm" not in sys.modules


def test_compute_maps_pointresult_fields(monkeypatch) -> None:
    log: list = []
    _install_stub(monkeypatch, log)
    res = energy.compute("fos", 92, 144, (2.0, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0))
    assert res.error is None and res.is_valid is True
    assert (res.mass_excess, res.total_energy, res.macro_energy, res.micro_energy,
            res.surface_energy, res.coulomb_energy) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert (res.proton_pairing_gap, res.neutron_pairing_gap) == (7.0, 8.0)
    assert (res.proton_k, res.neutron_k) == (9, 10)
    assert res.corrected_beta10 == 0.25
    assert ("compute", 92, 144, (2.0, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0)) in log


def test_model_cached_per_config(monkeypatch) -> None:
    log: list = []
    _install_stub(monkeypatch, log)
    energy.compute("legendre", 92, 144, [0.0] * 20, com_correction=False)
    energy.compute("legendre", 92, 144, [0.1] * 20, com_correction=False)
    energy.compute("legendre", 92, 144, [0.0] * 20, com_correction=True)
    assert len([e for e in log if e[0] == "init"]) == 2   # one per (type, com)
    assert [e for e in log if e[0] == "set"] == \
        [("set", "beta_10_com_shift", 0.0)]               # once, com-off model only


def test_compute_failure_returns_error_result(monkeypatch) -> None:
    stub = _install_stub(monkeypatch, [])

    class ExplodingModel:
        def __init__(self, param_type: str) -> None:
            raise RuntimeError("library not built")

    stub.Model = ExplodingModel
    res = energy.compute("fos", 92, 144, (1.0,) * 7)
    assert res.error is not None and "library not built" in res.error
    assert res.is_valid is False and res.total_energy == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_energy.py -v
```
Expected: collection error — `src.core.energy` does not exist.

- [ ] **Step 3: Implement `src/core/energy.py`**

```python
"""Optional WMMM energy adapter — the only module allowed to import wmmm.

The energy model is strictly local: wmmm is never a dependency of this
package (not even an optional extra) and is installed out-of-band as a local
editable install. This module carries the calling convention only — no model
code, data, or paths. The engine consults available() to decide whether the
Energy button exists; compute() returns EnergyResult, with failures in
EnergyResult.error rather than exceptions.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EnergyResult:
    """One WMMM point result (energies in MeV); zero-filled when error is set."""
    is_valid: bool
    mass_excess: float
    total_energy: float
    macro_energy: float
    micro_energy: float
    surface_energy: float
    coulomb_energy: float
    proton_pairing_gap: float
    neutron_pairing_gap: float
    proton_k: int
    neutron_k: int
    corrected_beta10: float     # cross-check channel (vs the render scalar), not displayed
    error: str | None = None


# Model handles are configuration-immutable: one per (param_type,
# com_correction) for the app lifetime (~20 ms per compute thereafter).
_models: dict[tuple[str, bool], object] = {}


def available() -> bool:
    """True when the wmmm package is importable; performs no import."""
    return importlib.util.find_spec("wmmm") is not None


def _error(message: str) -> EnergyResult:
    return EnergyResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0, 0, 0.0, error=message)


def _get_model(param_type: str, com_correction: bool):
    key = (param_type, com_correction)
    model = _models.get(key)
    if model is None:
        import wmmm  # deferred: loads the Fortran library on first use only
        model = wmmm.Model(param_type)
        if not com_correction:
            model.set("beta_10_com_shift", 0.0)
        _models[key] = model
    return model


def compute(param_type: str, z: int, n: int, shape: Sequence[float],
            com_correction: bool = True) -> EnergyResult:
    """One WMMM point computation; never raises.

    Parameters
    ----------
    param_type : str
        "legendre" (20 shape params) or "fos" (7).
    z, n : int
        Proton and neutron numbers.
    shape : Sequence[float]
        Shape parameters in WMMM's convention (raw slider values).
    com_correction : bool
        WMMM's beta_10_com_shift flag. False evaluates beta10 as given
        (legendre only — inert on the fos path).
    """
    try:
        model = _get_model(param_type, com_correction)
        r = model.compute(int(z), int(n), list(shape))
    except Exception as exc:  # any wmmm failure becomes a stats-box message
        return _error(f"{type(exc).__name__}: {exc}")
    return EnergyResult(
        is_valid=bool(r.is_valid), mass_excess=r.mass_excess,
        total_energy=r.total_energy, macro_energy=r.macro_energy,
        micro_energy=r.micro_energy, surface_energy=r.surface_energy,
        coulomb_energy=r.coulomb_energy,
        proton_pairing_gap=r.proton_pairing_gap,
        neutron_pairing_gap=r.neutron_pairing_gap,
        proton_k=int(r.proton_k), neutron_k=int(r.neutron_k),
        corrected_beta10=r.corrected_beta10)
```

- [ ] **Step 4: Run and commit**

```bash
python -m pytest tests/test_energy.py -v && python -m pytest -q
git add src/core/energy.py tests/test_energy.py
git commit -m "feat: WMMM energy adapter — find_spec guard, lazy import, per-config model cache"
```

---

### Task 8: Engine — Energy button, stats blocks, clear-on-change

**Files:**
- Modify: `src/core/engine.py` (import ~line 18; `__init__` ~line 39; `_build_widgets` ~line 134; `_suppress_widget_draws` ~line 157; `update()` stats line ~line 318; new methods after `_save`)
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `energy.available()`, `energy.compute(...)`, `energy.EnergyResult` (Task 7); `render.energy_requests(params, result)` (Task 6).
- Produces: `app.btn_energy` (`None` when unavailable), `app._on_energy(event=None)`, `app._energy_lines: list[str]`, `app._stats_base: str`, `app._refresh_stats()`, `app._energy_block(label, res) -> list[str]`. Stats-block headers: `"WMMM [MeV]:"` single-request, `"WMMM (<label>) [MeV]:"` multi-request.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_engine.py -v -k energy
```
Expected: FAIL — `src.core.engine` has no attribute `energy` / `btn_energy`.

- [ ] **Step 3: Implement in `src/core/engine.py`**

3a. Import (line 18): `from src.core import energy, quadrature`

3b. `__init__` — before `self._build_figure()`:

```python
        self._stats_base = ""
        self._energy_lines: list[str] = []
```

3c. `_build_widgets` — after the `self.checks` block (which computes `n`), add:

```python
        self.btn_energy = None
        if energy.available():
            # Right column, below the toggle block; absent (layout untouched)
            # when the local WMMM install is missing.
            self.btn_energy = Button(
                self.fig.add_axes((0.86, 0.26 - 0.04 * n - 0.045, 0.10, 0.030)),
                "Energy")
            self.btn_energy.on_clicked(self._on_energy)
```

3d. `_suppress_widget_draws` — include the button when present:

```python
        widgets: list = [self.btn_reset, self.btn_save, self.checks]
        if self.btn_energy is not None:
            widgets.append(self.btn_energy)
```

3e. `update()` — replace the `self.stats_text.set_text(self._stats_block(...))` line with:

```python
        self._stats_base = self._stats_block(result, scale, unit, v, s, zc)
        self._energy_lines = []   # any shape/Z/N/unit change invalidates energies
        self._refresh_stats()
```

3f. New methods after `_save`:

```python
    def _refresh_stats(self) -> None:
        self.stats_text.set_text("\n".join([self._stats_base, *self._energy_lines]))

    def _on_energy(self, _event=None) -> None:
        result = self.last_result
        if result is None or not result.ok:
            self._energy_lines = ["", "WMMM: shape invalid (not computed)"]
        else:
            params = {k: row.slider.val for k, row in self.rows.items()}
            requests = self.render.energy_requests(params, result)
            lines: list[str] = []
            for req in requests:
                res = energy.compute(
                    req.param_type, self.z_box.value, self.n_box.value,
                    req.shape, com_correction=req.com_correction)
                lines += self._energy_block(
                    req.label if len(requests) > 1 else None, res)
            self._energy_lines = lines
        self._refresh_stats()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _energy_block(label: str | None, res: energy.EnergyResult) -> list[str]:
        # Energies are MeV — deliberately outside the fm/R0 unit toggle.
        head = "WMMM" if label is None else f"WMMM ({label})"
        if res.error is not None:
            return ["", f"{head}: error", f"  {res.error}"]
        if not res.is_valid:
            return ["", f"{head}: invalid shape"]
        return ["", f"{head} [MeV]:",
                f"  E_total = {res.total_energy:.4f}",
                f"  E_macro = {res.macro_energy:.4f}",
                f"  E_micro = {res.micro_energy:.4f}",
                f"  mass_ex = {res.mass_excess:.4f}",
                f"  E_surf  = {res.surface_energy:.4f}",
                f"  E_coul  = {res.coulomb_energy:.4f}",
                f"  gap_p = {res.proton_pairing_gap:.4f}  k_p = {res.proton_k}",
                f"  gap_n = {res.neutron_pairing_gap:.4f}  k_n = {res.neutron_k}"]
```

- [ ] **Step 4: Full suite**

```bash
python -m pytest -q
```
Expected: all pass. Note the pre-existing engine tests construct apps with the real `energy.available()` — in this venv wmmm IS installed, so `btn_energy` exists there; none of them assert on widget counts, so they stay green either way.

- [ ] **Step 5: Commit**

```bash
git add src/core/engine.py tests/test_engine.py
git commit -m "feat: optional WMMM Energy button — labeled MeV stats blocks, cleared by any change"
```

---

### Task 9: Local-only smoke test + gate

**Files:**
- Create: `tests/test_energy_local.py`
- No production changes.

**Interfaces:**
- Consumes: everything above, plus the real local wmmm install.
- Produces: the release gate for this project.

- [ ] **Step 1: Write the local-only smoke test**

Create `tests/test_energy_local.py`:

```python
"""Local-only WMMM smoke tests — skipped wherever wmmm is not installed.

Structural assertions only: no WMMM-derived numbers are committed. The
corrected_beta10 cross-check compares two live in-process values (render
scalar vs WMMM result) — two independent call paths through the same
library."""
from __future__ import annotations

import math

import pytest

from src.core import energy
from src.renders.beta import BetaRender
from src.renders.fos import FoSRender

pytestmark = pytest.mark.skipif(
    not energy.available(),
    reason="wmmm not installed (strictly local, out-of-band)")

_ENERGY_FIELDS = ("mass_excess", "total_energy", "macro_energy", "micro_energy",
                  "surface_energy", "coulomb_energy",
                  "proton_pairing_gap", "neutron_pairing_gap")


def test_beta_both_variants_and_com_crosscheck() -> None:
    render = BetaRender()
    values = [0.0, 0.85, 0.35, 0.18, 0.0, 0.0, 0.0, 0.0]
    params = {f"beta{i}": v for i, v in zip(range(1, 9), values)}
    res = render.compute(params, {})
    assert res.ok and res.overlay_z is not None
    requests = render.energy_requests(params, res)
    assert len(requests) == 2
    results = [energy.compute(r.param_type, 92, 144, r.shape,
                              com_correction=r.com_correction) for r in requests]
    for er in results:
        assert er.error is None
        assert er.is_valid
        assert all(math.isfinite(getattr(er, f)) for f in _ENERGY_FIELDS)
    # com off: beta10 evaluated as given (the slider value, here 0).
    assert results[0].corrected_beta10 == pytest.approx(0.0, abs=1e-12)
    # com on: WMMM's corrected dipole equals the render's — same library,
    # two independent call paths.
    assert results[1].corrected_beta10 == pytest.approx(
        res.scalars["corrected_beta10"], abs=1e-9)
    # The two variants are genuinely different shapes.
    assert results[0].total_energy != results[1].total_energy


def test_fos_single_request_smoke() -> None:
    render = FoSRender()
    params = dict(c=1.5, a3=0.1, a4=0.05, a5=0.0, a6=0.0, a7=0.0, a8=0.0)
    res = render.compute(params, {})
    assert res.ok
    (req,) = render.energy_requests(params, res)
    er = energy.compute(req.param_type, 92, 144, req.shape,
                        com_correction=req.com_correction)
    assert er.error is None and er.is_valid
    assert all(math.isfinite(getattr(er, f)) for f in _ENERGY_FIELDS)
```

- [ ] **Step 2: Run it (this venv has wmmm — it must not skip here)**

```bash
python -m pytest tests/test_energy_local.py -v
```
Expected: 2 passed (first run takes seconds — the Fortran library loads and initializes once).

- [ ] **Step 3: Full suite + perf + grep gates**

```bash
python -m pytest -q
python -m pytest tests/test_perf.py -v
grep -rn "n_grid\|N_GRID" src/ ; echo "grid grep exit=$?"
grep -rn "import wmmm\|from wmmm" src/ | grep -v "src/core/energy.py" ; echo "isolation grep exit=$?"
grep -n "wmmm" pyproject.toml ; echo "pyproject grep exit=$?"
```
Expected: suite green; perf baselines unchanged (energy computes happen only on click — the drag path is untouched); all three greps exit 1 (no matches).

- [ ] **Step 4: Commit**

```bash
git add tests/test_energy_local.py
git commit -m "test: local-only WMMM smoke — both beta variants, corrected_beta10 cross-check"
```

- [ ] **Step 5: Manual WSLg gate (requires the user / a display)**

```bash
python main.py beta   # then: python main.py fos
```
Checklist:
1. Energy button visible (wmmm installed in this venv) below the toggle block.
2. Beta: click Energy on the default sphere → one "WMMM [MeV]:" block; plausible energies for Z=92, N=144.
3. Beta: set β₃ = 0.4 (orange overlay appears) → click Energy → two blocks, "WMMM (slider)" and "WMMM (COM corrected)", with different E_total.
4. Move any slider → both blocks disappear.
5. Energy → Save → PNG carries the energy block alongside the matching shape.
6. FoS: click Energy → single block.
7. Button-absent case: covered headless by `test_energy_button_absent_when_unavailable` (no wmmm-free venv needed).

---

## Self-Review Notes

- Spec §2 → Tasks 1–3; §3 → Task 4; §4 → Task 5; §5 → Task 7; §6 → Task 6; §7 → Task 8; §8.1 → Tasks 1–3 tests; §8.2 → Task 4 Step 4; §8.3 → Task 5; §8.4 → Task 6; §8.5 → Task 7; §8.6 → Task 8; §8.7 → Task 9 Steps 1–2; §8.8 + gate → Task 9 Steps 3–5.
- Type consistency: `EnergyRequest` fields (`label/param_type/shape/com_correction`) match across Tasks 6–9; `EnergyResult` field list identical in Task 7 code, Task 7 tests, Task 8 `_fake_energy_result`, Task 9 `_ENERGY_FIELDS`.
- Ordering: Task 4 hard-depends on Task 3 (venv import check is Step 1); Tasks 5–8 are ShapePlotters-sequential; Task 9 last.
