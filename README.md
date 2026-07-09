# 2dfc-matlab-mex

A drop-in, mex-accelerated port of [2dfc-matlab](https://github.com/ayang923/2dfc-matlab): the
same public API (`Curve_obj`, `Curve_seq_obj`, `Q_patch_obj`, `S_patch_obj`, `C1_patch_obj`,
`C2_patch_obj`, `R_cartesian_mesh_obj`, `FC2D`, `FC2D_patches`, ...), with the numerically
expensive inner routines backed by compiled C (via MEX) instead of pure MATLAB. Every public
function/class/method keeps its original name, signature, and behavior — only the implementation
underneath a handful of hot routines changes, and only when the compiled mex kernels are present
(see [Fallback behavior](#fallback-behavior)).

On the boomerang example (`examples/2DFC-examples/boomerang_2D_FC.m`, `h=0.01`, 192x282 output
grid), this port reproduces the original's `fc_err` to 3.4e-17 and `f_R` to 2.8e-10, in **0.35s**
vs the original's **54.0s** (~150x). On the 4-curve guitarbase example at its full shipped
resolution (`examples/2DFC-examples/guitarbase_2DFC.m`, `h=0.0005`, 3072x4374 output grid), this
port completes end-to-end (construction, POU normalization, FC extension, and interpolation onto
~13M grid points) in **8.0s**, with a relative L2 error of 4.5e-10.

## What changed vs. 2dfc-matlab, and why

Three things were asked for and are reflected directly in the design:

1. **Mex/C acceleration of the hot inner loops**, not a rewrite of the algorithm. See
   [Architecture](#architecture) below for exactly which routines and why.
2. **No Intel MKL / Intel compiler dependency.** The companion
   [2dfc-c](https://github.com/ayang923/2dfc-corners-c) reference implementation this port's C
   kernels are adapted from requires `icx` + MKL. This port only needs a standard CBLAS
   implementation (Accelerate on macOS, OpenBLAS or any other `cblas.h` provider on Linux/Windows)
   and compiles with any C11 compiler. See [No MKL](#no-mkl--no-lapack) below.
3. **FFT stays in MATLAB.** `R_cartesian_mesh_obj`'s spectral methods
   (`compute_fc_coeffs`, `ifft_interpolation`, `grad`, `div`, `lap`, `inv_lap`, `fft_filter`) were
   already pure MATLAB (`fft2`/`ifft2`) in the original — the *2dfc-c* C library additionally
   implemented its own MKL-DFTI-based FFT error-check routine (`r_cartesian_mesh_compute_fc_error`)
   for its standalone C examples, but that's not something this port needed to touch: no FFT code
   was added to the C side, and none needed to be.

## Architecture

### Full API mirror

Every `.m` file from the original is present under `src/` with the same name and public
signature. Most are untouched. A handful gained a mex dispatch at the top of the function body,
falling back to the original pure-MATLAB implementation when mex isn't available or doesn't apply:

| File | Status |
|---|---|
| `inpolygon_mesh.m` | dispatches to `inpolygon_mesh_mex` |
| `barylag.m` | dispatches to `barylag_mex` |
| `fcont_gram_blend_S.m` | dispatches to `fcont_gram_blend_S_mex` (CBLAS `dgemm`) |
| `Q_patch_obj.m` | `inverse_M_p` dispatches to `q_patch_inverse_M_p_mex` when the patch carries a `mex_spec` (see below); gained the `mex_spec` property and a 12th, optional constructor argument |
| `R_cartesian_mesh_obj.m` | `interpolate_patch` dispatches to `r_cartesian_mesh_interpolate_patch_mex` when the patch carries a `mex_spec` |
| `Curve_obj.m` | `construct_S_patch`/`construct_C_patch` now also build a `mex_spec` (curve samples + the patch's affine parameter remapping) and attach it to the patches they construct; new `get_mex_samples` method (memoized) |
| `S_patch_obj.m`, `C1_patch_obj.m`, `C2_patch_obj.m` | constructors and `.FC()` now thread `mex_spec` through to the `Q_patch_obj`s they build |
| `FC2D.m`, `FC2D_patches.m`, `Curve_seq_obj.m`, `fcont_gram_blend_C2.m` | **unchanged** — pure orchestration; they automatically benefit from the dispatch above with no changes of their own |
| `newton_solve.m`, `R_xi_eta_inversion.m`, `shift_idx_mesh.m` | **unchanged** — kept as the pure-MATLAB fallback path (used whenever a patch has no `mex_spec`, e.g. a `Q_patch_obj` built directly by a caller with an arbitrary `M_p`/`J` rather than through `Curve_obj`) |
| `mgs.m`, `precomp_fc_data.m` | **unchanged** — offline, symbolic-toolbox-based FC-matrix generation; not part of the runtime hot path (the runtime only ever loads already-generated `A`/`Q` matrices), so there's nothing to accelerate here |

Because the dispatch lives at the `Q_patch_obj`/`R_cartesian_mesh_obj` method level rather than at
the top of the `FC2D` pipeline, every caller of `.inverse_M_p()` — including the partition-of-unity
normalization code (`apply_w_normalization_*`, `compute_xi_corner`, `compute_eta_corner`) — benefits
automatically once a patch carries a `mex_spec`, with no changes needed in that code at all.

### The one real design problem: curve callbacks

Both the original MATLAB and C reference implementations represent a boundary curve as arbitrary
function handles (`l_1`, `l_2`, and their first/second derivatives) evaluated at arbitrary
parameter values. That evaluation happens *inside* the single hottest loop in the whole algorithm:
Newton's method for inverting the patch map `M_p`, which runs for every Cartesian output grid
point near a patch (`R_cartesian_mesh_obj.interpolate_patch`, via `R_xi_eta_inversion`). Calling
back into MATLAB from C for every curve evaluation inside that loop would erase most of the point
of moving it to C.

The fix (`csrc/include/curve_eval.h`, `csrc/src/curve_eval.c`): MATLAB samples each curve's 6
functions on a fine uniform grid once (`Curve_obj.get_mex_samples`, memoized, ~40,000 points), and
the mex kernels evaluate the curve at arbitrary parameter values via local M-point
barycentric-Lagrange interpolation through those samples — the *same* technique
(`barylag` + a shifted local stencil) this algorithm already uses elsewhere to interpolate patch
values, so it's not a new numerical method, just applied one level deeper. This only works because,
in this specific algorithm, `Q_patch_obj`'s `M_p`/`J` are never actually arbitrary in practice —
they're always one of exactly two closed-form templates (`Curve_obj.construct_S_patch`'s smooth
normal-offset map, or `construct_C_patch`'s corner-junction map), each parametrized by a curve's
sampled values plus a handful of scalars. See `csrc/include/patch_kernels.h`.

One subtlety that cost a debugging pass: the sampling domain can't be exactly `[0,1]`. The
corner/POU-normalization logic and the extra Gram-matching layers legitimately evaluate a curve's
remapped parameter slightly *outside* `[0,1]`, and Newton's method can transiently overshoot
further before converging. The original closed-form functions extrapolate fine on their own
(they're just `sin`/`cos`/polynomials); a local interpolant whose samples stop exactly at the
boundary does not — barycentric extrapolation diverges quickly past its node range. The fix is to
sample a padded domain (`[-0.5, 1.5]`, see `CURVE_EVAL_DOMAIN_LO/HI`) rather than exactly `[0,1]`,
which comfortably covers the actual excursions this algorithm's geometry produces.

A second subtlety, found via the 4-curve guitarbase example (which the boomerang example doesn't
exercise, since it's a single self-junction curve): a curve's derivative closure is allowed to
ignore its argument and return a theta-independent constant -- e.g. a quadratic Bezier segment's
`l_1_dprime`/`l_2_dprime` (guitarbase's curves 2 and 3, the "waist" connectors) are literally
constant. The original algorithm never notices, since it only ever evaluates these at a scalar
theta. But `get_mex_samples` calls each closure with the *entire* ~40,000-point theta vector at
once to build its sample array -- and a closure that ignores its input returns a **scalar**
regardless, silently producing a 1-element "sample array" instead of the intended one. That 1-point
array then breaks `sampled_fn_eval`'s uniform grid spacing (`h = domain_length/(n-1)`, i.e.
divide-by-zero), and `0 * Inf = NaN` poisons every downstream evaluation. `get_mex_samples`
broadcasts a scalar return value to the full sample length to fix this (see `bcast` in
`Curve_obj.m`). Relatedly, `patch_kernels.c`'s 2x2 Newton-step solve now explicitly rejects a
singular/NaN Jacobian and fails that one Newton attempt cleanly (so `patch_inverse_M_p` moves on to
its next initial guess) instead of letting `Inf`/`NaN` silently propagate through the remaining
999 iterations -- MATLAB's `\` doesn't hard-fail on a singular 2x2 system the way an unguarded
Cramer's-rule division does, so this guard is needed for behavioral parity even independent of the
broadcasting bug above.

The target function `f(x,y)` itself is not part of this bridge — it's only ever evaluated in a
handful of large, vectorized batches (once per patch, once for the interior), not per Newton
iteration, so it's evaluated directly from MATLAB before/after the mex calls that need it, exactly
as in the original.

### No MKL / no LAPACK

The `2dfc-c` C library this port's kernels are adapted from depends on:

- **`MKL_INT`, and the general `<mkl.h>` umbrella** — replaced with plain `int` throughout; no
  replacement header needed.
- **CBLAS (`cblas_dgemm`, `cblas_ddot`, ...)** — kept, but routed through
  `csrc/include/blas_compat.h`, which picks up the system Accelerate framework on macOS or
  `<cblas.h>` (e.g. OpenBLAS) elsewhere. This is the only remaining external numerical dependency.
- **LAPACKE (`LAPACKE_dgesv`)** — removed entirely. The only linear solve anywhere in this
  codebase is the 2x2 Newton-step Jacobian solve in `patch_kernels.c`; it's hand-solved via
  Cramer's rule instead of calling LAPACK, so there's no LAPACK dependency at all, not even a
  portable one.
- **MKL VML (`vdSub`, `vdMul`, `vdErfc`, `vdPackV`, ...) and MKL extended BLAS
  (`mkl_domatcopy`/`mkl_dimatcopy`)** — these were only needed by the parts of `2dfc-c` this port
  doesn't reuse (see below); nothing in `csrc/` calls them.
- **MKL DFTI (FFT)** — not used; see [FFT stays in MATLAB](#what-changed-vs-2dfc-matlab-and-why)
  above.

Note this port does **not** reuse `2dfc-c`'s own top-level orchestration
(`curve_seq_construct_patches`, `c1_patch_FC`/`c2_patch_FC`, `FC2D`/`FC2D_heap`, etc.) — that
machinery exists because `2dfc-c` is a *standalone* C implementation that does its own patch
construction and partition-of-unity normalization in C. Here, MATLAB keeps doing all of that (it's
not the bottleneck and there's no reason to duplicate working, correct orchestration code); the C
side only implements the leaf numerical kernels described above
(`csrc/include/{num_linalg,curve_eval,patch_kernels,fc_kernels,grid_interp,cartesian_kernels}.h`).

### Fallback behavior

Every dispatch point checks `use_mex_2dfc()` (cached `exist(...,'file')==3` check) before calling
into a mex gateway. If the mex kernels haven't been built, or a `Q_patch_obj` was constructed
directly with an arbitrary `M_p`/`J` (not through `Curve_obj`, so it has no `mex_spec`), every
function falls back to the exact original pure-MATLAB implementation — just slower, never wrong or
unavailable.

## Poisson / boundary-integral-equation solver

`2dfc-matlab`'s `examples/poisson-examples/` layers a Poisson solver on top of the FC2D core
above: `Delta(u) = f` with Dirichlet boundary data is split into (1) a particular solution `u_p`
via the already-mex-accelerated `FC2D`/`R_cartesian_mesh_obj.inv_lap`, and (2) a homogeneous
correction `u_h` (`Delta(u_h)=0`, `u_h|boundary = u_boundary - u_p|boundary`) solved with a
self-contained 2nd-kind Fredholm boundary-integral-equation (Nyström double-layer-potential)
solver (`curve_param_obj`, `IE_curve_obj`, `IE_curve_seq_obj`, driven by `laplace_solver.m`/
`laplace_solver_coarse.m`/`poisson_solver.m`/`poisson_solver_coarse.m`). Unlike the FC2D core
above, there was no existing C reference implementation to port from here — the C kernels below
were designed from scratch directly from the MATLAB algorithm.

### What's accelerated, and why

| File | Status |
|---|---|
| `IE_curve_seq_obj.m` | `u_num` dispatches to `ie_u_num_mex`; new `u_num_batch` method dispatches to `ie_u_num_batch_mex` (both fall back to pure MATLAB) |
| `R_cartesian_mesh_obj.m` | new `locally_compute_vec` method dispatches to `r_cartesian_mesh_locally_compute_vec_mex` |
| `IE_curve_obj.m` | `construct_interior_patch` builds a `mex_spec` (reusing `PATCH_KIND_S` verbatim — see below) so its `Q_patch_obj`-based interior patches get the same Newton-inversion/interpolation acceleration as the FC2D core |
| `curve_param_obj.m` | unchanged — pure bookkeeping |
| `poisson_solver.m`, `poisson_solver_coarse.m`, `laplace_solver.m`, `laplace_solver_coarse.m` | unchanged algorithm; call the new accelerated methods above instead of the original's scalar loops / local helper function |

Three things fell out of directly reading this algorithm (rather than a C reference) that shaped
the design:

1. **No curve-sampling bridge needed for the IE evaluation itself.** `IE_curve_obj.u_num`/
   `K_general` only ever evaluate a curve at its fixed, exactly-computable Kress sigmoidal
   quadrature mesh (`theta_mesh`) — never at an arbitrary, Newton-iterate-dependent point the way
   `R_xi_eta_inversion` needs for FC2D. So `ie_kernels.c` takes plain flat `double*` arrays that
   MATLAB fills in once per (curve, resolution) from the *exact* closures — no interpolation, no
   `curve_eval.c` involvement. The curve-sampling bridge is reused only for
   `IE_curve_obj.construct_interior_patch`, which *does* need Newton inversion and reuses
   `PATCH_KIND_S` from `patch_kernels.c` unmodified (its `M_p`/`J` is the identical closed-form
   inward-normal-offset template `Curve_obj.construct_S_patch` already uses).
2. **An exact algebraic simplification.** `K_general`'s `nu_norm = sqrt(l1p^2+l2p^2)` denominator
   cancels exactly against the same `sqrt(l1p^2+l2p^2)` arc-length factor in the quadrature weight,
   so `ie_kernels.c`'s inner loop needs no `sqrt` and no per-curve looping — just one flat sum over
   all `n_total` quadrature nodes:
   `sum_k weight[k] * [(x-l1[k])*l2p[k] - (y-l2[k])*l1p[k]] / [(x-l1[k])^2 + (y-l2[k])^2]`,
   `weight[k] = -1/(2*pi) * gr_phi[k] * ds[k]`. This is an identity, not an approximation.
3. **`construct_A_b`'s dense assembly and the `A\b` solve are not bottlenecks worth porting.**
   Assembly is already fully vectorized (no scalar loop; `n_curves` is 1-4), `n_total` tops out at
   a few thousand nodes even at the finest shipped resolution, and the backslash is already
   LAPACK-backed — reimplementing it without Intel MKL would just rebuild what
   Accelerate/OpenBLAS already does for free.

New from-scratch correctness checks (no C reference to diff against, so `csrc/examples/
smoke_test.c`'s `test_ie_u_num` validates two independent ways): `ie_u_num`/`ie_u_num_batch`
against a naive, unsimplified re-derivation of the same formula (i.e. without the `nu_norm`
cancellation), and a physics identity — a double-layer potential with constant density is exactly
1.0 at every point strictly inside the curve, independent of position.

### All four `u_num` evaluation sites are batched

`laplace_solver`/`laplace_solver_coarse` have four sites that repeatedly evaluate `u_num` under
adaptive refinement: the well-interior pass, the patch-grid-fill loop (`laplace_solver`) /
near-boundary interpolation-node loop (`laplace_solver_coarse`), and the corner-region loop. All
four were rewritten to the same pattern: every point in the current "active" (not-yet-converged)
set is evaluated together via `u_num_batch` at a shared `(curve_param, gr_phi)` resolution; points
that converge (two successive resolutions agree to within `int_eps`) are frozen at their value and
dropped from the active set; the rest move on to the next resolution together. This is a real
reduction in MATLAB-loop and mex-call overhead everywhere, not just a faster inner kernel.

This is safe to do everywhere, not just at the well-interior site, because the "coarse"/"fine"
resolution state was *always* a single pair shared across every point at a given site, even in the
original's per-point while-loops — points were processed one at a time, but they all read and
advanced the *same* global resolution counter. Batching just evaluates every still-active point
against that shared state at once, instead of visiting points one at a time and letting later
points inherit whatever resolution an earlier point's demand happened to leave the shared state at.

The practical consequence: a point's accepted value can differ slightly from the original, since it
is now always checked starting from the same base resolution as every other point at its site
(rather than possibly "free-riding" on a coarser-or-finer level some other point's demand left the
shared state at when the original's sequential walk reached it). Both values satisfy the same
`int_eps` stopping criterion, so the difference is bounded by a small multiple of `int_eps` — not
multiple orders of magnitude smaller the way the well-interior site's difference is (see the
Validation table below: ~1e-11 for well-interior-only vs. ~3-7x `int_eps` once the patch/corner
sites are included, since convergence right at the geometry's corners is slower and doesn't
"overshoot" the tolerance the way smooth well-interior evaluations do).

### Validation

Comparing the original `2dfc-matlab` against this port on the teardrop example
(`examples/poisson-examples/poisson_solver_teardrop.m`'s domain/problem, single self-junction
curve, manufactured solution), both at reduced resolution to keep the pure-MATLAB baseline
tractable, at `int_eps=1e-7` (matching the shipped driver scripts):

| `h` | `d` | grid points | original | mex | speedup | max `u_num_mat` diff |
|---|---|---|---|---|---|---|
| 0.04 | 6 | 14,231 | 5.44s | 0.51s | **10.7x** | 3.0e-07 |
| 0.02 | 6 | 29,045 | 9.98s | 0.68s | **14.6x** | 6.7e-07 |

`abs_max_err`/`rel_2_err` against the manufactured exact solution agree to 4-5 significant figures
between original and mex in both rows — both are dominated by the discretization error of this
(deliberately coarse, for fast baseline comparison) test resolution, not by the `int_eps`-scale
difference above. `IE_curve_seq_obj.u_num`/`u_num_batch` were additionally checked directly
(bypassing the full FC2D+IE pipeline) against the original MATLAB on a synthetic
`(curve_param, gr_phi)`, matching to 2.8e-16. The core FC2D port's own boomerang regression (see
[Validation](#validation) above) was re-run after the `Curve_obj.get_mex_samples`/
`build_curve_samples` refactor and shows no change in behavior.

An earlier version of this port batched only the well-interior site and left the patch-grid-fill/
corner-region loops as scalar per-point while-loops (still benefiting from `u_num`'s own mex
dispatch, but not from batched evaluation). That version's `max u_num_mat diff` was ~1e-11 to
5e-11 — far below `int_eps` — at the cost of a smaller speedup (4.7-10.4x) since more of the
runtime at finer boundary resolutions sat in the unbatched loops. Batching all four sites trades a
small, `int_eps`-bounded amount of order-dependence for a meaningfully larger speedup; if a
workload needs the tighter, closer-to-the-original numerical match instead, reverting the
patch-grid-fill/corner-region loops to scalar `u_num` calls (still mex-dispatched) is
straightforward.

**Multi-curve / concave-corner check**: the teardrop domain above has one self-junction curve with
a single convex corner. `laplace_solver`'s corner-region loop was additionally checked on
`examples/poisson-examples/poisson_solver_guitarbase.m`'s 4-curve domain (which has *concave*
corners) at a reduced test resolution (`h=0.006`, `d=5`, `int_eps=1e-7`, vs. the shipped
`h=0.004`). The batched result matched the uncapped original almost exactly (3.670e-05 vs.
3.668e-05 `abs_max_err`), confirming the batching logic itself is correct on this harder geometry
too — but this test also surfaced a real, pre-existing property of the algorithm worth knowing:
points evaluated very close to a concave curve junction can need well over 100 IE refinement
levels (`rho_fine_IE`) to converge to a tight `int_eps`, far more than the single-digit-to-low-tens
levels the teardrop/well-interior cases ever need. `laplace_solver.m`/`laplace_solver_coarse.m`
cap `rho_fine_IE` at `MAX_RHO_IE=100` (a hard ceiling against runaway/infinite refinement, with a
`warning('...:maxRhoIE', ...)` raised and the best-available value used for any point still
unconverged when the cap is hit); the reduced-resolution guitarbase check above hit exactly this
cap for 2 corner points, producing a large local error (`abs_max_err=0.31`) until the cap was
raised for that check. If a domain has sharp concave corners and near-corner accuracy matters,
raise `MAX_RHO_IE` (or watch for the warning and treat it as a signal to do so); the cap trades
worst-case accuracy at pathological corner points for a hard bound on runtime.

## Building

Requires a C11 compiler MATLAB's `mex` recognizes, and a CBLAS implementation:

```matlab
>> build_mex
```

This compiles `mex/*_mex.c` together with `csrc/src/*.c` and writes the resulting mex binaries to
`src/private/` (MATLAB's private-function convention — visible to the `.m` files in `src/` that
dispatch to them, not part of the public API surface). Re-run after editing anything under `csrc/`
or `mex/`.

- **macOS**: links against `-framework Accelerate`. No setup needed.
- **Linux**: links against `-lopenblas` (`apt install libopenblas-dev` or equivalent).
- **Windows**: also targets `-lopenblas`; provide an OpenBLAS-compatible `cblas.lib` on the linker
  path.

To build and sanity-check just the portable C core, independent of MATLAB/mex:

```sh
cd csrc && make test
```

## Repository layout

```
2dfc-matlab-mex/
├── src/                    Full MATLAB API mirror (see table above)
│   └── private/            Compiled mex binaries land here (build_mex output)
├── csrc/                   Portable C kernels (no MKL/Intel compiler needed)
│   ├── include/, src/      num_linalg, curve_eval, patch_kernels, fc_kernels,
│   │                       grid_interp, cartesian_kernels, ie_kernels, blas_compat
│   ├── examples/           smoke_test.c (standalone correctness checks)
│   └── Makefile
├── mex/                    Thin mexFunction gateways over csrc/, plus mex_common
├── build_mex.m             Top-level build script
├── data/FC_data/           Precomputed FC continuation matrices (same as 2dfc-matlab)
└── examples/
    ├── 2DFC-examples/      Same FC2D example scripts as 2dfc-matlab
    └── poisson-examples/   Poisson/IE solver (curve_param_obj, IE_curve_obj,
                             IE_curve_seq_obj, poisson_solver(_coarse), laplace_solver
                             (_coarse)) plus driver scripts, same layout as
                             2dfc-matlab (.mat outputs gitignored)
```

## Validation

- **Single-curve, full design resolution**: `examples/2DFC-examples/boomerang_2D_FC.m` as shipped
  (`h=0.01`, `d=5`, 192x282 output grid, one self-junction C-patch) run end-to-end through both
  the original `FC2D` and this port's `FC2D`. `fc_err` matches to 3.4e-17, `f_R` matches to 2.8e-10
  max pointwise difference (the residual from finite-sample curve interpolation vs. exact
  closed-form evaluation — see [the curve callback problem](#the-one-real-design-problem-curve-callbacks)),
  `fc_coeffs` to 3.9e-13. Runtime: 1.0s vs. 54.0s (~53x).
- **Mixed corner types and cross-curve references**: each of the 5 mex gateways
  (`inpolygon_mesh_mex`, `fcont_gram_blend_S_mex`, `barylag_mex`, `q_patch_inverse_M_p_mex`,
  `r_cartesian_mesh_interpolate_patch_mex`) was validated directly against its pure-MATLAB
  counterpart on hand-built S-type and C-type (distinct-curve, not self-referencing) patches,
  matching to 1e-15-1e-8 depending on the routine (see git history / development notes for the
  exact harness). This exercises both the concave (C1) and convex (C2) corner-patch code paths.
- **Multi-curve, full design resolution**: `examples/2DFC-examples/guitarbase_2DFC.m` as shipped
  (`h=0.0005`, `d=4`, 3072x4374 output grid, 4 curves mixing Bezier and teardrop-arc segments, both
  concave and convex corner types) run end-to-end. Relative L2 error 4.5e-10. Runtime: 8.0s
  end-to-end (patch construction + POU normalization + FC extension + interpolation onto ~13M grid
  points), including the redundant `plot_geometry` call the example script itself makes. This
  surfaced and fixed a real bug (the constant-derivative-closure broadcasting issue described
  under [curve callbacks](#the-one-real-design-problem-curve-callbacks) above) that boomerang's
  single self-junction curve doesn't exercise — worth knowing if you hit a `NaN`/nonconvergence
  warning with a curve type not covered here (see the next section).

## Known limitations

- If a curve's `l_1`/`l_2`/derivative closures include one that ignores its input and returns a
  scalar for any input size (a constant, like a Bezier segment's second derivative, or anything
  else that isn't elementwise-vectorized over `theta`), `get_mex_samples` broadcasts it to the full
  sample length automatically. If you add a *new* curve-defining pattern and hit a `NaN` or
  `Nonconvergence in computing boundary mesh values` warning that the original pure-MATLAB `FC2D`
  doesn't produce for the same input, check first whether one of the 6 closures fails to return a
  vector matching its input's shape — that was the root cause found in guitarbase's Bezier curves.
- `C1_patch_obj.refine_W` (barycentric upsampling ahead of the FC extension for concave corners)
  benefits from the accelerated `barylag`, but isn't itself a single combined mex kernel the way
  `R_cartesian_mesh_obj.interpolate_patch` is — its surrounding MATLAB loop overhead is still
  paid. It wasn't the bottleneck in either validation run above, but a dedicated mex kernel for it
  is a reasonable next step if a workload's corners dominate runtime.
- The curve-sampling bridge assumes a curve's remapped parameter never strays outside
  `[CURVE_EVAL_DOMAIN_LO, CURVE_EVAL_DOMAIN_HI] = [-0.5, 1.5]`. This has generous margin for this
  algorithm's actual matching-region geometry (validated above), but an unusually extreme domain
  (e.g. very large `frac_n_C`/`frac_n_S` overlap fractions) could in principle exceed it; the
  evaluator clamps rather than crashing in that case, which would show up as reduced accuracy
  rather than an error.
