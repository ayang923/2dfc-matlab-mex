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
│   │                       grid_interp, cartesian_kernels, blas_compat
│   ├── examples/           smoke_test.c (standalone correctness checks)
│   └── Makefile
├── mex/                    Thin mexFunction gateways over csrc/, plus mex_common
├── build_mex.m             Top-level build script
├── data/FC_data/           Precomputed FC continuation matrices (same as 2dfc-matlab)
└── examples/               Same example scripts as 2dfc-matlab
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
