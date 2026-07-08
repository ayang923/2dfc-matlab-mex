#ifndef __CURVE_EVAL_H__
#define __CURVE_EVAL_H__

/*
 * Bridge that lets MATLAB-defined boundary curves (arbitrary function
 * handles l_1(theta), l_2(theta), ... on theta in [0,1]) be evaluated
 * from C at arbitrary theta *without* calling back into MATLAB.
 *
 * Newton's method for inverting the patch map M_p (patch_kernels.c) needs
 * to evaluate the curve and its derivatives at arbitrary, not-known-ahead-
 * of-time theta values, many times per grid point, over potentially the
 * entire Cartesian output mesh. A MATLAB callback (mexCallMATLAB) at that
 * call frequency would erase most of the benefit of moving this loop to C.
 *
 * Instead, MATLAB samples each curve function on a fine uniform grid over a
 * padded domain (see CURVE_EVAL_DOMAIN_LO/HI below) and passes the plain
 * sample array into the mex gateway. This module evaluates the sampled
 * function at an arbitrary theta via local M-point barycentric Lagrange
 * interpolation -- the exact same technique (barylag + a shifted local
 * stencil) the algorithm already uses elsewhere to interpolate patch
 * function values, so it introduces no new numerical method, only reuses
 * one already in this codebase. With a fine sample grid (tens of thousands
 * of points) and M ~ 8, interpolation error is far below the Newton
 * tolerances (~1e-13) used in practice for the smooth analytic curves this
 * algorithm targets.
 *
 * Padding beyond the nominal theta in [0,1] is required, not optional: the
 * corner/POU-normalization logic (Q_patch_obj.apply_w*, compute_xi_corner/
 * compute_eta_corner) and the extra "d-1" Gram-matching layers deliberately
 * evaluate a curve's xi_tilde/eta_tilde slightly outside its own [0,1], and
 * Newton's method can transiently overshoot further than that before
 * converging. The original closed-form MATLAB functions (sin/cos/polynomials)
 * extrapolate fine on their own; a local interpolant through samples that
 * stop exactly at 0/1 does not -- barycentric extrapolation far past its
 * node range diverges quickly. Sampling a padded domain sidesteps this
 * entirely as long as the padding covers the actual excursion, which for
 * this algorithm's matching-region geometry is comfortably within +-0.5.
 */
#define CURVE_EVAL_DOMAIN_LO (-0.5)
#define CURVE_EVAL_DOMAIN_HI (1.5)

typedef struct sampled_fn {
    const double *samples;  /* n_samples values of the function on a uniform
                                grid over [CURVE_EVAL_DOMAIN_LO, CURVE_EVAL_DOMAIN_HI] */
    int n_samples;
} sampled_fn_t;

/** Default local barycentric-Lagrange stencil width used for curve evaluation. */
#define CURVE_EVAL_DEFAULT_M 8

/** Evaluates a sampled_fn_t at arbitrary theta via a local M-point stencil. */
double sampled_fn_eval(sampled_fn_t f, double theta, int M);

#endif
