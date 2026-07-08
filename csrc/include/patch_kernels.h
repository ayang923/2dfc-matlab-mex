#ifndef __PATCH_KERNELS_H__
#define __PATCH_KERNELS_H__

#include "curve_eval.h"

/*
 * Concrete parametrizations M_p(xi,eta) -> (x,y) for the two patch templates
 * actually used by the algorithm (mirroring Curve_obj.construct_S_patch and
 * Curve_obj.construct_C_patch in the MATLAB source): a smooth boundary-offset
 * patch (S) and a corner-junction patch (C). Q_patch_obj in MATLAB is generic
 * (arbitrary M_p/J handles), but in practice it is only ever instantiated
 * through these two templates, so accelerating exactly these two closed forms
 * covers every patch the algorithm builds.
 */
typedef enum patch_kind {
    PATCH_KIND_S,
    PATCH_KIND_C
} patch_kind_t;

/** S-type (smooth) patch: normal offset from a single boundary curve. */
typedef struct s_patch_curve_spec {
    double xi_diff, xi_0;
    sampled_fn_t l1, l2, l1p, l2p, l1pp, l2pp;
} s_patch_curve_spec_t;

/** C-type (corner) patch: junction of the current curve's end and the next curve's start. */
typedef struct c_patch_curve_spec {
    double xi_diff, xi_0, eta_diff, eta_0;
    double curr_l1_at_1, curr_l2_at_1;   /* exact l_1(1), l_2(1) (the corner point) */
    sampled_fn_t curr_l1, curr_l2, curr_l1p, curr_l2p;
    sampled_fn_t next_l1, next_l2, next_l1p, next_l2p;
} c_patch_curve_spec_t;

typedef struct patch_spec {
    patch_kind_t kind;
    int curve_M; /* barycentric stencil width for curve evaluation, e.g. CURVE_EVAL_DEFAULT_M */
    union {
        s_patch_curve_spec_t s;
        c_patch_curve_spec_t c;
    } u;
} patch_spec_t;

typedef struct patch_bounds {
    double xi_start, xi_end, eta_start, eta_end;
} patch_bounds_t;

typedef struct inverse_result {
    double xi, eta;
    int converged;
} inverse_result_t;

void patch_eval_M_p(const patch_spec_t *spec, double xi, double eta, double *x, double *y);

/** J_vals: [dM_p_x/dxi, dM_p_x/deta, dM_p_y/dxi, dM_p_y/deta] */
void patch_eval_J(const patch_spec_t *spec, double xi, double eta, double J_vals[4]);

int patch_in_bounds(patch_bounds_t b, double xi, double eta);
void patch_round_boundary_point(patch_bounds_t b, double eps_xi_eta, double *xi, double *eta);

/**
 * Default N boundary-edge initial guesses for Newton inversion, matching
 * Q_patch_obj.default_initial_guesses. out_xi/out_eta must have length
 * 4*ceil(N/4.0); returns that length.
 */
int patch_default_initial_guesses(patch_bounds_t b, int N, double *out_xi, double *out_eta);

/**
 * Inverts M_p at physical point (x,y) via Newton's method, trying each
 * supplied initial guess (or 20 default boundary guesses if init_xi/init_eta
 * are NULL) until one converges to an in-bounds solution. Mirrors
 * Q_patch_obj.inverse_M_p / newton_solve.m exactly (same iteration count,
 * same convergence test, same fallback-to-last-guess-on-failure behavior).
 */
inverse_result_t patch_inverse_M_p(const patch_spec_t *spec, patch_bounds_t bounds,
                                    double x, double y,
                                    const double *init_xi, const double *init_eta, int n_init,
                                    double eps_xi_eta, double eps_xy);

#endif
