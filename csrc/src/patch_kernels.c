#include <math.h>
#include <stddef.h>

#include "patch_kernels.h"
#include "num_linalg.h"

static double eval(sampled_fn_t f, double theta, int M) {
    return sampled_fn_eval(f, theta, M);
}

void patch_eval_M_p(const patch_spec_t *spec, double xi, double eta, double *x, double *y) {
    int M = spec->curve_M;

    if (spec->kind == PATCH_KIND_S) {
        const s_patch_curve_spec_t *p = &spec->u.s;
        double xi_tilde = p->xi_diff * xi + p->xi_0;

        double l1p_v = eval(p->l1p, xi_tilde, M);
        double l2p_v = eval(p->l2p, xi_tilde, M);
        double nu_norm = sqrt(l1p_v*l1p_v + l2p_v*l2p_v);

        *x = eval(p->l1, xi_tilde, M) - eta * l2p_v / nu_norm;
        *y = eval(p->l2, xi_tilde, M) + eta * l1p_v / nu_norm;
    } else {
        const c_patch_curve_spec_t *p = &spec->u.c;
        double xi_tilde  = p->xi_diff  * xi  + p->xi_0;
        double eta_tilde = p->eta_diff * eta + p->eta_0;

        *x = eval(p->curr_l1, xi_tilde, M) + eval(p->next_l1, eta_tilde, M) - p->curr_l1_at_1;
        *y = eval(p->curr_l2, xi_tilde, M) + eval(p->next_l2, eta_tilde, M) - p->curr_l2_at_1;
    }
}

void patch_eval_J(const patch_spec_t *spec, double xi, double eta, double J_vals[4]) {
    int M = spec->curve_M;

    if (spec->kind == PATCH_KIND_S) {
        const s_patch_curve_spec_t *p = &spec->u.s;
        double xi_tilde = p->xi_diff * xi + p->xi_0;

        double l1p_v  = eval(p->l1p,  xi_tilde, M);
        double l2p_v  = eval(p->l2p,  xi_tilde, M);
        double l1pp_v = eval(p->l1pp, xi_tilde, M);
        double l2pp_v = eval(p->l2pp, xi_tilde, M);

        double nu_norm = sqrt(l1p_v*l1p_v + l2p_v*l2p_v);
        double nu_norm2 = nu_norm * nu_norm;
        double nu_norm3 = nu_norm2 * nu_norm;
        double curv_term = l2pp_v*l2p_v + l1pp_v*l1p_v;

        /* dM_p_x/dxi */
        J_vals[0] = p->xi_diff * (l1p_v - eta * (l2pp_v*nu_norm2 - l2p_v*curv_term) / nu_norm3);
        /* dM_p_x/deta */
        J_vals[1] = -l2p_v / nu_norm;
        /* dM_p_y/dxi */
        J_vals[2] = p->xi_diff * (l2p_v + eta * (l1pp_v*nu_norm2 - l1p_v*curv_term) / nu_norm3);
        /* dM_p_y/deta */
        J_vals[3] = l1p_v / nu_norm;
    } else {
        const c_patch_curve_spec_t *p = &spec->u.c;
        double xi_tilde  = p->xi_diff  * xi  + p->xi_0;
        double eta_tilde = p->eta_diff * eta + p->eta_0;

        J_vals[0] = p->xi_diff  * eval(p->curr_l1p, xi_tilde, M);
        J_vals[1] = p->eta_diff * eval(p->next_l1p, eta_tilde, M);
        J_vals[2] = p->xi_diff  * eval(p->curr_l2p, xi_tilde, M);
        J_vals[3] = p->eta_diff * eval(p->next_l2p, eta_tilde, M);
    }
}

int patch_in_bounds(patch_bounds_t b, double xi, double eta) {
    return xi >= b.xi_start && xi <= b.xi_end && eta >= b.eta_start && eta <= b.eta_end;
}

void patch_round_boundary_point(patch_bounds_t b, double eps_xi_eta, double *xi, double *eta) {
    if (fabs(*xi - b.xi_start) < eps_xi_eta) *xi = b.xi_start;
    else if (fabs(*xi - b.xi_end) < eps_xi_eta) *xi = b.xi_end;
    if (fabs(*eta - b.eta_start) < eps_xi_eta) *eta = b.eta_start;
    else if (fabs(*eta - b.eta_end) < eps_xi_eta) *eta = b.eta_end;
}

int patch_default_initial_guesses(patch_bounds_t b, int N, double *out_xi, double *out_eta) {
    int N_segment = (N + 3) / 4; /* ceil(N/4) */

    double xi_mesh[N_segment + 1];
    double eta_mesh[N_segment + 1];
    for (int i = 0; i <= N_segment; i++) {
        xi_mesh[i]  = b.xi_start  + (b.xi_end  - b.xi_start)  * i / (double) N_segment;
        eta_mesh[i] = b.eta_start + (b.eta_end - b.eta_start) * i / (double) N_segment;
    }

    for (int i = 0; i < N_segment; i++) {
        out_xi[i]                = xi_mesh[i];
        out_xi[N_segment + i]    = xi_mesh[i];
        out_xi[2*N_segment + i]  = b.xi_start;
        out_xi[3*N_segment + i]  = b.xi_end;

        out_eta[i]               = b.eta_start;
        out_eta[N_segment + i]   = b.eta_end;
        out_eta[2*N_segment + i] = eta_mesh[i];
        out_eta[3*N_segment + i] = eta_mesh[i];
    }

    return 4 * N_segment;
}

/* Solves J*delta = f for a dense 2x2 system via Cramer's rule (replaces the
 * single LAPACKE_dgesv call in the original C library -- always a 2x2 solve
 * here, so no LAPACK dependency is needed at all). */
static int solve_2x2(const double J[4], const double f[2], double delta[2]) {
    /* J_vals layout: [dM_x/dxi, dM_x/deta, dM_y/dxi, dM_y/deta].
     * Returns 0 (and leaves delta untouched) if J is singular or close to
     * it -- e.g. one of the boundary-corner initial guesses (such as
     * (xi,eta) = (0,0)) can land exactly on a degenerate point of a
     * patch's parametrization, particularly for corner (C-type) patches.
     * MATLAB's `\` doesn't hard-fail on a singular 2x2 system (it warns
     * and returns a least-squares-ish result), so without this guard a
     * bad initial guess here would divide by ~0, produce Inf/NaN, and
     * (since NaN propagates through every subsequent iteration) silently
     * poison that whole Newton attempt instead of just failing it -- the
     * caller (patch_inverse_M_p) already tries multiple initial guesses
     * and only needs any *one* of them to fail cleanly so it can move on
     * to the next, matching the original algorithm's actual behavior. */
    double a = J[0], b = J[1], c = J[2], d = J[3];
    double det = a*d - b*c;
    double scale = fabs(a) + fabs(b) + fabs(c) + fabs(d);
    if (scale == 0.0 || fabs(det) < 1e-14 * scale * scale) {
        return 0;
    }
    delta[0] = (f[0]*d - f[1]*b) / det;
    delta[1] = (a*f[1] - c*f[0]) / det;
    return 1;
}

static inverse_result_t newton_solve_one(const patch_spec_t *spec, double x, double y,
                                          double xi0, double eta0, double eps_xi_eta, double eps_xy) {
    const int Nmax = 1000;
    double v[2] = {xi0, eta0};
    int broke = 0;
    int i;

    for (i = 2; i <= Nmax; i++) {
        double v_prev[2] = {v[0], v[1]};

        double mx, my;
        patch_eval_M_p(spec, v[0], v[1], &mx, &my);
        double fx[2] = {mx - x, my - y};

        double J[4];
        patch_eval_J(spec, v[0], v[1], J);

        double delta[2];
        if (!solve_2x2(J, fx, delta)) {
            /* Singular Jacobian at this iterate: fail this initial guess
             * cleanly (v stays at its last finite value) rather than
             * propagate Inf/NaN through the remaining iterations. */
            break;
        }
        v[0] -= delta[0];
        v[1] -= delta[1];

        double mx_new, my_new;
        patch_eval_M_p(spec, v[0], v[1], &mx_new, &my_new);
        double fx_new[2] = {mx_new - x, my_new - y};

        double f_inf = MAX(fabs(fx_new[0]), fabs(fx_new[1]));
        double d0 = fabs(v[0] - v_prev[0]);
        double d1 = fabs(v[1] - v_prev[1]);

        if (f_inf < eps_xy && d0 < eps_xi_eta && d1 < eps_xi_eta) {
            broke = 1;
            break;
        }
    }

    return (inverse_result_t) {v[0], v[1], broke && (i < Nmax)};
}

inverse_result_t patch_inverse_M_p(const patch_spec_t *spec, patch_bounds_t bounds,
                                    double x, double y,
                                    const double *init_xi, const double *init_eta, int n_init,
                                    double eps_xi_eta, double eps_xy) {
    double default_xi[20], default_eta[20];
    if (init_xi == NULL || init_eta == NULL || n_init == 0) {
        n_init = patch_default_initial_guesses(bounds, 20, default_xi, default_eta);
        init_xi = default_xi;
        init_eta = default_eta;
    }

    inverse_result_t last = {0, 0, 0};
    for (int k = 0; k < n_init; k++) {
        inverse_result_t r = newton_solve_one(spec, x, y, init_xi[k], init_eta[k], eps_xi_eta, eps_xy);
        patch_round_boundary_point(bounds, eps_xi_eta, &r.xi, &r.eta);
        last = r;
        if (r.converged && patch_in_bounds(bounds, r.xi, r.eta)) {
            return r;
        }
    }
    return last;
}
