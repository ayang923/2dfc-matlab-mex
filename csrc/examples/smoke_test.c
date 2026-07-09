/* Standalone smoke test for the portable C kernels -- no MATLAB/mex needed.
 * Exercises: sampled_fn_eval accuracy, S-type Newton inversion round-trip,
 * C-type Newton inversion round-trip, inpolygon_mesh, and fcont_gram_blend_S
 * (CBLAS dgemm) against a hand-computed reference. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cartesian_kernels.h"
#include "curve_eval.h"
#include "fc_kernels.h"
#include "ie_kernels.h"
#include "num_linalg.h"
#include "patch_kernels.h"

static int g_failures = 0;

static void check(const char *name, int cond) {
    printf("[%s] %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) g_failures++;
}

static void check_close(const char *name, double got, double expect, double tol) {
    double err = fabs(got - expect);
    printf("[%s] %s (got %.15g, expect %.15g, err %.3e)\n",
           err < tol ? "PASS" : "FAIL", name, got, expect, err);
    if (!(err < tol)) g_failures++;
}

static const int N_SAMPLE = 40001;

static void fill_samples(double *buf, double (*f)(double)) {
    for (int i = 0; i < N_SAMPLE; i++) {
        double theta = CURVE_EVAL_DOMAIN_LO
            + i * (CURVE_EVAL_DOMAIN_HI - CURVE_EVAL_DOMAIN_LO) / (double) (N_SAMPLE - 1);
        buf[i] = f(theta);
    }
}

/* Unit-circle-ish boundary curve: l_1(theta) = cos(2 pi theta), l_2 = sin(2 pi theta) */
static double c_l1(double t)   { return cos(2*M_PI*t); }
static double c_l2(double t)   { return sin(2*M_PI*t); }
static double c_l1p(double t)  { return -2*M_PI*sin(2*M_PI*t); }
static double c_l2p(double t)  { return  2*M_PI*cos(2*M_PI*t); }
static double c_l1pp(double t) { return -4*M_PI*M_PI*cos(2*M_PI*t); }
static double c_l2pp(double t) { return -4*M_PI*M_PI*sin(2*M_PI*t); }

static void test_sampled_fn_eval(void) {
    double *samples = malloc(N_SAMPLE * sizeof(double));
    fill_samples(samples, sin);
    sampled_fn_t f = {samples, N_SAMPLE};

    for (double theta = 0.05; theta < 1.0; theta += 0.1373) {
        double got = sampled_fn_eval(f, theta, 8);
        check_close("sampled_fn_eval matches sin(theta)", got, sin(theta), 1e-10);
    }
    free(samples);
}

static void test_s_patch_inverse(void) {
    double *l1 = malloc(N_SAMPLE*sizeof(double));
    double *l2 = malloc(N_SAMPLE*sizeof(double));
    double *l1p = malloc(N_SAMPLE*sizeof(double));
    double *l2p = malloc(N_SAMPLE*sizeof(double));
    double *l1pp = malloc(N_SAMPLE*sizeof(double));
    double *l2pp = malloc(N_SAMPLE*sizeof(double));
    fill_samples(l1, c_l1);
    fill_samples(l2, c_l2);
    fill_samples(l1p, c_l1p);
    fill_samples(l2p, c_l2p);
    fill_samples(l1pp, c_l1pp);
    fill_samples(l2pp, c_l2pp);

    patch_spec_t spec;
    spec.kind = PATCH_KIND_S;
    spec.curve_M = CURVE_EVAL_DEFAULT_M;
    spec.u.s.xi_diff = 1.0;
    spec.u.s.xi_0 = 0.0;
    spec.u.s.l1  = (sampled_fn_t) {l1, N_SAMPLE};
    spec.u.s.l2  = (sampled_fn_t) {l2, N_SAMPLE};
    spec.u.s.l1p = (sampled_fn_t) {l1p, N_SAMPLE};
    spec.u.s.l2p = (sampled_fn_t) {l2p, N_SAMPLE};
    spec.u.s.l1pp = (sampled_fn_t) {l1pp, N_SAMPLE};
    spec.u.s.l2pp = (sampled_fn_t) {l2pp, N_SAMPLE};

    patch_bounds_t bounds = {0.0, 1.0, 0.0, 0.05};

    /* Pick a target point by forward-evaluating M_p at a known (xi*, eta*),
     * then check Newton inversion recovers it. */
    double xi_true = 0.37, eta_true = 0.02;
    double x, y;
    patch_eval_M_p(&spec, xi_true, eta_true, &x, &y);

    inverse_result_t r = patch_inverse_M_p(&spec, bounds, x, y, NULL, NULL, 0, 1e-13, 1e-13);
    check("S-patch inverse_M_p converged", r.converged);
    check_close("S-patch inverse_M_p recovers xi", r.xi, xi_true, 1e-9);
    check_close("S-patch inverse_M_p recovers eta", r.eta, eta_true, 1e-9);

    free(l1); free(l2); free(l1p); free(l2p); free(l1pp); free(l2pp);
}

/* Second curve for the C-type junction test: a straight segment continuing
 * on from the unit-circle curve at theta=0, i.e. next_curve.l_1(0) coincides
 * with curr_curve.l_1(1) (both equal (1,0)), matching how curve_seq curves
 * are meant to join end-to-start. */
static double n_l1(double t)  { return 1.0 + t; }
static double n_l2(double t)  { return t; }
static double n_l1p(double t) { (void) t; return 1.0; }
static double n_l2p(double t) { (void) t; return 1.0; }

static void test_c_patch_inverse(void) {
    double *cl1 = malloc(N_SAMPLE*sizeof(double));
    double *cl2 = malloc(N_SAMPLE*sizeof(double));
    double *cl1p = malloc(N_SAMPLE*sizeof(double));
    double *cl2p = malloc(N_SAMPLE*sizeof(double));
    double *nl1 = malloc(N_SAMPLE*sizeof(double));
    double *nl2 = malloc(N_SAMPLE*sizeof(double));
    double *nl1p = malloc(N_SAMPLE*sizeof(double));
    double *nl2p = malloc(N_SAMPLE*sizeof(double));
    fill_samples(cl1, c_l1);
    fill_samples(cl2, c_l2);
    fill_samples(cl1p, c_l1p);
    fill_samples(cl2p, c_l2p);
    fill_samples(nl1, n_l1);
    fill_samples(nl2, n_l2);
    fill_samples(nl1p, n_l1p);
    fill_samples(nl2p, n_l2p);

    patch_spec_t spec;
    spec.kind = PATCH_KIND_C;
    spec.curve_M = CURVE_EVAL_DEFAULT_M;
    spec.u.c.xi_diff = 1.0;
    spec.u.c.xi_0 = 0.0;
    spec.u.c.eta_diff = 1.0;
    spec.u.c.eta_0 = 0.0;
    spec.u.c.curr_l1_at_1 = c_l1(1.0);
    spec.u.c.curr_l2_at_1 = c_l2(1.0);
    spec.u.c.curr_l1 = (sampled_fn_t) {cl1, N_SAMPLE};
    spec.u.c.curr_l2 = (sampled_fn_t) {cl2, N_SAMPLE};
    spec.u.c.curr_l1p = (sampled_fn_t) {cl1p, N_SAMPLE};
    spec.u.c.curr_l2p = (sampled_fn_t) {cl2p, N_SAMPLE};
    spec.u.c.next_l1 = (sampled_fn_t) {nl1, N_SAMPLE};
    spec.u.c.next_l2 = (sampled_fn_t) {nl2, N_SAMPLE};
    spec.u.c.next_l1p = (sampled_fn_t) {nl1p, N_SAMPLE};
    spec.u.c.next_l2p = (sampled_fn_t) {nl2p, N_SAMPLE};

    patch_bounds_t bounds = {0.8, 1.0, 0.0, 0.2};

    double xi_true = 0.91, eta_true = 0.06;
    double x, y;
    patch_eval_M_p(&spec, xi_true, eta_true, &x, &y);

    inverse_result_t r = patch_inverse_M_p(&spec, bounds, x, y, NULL, NULL, 0, 1e-13, 1e-13);
    check("C-patch inverse_M_p converged", r.converged);
    check_close("C-patch inverse_M_p recovers xi", r.xi, xi_true, 1e-9);
    check_close("C-patch inverse_M_p recovers eta", r.eta, eta_true, 1e-9);

    free(cl1); free(cl2); free(cl1p); free(cl2p);
    free(nl1); free(nl2); free(nl1p); free(nl2p);
}

static void test_inpolygon_mesh(void) {
    /* 5x5 grid over [0,4]x[0,4], unit square boundary [1,3]x[1,3] should
     * mark grid points (2,2) only... let's use exact grid-aligned square
     * boundary [1,3]x[1,3] on integer grid spacing h=1: interior points are
     * strictly inside, i.e. just (2,2). */
    int n = 5;
    double R_X_data[25], R_Y_data[25];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = sub2ind(n, n, (sub_t) {i, j});
            R_X_data[idx] = j;
            R_Y_data[idx] = i;
        }
    }
    rd_mat_t R_X = rd_mat_init(R_X_data, n, n);
    rd_mat_t R_Y = rd_mat_init(R_Y_data, n, n);

    /* Deliberately off-grid boundary (not aligned to integer grid lines) --
     * real usage always perturbs the bounding box by a fraction of h for
     * exactly this reason (see FC2D.m's `perturb` option): a boundary edge
     * landing exactly on a grid line/point is an ambiguous, algorithm-
     * dependent edge case for any scanline point-in-polygon test. */
    double bx[5] = {0.9, 3.1, 3.1, 0.9, 0.9};
    double by[5] = {0.9, 0.9, 3.1, 3.1, 0.9};
    rd_mat_t boundary_X = rd_mat_init(bx, 5, 1);
    rd_mat_t boundary_Y = rd_mat_init(by, 5, 1);

    int in_data[25];
    ri_mat_t in_msk = ri_mat_init(in_data, 0, 0);
    int n_interior = inpolygon_mesh(R_X, R_Y, boundary_X, boundary_Y, &in_msk);

    check_close("inpolygon_mesh interior count", n_interior, 9, 0.5);
    int center_idx = sub2ind(n, n, (sub_t) {2, 2});
    check("inpolygon_mesh marks center interior", in_msk.mat_data[center_idx] != 0);
}

static void test_fcont_gram_blend_S(void) {
    /* Validate the CBLAS-based implementation against a naive triple-loop
     * reference computing the same flipud(A*(Q'*flipud(fx(1:d,:)))) formula. */
    int d = 3, n_r_c = 4, n_xi = 2;
    double A_data[4*3] = {1,2,3,4, 5,6,7,8, 9,10,11,12};   /* 4x3, column-major */
    double Q_data[3*3] = {1,0,0, 0,1,0, 0,0,1};             /* identity */
    double fx_data[6*2];
    for (int i = 0; i < 12; i++) fx_data[i] = i + 1;        /* 6x2 */

    rd_mat_t A = rd_mat_init(A_data, n_r_c, d);
    rd_mat_t Q = rd_mat_init(Q_data, d, d);
    rd_mat_t fx = rd_mat_init(fx_data, 6, n_xi);

    double fcont_data[(n_r_c+1) * n_xi];
    rd_mat_t fcont = rd_mat_init_no_shape(fcont_data);
    fcont_gram_blend_S(fx, d, A, Q, &fcont);

    /* Naive reference in column-major layout matching MATLAB semantics:
     * fl = fx(1:d,:); fc = flipud(A*(Q'*flipud(fl))); fcont=[fc; fx(1,:)]. */
    double fl[3*2], fl_flip[3*2], proj[3*2], fc[4*2], fc_flip[4*2];
    for (int j = 0; j < n_xi; j++)
        for (int i = 0; i < d; i++)
            fl[sub2ind(d,n_xi,(sub_t){i,j})] = fx_data[sub2ind(6,n_xi,(sub_t){i,j})];
    for (int j = 0; j < n_xi; j++)
        for (int i = 0; i < d; i++)
            fl_flip[sub2ind(d,n_xi,(sub_t){i,j})] = fl[sub2ind(d,n_xi,(sub_t){d-1-i,j})];
    for (int j = 0; j < n_xi; j++)
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int k = 0; k < d; k++) s += Q_data[sub2ind(d,d,(sub_t){k,i})] * fl_flip[sub2ind(d,n_xi,(sub_t){k,j})];
            proj[sub2ind(d,n_xi,(sub_t){i,j})] = s;
        }
    for (int j = 0; j < n_xi; j++)
        for (int i = 0; i < n_r_c; i++) {
            double s = 0;
            for (int k = 0; k < d; k++) s += A_data[sub2ind(n_r_c,d,(sub_t){i,k})] * proj[sub2ind(d,n_xi,(sub_t){k,j})];
            fc[sub2ind(n_r_c,n_xi,(sub_t){i,j})] = s;
        }
    for (int j = 0; j < n_xi; j++)
        for (int i = 0; i < n_r_c; i++)
            fc_flip[sub2ind(n_r_c,n_xi,(sub_t){i,j})] = fc[sub2ind(n_r_c,n_xi,(sub_t){n_r_c-1-i,j})];

    int ok = 1;
    for (int j = 0; j < n_xi; j++) {
        for (int i = 0; i < n_r_c; i++) {
            double got = fcont.mat_data[sub2ind(fcont.rows,fcont.columns,(sub_t){i,j})];
            double expect = fc_flip[sub2ind(n_r_c,n_xi,(sub_t){i,j})];
            if (fabs(got-expect) > 1e-9) ok = 0;
        }
        double got_bnd = fcont.mat_data[sub2ind(fcont.rows,fcont.columns,(sub_t){n_r_c,j})];
        double expect_bnd = fx_data[sub2ind(6,n_xi,(sub_t){0,j})];
        if (fabs(got_bnd-expect_bnd) > 1e-9) ok = 0;
    }
    check("fcont_gram_blend_S matches naive reference", ok);
}

/* No existing C code implements this solver anywhere (unlike the rest of
 * this file's kernels, which mirror the 2dfc-c reference), so there's no
 * "port fidelity" check available. Instead: (1) validate ie_u_num's
 * cancelled-nu_norm shortcut against a literal, unsimplified
 * reimplementation of the original MATLAB formula (the one nontrivial
 * algebraic step in this new kernel), (2) check the batched entry point
 * matches the scalar one, (3) a physics sanity check: the double-layer
 * potential of a constant density over a closed curve is the same at every
 * interior point (a real, assumption-light property of this kernel, not a
 * hardcoded expected value/sign). */
static double ie_u_num_naive(double x, double y,
                              const double *l1, const double *l2,
                              const double *l1p, const double *l2p,
                              const double *gr_phi, double ds, int n_total) {
    double sum = 0.0;
    for (int k = 0; k < n_total; k++) {
        double dx = x - l1[k];
        double dy = y - l2[k];
        double num = dx * l2p[k] - dy * l1p[k];
        double nu = sqrt(l1p[k]*l1p[k] + l2p[k]*l2p[k]);
        double dist2 = dx*dx + dy*dy;
        double K = -1.0 / (2.0*M_PI) * num / (nu * dist2);
        sum += K * gr_phi[k] * nu * ds;
    }
    return sum;
}

static void fill_unit_circle(int n, double *l1, double *l2, double *l1p, double *l2p) {
    for (int k = 0; k < n; k++) {
        double theta = (2.0*M_PI*k) / n;
        l1[k] = cos(theta);
        l2[k] = sin(theta);
        l1p[k] = -2.0*M_PI*sin(theta) / 1.0; /* d(l1)/d(theta_normalized) style scaling doesn't matter for this test */
        l2p[k] =  2.0*M_PI*cos(theta);
    }
}

static void test_ie_u_num(void) {
    const int n = 64;
    double l1[64], l2[64], l1p[64], l2p[64], gr_phi[64], weight[64];
    fill_unit_circle(n, l1, l2, l1p, l2p);
    double ds = 1.0 / n;
    for (int k = 0; k < n; k++) {
        gr_phi[k] = 2.0 + sin(3.0 * (2.0*M_PI*k)/n);
        weight[k] = -1.0/(2.0*M_PI) * gr_phi[k] * ds;
    }

    double test_pts_x[3] = {0.3, -0.2, 0.05};
    double test_pts_y[3] = {0.1, 0.4, -0.15};
    double max_err = 0.0;
    for (int i = 0; i < 3; i++) {
        double got = ie_u_num(test_pts_x[i], test_pts_y[i], l1, l2, l1p, l2p, weight, n);
        double expect = ie_u_num_naive(test_pts_x[i], test_pts_y[i], l1, l2, l1p, l2p, gr_phi, ds, n);
        double err = fabs(got - expect);
        if (err > max_err) max_err = err;
    }
    check_close("ie_u_num matches naive (unsimplified) formula", max_err, 0.0, 1e-12);

    /* batched vs scalar */
    double batch_out[3];
    ie_u_num_batch(test_pts_x, test_pts_y, 3, l1, l2, l1p, l2p, weight, n, batch_out);
    double max_batch_err = 0.0;
    for (int i = 0; i < 3; i++) {
        double scalar_val = ie_u_num(test_pts_x[i], test_pts_y[i], l1, l2, l1p, l2p, weight, n);
        double err = fabs(batch_out[i] - scalar_val);
        if (err > max_batch_err) max_batch_err = err;
    }
    check_close("ie_u_num_batch matches scalar ie_u_num", max_batch_err, 0.0, 1e-15);

    /* Physics sanity check: constant density -> position-independent value
     * for interior points (finer discretization for a tighter check). */
    const int n_fine = 4000;
    double *l1f = malloc(n_fine*sizeof(double));
    double *l2f = malloc(n_fine*sizeof(double));
    double *l1pf = malloc(n_fine*sizeof(double));
    double *l2pf = malloc(n_fine*sizeof(double));
    double *weightf = malloc(n_fine*sizeof(double));
    fill_unit_circle(n_fine, l1f, l2f, l1pf, l2pf);
    double dsf = 1.0 / n_fine;
    for (int k = 0; k < n_fine; k++) {
        weightf[k] = -1.0/(2.0*M_PI) * 1.0 * dsf; /* constant density gr_phi = 1 */
    }
    double v1 = ie_u_num(0.3, 0.1, l1f, l2f, l1pf, l2pf, weightf, n_fine);
    double v2 = ie_u_num(-0.5, 0.2, l1f, l2f, l1pf, l2pf, weightf, n_fine);
    double v3 = ie_u_num(0.0, 0.0, l1f, l2f, l1pf, l2pf, weightf, n_fine);
    printf("[INFO] constant-density double-layer potential at 3 interior points: %.10g, %.10g, %.10g\n", v1, v2, v3);
    check_close("constant-density double-layer potential is position-independent inside the curve",
                fmax(fabs(v1-v2), fabs(v1-v3)), 0.0, 1e-6);
    free(l1f); free(l2f); free(l1pf); free(l2pf); free(weightf);
}

int main(void) {
    test_sampled_fn_eval();
    test_s_patch_inverse();
    test_c_patch_inverse();
    test_inpolygon_mesh();
    test_fcont_gram_blend_S();
    test_ie_u_num();

    printf("\n%s (%d failure(s))\n", g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
