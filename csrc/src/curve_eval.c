#include <math.h>

#include "curve_eval.h"
#include "num_linalg.h"

double sampled_fn_eval(sampled_fn_t f, double theta, int M) {
    int n = f.n_samples;
    double lo = CURVE_EVAL_DOMAIN_LO;
    double hi = CURVE_EVAL_DOMAIN_HI;
    double h = (hi - lo) / (n - 1);

    if (M > n) {
        M = n;
    }

    /* Clamp defensively: theta should stay within the padded domain (see
     * the rationale in curve_eval.h), but guard against a pathological
     * excursion producing garbage instead of failing silently in a way
     * that's hard to trace back. */
    if (theta < lo) theta = lo;
    if (theta > hi) theta = hi;

    int j = (int) floor((theta - lo) / h);
    int half_M = M / 2;

    int idx_data[M];
    ri_mat_t idx_mesh = ri_mat_init(idx_data, M, 1);
    if (M % 2) {
        ri_range(j - half_M, 1, j + half_M, &idx_mesh);
    } else {
        ri_range(j - half_M + 1, 1, j + half_M, &idx_mesh);
    }
    shift_idx_mesh(&idx_mesh, 0, n - 1);

    double node_theta_data[M];
    double node_val_data[M];
    for (int k = 0; k < M; k++) {
        node_theta_data[k] = lo + idx_data[k] * h;
        node_val_data[k] = f.samples[idx_data[k]];
    }

    rd_mat_t node_theta = rd_mat_init(node_theta_data, M, 1);
    rd_mat_t node_val = rd_mat_init(node_val_data, M, 1);

    return barylag(node_theta, node_val, theta);
}
