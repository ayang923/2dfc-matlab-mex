#include "ie_kernels.h"

double ie_u_num(double x, double y,
                 const double *l1, const double *l2,
                 const double *l1p, const double *l2p,
                 const double *weight, int n_total) {
    double sum = 0.0;
    for (int k = 0; k < n_total; k++) {
        double dx = x - l1[k];
        double dy = y - l2[k];
        double num = dx * l2p[k] - dy * l1p[k];
        double dist2 = dx*dx + dy*dy;
        sum += weight[k] * num / dist2;
    }
    return sum;
}

void ie_u_num_batch(const double *targets_x, const double *targets_y, int n_targets,
                     const double *l1, const double *l2,
                     const double *l1p, const double *l2p,
                     const double *weight, int n_total,
                     double *out) {
    for (int t = 0; t < n_targets; t++) {
        out[t] = ie_u_num(targets_x[t], targets_y[t], l1, l2, l1p, l2p, weight, n_total);
    }
}
