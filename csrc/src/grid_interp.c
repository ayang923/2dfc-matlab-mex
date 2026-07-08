#include <math.h>

#include "grid_interp.h"

locally_compute_result_t grid_locally_compute(
    rd_mat_t f,
    double dim2_start, double dim2_end,
    double dim1_start, double dim1_end,
    double dim2_query, double dim1_query,
    int M) {

    int n_dim1 = f.rows;
    int n_dim2 = f.columns;

    if (dim2_query < dim2_start || dim2_query > dim2_end ||
        dim1_query < dim1_start || dim1_query > dim1_end) {
        return (locally_compute_result_t) {NAN, 0};
    }

    double h_dim2 = (dim2_end - dim2_start) / (n_dim2 - 1);
    double h_dim1 = (dim1_end - dim1_start) / (n_dim1 - 1);

    int dim2_j = (int) floor((dim2_query - dim2_start) / h_dim2);
    int dim1_j = (int) floor((dim1_query - dim1_start) / h_dim1);

    int half_M = M / 2;

    int interpol_dim2_j[M];
    int interpol_dim1_j[M];
    ri_mat_t dim2_j_mesh = ri_mat_init(interpol_dim2_j, M, 1);
    ri_mat_t dim1_j_mesh = ri_mat_init(interpol_dim1_j, M, 1);

    if (M % 2) {
        ri_range(dim2_j - half_M, 1, dim2_j + half_M, &dim2_j_mesh);
        ri_range(dim1_j - half_M, 1, dim1_j + half_M, &dim1_j_mesh);
    } else {
        ri_range(dim2_j - half_M + 1, 1, dim2_j + half_M, &dim2_j_mesh);
        ri_range(dim1_j - half_M + 1, 1, dim1_j + half_M, &dim1_j_mesh);
    }

    shift_idx_mesh(&dim2_j_mesh, 0, n_dim2 - 1);
    shift_idx_mesh(&dim1_j_mesh, 0, n_dim1 - 1);

    double interpol_dim2_mesh[M];
    double interpol_dim1_mesh[M];
    for (int i = 0; i < M; i++) {
        interpol_dim2_mesh[i] = interpol_dim2_j[i] * h_dim2 + dim2_start;
        interpol_dim1_mesh[i] = interpol_dim1_j[i] * h_dim1 + dim1_start;
    }
    rd_mat_t dim2_mesh = rd_mat_init(interpol_dim2_mesh, M, 1);
    rd_mat_t dim1_mesh = rd_mat_init(interpol_dim1_mesh, M, 1);

    /* Pass 1: interpolate along dim2 for each of the M candidate dim1 rows */
    double interpol_dim2_exact[M];
    double row_vals[M];
    rd_mat_t row_vals_mat = rd_mat_init(row_vals, M, 1);
    for (int horz = 0; horz < M; horz++) {
        for (int i = 0; i < M; i++) {
            int idx = sub2ind(n_dim1, n_dim2, (sub_t) {interpol_dim1_j[horz], interpol_dim2_j[i]});
            row_vals[i] = f.mat_data[idx];
        }
        interpol_dim2_exact[horz] = barylag(dim2_mesh, row_vals_mat, dim2_query);
    }

    /* Pass 2: interpolate the M intermediate values along dim1 */
    rd_mat_t dim2_exact_mat = rd_mat_init(interpol_dim2_exact, M, 1);
    double value = barylag(dim1_mesh, dim2_exact_mat, dim1_query);

    return (locally_compute_result_t) {value, 1};
}
