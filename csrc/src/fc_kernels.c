#include "fc_kernels.h"
#include "blas_compat.h"

void fcont_gram_blend_S(rd_mat_t fx, int d, rd_mat_t A, rd_mat_t Q, rd_mat_t *fcont) {
    /*
     * Build the (d x n_cols) matching-point matrix from the first d rows of
     * fx, reversed in row order (flipud), so the last sample is the 1D-BTZ
     * matching mesh point closest to the continuation interval.
     */
    double f_matching_data[d * fx.columns];

    fcont->rows = A.rows + 1;
    fcont->columns = fx.columns;

    for (int j = 0; j < fx.columns; j++) {
        for (int i = 0; i < d; i++) {
            int fx_idx = sub2ind(fx.rows, fx.columns, (sub_t) {i, j});
            int f_matching_idx = sub2ind(d, fx.columns, (sub_t) {d-1-i, j});
            f_matching_data[f_matching_idx] = fx.mat_data[fx_idx];

            if (i == 0) {
                int fcont_idx = sub2ind(fcont->rows, fcont->columns, (sub_t) {fcont->rows-1, j});
                fcont->mat_data[fcont_idx] = fx.mat_data[fx_idx];
            }
        }
    }

    /* data_projection = Q^T * f_matching */
    double data_projection[d * fx.columns];
    double fcont_no_boundary[A.rows * fx.columns];
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, fx.columns, d,
                1.0, Q.mat_data, d, f_matching_data, d,
                0.0, data_projection, d);

    /* fcont_no_boundary = A * data_projection */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                A.rows, fx.columns, d,
                1.0, A.mat_data, A.rows, data_projection, d,
                0.0, fcont_no_boundary, A.rows);

    /* Write continuation values in reverse row order (excluding the last
     * row, already filled with the boundary value above). */
    for (int j = 0; j < fcont->columns; j++) {
        for (int i = 0; i < fcont->rows - 1; i++) {
            int fcont_idx = sub2ind(fcont->rows, fcont->columns, (sub_t) {i, j});
            int fcont_no_boundary_idx = sub2ind(A.rows, fx.columns, (sub_t) {fcont->rows-2-i, j});
            fcont->mat_data[fcont_idx] = fcont_no_boundary[fcont_no_boundary_idx];
        }
    }
}
