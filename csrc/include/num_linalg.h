#ifndef __NUM_LINALG_H__
#define __NUM_LINALG_H__

#include <stddef.h>

/*
 * Portability note: the original 2dfc-c library used Intel MKL's MKL_INT
 * (and cblas, LAPACKE, and VML calls) throughout. This port needs no Intel
 * toolchain: plain int is used for indices/sizes, and only standard CBLAS
 * (see blas_compat.h) is required for the one matrix multiply in the whole
 * kernel set (fc_kernels.c). No LAPACK dependency at all -- the only linear
 * solve in the codebase is a fixed 2x2 system (Newton's method for a planar
 * map), which is hand-solved in patch_kernels.c instead of calling LAPACKE.
 */

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/** Real double-precision matrix stored in column-major order. */
typedef struct rd_mat {
    double *mat_data;
    int rows;
    int columns;
} rd_mat_t;

/** Integer matrix stored in column-major order. */
typedef struct ri_mat {
    int *mat_data;
    int rows;
    int columns;
} ri_mat_t;

/** (row, column) subscript into a 2D matrix. */
typedef struct sub {
    int i;
    int j;
} sub_t;

int sub2ind(int rows, int columns, sub_t sub);
sub_t ind2sub(int rows, int columns, int idx);

void rd_linspace(double start, double end, int n, rd_mat_t *mat_addr);
void rd_meshgrid(rd_mat_t x, rd_mat_t y, rd_mat_t *X, rd_mat_t *Y);

/* Clamps an integer index window to [min_bound, max_bound] by shifting
 * (preserving its length). Used to keep local interpolation stencils
 * in-bounds near the edges of a patch/array. */
void shift_idx_mesh(ri_mat_t *mat, int min_bound, int max_bound);

rd_mat_t rd_mat_init(double *mat_data_addr, int rows, int columns);
rd_mat_t rd_mat_init_no_shape(double *mat_data_addr);
ri_mat_t ri_mat_init(int *mat_data_addr, int rows, int columns);
ri_mat_t ri_mat_init_no_shape(int *mat_data_addr);

void rd_mat_shape(rd_mat_t *mat, int rows, int columns);
void ri_mat_shape(ri_mat_t *mat, int rows, int columns);

void ri_range(int start, int step_size, int end, ri_mat_t *mat_addr);
void ri_meshgrid(ri_mat_t x, ri_mat_t y, ri_mat_t *X, ri_mat_t *Y);

/*
 * Evaluates the barycentric Lagrange interpolating polynomial through
 * (ix[k], iy[k]) at x. ix, iy are column vectors of the same length
 * (assumed distinct nodes). If x coincides with a node, that node's value
 * is returned exactly.
 */
double barylag(rd_mat_t ix, rd_mat_t iy, double x);

#endif
