#ifndef __GRID_INTERP_H__
#define __GRID_INTERP_H__

#include "num_linalg.h"

typedef struct locally_compute_result {
    double value;
    int in_range;
} locally_compute_result_t;

/*
 * Two-pass local barycentric-Lagrange interpolation on a uniform 2D grid of
 * sampled values, shared by Q_patch_obj.locally_compute (xi,eta grid) and
 * R_cartesian_mesh_obj.locally_compute (x,y grid) -- the two are otherwise
 * identical algorithms operating on different grids, so one kernel serves
 * both mex gateways.
 *
 * f is stored rows-major-in-"dim1", i.e. f.rows correspond to the dim1 axis
 * (eta, or y) and f.columns to the dim2 axis (xi, or x) -- the same
 * column-major (dim1 x dim2) convention MATLAB uses for f_XY / f_R.
 */
locally_compute_result_t grid_locally_compute(
    rd_mat_t f,
    double dim2_start, double dim2_end, /* xi or x range */
    double dim1_start, double dim1_end, /* eta or y range */
    double dim2_query, double dim1_query,
    int M);

#endif
