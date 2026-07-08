#ifndef __CARTESIAN_KERNELS_H__
#define __CARTESIAN_KERNELS_H__

#include "num_linalg.h"
#include "patch_kernels.h"

/**
 * Point-in-polygon test on a uniform Cartesian mesh via scanline/ray-casting
 * (ported from inpolygon_mesh.m / the original r_cartesian_mesh_lib.c).
 * boundary_X/boundary_Y must describe a closed polygon (first point repeated
 * at the end). Returns the number of interior points; in_msk is shaped to
 * match R_X.
 */
int inpolygon_mesh(rd_mat_t R_X, rd_mat_t R_Y, rd_mat_t boundary_X, rd_mat_t boundary_Y, ri_mat_t *in_msk);

typedef struct cartesian_mesh_geom {
    double x_start, y_start, h;
    int n_x, n_y;
    rd_mat_t R_X, R_Y;      /* (n_y x n_x) */
    ri_mat_t in_interior;   /* (n_y x n_x), nonzero => already-filled interior point */
} cartesian_mesh_geom_t;

typedef struct patch_grid {
    patch_spec_t spec;
    patch_bounds_t bounds;
    rd_mat_t f_XY;   /* n_eta x n_xi, rows=eta, columns=xi */
} patch_grid_t;

typedef struct interp_patch_result {
    int *r_patch_idxs;  /* 0-based linear (column-major) indices into the n_y x n_x grid */
    double *f_r_patch;  /* interpolated values, same length */
    int n_out;
} interp_patch_result_t;

/**
 * Combined replacement for R_cartesian_mesh_obj.interpolate_patch:
 * traces the patch's (n_r-refined) boundary polygon, finds Cartesian grid
 * points inside it but not already filled by the interior, inverts the
 * patch map at each via Newton's method (proximity-heuristic + propagation,
 * mirroring R_xi_eta_inversion.m exactly), then evaluates the patch's
 * function values there via local barycentric interpolation
 * (Q_patch_obj.locally_compute). Runs entirely in C, including the Newton
 * inversion's curve evaluations (see curve_eval.h) -- no MATLAB callbacks.
 */
interp_patch_result_t cartesian_mesh_interpolate_patch(
    cartesian_mesh_geom_t geom, patch_grid_t patch, int n_r, int M,
    double eps_xi_eta, double eps_xy);

void interp_patch_result_free(interp_patch_result_t *r);

#endif
