#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cartesian_kernels.h"
#include "grid_interp.h"
#include "num_linalg.h"

int inpolygon_mesh(rd_mat_t R_X, rd_mat_t R_Y, rd_mat_t boundary_X, rd_mat_t boundary_Y, ri_mat_t *in_msk) {
    ri_mat_shape(in_msk, R_X.rows, R_X.columns);
    memset(in_msk->mat_data, 0, (size_t) in_msk->rows * in_msk->columns * sizeof(int));

    int n_edges = boundary_X.rows - 1;

    rd_mat_t boundary_x_edge_1 = rd_mat_init(boundary_X.mat_data,     n_edges, 1);
    rd_mat_t boundary_x_edge_2 = rd_mat_init(boundary_X.mat_data + 1, n_edges, 1);
    rd_mat_t boundary_y_edge_1 = rd_mat_init(boundary_Y.mat_data,     n_edges, 1);
    rd_mat_t boundary_y_edge_2 = rd_mat_init(boundary_Y.mat_data + 1, n_edges, 1);

    double x_start = R_X.mat_data[0];
    double y_start = R_Y.mat_data[0];
    double h_x = R_X.mat_data[R_X.rows] - R_X.mat_data[0];
    double h_y = R_Y.mat_data[1] - R_Y.mat_data[0];

    double *boundary_y_j = malloc((size_t) boundary_Y.rows * sizeof(double));
    for (int i = 0; i < boundary_Y.rows; i++) {
        boundary_y_j[i] = (boundary_Y.mat_data[i] - y_start) / h_y;
    }

    double *boundary_y_edge_1_j = boundary_y_j;
    double *boundary_y_edge_2_j = boundary_y_j + 1;

    int *intersection_idxs = malloc((size_t) n_edges * sizeof(int));
    int n_intersection_edges = 0;
    for (int i = 0; i < n_edges; i++) {
        double e1 = boundary_y_edge_1_j[i];
        double e2 = boundary_y_edge_2_j[i];
        if (fabs(e1 - round(e1)) < DBL_EPSILON) e1 = round(e1);
        if (fabs(e2 - round(e2)) < DBL_EPSILON) e2 = round(e2);
        boundary_y_edge_1_j[i] = e1;
        boundary_y_edge_2_j[i] = e2;

        if (((int) floor(e1)) != ((int) floor(e2))) {
            intersection_idxs[n_intersection_edges++] = i;
        }
    }

    for (int k = 0; k < n_intersection_edges; k++) {
        int idx = intersection_idxs[k];

        double x_edge_1 = boundary_x_edge_1.mat_data[idx];
        double x_edge_2 = boundary_x_edge_2.mat_data[idx];
        double y_edge_1 = boundary_y_edge_1.mat_data[idx];
        double y_edge_2 = boundary_y_edge_2.mat_data[idx];
        int y_edge_1_j = (int) floor(boundary_y_edge_1_j[idx]);
        int y_edge_2_j = (int) floor(boundary_y_edge_2_j[idx]);

        int lo = MIN(y_edge_1_j, y_edge_2_j);
        int hi = MAX(y_edge_1_j, y_edge_2_j);

        for (int yj = lo + 1; yj <= hi; yj++) {
            double intersection_y = yj * h_y + y_start;
            double intersection_x = x_edge_1 + (x_edge_2 - x_edge_1) * (intersection_y - y_edge_1) / (y_edge_2 - y_edge_1);

            int row = (int) round((intersection_y - y_start) / h_y);
            int col = (int) floor((intersection_x - x_start) / h_x);
            int mesh_idx = sub2ind(in_msk->rows, in_msk->columns, (sub_t) {row, col});
            in_msk->mat_data[mesh_idx] = !in_msk->mat_data[mesh_idx];
        }
    }

    free(intersection_idxs);
    free(boundary_y_j);

    int n_points_interior = 0;
    for (int row = 0; row < in_msk->rows; row++) {
        bool in_interior = false;
        for (int col = 0; col < in_msk->columns; col++) {
            int idx = sub2ind(in_msk->rows, in_msk->columns, (sub_t) {row, col});
            if (in_msk->mat_data[idx] && !in_interior) {
                in_interior = true;
                in_msk->mat_data[idx] = 0;
            } else if (in_msk->mat_data[idx] && in_interior) {
                in_interior = false;
                n_points_interior += 1;
            } else if (in_interior) {
                in_msk->mat_data[idx] = 1;
                n_points_interior += 1;
            }
        }
    }

    return n_points_interior;
}

/* Traces the patch's parameter-space boundary at n_r refinement (matching
 * Q_patch_obj.boundary_mesh(n_r, false) exactly) and maps it to physical
 * space via M_p. */
static void patch_boundary_mesh_xy(const patch_spec_t *spec, patch_bounds_t b, int n_xi, int n_eta, int n_r,
                                    double **out_x, double **out_y, int *out_len) {
    int n_xi_r = n_xi * n_r;
    int n_eta_r = n_eta * n_r;
    int total = 2 * (n_xi_r + n_eta_r) + 1;

    double *bxi = malloc((size_t) total * sizeof(double));
    double *beta = malloc((size_t) total * sizeof(double));

    int p = 0;
    for (int i = 0; i < n_eta_r; i++) { bxi[p] = b.xi_start; beta[p] = b.eta_start + (b.eta_end - b.eta_start) * i / (double) (n_eta_r - 1); p++; }
    for (int i = 0; i < n_xi_r; i++)  { bxi[p] = b.xi_start + (b.xi_end - b.xi_start) * i / (double) (n_xi_r - 1); beta[p] = b.eta_end; p++; }
    for (int i = 0; i < n_eta_r; i++) { bxi[p] = b.xi_end; beta[p] = b.eta_end - (b.eta_end - b.eta_start) * i / (double) (n_eta_r - 1); p++; }
    for (int i = 0; i < n_xi_r; i++)  { bxi[p] = b.xi_end - (b.xi_end - b.xi_start) * i / (double) (n_xi_r - 1); beta[p] = b.eta_start; p++; }
    bxi[p] = b.xi_start; beta[p] = b.eta_start; p++;

    double *bx = malloc((size_t) total * sizeof(double));
    double *by = malloc((size_t) total * sizeof(double));
    for (int i = 0; i < total; i++) {
        patch_eval_M_p(spec, bxi[i], beta[i], &bx[i], &by[i]);
    }

    free(bxi);
    free(beta);
    *out_x = bx;
    *out_y = by;
    *out_len = total;
}

void interp_patch_result_free(interp_patch_result_t *r) {
    free(r->r_patch_idxs);
    free(r->f_r_patch);
    r->r_patch_idxs = NULL;
    r->f_r_patch = NULL;
    r->n_out = 0;
}

interp_patch_result_t cartesian_mesh_interpolate_patch(
    cartesian_mesh_geom_t geom, patch_grid_t patch, int n_r, int M,
    double eps_xi_eta, double eps_xy) {

    int n_xi = patch.f_XY.columns;
    int n_eta = patch.f_XY.rows;
    int n_grid = geom.n_x * geom.n_y;

    double *bound_x, *bound_y;
    int bound_len;
    patch_boundary_mesh_xy(&patch.spec, patch.bounds, n_xi, n_eta, n_r, &bound_x, &bound_y, &bound_len);

    rd_mat_t bound_X = rd_mat_init(bound_x, bound_len, 1);
    rd_mat_t bound_Y = rd_mat_init(bound_y, bound_len, 1);

    int *in_patch_data = malloc((size_t) n_grid * sizeof(int));
    ri_mat_t in_patch = ri_mat_init(in_patch_data, geom.n_y, geom.n_x);
    int n_in_patch = inpolygon_mesh(geom.R_X, geom.R_Y, bound_X, bound_Y, &in_patch);

    free(bound_x);
    free(bound_y);

    for (int i = 0; i < n_grid; i++) {
        if (geom.in_interior.mat_data[i] && in_patch_data[i]) {
            n_in_patch -= 1;
        }
        in_patch_data[i] = in_patch_data[i] && !(geom.in_interior.mat_data[i]);
    }

    int *r_patch_idxs = malloc((size_t) n_in_patch * sizeof(int));
    int curr_idx = 0;
    for (int i = 0; i < n_grid; i++) {
        if (in_patch_data[i]) {
            r_patch_idxs[curr_idx] = i;
            in_patch_data[i] = curr_idx + 1; /* 1-based marker into P_xi/P_eta */
            curr_idx += 1;
        }
    }

    /* Patch's own (xi,eta) grid mapped to physical space, used as the
     * proximity-heuristic initial guess source (pass 1). */
    int n_patch_grid = n_xi * n_eta;
    double *patch_XI = malloc((size_t) n_patch_grid * sizeof(double));
    double *patch_ETA = malloc((size_t) n_patch_grid * sizeof(double));
    double *patch_X = malloc((size_t) n_patch_grid * sizeof(double));
    double *patch_Y = malloc((size_t) n_patch_grid * sizeof(double));

    double h_xi = (patch.bounds.xi_end - patch.bounds.xi_start) / (n_xi - 1);
    double h_eta = (patch.bounds.eta_end - patch.bounds.eta_start) / (n_eta - 1);

    for (int i = 0; i < n_eta; i++) {
        for (int j = 0; j < n_xi; j++) {
            int idx = sub2ind(n_eta, n_xi, (sub_t) {i, j});
            double xi = patch.bounds.xi_start + j * h_xi;
            double eta = patch.bounds.eta_start + i * h_eta;
            patch_XI[idx] = xi;
            patch_ETA[idx] = eta;
            patch_eval_M_p(&patch.spec, xi, eta, &patch_X[idx], &patch_Y[idx]);
        }
    }

    double *P_xi = malloc((size_t) n_in_patch * sizeof(double));
    double *P_eta = malloc((size_t) n_in_patch * sizeof(double));
    for (int i = 0; i < n_in_patch; i++) { P_xi[i] = NAN; P_eta[i] = NAN; }

    for (int i = 0; i < n_patch_grid; i++) {
        int floor_X_j = (int) floor((patch_X[i] - geom.x_start) / geom.h);
        int ceil_X_j  = (int) ceil ((patch_X[i] - geom.x_start) / geom.h);
        int floor_Y_j = (int) floor((patch_Y[i] - geom.y_start) / geom.h);
        int ceil_Y_j  = (int) ceil ((patch_Y[i] - geom.y_start) / geom.h);

        int neighbors_X[4] = {floor_X_j, floor_X_j, ceil_X_j, ceil_X_j};
        int neighbors_Y[4] = {floor_Y_j, ceil_Y_j, floor_Y_j, ceil_Y_j};

        for (int nb = 0; nb < 4; nb++) {
            int nx = neighbors_X[nb];
            int ny = neighbors_Y[nb];
            if (nx > geom.n_x - 1 || nx < 0 || ny > geom.n_y - 1 || ny < 0) continue;

            int patch_idx = sub2ind(geom.n_y, geom.n_x, (sub_t) {ny, nx});
            int marker = in_patch_data[patch_idx];
            if (marker != 0 && isnan(P_xi[marker - 1])) {
                double nx_coord = nx * geom.h + geom.x_start;
                double ny_coord = ny * geom.h + geom.y_start;
                inverse_result_t r = patch_inverse_M_p(&patch.spec, patch.bounds, nx_coord, ny_coord,
                                                        &patch_XI[i], &patch_ETA[i], 1, eps_xi_eta, eps_xy);
                if (r.converged) {
                    P_xi[marker - 1] = r.xi;
                    P_eta[marker - 1] = r.eta;
                } else {
                    fprintf(stderr, "Nonconvergence in interpolation\n");
                }
            }
        }
    }

    /* Pass 2: propagate to any points pass 1 missed via 8-connected neighbors. */
    int neighbor_shift_x[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    int neighbor_shift_y[8] = {0, 0, -1, 1, 1, -1, 1, -1};

    /* Bounded by nan_count actually decreasing each full pass (not just a
     * generous iteration cap): each pass either shrinks the unresolved set
     * or, if a subset is topologically unreachable from any resolved seed
     * (isolated from the rest of the patch's footprint on this Cartesian
     * mesh), repeats forever with zero progress. Bail out in that case
     * instead of hanging -- this is a real (if rare) possibility inherited
     * from the original algorithm's own equivalent loop (R_xi_eta_inversion.m
     * / the 2dfc-c reference), not something specific to this port. */
    int prev_nan_count = n_in_patch + 1;
    while (true) {
        int nan_count = 0;
        for (int i = 0; i < n_in_patch; i++) {
            if (!isnan(P_xi[i])) continue;

            sub_t idx = ind2sub(geom.n_y, geom.n_x, r_patch_idxs[i]);
            bool touched = false;

            for (int s = 0; s < 8; s++) {
                int ni = idx.i + neighbor_shift_y[s];
                int nj = idx.j + neighbor_shift_x[s];
                if (nj > geom.n_x - 1 || nj < 0 || ni > geom.n_y - 1 || ni < 0) continue;

                int neighbor = sub2ind(geom.n_y, geom.n_x, (sub_t) {ni, nj});
                int marker = in_patch_data[neighbor];
                if (marker != 0 && !isnan(P_xi[marker - 1])) {
                    double x_coord = idx.j * geom.h + geom.x_start;
                    double y_coord = idx.i * geom.h + geom.y_start;
                    inverse_result_t r = patch_inverse_M_p(&patch.spec, patch.bounds, x_coord, y_coord,
                                                            &P_xi[marker - 1], &P_eta[marker - 1], 1,
                                                            eps_xi_eta, eps_xy);
                    if (r.converged) {
                        P_xi[i] = r.xi;
                        P_eta[i] = r.eta;
                        touched = true;
                        break;
                    } else {
                        fprintf(stderr, "Nonconvergence in interpolation\n");
                    }
                }
            }
            if (!touched) nan_count += 1;
        }

        if (nan_count == 0) break;
        if (nan_count >= prev_nan_count) {
            fprintf(stderr, "WARNING: %d point(s) unreachable during patch interpolation propagation; "
                             "leaving them at their interior-fill/zero default\n", nan_count);
            break;
        }
        prev_nan_count = nan_count;
    }

    double *f_r_patch = malloc((size_t) n_in_patch * sizeof(double));
    for (int i = 0; i < n_in_patch; i++) {
        if (isnan(P_xi[i]) || isnan(P_eta[i])) {
            /* Left unresolved by a stalled propagation (see warning above);
             * grid_locally_compute's range check can't be trusted with a
             * NaN query, so skip it explicitly rather than risk an
             * out-of-bounds access from an undefined NaN-to-int cast. */
            f_r_patch[i] = 0.0;
            continue;
        }
        locally_compute_result_t lc = grid_locally_compute(
            patch.f_XY, patch.bounds.xi_start, patch.bounds.xi_end,
            patch.bounds.eta_start, patch.bounds.eta_end, P_xi[i], P_eta[i], M);
        if (lc.in_range) {
            f_r_patch[i] = lc.value;
        } else {
            f_r_patch[i] = 0.0;
            fprintf(stderr, "WARNING: interpolating point not in patch\n");
        }
    }

    free(in_patch_data);
    free(patch_XI);
    free(patch_ETA);
    free(patch_X);
    free(patch_Y);
    free(P_xi);
    free(P_eta);

    return (interp_patch_result_t) {r_patch_idxs, f_r_patch, n_in_patch};
}
