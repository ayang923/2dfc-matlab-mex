/* [R_patch_idxs, f_R_patch] = r_cartesian_mesh_interpolate_patch_mex( ...
 *     mex_curve_spec, xi_start, xi_end, eta_start, eta_end, f_XY, ...
 *     R_X, R_Y, in_interior, x_start, y_start, h, n_r, M, eps_xi_eta, eps_xy)
 *
 * Combined mex-accelerated replacement for R_cartesian_mesh_obj.interpolate_patch
 * (which otherwise calls inpolygon_mesh + R_xi_eta_inversion + a per-point
 * locally_compute loop): does all of that in one C call, including every
 * Newton inversion, with no MATLAB callbacks. R_patch_idxs is returned
 * 1-based (MATLAB linear-index convention) so the caller can do
 * `obj.f_R(R_patch_idxs) = obj.f_R(R_patch_idxs) + f_R_patch` exactly as
 * the original method does. */
#include <stdlib.h>

#include "mex.h"
#include "cartesian_kernels.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 16) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "r_cartesian_mesh_interpolate_patch_mex expects 16 inputs.");
    }

    patch_grid_t patch;
    unpack_patch_spec(prhs[0], &patch.spec);
    patch.bounds.xi_start  = mxGetScalar(prhs[1]);
    patch.bounds.xi_end    = mxGetScalar(prhs[2]);
    patch.bounds.eta_start = mxGetScalar(prhs[3]);
    patch.bounds.eta_end   = mxGetScalar(prhs[4]);
    patch.f_XY = rd_mat_from_mx(prhs[5]);

    cartesian_mesh_geom_t geom;
    geom.R_X = rd_mat_from_mx(prhs[6]);
    geom.R_Y = rd_mat_from_mx(prhs[7]);
    geom.n_x = geom.R_X.columns;
    geom.n_y = geom.R_X.rows;

    const mxArray *in_interior_mx = prhs[8];
    int n_grid = geom.n_x * geom.n_y;
    int *in_interior_data = malloc((size_t) n_grid * sizeof(int));
    if (mxIsLogical(in_interior_mx)) {
        mxLogical *li = mxGetLogicals(in_interior_mx);
        for (int i = 0; i < n_grid; i++) in_interior_data[i] = li[i] != 0;
    } else {
        double *ld = mxGetPr(in_interior_mx);
        for (int i = 0; i < n_grid; i++) in_interior_data[i] = ld[i] != 0;
    }
    geom.in_interior = ri_mat_init(in_interior_data, geom.n_y, geom.n_x);

    geom.x_start = mxGetScalar(prhs[9]);
    geom.y_start = mxGetScalar(prhs[10]);
    geom.h       = mxGetScalar(prhs[11]);
    int n_r = (int) mxGetScalar(prhs[12]);
    int M   = (int) mxGetScalar(prhs[13]);
    double eps_xi_eta = mxGetScalar(prhs[14]);
    double eps_xy     = mxGetScalar(prhs[15]);

    interp_patch_result_t result = cartesian_mesh_interpolate_patch(geom, patch, n_r, M, eps_xi_eta, eps_xy);

    plhs[0] = mxCreateDoubleMatrix(result.n_out, 1, mxREAL);
    double *idxs_out = mxGetPr(plhs[0]);
    for (int i = 0; i < result.n_out; i++) {
        idxs_out[i] = result.r_patch_idxs[i] + 1; /* 0-based -> MATLAB 1-based */
    }

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(result.n_out, 1, mxREAL);
        double *f_out = mxGetPr(plhs[1]);
        for (int i = 0; i < result.n_out; i++) f_out[i] = result.f_r_patch[i];
    }

    interp_patch_result_free(&result);
    free(in_interior_data);
}
