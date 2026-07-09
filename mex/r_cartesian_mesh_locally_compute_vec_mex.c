/* vals = r_cartesian_mesh_locally_compute_vec_mex(x_start, x_end, y_start, y_end, f_R, X, Y, M)
 * Batched drop-in backend for R_cartesian_mesh_obj.locally_compute_vec (a
 * new method), replacing the scalar-loop-over-locally_compute pattern
 * (R_locally_compute_vec in poisson_solver.m/poisson_solver_coarse.m) with
 * one mex call. Reuses grid_locally_compute verbatim -- no new C algorithm,
 * just a loop over targets and the mxArray marshaling. Out-of-range targets
 * get NaN (matching locally_compute's f_xy output on that path); unlike the
 * scalar MATLAB method, this does not emit a warning per out-of-range point
 * (diagnostic-only difference, not a numerical one -- call sites only ever
 * pass points already known to be in range). */
#include "mex.h"
#include "grid_interp.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "r_cartesian_mesh_locally_compute_vec_mex expects 8 inputs.");
    }

    double x_start = mxGetScalar(prhs[0]);
    double x_end   = mxGetScalar(prhs[1]);
    double y_start = mxGetScalar(prhs[2]);
    double y_end   = mxGetScalar(prhs[3]);
    rd_mat_t f_R = rd_mat_from_mx(prhs[4]);
    const double *X = mxGetPr(prhs[5]);
    const double *Y = mxGetPr(prhs[6]);
    int M = (int) mxGetScalar(prhs[7]);
    int n_targets = (int) mxGetNumberOfElements(prhs[5]);

    mwSize rows = mxGetM(prhs[5]);
    mwSize cols = mxGetN(prhs[5]);
    plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double *out = mxGetPr(plhs[0]);

    for (int i = 0; i < n_targets; i++) {
        locally_compute_result_t r = grid_locally_compute(f_R, x_start, x_end, y_start, y_end, X[i], Y[i], M);
        out[i] = r.in_range ? r.value : mxGetNaN();
    }
    (void) nlhs;
}
