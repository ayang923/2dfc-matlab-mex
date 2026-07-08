/* p = barylag_mex(data, x)
 * Drop-in mex-accelerated backend for barylag.m. data is an (N x 2) matrix
 * [nodes, values]; x is a vector of evaluation points (matching barylag.m's
 * vectorized-x contract); p is a vector the same length as x. */
#include "mex.h"
#include "num_linalg.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "barylag_mex expects 2 inputs.");
    }

    int n = (int) mxGetM(prhs[0]);
    double *data = mxGetPr(prhs[0]);
    rd_mat_t ix = rd_mat_init(data, n, 1);
    rd_mat_t iy = rd_mat_init(data + n, n, 1);

    int n_x = (int) mxGetNumberOfElements(prhs[1]);
    double *x = mxGetPr(prhs[1]);

    mwSize rows = mxGetM(prhs[1]);
    mwSize cols = mxGetN(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double *p = mxGetPr(plhs[0]);

    for (int k = 0; k < n_x; k++) {
        p[k] = barylag(ix, iy, x[k]);
    }
    (void) nlhs;
}
