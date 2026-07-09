/* u_vals = ie_u_num_batch_mex(X, Y, l1, l2, l1p, l2p, weight)
 * Batched double-layer-potential evaluation backing IE_curve_seq_obj.u_num_batch.
 * X, Y are arrays of target points (any shape; output matches X's shape).
 * l1,l2,l1p,l2p,weight are length-n_total column vectors as in ie_u_num_mex. */
#include "mex.h"
#include "ie_kernels.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "ie_u_num_batch_mex expects 7 inputs.");
    }

    const double *X = mxGetPr(prhs[0]);
    const double *Y = mxGetPr(prhs[1]);
    int n_targets = (int) mxGetNumberOfElements(prhs[0]);

    const double *l1 = mxGetPr(prhs[2]);
    const double *l2 = mxGetPr(prhs[3]);
    const double *l1p = mxGetPr(prhs[4]);
    const double *l2p = mxGetPr(prhs[5]);
    const double *weight = mxGetPr(prhs[6]);
    int n_total = (int) mxGetNumberOfElements(prhs[2]);

    mwSize rows = mxGetM(prhs[0]);
    mwSize cols = mxGetN(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
    double *out = mxGetPr(plhs[0]);

    ie_u_num_batch(X, Y, n_targets, l1, l2, l1p, l2p, weight, n_total, out);
    (void) nlhs;
}
