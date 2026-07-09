/* u = ie_u_num_mex(x, y, l1, l2, l1p, l2p, weight)
 * Scalar double-layer-potential evaluation backing IE_curve_seq_obj.u_num's
 * mex-dispatch branch. l1,l2,l1p,l2p,weight are length-n_total column
 * vectors: the curve's exact closures evaluated at its fixed quadrature
 * mesh theta_mesh (flattened across all curves), and the precomputed
 * weight = -1/(2*pi) * gr_phi .* ds (see csrc/include/ie_kernels.h for the
 * algebraic identity that lets the per-target kernel skip nu_norm/sqrt
 * entirely). */
#include "mex.h"
#include "ie_kernels.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "ie_u_num_mex expects 7 inputs.");
    }

    double x = mxGetScalar(prhs[0]);
    double y = mxGetScalar(prhs[1]);
    const double *l1 = mxGetPr(prhs[2]);
    const double *l2 = mxGetPr(prhs[3]);
    const double *l1p = mxGetPr(prhs[4]);
    const double *l2p = mxGetPr(prhs[5]);
    const double *weight = mxGetPr(prhs[6]);
    int n_total = (int) mxGetNumberOfElements(prhs[2]);

    double u = ie_u_num(x, y, l1, l2, l1p, l2p, weight, n_total);

    plhs[0] = mxCreateDoubleScalar(u);
    (void) nlhs;
}
