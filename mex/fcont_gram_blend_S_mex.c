/* fcont = fcont_gram_blend_S_mex(fx, d, A, Q)
 * Drop-in mex-accelerated (CBLAS dgemm) backend for fcont_gram_blend_S.m. */
#include "mex.h"
#include "fc_kernels.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "fcont_gram_blend_S_mex expects 4 inputs.");
    }

    rd_mat_t fx = rd_mat_from_mx(prhs[0]);
    int d = (int) mxGetScalar(prhs[1]);
    rd_mat_t A = rd_mat_from_mx(prhs[2]);
    rd_mat_t Q = rd_mat_from_mx(prhs[3]);

    plhs[0] = mxCreateDoubleMatrix(A.rows + 1, fx.columns, mxREAL);
    rd_mat_t fcont = rd_mat_init(mxGetPr(plhs[0]), A.rows + 1, fx.columns);

    fcont_gram_blend_S(fx, d, A, Q, &fcont);
    (void) nlhs;
}
