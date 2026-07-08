/* [xi, eta, converged] = q_patch_inverse_M_p_mex(mex_curve_spec, ...
 *     xi_start, xi_end, eta_start, eta_end, eps_xi_eta, eps_xy, x, y, initial_guesses)
 *
 * Drop-in mex-accelerated backend for Q_patch_obj.inverse_M_p, used only
 * when the patch was built by Curve_obj (i.e. carries a mex_curve_spec) --
 * see patch_kernels.h for why this is limited to the two closed-form
 * S-type/C-type templates rather than fully arbitrary M_p/J handles.
 * initial_guesses is a (2 x k) matrix [xi_row; eta_row], or NaN to request
 * the built-in default boundary-edge guesses. */
#include <stdlib.h>

#include "mex.h"
#include "patch_kernels.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 10) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "q_patch_inverse_M_p_mex expects 10 inputs.");
    }

    patch_spec_t spec;
    unpack_patch_spec(prhs[0], &spec);

    patch_bounds_t bounds;
    bounds.xi_start  = mxGetScalar(prhs[1]);
    bounds.xi_end    = mxGetScalar(prhs[2]);
    bounds.eta_start = mxGetScalar(prhs[3]);
    bounds.eta_end   = mxGetScalar(prhs[4]);
    double eps_xi_eta = mxGetScalar(prhs[5]);
    double eps_xy     = mxGetScalar(prhs[6]);
    double x = mxGetScalar(prhs[7]);
    double y = mxGetScalar(prhs[8]);

    const mxArray *ig = prhs[9];
    double *init_xi = NULL;
    double *init_eta = NULL;
    int n_init = 0;

    if (!(mxGetNumberOfElements(ig) == 1 && mxIsNaN(mxGetScalar(ig)))) {
        /* ig is a (2 x k) column-major matrix [xi;eta] per column: unpack
         * the interleaved (xi0,eta0,xi1,eta1,...) storage into two
         * contiguous arrays for patch_inverse_M_p. */
        int k = (int) mxGetN(ig);
        const double *data = mxGetPr(ig);
        init_xi = malloc((size_t) k * sizeof(double));
        init_eta = malloc((size_t) k * sizeof(double));
        for (int i = 0; i < k; i++) {
            init_xi[i]  = data[2*i];
            init_eta[i] = data[2*i + 1];
        }
        n_init = k;
    }

    inverse_result_t r = patch_inverse_M_p(&spec, bounds, x, y, init_xi, init_eta, n_init, eps_xi_eta, eps_xy);

    free(init_xi);
    free(init_eta);

    plhs[0] = mxCreateDoubleScalar(r.xi);
    if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(r.eta);
    if (nlhs > 2) plhs[2] = mxCreateLogicalScalar(r.converged != 0);
}
