/* in = inpolygon_mesh_mex(R_X, R_Y, boundary_X, boundary_Y)
 * Drop-in mex-accelerated backend for inpolygon_mesh.m. */
#include <stdlib.h>

#include "mex.h"
#include "cartesian_kernels.h"
#include "mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("2dfcMex:nargin", "inpolygon_mesh_mex expects 4 inputs.");
    }

    rd_mat_t R_X = rd_mat_from_mx(prhs[0]);
    rd_mat_t R_Y = rd_mat_from_mx(prhs[1]);
    rd_mat_t boundary_X = rd_mat_from_mx(prhs[2]);
    rd_mat_t boundary_Y = rd_mat_from_mx(prhs[3]);

    int *in_data = malloc((size_t) R_X.rows * R_X.columns * sizeof(int));
    ri_mat_t in_msk = ri_mat_init(in_data, 0, 0);
    inpolygon_mesh(R_X, R_Y, boundary_X, boundary_Y, &in_msk);

    plhs[0] = mxCreateLogicalMatrix(R_X.rows, R_X.columns);
    mxLogical *out = mxGetLogicals(plhs[0]);
    int n = R_X.rows * R_X.columns;
    for (int i = 0; i < n; i++) {
        out[i] = in_data[i] != 0;
    }

    free(in_data);
    (void) nlhs;
}
