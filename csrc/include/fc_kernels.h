#ifndef __FC_KERNELS_H__
#define __FC_KERNELS_H__

#include "num_linalg.h"

/**
 * 1D blending-to-zero Fourier continuation for S-type patches (ported
 * directly from fcont_gram_blend_S.m / the original fc_lib.c, CBLAS dgemm
 * only -- no MKL). Operates column-wise: each column of fx is an independent
 * 1D signal; fx(0,:) is the boundary row.
 *
 * fx      - (n_eta x n_xi) matrix; only the first d rows are used
 * d       - number of Gram matching points
 * A       - (C*n_r x d) FC continuation matrix
 * Q       - (d x d) Gram orthogonalization matrix
 * fcont   - output (C*n_r+1 x n_xi) matrix; must point to pre-allocated data
 */
void fcont_gram_blend_S(rd_mat_t fx, int d, rd_mat_t A, rd_mat_t Q, rd_mat_t *fcont);

#endif
