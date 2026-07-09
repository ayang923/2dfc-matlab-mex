#ifndef __IE_KERNELS_H__
#define __IE_KERNELS_H__

/*
 * Kernel evaluation for the 2nd-kind boundary-integral-equation (Nystrom
 * double-layer-potential) Laplace solver used by the Poisson examples
 * (examples/poisson-examples/{IE_curve_obj,IE_curve_seq_obj,laplace_solver}.m).
 * There is no existing C implementation of this solver anywhere (unlike the
 * core FC2D port, which mirrors 2dfc-c) -- these kernels are derived fresh
 * from the MATLAB algorithm.
 *
 * Unlike the core port's Newton-inversion bridge (csrc/include/curve_eval.h),
 * this does NOT need arbitrary-theta curve evaluation or interpolation: the
 * double-layer potential u_num(x,y) = sum over all boundary quadrature nodes
 * of K_general(target, node) * (density * arc-length * quadrature weight) is
 * only ever evaluated at a curve's fixed, MATLAB-precomputable graded
 * (Kress sigmoidal) quadrature mesh -- never at a Newton-iterate-dependent
 * point. So MATLAB computes exact closure values at that fixed mesh once per
 * (curve, resolution) and hands these kernels plain arrays.
 *
 * Algebraic simplification (exact identity, not an approximation): the
 * original MATLAB per-node term is
 *   K_general(x,theta) * gr_phi(theta) * sqrt(l1p(theta)^2+l2p(theta)^2) * ds
 * where
 *   K_general(x,theta) = -1/(2*pi) * [(x1-l1)*l2p - (x2-l2)*l1p]
 *                         / (sqrt(l1p^2+l2p^2) * [(x1-l1)^2+(x2-l2)^2])
 * The sqrt(l1p^2+l2p^2) factor cancels exactly between K_general's
 * denominator and the arc-length element, leaving
 *   weight * [(x1-l1)*l2p - (x2-l2)*l1p] / [(x1-l1)^2+(x2-l2)^2]
 * where weight = -1/(2*pi) * gr_phi * ds is precomputed once per (curve_param,
 * gr_phi) pair, folded across all curves into one flat length-n_total array
 * (together with l1,l2,l1p,l2p sampled at each curve's own theta_mesh) --
 * no per-curve looping or sqrt needed in the hot per-target loop at all.
 */

/** Evaluates the double-layer potential at a single target point. */
double ie_u_num(double x, double y,
                 const double *l1, const double *l2,
                 const double *l1p, const double *l2p,
                 const double *weight, int n_total);

/** Evaluates the double-layer potential at an array of n_targets target points. */
void ie_u_num_batch(const double *targets_x, const double *targets_y, int n_targets,
                     const double *l1, const double *l2,
                     const double *l1p, const double *l2p,
                     const double *weight, int n_total,
                     double *out);

#endif
