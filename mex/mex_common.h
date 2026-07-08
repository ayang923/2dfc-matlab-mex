#ifndef __MEX_COMMON_H__
#define __MEX_COMMON_H__

#include "mex.h"
#include "num_linalg.h"
#include "patch_kernels.h"

/* Wraps an mxArray's (column-major) double data directly as an rd_mat_t --
 * no copy, since MATLAB's storage layout already matches rd_mat_t exactly. */
rd_mat_t rd_mat_from_mx(const mxArray *a);

/*
 * Unpacks a MATLAB struct (see Curve_obj.m's build_mex_curve_spec) into a
 * patch_spec_t. The struct has a numeric `kind` field (0 = S-type, 1 =
 * C-type) and the corresponding sampled-curve fields; see patch_kernels.h
 * for what each kind requires. Does not copy the sample arrays -- the
 * returned patch_spec_t's sampled_fn_t entries point directly into the
 * mxArray's data, so `spec` must not outlive `s`.
 */
void unpack_patch_spec(const mxArray *s, patch_spec_t *spec);

double mx_get_scalar_field(const mxArray *s, const char *name);
double mx_get_scalar_field_default(const mxArray *s, const char *name, double default_val);

#endif
