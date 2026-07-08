#include <string.h>

#include "mex_common.h"

rd_mat_t rd_mat_from_mx(const mxArray *a) {
    return rd_mat_init(mxGetPr(a), (int) mxGetM(a), (int) mxGetN(a));
}

double mx_get_scalar_field(const mxArray *s, const char *name) {
    mxArray *f = mxGetField(s, 0, name);
    if (f == NULL) {
        mexErrMsgIdAndTxt("2dfcMex:missingField", "mex_curve_spec missing required field '%s'", name);
    }
    return mxGetScalar(f);
}

double mx_get_scalar_field_default(const mxArray *s, const char *name, double default_val) {
    mxArray *f = mxGetField(s, 0, name);
    if (f == NULL || mxIsEmpty(f)) {
        return default_val;
    }
    return mxGetScalar(f);
}

static sampled_fn_t get_sampled_fn(const mxArray *s, const char *name) {
    mxArray *f = mxGetField(s, 0, name);
    if (f == NULL) {
        mexErrMsgIdAndTxt("2dfcMex:missingField", "mex_curve_spec missing required field '%s'", name);
    }
    sampled_fn_t fn;
    fn.samples = mxGetPr(f);
    fn.n_samples = (int) mxGetNumberOfElements(f);
    return fn;
}

void unpack_patch_spec(const mxArray *s, patch_spec_t *spec) {
    int kind = (int) mx_get_scalar_field(s, "kind");
    spec->curve_M = (int) mx_get_scalar_field_default(s, "curve_M", CURVE_EVAL_DEFAULT_M);

    if (kind == 0) {
        spec->kind = PATCH_KIND_S;
        s_patch_curve_spec_t *p = &spec->u.s;
        p->xi_diff = mx_get_scalar_field(s, "xi_diff");
        p->xi_0    = mx_get_scalar_field(s, "xi_0");
        p->l1   = get_sampled_fn(s, "l1");
        p->l2   = get_sampled_fn(s, "l2");
        p->l1p  = get_sampled_fn(s, "l1p");
        p->l2p  = get_sampled_fn(s, "l2p");
        p->l1pp = get_sampled_fn(s, "l1pp");
        p->l2pp = get_sampled_fn(s, "l2pp");
    } else {
        spec->kind = PATCH_KIND_C;
        c_patch_curve_spec_t *p = &spec->u.c;
        p->xi_diff  = mx_get_scalar_field(s, "xi_diff");
        p->xi_0     = mx_get_scalar_field(s, "xi_0");
        p->eta_diff = mx_get_scalar_field(s, "eta_diff");
        p->eta_0    = mx_get_scalar_field(s, "eta_0");
        p->curr_l1_at_1 = mx_get_scalar_field(s, "curr_l1_at_1");
        p->curr_l2_at_1 = mx_get_scalar_field(s, "curr_l2_at_1");
        p->curr_l1  = get_sampled_fn(s, "curr_l1");
        p->curr_l2  = get_sampled_fn(s, "curr_l2");
        p->curr_l1p = get_sampled_fn(s, "curr_l1p");
        p->curr_l2p = get_sampled_fn(s, "curr_l2p");
        p->next_l1  = get_sampled_fn(s, "next_l1");
        p->next_l2  = get_sampled_fn(s, "next_l2");
        p->next_l1p = get_sampled_fn(s, "next_l1p");
        p->next_l2p = get_sampled_fn(s, "next_l2p");
    }
}
