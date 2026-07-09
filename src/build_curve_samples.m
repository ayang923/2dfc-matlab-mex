function samples = build_curve_samples(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime)
% BUILD_CURVE_SAMPLES  Samples a curve's l_1, l_2, and their first/second
% derivatives on a fine padded-domain uniform grid, for use in a
% mex_curve_spec struct (see Q_patch_obj.mex_spec) so the mex kernels can
% evaluate the curve at arbitrary theta without calling back into MATLAB --
% see csrc/include/curve_eval.h for why and how (local barycentric
% interpolation through the samples, the same technique the algorithm
% already uses to interpolate patch values).
%
% Shared by Curve_obj.get_mex_samples and IE_curve_obj.get_mex_samples --
% both wrap the same kind of arbitrary MATLAB closures and are equally
% exposed to the broadcasting subtlety handled here, so this logic (and
% its bcast fix) lives in exactly one place rather than being duplicated.
%
% The sampling domain [DOMAIN_LO, DOMAIN_HI] intentionally extends past the
% curve's nominal theta in [0,1] -- Newton's method and the corner/POU-
% normalization logic legitimately evaluate a curve slightly outside [0,1]
% (matching-region overlap), and a local interpolant needs real samples out
% there too, not just an extrapolation off the [0,1] endpoint. This must
% match CURVE_EVAL_DOMAIN_LO/HI in csrc/include/curve_eval.h exactly.
%
% N_SAMPLE and curve_M are chosen so interpolation error is far below the
% Newton tolerances (~1e-13) used in practice for the smooth analytic
% curves this algorithm targets; see csrc/include/curve_eval.h for the
% error-scaling argument.
%
% Inputs:
%   l_1, l_2                - Curve parametrization function handles
%   l_1_prime, l_2_prime    - First-derivative function handles
%   l_1_dprime, l_2_dprime  - Second-derivative function handles
%
% Output:
%   samples - struct with fields l1, l2, l1p, l2p, l1pp, l2pp (each a
%             N_SAMPLE-length column vector) and curve_M (scalar)

    DOMAIN_LO = -0.5;
    DOMAIN_HI = 1.5;
    N_SAMPLE = 40001;
    theta = transpose(linspace(DOMAIN_LO, DOMAIN_HI, N_SAMPLE));

    % Broadcast to a full N_SAMPLE-length vector: a curve's derivative (or
    % any of the six) may be a constant closure that ignores its argument
    % entirely and returns a scalar regardless of theta's length -- e.g. a
    % quadratic Bezier segment's l_1_dprime/l_2_dprime, which are literally
    % theta-independent constants. That's fine for the original algorithm
    % (it only ever evaluates these at scalar theta), but silently produces
    % a 1-element "sample array" here instead of the intended N_SAMPLE-
    % length one, which breaks sampled_fn_eval's uniform-grid spacing
    % (h = domain_length/(n-1) -> divide-by-zero -> NaN).
    samples = struct( ...
        'l1',   bcast(l_1(theta), N_SAMPLE), ...
        'l2',   bcast(l_2(theta), N_SAMPLE), ...
        'l1p',  bcast(l_1_prime(theta), N_SAMPLE), ...
        'l2p',  bcast(l_2_prime(theta), N_SAMPLE), ...
        'l1pp', bcast(l_1_dprime(theta), N_SAMPLE), ...
        'l2pp', bcast(l_2_dprime(theta), N_SAMPLE), ...
        'curve_M', 8);
end

function v = bcast(v, n)
% BCAST  Expands a scalar to a length-n column vector; passes a length-n
% vector through unchanged. See the header comment above for why this is
% needed (a curve-derivative closure that ignores its argument, such as a
% Bezier segment's constant second derivative, returns a scalar even when
% called with a length-n theta vector).
    if isscalar(v)
        v = repmat(v, n, 1);
    end
end
