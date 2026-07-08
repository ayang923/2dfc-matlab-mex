% CURVE_OBJ  Stores a single C^2 boundary curve and constructs boundary patches.
%
% Each curve is parameterized as (l_1(theta), l_2(theta)) for theta in [0,1].
% The curve is discretized into n uniformly-spaced points in theta. Two types
% of patches are constructed from each curve:
%
%   S-patch  (smooth): covers the interior of the curve, trimmed at each end
%            to avoid overlap with the corner patches of adjacent curves
%   C-patch  (corner): covers the junction between this curve's end (theta=1)
%            and the next curve's start (theta=0)
%
% The corner type (convex vs. concave) is determined from the sign of the
% cross product of the incoming and outgoing tangent vectors:
%   cross >= 0  =>  convex corner  (interior angle < 180 deg)  =>  C2_patch_obj
%   cross < 0   =>  concave corner (interior angle > 180 deg)  =>  C1_patch_obj
%
% Properties:
%   l_1, l_2             - Curve parametrization functions (theta -> x, theta -> y)
%   l_1_prime, l_2_prime - First derivatives of l_1, l_2
%   l_1_dprime, l_2_dprime - Second derivatives of l_1, l_2
%   n                    - Number of discretization points along the curve
%   n_C_0                - Number of points used for corner patch at theta=0
%   n_C_1                - Number of points used for corner patch at theta=1
%   n_S_0                - Overlap (in points) between S-patch and C-patch at theta=0
%   n_S_1                - Overlap (in points) between S-patch and C-patch at theta=1
%   h_tan                - Tangential step size: 1/(n-1)
%   h_norm               - Normal step size (physical distance into domain)
%   next_curve           - Handle to the next Curve_obj in the sequence
%
% Methods:
%   Curve_obj        - Constructor
%   construct_S_patch - Builds the smooth S-type patch for this curve
%   construct_C_patch - Builds the corner C-type patch at theta=1
%   compute_length    - Arc-length integral of the curve
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef Curve_obj < handle
    properties
        l_1
        l_2
        l_1_prime
        l_2_prime
        l_1_dprime
        l_2_dprime

        n
        n_C_0
        n_C_1
        n_S_0
        n_S_1

        h_tan
        h_norm

        next_curve

        % Lazily-computed, memoized fine-grid samples of l_1/l_2 and their
        % derivatives, used to build mex_curve_spec structs (see
        % get_mex_samples). [] until first requested.
        mex_samples = []
    end

    methods
        function obj = Curve_obj(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
                n, frac_n_C_0, frac_n_C_1, frac_n_S_0, frac_n_S_1, h_norm, next_curve)
            % CURVE_OBJ  Constructor.
            %
            % Inputs:
            %   l_1, l_2             - Curve parametrization function handles
            %   l_1_prime, l_2_prime - First-derivative function handles
            %   l_1_dprime, l_2_dprime - Second-derivative function handles
            %   n           - Number of discretization points; if 0, computed
            %                 automatically as ceil(arc_length / h_norm) + 1
            %   frac_n_C_0  - Fraction of n used for the corner patch at theta=0;
            %                 if 0, defaults to 1/10
            %   frac_n_C_1  - Fraction of n used for the corner patch at theta=1;
            %                 if 0, defaults to 1/10
            %   frac_n_S_0  - Fraction of n_C_0 by which the S-patch overlaps the
            %                 theta=0 corner patch; if 0, defaults to 2/3
            %   frac_n_S_1  - Fraction of n_C_1 by which the S-patch overlaps the
            %                 theta=1 corner patch; if 0, defaults to 2/3
            %   h_norm      - Normal step size in physical space
            %   next_curve  - Handle to the next Curve_obj (or [] for single-curve domains)

            obj.l_1        = l_1;
            obj.l_2        = l_2;
            obj.l_1_prime  = l_1_prime;
            obj.l_2_prime  = l_2_prime;
            obj.l_1_dprime = l_1_dprime;
            obj.l_2_dprime = l_2_dprime;

            % Determine n, defaulting to arc-length / h_norm
            if n == 0
                L     = obj.compute_length();
                obj.n = ceil(L / h_norm) + 1;
            else
                obj.n = n;
            end

            % Determine corner-patch widths, defaulting to 1/10 of n
            if frac_n_C_0 == 0
                obj.n_C_0 = ceil(1/10 * obj.n);
            else
                obj.n_C_0 = ceil(frac_n_C_0 * obj.n);
            end
            if frac_n_C_1 == 0
                obj.n_C_1 = ceil(1/10 * obj.n);
            else
                obj.n_C_1 = ceil(frac_n_C_1 * obj.n);
            end

            % Determine S-patch overlap with corner patches, defaulting to 2/3 of n_C
            if frac_n_S_0 == 0
                obj.n_S_0 = ceil(2/3 * obj.n_C_0);
            else
                obj.n_S_0 = ceil(frac_n_S_0 * obj.n_C_0);
            end
            if frac_n_S_1 == 0
                obj.n_S_1 = ceil(2/3 * obj.n_C_1);
            else
                obj.n_S_1 = ceil(frac_n_S_1 * obj.n_C_1);
            end

            obj.h_tan  = 1 / (obj.n - 1);
            obj.h_norm = h_norm;

            if isempty(next_curve)
                obj.next_curve = obj;
            else
                obj.next_curve = next_curve;
            end
        end

        function S_patch = construct_S_patch(obj, f, d, eps_xi_eta, eps_xy)
            % CONSTRUCT_S_PATCH  Builds the smooth (S-type) boundary patch.
            %
            % The patch parametrization M_p uses the Frenet-Serret frame:
            %   xi  maps to theta via a linear rescaling that trims the ends to
            %       avoid overlap with the two corner patches
            %   eta moves inward along the unit outward normal at each boundary point
            %
            % The normal direction at theta is:
            %   n_hat = [-l_2'(theta), l_1'(theta)] / ||l'(theta)||
            %
            % Inputs:
            %   f          - Function handle f(x,y) to sample on the patch
            %   d          - Number of normal layers (Gram matching depth)
            %   eps_xi_eta - Newton inversion tolerance in parameter space
            %   eps_xy     - Newton inversion tolerance in physical space
            %
            % Outputs:
            %   S_patch - Constructed and filled S_patch_obj

            % Rescaling: xi in [0,1] maps to theta in [xi_0, xi_0 + xi_diff]
            % where xi_0 and xi_diff trim off the n_S_0 and n_S_1 overlap regions
            xi_diff  = 1 - (obj.n_C_1 - obj.n_S_1)*obj.h_tan - (obj.n_C_0 - obj.n_S_0)*obj.h_tan;
            xi_0     = (obj.n_C_0 - obj.n_S_0) * obj.h_tan;
            xi_tilde = @(xi) xi_diff*xi + xi_0;

            nu_norm = @(theta) sqrt(obj.l_1_prime(theta).^2 + obj.l_2_prime(theta).^2);

            % Parametrization: translate along the inward unit normal
            M_p_1 = @(xi, eta) obj.l_1(xi_tilde(xi)) - eta .* obj.l_2_prime(xi_tilde(xi)) ./ nu_norm(xi_tilde(xi));
            M_p_2 = @(xi, eta) obj.l_2(xi_tilde(xi)) + eta .* obj.l_1_prime(xi_tilde(xi)) ./ nu_norm(xi_tilde(xi));
            M_p   = @(xi, eta) [M_p_1(xi, eta), M_p_2(xi, eta)];

            % Jacobian components (chain rule with curvature correction in the xi terms)
            dM_p_1_dxi = @(xi, eta) xi_diff * ( ...
                obj.l_1_prime(xi_tilde(xi)) - eta .* ( ...
                    obj.l_2_dprime(xi_tilde(xi)) .* nu_norm(xi_tilde(xi)).^2 - ...
                    obj.l_2_prime(xi_tilde(xi)) .* (obj.l_2_dprime(xi_tilde(xi)) .* obj.l_2_prime(xi_tilde(xi)) + ...
                                                     obj.l_1_dprime(xi_tilde(xi)) .* obj.l_1_prime(xi_tilde(xi)))) ...
                ./ nu_norm(xi_tilde(xi)).^3);

            dM_p_2_dxi = @(xi, eta) xi_diff * ( ...
                obj.l_2_prime(xi_tilde(xi)) + eta .* ( ...
                    obj.l_1_dprime(xi_tilde(xi)) .* nu_norm(xi_tilde(xi)).^2 - ...
                    obj.l_1_prime(xi_tilde(xi)) .* (obj.l_2_dprime(xi_tilde(xi)) .* obj.l_2_prime(xi_tilde(xi)) + ...
                                                     obj.l_1_dprime(xi_tilde(xi)) .* obj.l_1_prime(xi_tilde(xi)))) ...
                ./ nu_norm(xi_tilde(xi)).^3);

            dM_p_1_deta = @(xi, eta) -obj.l_2_prime(xi_tilde(xi)) ./ nu_norm(xi_tilde(xi));
            dM_p_2_deta = @(xi, eta)  obj.l_1_prime(xi_tilde(xi)) ./ nu_norm(xi_tilde(xi));

            J = @(v) [dM_p_1_dxi(v(1), v(2)), dM_p_1_deta(v(1), v(2)); ...
                      dM_p_2_dxi(v(1), v(2)), dM_p_2_deta(v(1), v(2))];

            n_xi_S = obj.n - (obj.n_C_1 - obj.n_S_1) - (obj.n_C_0 - obj.n_S_0);

            samples = obj.get_mex_samples();
            mex_spec = struct('kind', 0, 'xi_diff', xi_diff, 'xi_0', xi_0, ...
                'l1', samples.l1, 'l2', samples.l2, ...
                'l1p', samples.l1p, 'l2p', samples.l2p, ...
                'l1pp', samples.l1pp, 'l2pp', samples.l2pp, ...
                'curve_M', samples.curve_M);

            S_patch = S_patch_obj(M_p, J, obj.h_norm, eps_xi_eta, eps_xy, n_xi_S, d, nan, mex_spec);
            [X, Y] = S_patch.Q.xy_mesh();
            S_patch.Q.f_XY = f(X, Y);
        end

        function samples = get_mex_samples(obj)
            % GET_MEX_SAMPLES  Lazily samples l_1, l_2, and their first/second
            % derivatives on a fine uniform grid, memoized on this curve. Used
            % to build mex_curve_spec structs so the mex kernels can evaluate
            % this (arbitrary, MATLAB-defined) curve at arbitrary theta
            % without calling back into MATLAB -- see csrc/include/curve_eval.h
            % for why and how (local barycentric interpolation through the
            % samples, the same technique the algorithm already uses to
            % interpolate patch values).
            %
            % The sampling domain [DOMAIN_LO, DOMAIN_HI] intentionally extends
            % past the curve's nominal theta in [0,1] -- Newton's method and
            % the corner/POU-normalization logic legitimately evaluate a
            % curve slightly outside [0,1] (matching-region overlap), and a
            % local interpolant needs real samples out there too, not just an
            % extrapolation off the [0,1] endpoint. This must match
            % CURVE_EVAL_DOMAIN_LO/HI in csrc/include/curve_eval.h exactly.
            %
            % N_SAMPLE and curve_M are chosen so interpolation error is far
            % below the Newton tolerances (~1e-13) used in practice for the
            % smooth analytic curves this algorithm targets; see
            % csrc/include/curve_eval.h for the error-scaling argument.

            if ~isempty(obj.mex_samples)
                samples = obj.mex_samples;
                return
            end

            DOMAIN_LO = -0.5;
            DOMAIN_HI = 1.5;
            N_SAMPLE = 40001;
            theta = transpose(linspace(DOMAIN_LO, DOMAIN_HI, N_SAMPLE));

            % Broadcast to a full N_SAMPLE-length vector: a curve's second
            % derivative (or any of the six) may be a constant closure that
            % ignores its argument entirely and returns a scalar regardless
            % of theta's length -- e.g. a quadratic Bezier segment's
            % l_1_dprime/l_2_dprime, which are literally theta-independent
            % constants. That's fine for the original algorithm (it only
            % ever evaluates these at scalar theta), but silently produces a
            % 1-element "sample array" here instead of the intended
            % N_SAMPLE-length one, which breaks sampled_fn_eval's uniform-
            % grid spacing (h = domain_length/(n-1) -> divide-by-zero -> NaN).
            samples = struct( ...
                'l1',   bcast(obj.l_1(theta), N_SAMPLE), ...
                'l2',   bcast(obj.l_2(theta), N_SAMPLE), ...
                'l1p',  bcast(obj.l_1_prime(theta), N_SAMPLE), ...
                'l2p',  bcast(obj.l_2_prime(theta), N_SAMPLE), ...
                'l1pp', bcast(obj.l_1_dprime(theta), N_SAMPLE), ...
                'l2pp', bcast(obj.l_2_dprime(theta), N_SAMPLE), ...
                'curve_M', 8);

            obj.mex_samples = samples;
        end

        function C_patch = construct_C_patch(obj, f, d, eps_xi_eta, eps_xy)
            % CONSTRUCT_C_PATCH  Builds the corner patch at theta=1 of this curve.
            %
            % Detects whether the corner is convex or concave by computing the
            % cross product of the incoming and outgoing tangent vectors.
            % The corner patch parametrization is:
            %
            %   M_p(xi, eta) = l_curr(xi_tilde(xi)) + l_next(eta_tilde(eta)) - corner
            %
            % Inputs:
            %   f          - Function handle f(x,y) to sample on the patch
            %   d          - Number of Gram matching points
            %   eps_xi_eta - Newton inversion tolerance in parameter space
            %   eps_xy     - Newton inversion tolerance in physical space
            %
            % Outputs:
            %   C_patch - Constructed and filled C2_patch_obj (convex) or
            %             C1_patch_obj (concave)

            curr_v = [obj.l_1(1); obj.l_2(1)] - ...
                     [obj.l_1(1 - 1/(obj.n-1)); obj.l_2(1 - 1/(obj.n-1))];
            next_v = [obj.next_curve.l_1(1/(obj.next_curve.n-1)); ...
                      obj.next_curve.l_2(1/(obj.next_curve.n-1))] - ...
                     [obj.l_1(1); obj.l_2(1)];

            cross_product = curr_v(1)*next_v(2) - curr_v(2)*next_v(1);

            if cross_product >= 0
                % Convex corner (interior angle < 180 deg) -> C2_patch_obj
                xi_diff  = -(obj.n_C_1 - 1) * obj.h_tan;
                eta_diff =  (obj.next_curve.n_C_0 - 1) * obj.next_curve.h_tan;
                xi_0     = 1;
                eta_0    = 0;

                xi_tilde  = @(xi)  xi_diff*xi  + xi_0;
                eta_tilde = @(eta) eta_diff*eta + eta_0;

                M_p_1 = @(xi, eta) obj.l_1(xi_tilde(xi)) + obj.next_curve.l_1(eta_tilde(eta)) - obj.l_1(1);
                M_p_2 = @(xi, eta) obj.l_2(xi_tilde(xi)) + obj.next_curve.l_2(eta_tilde(eta)) - obj.l_2(1);
                M_p   = @(xi, eta) [M_p_1(xi, eta), M_p_2(xi, eta)];

                dM_p_1_dxi  = @(xi, eta) xi_diff  * obj.l_1_prime(xi_tilde(xi));
                dM_p_2_dxi  = @(xi, eta) xi_diff  * obj.l_2_prime(xi_tilde(xi));
                dM_p_1_deta = @(xi, eta) eta_diff * obj.next_curve.l_1_prime(eta_tilde(eta));
                dM_p_2_deta = @(xi, eta) eta_diff * obj.next_curve.l_2_prime(eta_tilde(eta));

                J = @(v) [dM_p_1_dxi(v(1), v(2)), dM_p_1_deta(v(1), v(2)); ...
                          dM_p_2_dxi(v(1), v(2)), dM_p_2_deta(v(1), v(2))];

                mex_spec = obj.build_c_patch_mex_spec(xi_diff, xi_0, eta_diff, eta_0);
                C_patch = C2_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                    obj.n_C_1, obj.next_curve.n_C_0, d, nan, nan, mex_spec);
                [X_L, Y_L] = C_patch.L.xy_mesh();
                C_patch.L.f_XY = f(X_L, Y_L);
                [X_W, Y_W] = C_patch.W.xy_mesh();
                C_patch.W.f_XY = f(X_W, Y_W);

            else
                % Concave corner (interior angle > 180 deg) -> C1_patch_obj
                xi_diff  =  2 * (obj.n_C_1 - 1) * obj.h_tan;
                eta_diff = -2 * (obj.next_curve.n_C_0 - 1) * obj.next_curve.h_tan;
                xi_0     = 1 - (obj.n_C_1 - 1) * obj.h_tan;
                eta_0    =     (obj.next_curve.n_C_0 - 1) * obj.next_curve.h_tan;

                xi_tilde  = @(xi)  xi_diff*xi  + xi_0;
                eta_tilde = @(eta) eta_diff*eta + eta_0;

                M_p_1 = @(xi, eta) obj.l_1(xi_tilde(xi)) + obj.next_curve.l_1(eta_tilde(eta)) - obj.l_1(1);
                M_p_2 = @(xi, eta) obj.l_2(xi_tilde(xi)) + obj.next_curve.l_2(eta_tilde(eta)) - obj.l_2(1);
                M_p   = @(xi, eta) [M_p_1(xi, eta), M_p_2(xi, eta)];

                dM_p_1_dxi  = @(xi, eta) xi_diff  * obj.l_1_prime(xi_tilde(xi));
                dM_p_2_dxi  = @(xi, eta) xi_diff  * obj.l_2_prime(xi_tilde(xi));
                dM_p_1_deta = @(xi, eta) eta_diff * obj.next_curve.l_1_prime(eta_tilde(eta));
                dM_p_2_deta = @(xi, eta) eta_diff * obj.next_curve.l_2_prime(eta_tilde(eta));

                J = @(v) [dM_p_1_dxi(v(1), v(2)), dM_p_1_deta(v(1), v(2)); ...
                          dM_p_2_dxi(v(1), v(2)), dM_p_2_deta(v(1), v(2))];

                mex_spec = obj.build_c_patch_mex_spec(xi_diff, xi_0, eta_diff, eta_0);
                C_patch = C1_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                    obj.n_C_1*2 - 1, obj.next_curve.n_C_0*2 - 1, d, nan, nan, mex_spec);
                [X_L, Y_L] = C_patch.L.xy_mesh();
                C_patch.L.f_XY = f(X_L, Y_L);
                [X_W, Y_W] = C_patch.W.xy_mesh();
                C_patch.W.f_XY = f(X_W, Y_W);
            end
        end

        function mex_spec = build_c_patch_mex_spec(obj, xi_diff, xi_0, eta_diff, eta_0)
            % BUILD_C_PATCH_MEX_SPEC  Assembles the mex_curve_spec for a
            % C-type (corner) patch joining this curve's end to
            % obj.next_curve's start. Only called from construct_C_patch,
            % after the full Curve_seq_obj has finished being built (so
            % obj.next_curve is guaranteed to be finalized -- see
            % Curve_seq_obj.add_curve).
            curr_samples = obj.get_mex_samples();
            next_samples = obj.next_curve.get_mex_samples();

            mex_spec = struct('kind', 1, ...
                'xi_diff', xi_diff, 'xi_0', xi_0, ...
                'eta_diff', eta_diff, 'eta_0', eta_0, ...
                'curr_l1_at_1', obj.l_1(1), 'curr_l2_at_1', obj.l_2(1), ...
                'curr_l1', curr_samples.l1, 'curr_l2', curr_samples.l2, ...
                'curr_l1p', curr_samples.l1p, 'curr_l2p', curr_samples.l2p, ...
                'next_l1', next_samples.l1, 'next_l2', next_samples.l2, ...
                'next_l1p', next_samples.l1p, 'next_l2p', next_samples.l2p, ...
                'curve_M', curr_samples.curve_M);
        end

        function curve_length = compute_length(obj)
            % COMPUTE_LENGTH  Computes the arc length of the curve via quadrature.
            curve_length = integral( ...
                @(theta) sqrt(obj.l_1_prime(theta).^2 + obj.l_2_prime(theta).^2), 0, 1);
        end
    end
end

function v = bcast(v, n)
% BCAST  Expands a scalar to a length-n column vector; passes a
% length-n vector through unchanged. See get_mex_samples for why this
% is needed (a curve-derivative closure that ignores its argument,
% such as a Bezier segment's constant second derivative, returns a
% scalar even when called with a length-n theta vector).
    if isscalar(v)
        v = repmat(v, n, 1);
    end
end
