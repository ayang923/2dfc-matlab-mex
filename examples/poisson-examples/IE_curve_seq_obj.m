classdef IE_curve_seq_obj < handle
    %IE_CURVE_SEQ_OBJ Obj for integral equation solver
    %
    % Full mirror of the original 2dfc-matlab IE_curve_seq_obj, plus mex
    % acceleration: u_num dispatches to a mex kernel (ie_u_num_mex) when
    % available, falling back to the original per-curve loop unchanged.
    % A new u_num_batch method evaluates the double-layer potential at an
    % array of target points in one mex call (ie_u_num_batch_mex); its
    % fallback is simply a loop calling the original scalar u_num, so
    % behavior without mex is bit-identical to calling u_num in a loop
    % (which is exactly what laplace_solver.m's well-interior evaluation
    % loops did before this port).
    %
    % Neither u_num nor u_num_batch's mex path needs the curve-sampling/
    % barylag bridge used elsewhere in this codebase for Newton inversion:
    % they only ever evaluate a curve at its fixed, exactly-computable
    % graded quadrature mesh theta_mesh, never at an arbitrary Newton-
    % iterate-dependent point. See build_flat_nodes and
    % csrc/include/ie_kernels.h for the (exact-identity) algebraic
    % simplification used by the mex kernel.
    %
    % Note: construct_A_b_unstable, int_segment_general, int_segment_boundary,
    % and interp_int_seg from the original class are dead code (confirmed
    % zero call sites anywhere in examples/poisson-examples/) and are not
    % ported here.

    properties
        n_curves
        first_curve
        last_curve
    end

    methods
        function obj = IE_curve_seq_obj(curve_seq, p)
            %IE_CURVE_SEQ_OBJ Construct an instance of this class

            obj.n_curves = curve_seq.n_curves;
            obj.first_curve = IE_curve_obj(curve_seq.first_curve, 1, p);

            curr_curve = curve_seq.first_curve;
            curr_ie_curve = obj.first_curve;
            for i = 2:obj.n_curves
                curr_curve= curr_curve.next_curve;
                curr_ie_curve.next_curve = IE_curve_obj(curr_curve, i, p);
                curr_ie_curve = curr_ie_curve.next_curve;
            end
            obj.last_curve = curr_ie_curve;
            curr_ie_curve.next_curve = obj.first_curve;
        end

        function [A, b] = construct_A_b(obj, curve_param, f)
            % Dense BIE system assembly. Not mex-accelerated: already fully
            % vectorized (no scalar loop; the loop below is only over curve
            % PAIRS, and n_curves is 1-4 in every example in this repo),
            % n_total tops out at a few thousand, and the solve
            % (A_rho1 \ b_rho1, in laplace_solver.m) is already LAPACK-backed
            % by MATLAB's own backslash -- reimplementing it without MKL
            % would just rebuild what OpenBLAS/Accelerate already does.
            curve_n = curve_param.curve_n;
            n_total = curve_param.n_total;
            start_idx = curve_param. start_idx;
            end_idx = curve_param.end_idx;

            A = zeros(n_total, n_total);
            b = zeros(n_total, 1);


            curr_y = obj.first_curve;
            for i = 1:obj.n_curves
                [s_mesh_y, ds_y] = curr_y.s_mesh(curve_n(i));
                theta_mesh_y = curr_y.w(s_mesh_y);
                b(start_idx(i):end_idx(i)) = curr_y.w_prime(s_mesh_y).*f(curr_y.l_1(theta_mesh_y), curr_y.l_2(theta_mesh_y));

                curr_x = obj.first_curve;
                for j = 1:obj.n_curves
                    [s_mesh_x, ds_x] = curr_x.s_mesh(curve_n(j));
                    theta_mesh_x = curr_x.w(s_mesh_x);
                    [theta_2, theta_1] = meshgrid(theta_mesh_y, theta_mesh_x);

                    A_local = curr_x.w_prime(s_mesh_x).*curr_x.K_boundary(theta_1, theta_2, curr_y).*sqrt(curr_y.l_1_prime(theta_mesh_y').^2+curr_y.l_2_prime(theta_mesh_y').^2)*ds_y;

                    if i == j
                        msk_diagonals = logical(diag(ones(curve_n(j), 1)));
                        A_local(msk_diagonals) = curr_x.w_prime(s_mesh_x).*curr_x.K_boundary_same_point(theta_mesh_y).*sqrt(curr_x.l_1_prime(theta_mesh_x).^2+curr_x.l_2_prime(theta_mesh_x).^2)*ds_x + 1/2;
                    end

                    A(start_idx(j):end_idx(j), start_idx(i):end_idx(i)) = A_local;

                    curr_x = curr_x.next_curve;
                end

                curr_y = curr_y.next_curve;
            end
            A(isnan(A)) = 0;
        end

        function u_num = u_num(obj, x, y, curve_param, gr_phi)
            if use_mex_2dfc()
                [l1, l2, l1p, l2p, ds_per_node] = obj.build_flat_nodes(curve_param);
                weight = -1/(2*pi) * gr_phi .* ds_per_node;
                u_num = ie_u_num_mex(x, y, l1, l2, l1p, l2p, weight);
                return
            end

            curr = obj.first_curve;
            u_num = 0;
            for i = 1:obj.n_curves
                u_num =  u_num + curr.u_num_curve(x, y, curve_param, gr_phi);
                curr = curr.next_curve;
            end
        end

        function u_vals = u_num_batch(obj, X, Y, curve_param, gr_phi)
            % U_NUM_BATCH  Evaluates the double-layer potential at every
            % point in X,Y (same shape) in one call. Replaces the
            % well-interior evaluation loops in laplace_solver.m/
            % laplace_solver_coarse.m, which otherwise call u_num once per
            % point with the same (curve_param, gr_phi) for every point in
            % a pass -- the natural, risk-free batching opportunity (no
            % per-point branching/convergence state, unlike the patch-grid
            % and corner-region loops, which keep calling scalar u_num).
            if use_mex_2dfc()
                [l1, l2, l1p, l2p, ds_per_node] = obj.build_flat_nodes(curve_param);
                weight = -1/(2*pi) * gr_phi .* ds_per_node;
                u_vals = ie_u_num_batch_mex(X, Y, l1, l2, l1p, l2p, weight);
                return
            end

            u_vals = zeros(size(X));
            for i = 1:numel(X)
                u_vals(i) = obj.u_num(X(i), Y(i), curve_param, gr_phi);
            end
        end

        function [l1, l2, l1p, l2p, ds_per_node] = build_flat_nodes(obj, curve_param)
            % BUILD_FLAT_NODES  Concatenates every curve's exact-closure
            % evaluation at its own fixed quadrature mesh theta_mesh into
            % flat length-n_total arrays indexed the same way as gr_phi
            % (via curve_param.start_idx/end_idx). Pure O(n_total) vectorized
            % work using the curves' true closures -- no interpolation, since
            % theta_mesh is always exactly known in advance, never an
            % arbitrary Newton iterate.
            n_total = curve_param.n_total;
            l1 = zeros(n_total, 1);
            l2 = zeros(n_total, 1);
            l1p = zeros(n_total, 1);
            l2p = zeros(n_total, 1);
            ds_per_node = zeros(n_total, 1);

            curr = obj.first_curve;
            for i = 1:obj.n_curves
                curve_n = curve_param.curve_n(i);
                [s_mesh, ds] = curr.s_mesh(curve_n);
                theta_mesh = curr.w(s_mesh);
                idx = curve_param.start_idx(i):curve_param.end_idx(i);

                l1(idx) = curr.l_1(theta_mesh);
                l2(idx) = curr.l_2(theta_mesh);
                l1p(idx) = curr.l_1_prime(theta_mesh);
                l2p(idx) = curr.l_2_prime(theta_mesh);
                ds_per_node(idx) = ds;

                curr = curr.next_curve;
            end
        end

        function [s_patches, c_0_patches, c_1_patches] = construct_interior_patches(obj, curve_param, h_norm, M, eps_xi_eta, eps_xy)
            % Computing theta thresholds
            curr = obj.first_curve;
            for i = 1:obj.n_curves
                curr_n = curve_param.curve_n(i);
                next_n = curve_param.curve_n(mod(i, obj.n_curves)+1);
                curr_v = [curr.l_1(1); curr.l_2(1)] - [curr.l_1(1-1/curr_n); curr.l_2(1-1/curr_n)];
                next_v = [curr.next_curve.l_1(1/next_n); curr.next_curve.l_2(1/next_n)] - [curr.l_1(1); curr.l_2(1)];

                % C2-type patch means cross product is positive
                if curr_v(1)*next_v(2) - curr_v(2)*next_v(1) >= 0
                    curr.C2_corner = true;
                    [theta_1, theta_2] = compute_normal_intersection(curr, curr.next_curve, M, h_norm, eps_xy, [1; 0]);
                    curr.c_1_theta_thresh = floor(theta_1 * curr_n) / curr_n;
                    curr.next_curve.c_0_theta_thresh =  ceil(theta_2 * next_n) / next_n;
                 % C1-type patch otherwise
                else
                    curr.C2_corner = false;
                    curr.c_1_theta_thresh = 1;
                    curr.next_curve.c_0_theta_thresh = 0;
                end
                curr = curr.next_curve;
            end

            s_patches = cell(obj.n_curves, 1);
            c_0_patches = cell(obj.n_curves, 1);
            c_1_patches = cell(obj.n_curves, 1);
            curr = obj.first_curve;
            for i = 1:obj.n_curves
                [s_patches{i}, c_0_patches{i}, c_1_patches{i}] = curr.construct_interior_patch(curve_param, h_norm, M, eps_xi_eta, eps_xy);
                curr = curr.next_curve;
            end
        end
    end
end


function  [theta_1, theta_2] = compute_normal_intersection(curve_1, curve_2, M, h_norm, eps_xy, initial_guess)
    nu_norm = @(theta, curve) sqrt(curve.l_1_prime(theta).^2 + curve.l_2_prime(theta).^2);
    err_guess_x = @(theta_1, theta_2) curve_1.l_1(theta_1) - (M-1)*h_norm*(curve_1.l_2_prime(theta_1)./nu_norm(theta_1, curve_1)) - (curve_2.l_1(theta_2) - (M-1)*h_norm*(curve_2.l_2_prime(theta_2)./nu_norm(theta_2, curve_2)));
    err_guess_y = @(theta_1, theta_2) curve_1.l_2(theta_1) + (M-1)*h_norm*(curve_1.l_1_prime(theta_1)./nu_norm(theta_1, curve_1)) - (curve_2.l_2(theta_2) + (M-1)*h_norm*(curve_2.l_1_prime(theta_2)./nu_norm(theta_2, curve_2)));

    derr_x_d1 = @(theta_1, theta_2) curve_1.l_1_prime(theta_1)-(M-1)*h_norm*(curve_1.l_2_dprime(theta_1).*nu_norm(theta_1, curve_1).^2-curve_1.l_2_prime(theta_1).*(curve_1.l_2_dprime(theta_1).*curve_1.l_2_prime(theta_1)+curve_1.l_1_dprime(theta_1).*curve_1.l_1_prime(theta_1)))./nu_norm(theta_1, curve_1).^3;
    derr_y_d1 = @(theta_1, theta_2) curve_1.l_2_prime(theta_1)+(M-1)*h_norm*(curve_1.l_1_dprime(theta_1).*nu_norm(theta_1, curve_1).^2-curve_1.l_1_prime(theta_1).*(curve_1.l_2_dprime(theta_1).*curve_1.l_2_prime(theta_1)+curve_1.l_1_dprime(theta_1).*curve_1.l_1_prime(theta_1)))./nu_norm(theta_1, curve_1).^3;
    derr_x_d2 = @(theta_1, theta_2) -(curve_2.l_1_prime(theta_2)-(M-1)*h_norm*(curve_2.l_2_dprime(theta_2).*nu_norm(theta_2, curve_2).^2-curve_2.l_2_prime(theta_2).*(curve_2.l_2_dprime(theta_2).*curve_2.l_2_prime(theta_2)+curve_2.l_1_dprime(theta_2).*curve_2.l_1_prime(theta_2)))./nu_norm(theta_2, curve_2).^3);
    derr_y_d2 = @(theta_1, theta_2) -( curve_2.l_2_prime(theta_2)+(M-1)*h_norm*(curve_2.l_1_dprime(theta_2).*nu_norm(theta_2, curve_2).^2-curve_2.l_1_prime(theta_2).*(curve_2.l_2_dprime(theta_2).*curve_2.l_2_prime(theta_2)+curve_2.l_1_dprime(theta_2).*curve_2.l_1_prime(theta_2)))./nu_norm(theta_2, curve_2).^3);

    err_guess = @(v) [err_guess_x(v(1), v(2)); err_guess_y(v(1), v(2))];
    J_err = @(v) [derr_x_d1(v(1), v(2)) derr_x_d2(v(1), v(2)); derr_y_d1(v(1), v(2)) derr_y_d2(v(1), v(2))];

    [v_guess, converged] = newton_solve(err_guess, J_err, initial_guess, eps_xy, 100);
    theta_1 = v_guess(1); theta_2 = v_guess(2);
    if ~converged
        warning('Nonconvergence in normal-boundary intersection')
    end
end
