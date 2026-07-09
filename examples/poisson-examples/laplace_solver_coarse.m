function [u_num_mat] = laplace_solver_coarse(R, R_eval, curve_seq, u_G, cfac, p, M, int_eps, eps_xi_eta, eps_xy, rho)
% LAPLACE_SOLVER_COARSE  Solves Delta(u) = 0 and returns values on a coarser grid.
%
% Same BIE algorithm as laplace_solver, but the solution is evaluated on the
% coarser mesh R_eval rather than the fine FC grid R. The patch geometry is
% built using R.h, but the output matrix is sized to R_eval.
%
% For smooth-patch points in R_eval, an additional check is made: if the
% inverse map gives an eta coordinate smaller than the patch's minimum node
% spacing h_thresh, the point is handled by polynomial interpolation from
% M pre-computed layer values (at eta = 0, h_thresh, ..., (M-1)*h_thresh).
%
% As in laplace_solver, all four IE evaluation sites (well-interior,
% near-boundary interpolation-node, corner-region) are batched via
% IE_curve_seq_obj.u_num_batch (mex-accelerated): every point in the active
% set is evaluated at a shared (curve_param, gr_phi) resolution, converged
% points are frozen and dropped, and the remaining active points move on to
% the next resolution together -- see laplace_solver.m's header comment for
% why this is safe (the coarse/fine resolution state was always a single
% pair shared across all points at a given site, even in the original
% per-point while-loops).
%
% Inputs:
%   R          - Fine R_cartesian_mesh_obj used for the FC step (h fine)
%   R_eval     - Coarse R_cartesian_mesh_obj on which the solution is returned
%   curve_seq  - Curve_seq_obj describing the domain boundary
%   u_G        - Function handle for the Dirichlet boundary data u_G(x,y)
%   cfac       - Quadrature refinement factor for the IE boundary discretization
%   p          - Polynomial degree for the IE graded-mesh quadrature
%   M          - Polynomial interpolation degree
%   int_eps    - Convergence tolerance for adaptive IE integration
%   eps_xi_eta - Newton inversion tolerance in (xi,eta) space
%   eps_xy     - Newton inversion tolerance in (x,y) space
%   rho        - Patch-exclusion radius multiplier (in units of h)
%
% Output:
%   u_num_mat - (n_y_eval x n_x_eval) solution values on R_eval; NaN outside domain

    % Hard cap on the IE refinement level -- see laplace_solver.m's header
    % comment for why this guards against an infinite loop and is shared
    % across all evaluation sites below. Every rho_fine_IE advance uses
    % next_rho_IE (step size scales with the current level's order of
    % magnitude), so reaching this cap from rho=2 takes on the order of
    % tens of refinement rounds, not thousands.
    MAX_RHO_IE = 10000;

    % ------------------------------------------------------------------ %
    % Build boundary quadrature                                           %
    % ------------------------------------------------------------------ %
    curve_n_rho1 = zeros(curve_seq.n_curves, 1);
    curr = curve_seq.first_curve;
    for i = 1:curve_seq.n_curves
        curve_n_rho1(i) = ceil((curr.n - 1) * cfac);
        curr = curr.next_curve;
    end
    curve_param_rho1 = curve_param_obj(curve_n_rho1);

    % ------------------------------------------------------------------ %
    % Solve the BIE for the double-layer density phi                      %
    % ------------------------------------------------------------------ %
    IE_curve_seq = IE_curve_seq_obj(curve_seq, p);
    [A_rho1, b_rho1] = IE_curve_seq.construct_A_b(curve_param_rho1, u_G);
    gr_phi_rho1     = A_rho1 \ b_rho1;
    gr_phi_fft_rho1 = fftshift(fft(gr_phi_rho1)) / curve_param_rho1.n_total;

    [s_patches, c_0_patches, c_1_patches] = IE_curve_seq.construct_interior_patches( ...
        curve_param_rho1, R.h, M, eps_xi_eta, eps_xy);
    [well_interior_msk, s_patch_msks] = gen_R_msks(R_eval, rho, s_patches, c_0_patches, c_1_patches);

    % ------------------------------------------------------------------ %
    % Identify patch points that need near-boundary interpolation         %
    % ------------------------------------------------------------------ %
    interpol_nodes_x   = [];
    interpol_nodes_y   = [];
    interpol_target_eta = [];
    interpol_target_idx = [];

    for i = 1:curve_seq.n_curves
        s_patch = s_patches{i};
        [~, h_thresh] = s_patch.h_mesh;
        in_patch  = s_patch_msks{i};
        R_s_idxs  = R_eval.R_idxs(in_patch);

        needs_interpol_msk = false(length(R_s_idxs), 1);
        [P_xi_s, P_eta_s]  = R_xi_eta_inversion(R_eval, s_patch, in_patch);

        for idx = 1:length(R_s_idxs)
            if s_patch.in_patch(P_xi_s(idx), P_eta_s(idx))
                if P_eta_s(idx) < h_thresh
                    needs_interpol_msk(idx) = true;
                else
                    well_interior_msk(R_s_idxs(idx)) = true;
                end
            else
                s_patch_msks{i}(R_s_idxs(idx)) = false;
            end
        end

        n_interpol = sum(needs_interpol_msk);
        interpol_target_eta = [interpol_target_eta; P_eta_s(needs_interpol_msk)];
        interpol_target_idx = [interpol_target_idx; R_s_idxs(needs_interpol_msk)];

        interpol_mesh_xi_patch  = repmat(P_xi_s(needs_interpol_msk), 1, M);
        interpol_mesh_eta_patch = repmat((0:(M-1)) * h_thresh, n_interpol, 1);
        [interpol_nodes_x_patch, interpol_nodes_y_patch] = ...
            s_patch.convert_to_XY(interpol_mesh_xi_patch, interpol_mesh_eta_patch);
        interpol_nodes_x = [interpol_nodes_x; interpol_nodes_x_patch];
        interpol_nodes_y = [interpol_nodes_y; interpol_nodes_y_patch];
    end

    % ------------------------------------------------------------------ %
    % Evaluate well-interior points with adaptive IE refinement           %
    % ------------------------------------------------------------------ %
    u_num_mat      = zeros(size(R_eval.f_R));
    u_num_mat_fine = zeros(size(R_eval.f_R));

    disp('laplace_solver_coarse: evaluating interior points');
    init_idxs = R_eval.R_idxs(well_interior_msk);
    u_num_mat(init_idxs) = IE_curve_seq.u_num_batch(R_eval.R_X(init_idxs), R_eval.R_Y(init_idxs), curve_param_rho1, gr_phi_rho1);

    to_update   = well_interior_msk;
    rho_fine_IE = 2;

    while (sum(to_update, 'all') > 0 || rho_fine_IE == 2) && rho_fine_IE <= MAX_RHO_IE
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        update_idxs = R_eval.R_idxs(to_update);
        u_num_mat_fine(update_idxs) = IE_curve_seq.u_num_batch(R_eval.R_X(update_idxs), R_eval.R_Y(update_idxs), curve_param_fine, gr_phi_fine);

        resid = abs(u_num_mat_fine - u_num_mat);
        to_update   = to_update & resid > int_eps;
        u_num_mat(to_update) = u_num_mat_fine(to_update);
        rho_fine_IE = next_rho_IE(rho_fine_IE);
    end

    if sum(to_update, 'all') > 0
        warning('laplace_solver_coarse:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d well-interior point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update, 'all'), int_eps, max(resid(to_update)));
    end

    % ------------------------------------------------------------------ %
    % Evaluate interpolation-node values (near-boundary patch rows)       %
    % ------------------------------------------------------------------ %
    disp('laplace_solver_coarse: evaluating smooth patch points');

    rho_coarse_IE      = rho_fine_IE;
    gr_phi_coarse      = gr_phi_fine;
    curve_param_coarse = curve_param_fine;

    rho_fine_IE = next_rho_IE(rho_coarse_IE);
    [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

    n_interpol_total = length(interpol_target_idx);

    sub_X = interpol_nodes_x(:, 2:M);
    sub_Y = interpol_nodes_y(:, 2:M);
    interpol_pt_x = sub_X(:);
    interpol_pt_y = sub_Y(:);

    u_interpol_coarse = IE_curve_seq.u_num_batch(interpol_pt_x, interpol_pt_y, curve_param_coarse, gr_phi_coarse);
    u_interpol_fine   = IE_curve_seq.u_num_batch(interpol_pt_x, interpol_pt_y, curve_param_fine,   gr_phi_fine);

    to_update = abs(u_interpol_coarse - u_interpol_fine) > int_eps;

    while any(to_update) && rho_fine_IE <= MAX_RHO_IE
        u_interpol_coarse(to_update) = u_interpol_fine(to_update);

        rho_coarse_IE      = rho_fine_IE;
        curve_param_coarse = curve_param_fine;
        gr_phi_coarse      = gr_phi_fine;

        rho_fine_IE = next_rho_IE(rho_coarse_IE);
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        u_interpol_fine(to_update) = IE_curve_seq.u_num_batch( ...
            interpol_pt_x(to_update), interpol_pt_y(to_update), curve_param_fine, gr_phi_fine);
        resid = abs(u_interpol_coarse(to_update) - u_interpol_fine(to_update));
        to_update(to_update) = resid > int_eps;
    end

    if any(to_update)
        warning('laplace_solver_coarse:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d interpolation-node point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update), int_eps, max(resid(resid > int_eps)));
    end

    interpol_u = zeros(n_interpol_total, M);
    interpol_u(:, 2:M) = reshape(u_interpol_coarse, n_interpol_total, M-1);

    % Boundary row: enforce exact Dirichlet data
    interpol_u(:, 1) = u_G(interpol_nodes_x(:, 1), interpol_nodes_y(:, 1));

    % Evaluate each near-boundary R_eval point by 1D barycentric interpolation in eta
    for idx = 1:length(interpol_target_idx)
        u_num_mat(interpol_target_idx(idx)) = barylag( ...
            [(0:(M-1))' * R.h, interpol_u(idx, :)'], interpol_target_eta(idx));
    end

    % ------------------------------------------------------------------ %
    % Evaluate corner-region points with adaptive IE refinement           %
    % ------------------------------------------------------------------ %
    disp('laplace_solver_coarse: evaluating corner points');
    c_pts_msk = R_eval.in_interior & ~well_interior_msk;
    for i = 1:length(s_patch_msks)
        c_pts_msk = c_pts_msk & ~s_patch_msks{i};
    end

    R_idxs_c_left = R_eval.R_idxs(c_pts_msk);

    % Batched adaptive refinement -- see laplace_solver.m's corner-region
    % loop for why the original's distance-to-boundary traversal order is
    % no longer needed once every still-active point is refined together.
    u_corner_coarse = IE_curve_seq.u_num_batch(R_eval.R_X(R_idxs_c_left), R_eval.R_Y(R_idxs_c_left), curve_param_coarse, gr_phi_coarse);
    u_corner_fine   = IE_curve_seq.u_num_batch(R_eval.R_X(R_idxs_c_left), R_eval.R_Y(R_idxs_c_left), curve_param_fine,   gr_phi_fine);

    to_update = abs(u_corner_coarse - u_corner_fine) > int_eps;

    while any(to_update) && rho_fine_IE <= MAX_RHO_IE
        u_corner_coarse(to_update) = u_corner_fine(to_update);

        rho_coarse_IE      = rho_fine_IE;
        curve_param_coarse = curve_param_fine;
        gr_phi_coarse      = gr_phi_fine;

        rho_fine_IE = next_rho_IE(rho_coarse_IE);
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        idxs_active = R_idxs_c_left(to_update);
        u_corner_fine(to_update) = IE_curve_seq.u_num_batch(R_eval.R_X(idxs_active), R_eval.R_Y(idxs_active), curve_param_fine, gr_phi_fine);
        resid = abs(u_corner_coarse(to_update) - u_corner_fine(to_update));
        to_update(to_update) = resid > int_eps;
    end

    if any(to_update)
        warning('laplace_solver_coarse:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d corner-region point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update), int_eps, max(resid(resid > int_eps)));
    end

    u_num_mat(R_idxs_c_left) = u_corner_coarse;

    u_num_mat(~R_eval.in_interior) = nan;
end

% =========================================================================

function [gr_phi_rho, curve_param_rho] = refine_gr_phi(curve_param_rho1, rho_IE, gr_phi_fft_rho1)
% REFINE_GR_PHI  Zero-pads FFT coefficients to upsample the density by rho_IE.
    curve_param_rho = curve_param_obj(curve_param_rho1.curve_n * rho_IE);
    n_base          = curve_param_rho1.n_total;
    n_fine          = n_base * rho_IE;

    padded_fft_coeffs = [ ...
        zeros(ceil((n_fine - n_base) / 2), 1); ...
        gr_phi_fft_rho1; ...
        zeros(floor((n_fine - n_base) / 2), 1)];
    gr_phi_rho = rho_IE * n_base * real(ifft(ifftshift(padded_fft_coeffs)));
end

% =========================================================================

function rho_next = next_rho_IE(rho)
% NEXT_RHO_IE  Advances the IE refinement level with a step size that scales
% with rho's current order of magnitude -- see laplace_solver.m's copy of
% this function for the rationale.
    step = 1;
    while rho >= step * 10
        step = step * 10;
    end
    rho_next = rho + step;
end

% =========================================================================

function [well_interior_msk, s_patch_msks] = gen_R_msks(R, n_r, s_patches, c_0_patches, c_1_patches)
% GEN_R_MSKS  Partitions R.in_interior into well-interior and patch-region masks.
    well_interior_msk = R.in_interior;
    s_patch_msks = cell(size(s_patches));

    for i = 1:length(s_patch_msks)
        s_patch = s_patches{i};
        [bound_X, bound_Y] = s_patch.boundary_mesh_xy(n_r, false);
        in_patch = inpolygon_mesh(R.R_X, R.R_Y, bound_X, bound_Y) & R.in_interior;
        s_patch_msks{i} = in_patch;
        well_interior_msk = well_interior_msk & ~in_patch;

        c_0_patch = c_0_patches{i};
        c_1_patch = c_1_patches{i};

        if isobject(c_0_patch)
            [bound_X, bound_Y] = c_0_patch.boundary_mesh_xy(n_r, false);
            in_patch = inpolygon_mesh(R.R_X, R.R_Y, bound_X, bound_Y) & R.in_interior;
            well_interior_msk = well_interior_msk & ~in_patch;
        end

        if isobject(c_1_patch)
            [bound_X, bound_Y] = c_1_patch.boundary_mesh_xy(n_r, false);
            in_patch = inpolygon_mesh(R.R_X, R.R_Y, bound_X, bound_Y) & R.in_interior;
            well_interior_msk = well_interior_msk & ~in_patch;
        end
    end

    for i = 1:length(s_patch_msks)
        M      = s_patches{i}.n_eta;
        corner = s_patches{i}.M_p(1, 0)';
        dist2  = (corner(1) - R.R_X).^2 + (corner(2) - R.R_Y).^2;
        r2     = (M * R.h)^2;

        well_interior_msk  = well_interior_msk  & (dist2 > r2);
        s_patch_msks{i}    = s_patch_msks{i}    & (dist2 > r2);

        next_i = mod(i, length(s_patch_msks)) + 1;
        s_patch_msks{next_i} = s_patch_msks{next_i} & (dist2 > r2);
    end
end
