function [u_num_mat] = laplace_solver(R, curve_seq, u_G, cfac, p, M, int_eps, eps_xi_eta, eps_xy, rho, corner_r)
% LAPLACE_SOLVER  Solves Delta(u) = 0 on a 2D domain with Dirichlet boundary data.
%
% Uses a 2nd-kind Fredholm boundary integral equation (BIE) formulation.
% The double-layer density phi is computed by solving a linear system, then the
% solution is evaluated via adaptive quadrature refinement:
%
%   - Points well inside the domain use direct IE evaluation, refined
%     until two successive resolutions agree to within int_eps. This pass
%     evaluates IE_curve_seq_obj.u_num_batch once per refinement level
%     (mex-accelerated double-layer-potential evaluation over every
%     not-yet-converged point at once) rather than one point at a time --
%     safe to batch since every point in a given pass shares the same
%     (curve_param, gr_phi) resolution with no per-point branching.
%   - Points near smooth boundary patches use patch-local polynomial
%     interpolation from precomputed values on the patch grid.
%   - Remaining corner-region points are handled by the same adaptive
%     IE refinement, processed in order of decreasing distance to the boundary.
%     This loop and the patch-grid-fill loop below keep their original
%     per-point independent while-loop structure (each point refines until
%     it personally converges, not synchronized with other points), so they
%     are not batched the way the well-interior pass is -- they still
%     benefit from IE_curve_seq_obj.u_num's own mex dispatch, just without
%     removing the per-call redundant node-array rebuild. See the mex
%     port's README section for why this is scoped as a follow-up rather
%     than done here.
%
% Inputs:
%   R          - R_cartesian_mesh_obj (grid, interior mask, etc.)
%   curve_seq  - Curve_seq_obj describing the domain boundary
%   u_G        - Function handle for the Dirichlet boundary data u_G(x,y)
%   cfac       - Quadrature refinement factor (each curve gets ceil((n-1)*cfac) points)
%   p          - Polynomial degree for the IE graded-mesh quadrature
%   M          - Polynomial interpolation degree
%   int_eps    - Convergence tolerance for adaptive IE integration
%   eps_xi_eta - Newton inversion tolerance in (xi,eta) space
%   eps_xy     - Newton inversion tolerance in (x,y) space
%   rho        - Patch-exclusion radius multiplier (in units of h)
%   corner_r   - All points within corner_r of a corner are evaluated with
%   direct refinement rather than polynomial interpolation. If not given,
%   set to M*h
%
% Output:
%   u_num_mat - (n_y x n_x) matrix of solution values; NaN outside the domain

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

    if ~exist('corner_r', 'var') || isempty(corner_r)
        corner_r = M*R.h;
    end

    [well_interior_msk, s_patch_msks] = gen_R_msks(R, rho, s_patches, c_0_patches, c_1_patches, corner_r);

    % ------------------------------------------------------------------ %
    % Evaluate at well-interior points with adaptive IE refinement        %
    % ------------------------------------------------------------------ %
    u_num_mat      = zeros(size(R.f_R));
    u_num_mat_fine = zeros(size(R.f_R));

    disp('laplace_solver: evaluating interior points');
    init_idxs = R.R_idxs(well_interior_msk);
    u_num_mat(init_idxs) = IE_curve_seq.u_num_batch(R.R_X(init_idxs), R.R_Y(init_idxs), curve_param_rho1, gr_phi_rho1);

    to_update  = well_interior_msk;
    rho_fine_IE = 2;

    while sum(to_update, 'all') > 0
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        update_idxs = R.R_idxs(to_update);
        u_num_mat_fine(update_idxs) = IE_curve_seq.u_num_batch(R.R_X(update_idxs), R.R_Y(update_idxs), curve_param_fine, gr_phi_fine);

        to_update  = to_update & abs(u_num_mat_fine - u_num_mat) > int_eps;
        u_num_mat(to_update) = u_num_mat_fine(to_update);
        rho_fine_IE = rho_fine_IE + 1;
    end

    % ------------------------------------------------------------------ %
    % Evaluate smooth-patch points via patch-local interpolation          %
    % ------------------------------------------------------------------ %
    disp('laplace_solver: evaluating smooth patch points');

    rho_coarse_IE    = rho_fine_IE;
    gr_phi_coarse    = gr_phi_fine;
    curve_param_coarse = curve_param_fine;

    rho_fine_IE = rho_coarse_IE + 1;
    [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

    % Fill patch grid values (excludes the boundary row eta=0)
    for eta_idx = M:-1:2
        for i = 1:curve_seq.n_curves
            s_patch = s_patches{i};
            [patch_X_s, patch_Y_s] = s_patch.xy_mesh;
            for xi_idx = 1:s_patch.n_xi
                x = patch_X_s(eta_idx, xi_idx);
                y = patch_Y_s(eta_idx, xi_idx);
                u_num_coarse_val = IE_curve_seq.u_num(x, y, curve_param_coarse, gr_phi_coarse);
                u_num_fine_val   = IE_curve_seq.u_num(x, y, curve_param_fine,   gr_phi_fine);

                while abs(u_num_coarse_val - u_num_fine_val) > int_eps
                    rho_coarse_IE      = rho_fine_IE;
                    curve_param_coarse = curve_param_fine;
                    gr_phi_coarse      = gr_phi_fine;

                    rho_fine_IE = rho_coarse_IE + 1;
                    [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

                    u_num_coarse_val = u_num_fine_val;
                    u_num_fine_val   = IE_curve_seq.u_num(x, y, curve_param_fine, gr_phi_fine);
                end
                s_patch.f_XY(eta_idx, xi_idx) = u_num_coarse_val;
            end
        end
    end

    % Boundary row: enforce exact Dirichlet data
    for i = 1:curve_seq.n_curves
        s_patch = s_patches{i};
        [patch_X, patch_Y] = s_patch.xy_mesh;
        s_patch.f_XY(1, :) = u_G(patch_X(1, :), patch_Y(1, :));
    end

    % Interpolate patch values onto R grid points
    for i = 1:curve_seq.n_curves
        s_patch   = s_patches{i};
        in_patch  = s_patch_msks{i};
        R_s_idxs  = R.R_idxs(in_patch);

        [P_xi_s, P_eta_s] = R_xi_eta_inversion(R, s_patch, in_patch);

        for idx = 1:length(R_s_idxs)
            if s_patch.in_patch(P_xi_s(idx), P_eta_s(idx))
                u_num_mat(R_s_idxs(idx)) = s_patch.locally_compute(P_xi_s(idx), P_eta_s(idx), M);
            else
                s_patch_msks{i}(R_s_idxs(idx)) = false;
            end
        end
    end

    % ------------------------------------------------------------------ %
    % Evaluate corner-region points with adaptive IE refinement           %
    % ------------------------------------------------------------------ %
    c_pts_msk = R.in_interior & ~well_interior_msk;
    for i = 1:length(s_patch_msks)
        c_pts_msk = c_pts_msk & ~s_patch_msks{i};
    end

    R_idxs_c_left = R.R_idxs(c_pts_msk);

    % Compute distance to boundary for each remaining point
    dist_to_boundary = zeros(size(R_idxs_c_left));
    for i = 1:length(R_idxs_c_left)
        dist_to_boundary(i) = sqrt(min( ...
            (R.boundary_X - R.R_X(R_idxs_c_left(i))).^2 + ...
            (R.boundary_Y - R.R_Y(R_idxs_c_left(i))).^2));
    end

    % Process in order of decreasing distance (avoids re-refinement cascades)
    [~, trav_order]  = sort(dist_to_boundary, 'descend');
    R_idxs_c_left    = R_idxs_c_left(trav_order);

    for i = 1:size(R_idxs_c_left, 1)
        x = R.R_X(R_idxs_c_left(i));
        y = R.R_Y(R_idxs_c_left(i));

        u_num_coarse_val = IE_curve_seq.u_num(x, y, curve_param_coarse, gr_phi_coarse);
        u_num_fine_val   = IE_curve_seq.u_num(x, y, curve_param_fine,   gr_phi_fine);

        while abs(u_num_coarse_val - u_num_fine_val) > int_eps
            rho_coarse_IE      = rho_fine_IE;
            curve_param_coarse = curve_param_fine;
            gr_phi_coarse      = gr_phi_fine;

            rho_fine_IE = rho_coarse_IE + 1;
            [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

            u_num_coarse_val = u_num_fine_val;
            u_num_fine_val   = IE_curve_seq.u_num(x, y, curve_param_fine, gr_phi_fine);
        end
        u_num_mat(R_idxs_c_left(i)) = u_num_coarse_val;
    end

    u_num_mat(~R.in_interior) = nan;
end

% =========================================================================

function [gr_phi_rho, curve_param_rho] = refine_gr_phi(curve_param_rho1, rho_IE, gr_phi_fft_rho1)
% REFINE_GR_PHI  Zero-pads FFT coefficients to upsample the density by rho_IE.
%
% Inputs:
%   curve_param_rho1 - Coarsest curve_param_obj (base resolution)
%   rho_IE           - Integer refinement factor
%   gr_phi_fft_rho1  - Fourier coefficients of the density at base resolution
%
% Outputs:
%   gr_phi_rho    - Density vector at the refined resolution
%   curve_param_rho - curve_param_obj at the refined resolution

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

function [well_interior_msk, s_patch_msks] = gen_R_msks(R, n_r, s_patches, c_0_patches, c_1_patches, corner_r)
% GEN_R_MSKS  Partitions R.in_interior into well-interior and patch-region masks.
%
% A grid point is removed from well_interior_msk if it lies inside the
% bounding polygon of any smooth-patch (s_patch) or corner-patch.
% Additionally, points within corner_r of any curve corner are excluded from
% both well_interior_msk and s_patch_msks (handled as corner points instead).
%
% Inputs:
%   R            - R_cartesian_mesh_obj
%   n_r          - Patch polygon exclusion parameter (passed to boundary_mesh_xy)
%   s_patches    - Cell array of S_patch_obj
%   c_0_patches  - Cell array of C1_patch_obj (concave corners; may contain [])
%   c_1_patches  - Cell array of C2_patch_obj (convex  corners; may contain [])
%
% Outputs:
%   well_interior_msk - Logical mask over R grid; true = well inside domain
%   s_patch_msks      - Cell array of per-patch logical masks over R grid

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

    % Exclude a ball of radius M*h around each curve junction (corner) from
    % both the well-interior mask and the adjacent smooth-patch masks.
    for i = 1:length(s_patch_msks)
        M      = s_patches{i}.n_eta;
        corner = s_patches{i}.M_p(1, 0)';
        dist2  = (corner(1) - R.R_X).^2 + (corner(2) - R.R_Y).^2;
        r2     = (corner_r)^2;

        well_interior_msk = well_interior_msk & (dist2 > r2);
        s_patch_msks{i}   = s_patch_msks{i}   & (dist2 > r2);

        next_i = mod(i, length(s_patch_msks)) + 1;
        s_patch_msks{next_i} = s_patch_msks{next_i} & (dist2 > r2);
    end
end
