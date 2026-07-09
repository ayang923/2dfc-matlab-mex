function [u_num_mat] = laplace_solver(R, curve_seq, u_G, cfac, p, M, int_eps, eps_xi_eta, eps_xy, rho, corner_r)
% LAPLACE_SOLVER  Solves Delta(u) = 0 on a 2D domain with Dirichlet boundary data.
%
% Uses a 2nd-kind Fredholm boundary integral equation (BIE) formulation.
% The double-layer density phi is computed by solving a linear system, then the
% solution is evaluated via adaptive quadrature refinement:
%
%   - Points well inside the domain use direct IE evaluation, refined
%     until two successive resolutions agree to within int_eps.
%   - Points near smooth boundary patches use patch-local polynomial
%     interpolation from precomputed values on the patch grid; those patch
%     grid values are themselves filled via the same adaptive IE refinement.
%   - Remaining corner-region points are handled by the same adaptive
%     IE refinement.
%
% All three evaluation sites (well-interior, patch-grid-fill, corner-region)
% share the same batching pattern: every point in the active set is evaluated
% at a shared (curve_param, gr_phi) resolution via IE_curve_seq_obj.u_num_batch
% (mex-accelerated), points that have converged (two successive resolutions
% agree to within int_eps) are frozen at their converged value and dropped
% from the active set, and the remaining active points are re-evaluated
% together at the next resolution. This is safe because the "coarse"/"fine"
% resolution state is (and always was, even in the original per-point
% while-loops) a single pair shared across all points in a given site, only
% ever ratcheting monotonically upward -- batching just evaluates every
% still-active point against that shared state at once instead of walking
% points one at a time and having later points inherit whatever resolution
% an earlier point's convergence check happened to leave the shared state at.
% The final accepted value for a given point can therefore differ from the
% original's sequential-order-dependent result, since a point here is always
% checked against its own natural first-satisfying resolution (starting from
% the same base level as every other point in the site), rather than
% "free-riding" on a coarser-or-finer level that some other point's demand
% happened to leave the shared state at when the original reached it. Both
% values satisfy the same int_eps stopping criterion, so this difference is
% bounded by a small multiple of int_eps, not multiple orders of magnitude
% smaller -- validated empirically at ~3-7x int_eps for the patch-grid-fill
% and corner-region sites (where convergence near the geometry's corners is
% slower and doesn't "overshoot" the tolerance the way well-interior points
% do), and at <<int_eps for the well-interior site itself. In practice this
% is far below the discretization error of the overall Poisson solve at any
% resolution where the IE tolerance is tight enough to matter (see the mex
% port's README).
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

    % Hard cap on the IE refinement level (shared across all three
    % evaluation sites below, since rho_fine_IE only ever increases and
    % carries forward from one site to the next). Guards against an
    % infinite loop if int_eps is set tighter than the quadrature can
    % actually resolve for some point (e.g. degenerate geometry); points
    % still unconverged when the cap is hit keep their best (coarse) value
    % and a warning is raised rather than hanging. Points evaluated very
    % close to a sharp/concave curve junction can genuinely need well over
    % 100 refinement levels to converge to a tight int_eps (validated on a
    % reduced-resolution 4-curve concave-corner test case, where raising
    % this cap resolved a hit-the-cap warning and matched the uncapped
    % original to 3.7e-5); lower this if runtime matters more than
    % accuracy near such corners, or raise it if the reverse. Every
    % rho_fine_IE advance below uses next_rho_IE (step size scales with the
    % current level's order of magnitude: +1 below 10, +10 below 100, +100
    % below 1000, etc.), so reaching this cap from rho=2 takes on the order
    % of tens of refinement rounds, not thousands.
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

    while sum(to_update, 'all') > 0 && rho_fine_IE <= MAX_RHO_IE
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        update_idxs = R.R_idxs(to_update);
        u_num_mat_fine(update_idxs) = IE_curve_seq.u_num_batch(R.R_X(update_idxs), R.R_Y(update_idxs), curve_param_fine, gr_phi_fine);

        resid = abs(u_num_mat_fine - u_num_mat);
        to_update  = to_update & resid > int_eps;
        u_num_mat(to_update) = u_num_mat_fine(to_update);
        rho_fine_IE = next_rho_IE(rho_fine_IE);
    end

    if sum(to_update, 'all') > 0
        warning('laplace_solver:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d well-interior point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update, 'all'), int_eps, max(resid(to_update)));
    end

    % ------------------------------------------------------------------ %
    % Evaluate smooth-patch points via patch-local interpolation          %
    % ------------------------------------------------------------------ %
    disp('laplace_solver: evaluating smooth patch points');

    rho_coarse_IE    = rho_fine_IE;
    gr_phi_coarse    = gr_phi_fine;
    curve_param_coarse = curve_param_fine;

    rho_fine_IE = next_rho_IE(rho_coarse_IE);
    [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

    % Fill patch grid values (excludes the boundary row eta=0), batched across
    % every curve's interior rows (eta=2..M) at once -- see the header comment.
    patch_pt_x = [];
    patch_pt_y = [];
    patch_pt_counts = zeros(curve_seq.n_curves, 1);
    for i = 1:curve_seq.n_curves
        s_patch = s_patches{i};
        [patch_X_s, patch_Y_s] = s_patch.xy_mesh;
        sub_X = patch_X_s(2:M, 1:s_patch.n_xi);
        sub_Y = patch_Y_s(2:M, 1:s_patch.n_xi);
        patch_pt_x = [patch_pt_x; sub_X(:)];
        patch_pt_y = [patch_pt_y; sub_Y(:)];
        patch_pt_counts(i) = numel(sub_X);
    end

    u_patch_coarse = IE_curve_seq.u_num_batch(patch_pt_x, patch_pt_y, curve_param_coarse, gr_phi_coarse);
    u_patch_fine   = IE_curve_seq.u_num_batch(patch_pt_x, patch_pt_y, curve_param_fine,   gr_phi_fine);

    to_update = abs(u_patch_coarse - u_patch_fine) > int_eps;

    while any(to_update) && rho_fine_IE <= MAX_RHO_IE
        u_patch_coarse(to_update) = u_patch_fine(to_update);

        rho_coarse_IE      = rho_fine_IE;
        curve_param_coarse = curve_param_fine;
        gr_phi_coarse      = gr_phi_fine;

        rho_fine_IE = next_rho_IE(rho_coarse_IE);
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        u_patch_fine(to_update) = IE_curve_seq.u_num_batch( ...
            patch_pt_x(to_update), patch_pt_y(to_update), curve_param_fine, gr_phi_fine);
        resid = abs(u_patch_coarse(to_update) - u_patch_fine(to_update));
        to_update(to_update) = resid > int_eps;
    end

    if any(to_update)
        warning('laplace_solver:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d patch-grid point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update), int_eps, max(resid(resid > int_eps)));
    end

    offset = 0;
    for i = 1:curve_seq.n_curves
        s_patch = s_patches{i};
        n = patch_pt_counts(i);
        s_patch.f_XY(2:M, 1:s_patch.n_xi) = reshape(u_patch_coarse(offset+1:offset+n), M-1, s_patch.n_xi);
        offset = offset + n;
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

    % Batched adaptive refinement, same pattern as the well-interior and
    % patch-grid-fill passes above. The original's distance-to-boundary
    % traversal order was a performance heuristic for the sequential
    % per-point while-loops (process likely-easy points first to avoid
    % forcing them through expensive re-refinement cascades triggered by a
    % harder point processed earlier); batching already evaluates only the
    % still-active subset each round regardless of order, so it no longer
    % serves a purpose and is dropped.
    u_corner_coarse = IE_curve_seq.u_num_batch(R.R_X(R_idxs_c_left), R.R_Y(R_idxs_c_left), curve_param_coarse, gr_phi_coarse);
    u_corner_fine   = IE_curve_seq.u_num_batch(R.R_X(R_idxs_c_left), R.R_Y(R_idxs_c_left), curve_param_fine,   gr_phi_fine);

    to_update = abs(u_corner_coarse - u_corner_fine) > int_eps;

    while any(to_update) && rho_fine_IE <= MAX_RHO_IE
        u_corner_coarse(to_update) = u_corner_fine(to_update);

        rho_coarse_IE      = rho_fine_IE;
        curve_param_coarse = curve_param_fine;
        gr_phi_coarse      = gr_phi_fine;

        rho_fine_IE = next_rho_IE(rho_coarse_IE);
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        idxs_active = R_idxs_c_left(to_update);
        u_corner_fine(to_update) = IE_curve_seq.u_num_batch(R.R_X(idxs_active), R.R_Y(idxs_active), curve_param_fine, gr_phi_fine);
        resid = abs(u_corner_coarse(to_update) - u_corner_fine(to_update));
        to_update(to_update) = resid > int_eps;
    end

    if any(to_update)
        warning('laplace_solver:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d corner-region point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update), int_eps, max(resid(resid > int_eps)));
    end

    u_num_mat(R_idxs_c_left) = u_corner_coarse;

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

function rho_next = next_rho_IE(rho)
% NEXT_RHO_IE  Advances the IE refinement level with a step size that scales
% with rho's current order of magnitude: +1 while rho < 10, +10 while
% rho < 100, +100 while rho < 1000, etc. Reaching MAX_RHO_IE=10000 from
% rho=2 this way takes on the order of tens of refinement rounds rather
% than thousands of unit steps, while still refining cheaply (step 1) in
% the common case where a point converges at a low level.
    step = 1;
    while rho >= step * 10
        step = step * 10;
    end
    rho_next = rho + step;
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
