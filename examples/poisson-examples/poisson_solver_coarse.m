function [u_num_mat, R, R_eval] = poisson_solver_coarse(curve_seq, f, u_boundary, h, G_cf, p, int_eps, eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, h_eval, n_x_padded, n_y_padded, perturb, timing)
% POISSON_SOLVER_COARSE  Solves Delta(u) = f using a coarser evaluation grid.
%
% Identical algorithm to poisson_solver, but the solution is evaluated on a
% separate coarser Cartesian mesh R_eval (step size h_eval > h) rather than
% on the fine FC grid R. This saves memory and time when visualizing or
% comparing against an exact solution at moderate resolution.
%
% Inputs:
%   curve_seq  - Curve_seq_obj describing the domain boundary
%   f          - Function handle for the Poisson right-hand side f(x,y)
%   u_boundary - Function handle for the Dirichlet boundary data u(x,y)
%   h          - Fine Cartesian mesh step size for the FC computation
%   G_cf       - Quadrature refinement factor for the IE boundary discretization
%   p          - Polynomial degree for the IE graded-mesh quadrature
%   int_eps    - Convergence tolerance for adaptive IE integration
%   eps_xi_eta - Newton inversion tolerance in parameter (xi,eta) space
%   eps_xy     - Newton inversion tolerance in physical (x,y) space
%   d          - Number of Gram matching points for 1D FC
%   C          - Number of continuation points for FC
%   n_r        - Refinement factor for the FC continuation grid
%   A          - Precomputed FC continuation matrix (n_r*C x d)
%   Q          - Precomputed FC Gram polynomial matrix (d x d)
%   M          - Polynomial interpolation degree
%   h_eval     - Either:
%                (1) numeric step size of the coarse evaluation grid, or
%                (2) prebuilt R_cartesian_mesh_obj used as R_eval%   n_x_padded - (optional) Override for R grid size in x; pass [] to skip
%   n_y_padded - (optional) Override for R grid size in y; pass [] to skip
%   perturb    - (optional) If true, expand bounding box by rand(1)*h per side;
%                if false or omitted, expand by the fixed h
%   timing     - (optional) If true, print elapsed time for each major step
%                (2DFC construction -- excluding FC2D's own error-check
%                evaluation -- obtaining and evaluating the particular
%                solution, and -- inside laplace_solver_coarse -- the
%                boundary solve and the interior/smooth-patch/corner-point
%                evaluation passes). Default false.
%
% Outputs:
%   u_num_mat - (n_y_eval x n_x_eval) solution values on R_eval; NaN outside domain
%   R         - Fine R_cartesian_mesh_obj (holds the particular solution in R.f_R)
%   R_eval    - Coarse R_cartesian_mesh_obj on which the solution is returned

    if nargin < 20 || isempty(timing)
        timing = false;
    end

    % Build the FC grid, forwarding any optional arguments to FC2D. n_x_padded/
    % n_y_padded/perturb are padded out to their defaults (rather than left
    % unset) so that `timing`, appended last, always lands in FC2D's 18th
    % (timing) position regardless of which optional args the caller supplied.
    if nargin >= 18
        fc2d_args = {n_x_padded, n_y_padded};
    else
        fc2d_args = {[], []};
    end
    if nargin >= 19
        fc2d_args{end+1} = perturb;
    else
        fc2d_args{end+1} = false;
    end
    fc2d_args{end+1} = timing;

    [R, ~, ~, ~] = FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, C, A, Q, M, fc2d_args{:});

    % Build or accept the coarse evaluation grid
    if isa(h_eval, 'R_cartesian_mesh_obj')
        R_eval_old = h_eval;

        % Infer the evaluation spacing from the old R_eval grid
        h_eval_new = R_eval_old.h;

        % Extract old grid coordinates
        x_old = R_eval_old.R_X(1, :);
        y_old = R_eval_old.R_Y(:, 1);

        % Keep only old R_eval grid points that lie inside the fine R box
        x_new = x_old(x_old >= R.x_start & x_old <= R.x_end);
        y_new = y_old(y_old >= R.y_start & y_old <= R.y_end);

        if isempty(x_new) || isempty(y_new)
            error('The supplied R_eval has no grid points contained in R.');
        end

        % Create a new R_eval contained in R, while preserving old R_eval grid alignment
        R_eval = R_cartesian_mesh_obj( ...
            x_new(1), x_new(end)-h_eval_new, ...
            y_new(1), y_new(end)-h_eval_new, ...
            h_eval_new, ...
            R.boundary_X, R.boundary_Y);
    else
        R_eval = R_cartesian_mesh_obj(R.x_start, R.x_end - h_eval, ...
            R.y_start, R.y_end - h_eval, ...
            h_eval, R.boundary_X, R.boundary_Y);
    end

    % Particular solution on the fine grid; interpolate onto the coarse grid.
    % This full-grid interpolation (potentially millions of R_eval points) is
    % the single biggest bottleneck in the whole poisson solver -- see
    % R_cartesian_mesh_obj.locally_compute_vec.
    if timing; t_step = tic; end
    R.f_R = R.inv_lap();
    R_eval.f_R = R.locally_compute_vec(R_eval.R_X, R_eval.R_Y, M);
    if timing; fprintf('poisson_solver_coarse: obtain and evaluate particular solution: %.3fs\n', toc(t_step)); end

    % Homogeneous BVP with modified boundary data u - u_p|boundary
    u_m_up_boundary = @(x, y) u_boundary(x, y) - R.locally_compute_vec(x, y, M);
    uh = laplace_solver_coarse(R, R_eval, curve_seq, u_m_up_boundary, G_cf, p, M, int_eps, eps_xi_eta, eps_xy, n_r, timing);

    u_num_mat = uh + R_eval.f_R;
end
