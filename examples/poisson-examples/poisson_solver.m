function [u_num_mat, R] = poisson_solver(curve_seq, f, u_boundary, h, G_cf, p, int_eps, eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x_padded, n_y_padded, perturb, corner_r, timing)
% POISSON_SOLVER  Solves Delta(u) = f on a 2D domain with Dirichlet boundary conditions.
%
% Strategy:
%   1. Compute a particular solution u_p satisfying Delta(u_p) = f everywhere
%      on the bounding Cartesian rectangle via 2DFC + spectral inverse Laplacian.
%   2. Solve the homogeneous Laplace BVP:
%        Delta(u_h) = 0  in Omega
%        u_h = u_boundary - u_p  on d(Omega)
%      using a 2nd-kind boundary integral equation (BIE) method.
%   3. Return u = u_p + u_h.
%
% Inputs:
%   curve_seq  - Curve_seq_obj describing the domain boundary
%   f          - Function handle for the Poisson right-hand side f(x,y)
%   u_boundary - Function handle for the Dirichlet boundary data u(x,y)
%   h          - Cartesian mesh step size for the FC grid
%   G_cf       - Quadrature refinement factor for the IE boundary discretization
%                (cfac: each curve gets ceil((n-1)*G_cf) quadrature points)
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
%   n_x_padded - (optional) Override for R grid size in x; pass [] to skip
%   n_y_padded - (optional) Override for R grid size in y; pass [] to skip
%   perturb    - (optional) If true, expand bounding box by rand(1)*h per side;
%                if false or omitted, expand by the fixed h
%   corner_r   - (optional) See laplace_solver.m
%   timing     - (optional) If true, print elapsed time for each major step
%                (2DFC construction -- excluding FC2D's own error-check
%                evaluation -- obtaining the particular solution, and --
%                inside laplace_solver -- the boundary solve and the
%                interior/smooth-patch/corner-point evaluation passes).
%                Default false.
%
% Outputs:
%   u_num_mat - (n_y x n_x) matrix of solution values on R's Cartesian grid;
%               NaN outside the domain
%   R         - R_cartesian_mesh_obj used for the FC computation; R.f_R holds
%               the particular solution u_p after this call returns

    if nargin >= 19
    else
        corner_r = M*h;
    end
    if nargin < 20 || isempty(timing)
        timing = false;
    end

    % Build the FC grid, forwarding any optional arguments to FC2D. n_x_padded/
    % n_y_padded/perturb are padded out to their defaults (rather than left
    % unset) so that `timing`, appended last, always lands in FC2D's 18th
    % (timing) position regardless of which optional args the caller supplied.
    if nargin >= 17
        fc2d_args = {n_x_padded, n_y_padded};
    else
        fc2d_args = {[], []};
    end
    if nargin >= 18
        fc2d_args{end+1} = perturb;
    else
        fc2d_args{end+1} = false;
    end
    fc2d_args{end+1} = timing;

    [R, ~, ~, ~] = FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, C, A, Q, M, fc2d_args{:});

    % Particular solution: u_p satisfies Delta(u_p) = f spectrally
    if timing; t_step = tic; end
    R.f_R = R.inv_lap();
    if timing; fprintf('poisson_solver: obtain and evaluate particular solution: %.3fs\n', toc(t_step)); end

    % Homogeneous BVP: u_h satisfies Delta(u_h) = 0, u_h|boundary = u - u_p|boundary
    u_m_up_boundary = @(x, y) u_boundary(x, y) - R.locally_compute_vec(x, y, M);
    uh = laplace_solver(R, curve_seq, u_m_up_boundary, G_cf, p, M, int_eps, eps_xi_eta, eps_xy, n_r, corner_r, timing);

    u_num_mat = uh + R.f_R;
end
