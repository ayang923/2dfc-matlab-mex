function [u_num_mat, R] = poisson_solver(curve_seq, f, u_boundary, h, G_cf, p, int_eps, eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x_padded, n_y_padded, perturb, corner_r)
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
%
% Outputs:
%   u_num_mat - (n_y x n_x) matrix of solution values on R's Cartesian grid;
%               NaN outside the domain
%   R         - R_cartesian_mesh_obj used for the FC computation; R.f_R holds
%               the particular solution u_p after this call returns

    % Build the FC grid, forwarding any optional arguments to FC2D
    fc2d_args = {};
    if nargin >= 17
        fc2d_args{end+1} = n_x_padded;
        fc2d_args{end+1} = n_y_padded;
    end
    if nargin >= 18;  fc2d_args{end+1} = perturb;  end
    if nargin >= 19
        corner_r = corner_r;
    else
        corner_r = M*h;
    end

    [R, ~, ~, ~] = FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, C, A, Q, M, fc2d_args{:});

    % Particular solution: u_p satisfies Delta(u_p) = f spectrally
    R.f_R = R.inv_lap();

    % Homogeneous BVP: u_h satisfies Delta(u_h) = 0, u_h|boundary = u - u_p|boundary
    u_m_up_boundary = @(x, y) u_boundary(x, y) - R.locally_compute_vec(x, y, M);
    uh = laplace_solver(R, curve_seq, u_m_up_boundary, G_cf, p, M, int_eps, eps_xi_eta, eps_xy, n_r, corner_r);

    u_num_mat = uh + R.f_R;
end
