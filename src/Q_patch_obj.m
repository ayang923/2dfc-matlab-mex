% Q_PATCH_OBJ  Base class for all boundary patch types in the 2DFC algorithm.
%
% Represents a rectangular region [xi_start, xi_end] x [eta_start, eta_end]
% in parameter space, mapped to physical (x,y) space via the parametrization
% M_p. Stores function values on the uniform (n_xi x n_eta) mesh and provides
% methods for mesh generation, coordinate conversion, Newton-based M_p
% inversion, polynomial interpolation, and partition-of-unity (POU)
% normalization.
%
% The POU window function is:
%   w_1D(t) = erfc(6*(-2*t+1)) / 2
% which is a smooth, monotone function mapping [0,1] -> [1,0].
%
% Properties:
%   M_p        - Vectorized parametrization handle: (xi_col, eta_col) -> (n x 2)
%   J          - Jacobian handle: ([xi; eta]) -> (2 x 2)
%   n_xi       - Number of mesh points along xi
%   n_eta      - Number of mesh points along eta
%   xi_start   - Lower bound of xi range
%   xi_end     - Upper bound of xi range
%   eta_start  - Lower bound of eta range
%   eta_end    - Upper bound of eta range
%   f_XY       - (n_eta x n_xi) matrix of function values on patch mesh
%   x_min, x_max, y_min, y_max - Bounding box of M_p(patch)
%   w_1D       - 1D POU window function handle
%   eps_xi_eta - Convergence tolerance for Newton inversion in (xi,eta) space
%   eps_xy     - Convergence tolerance for Newton inversion in (x,y) space
%
% Methods:
%   Q_patch_obj                      - Constructor
%   h_mesh                           - Returns (h_xi, h_eta) mesh spacings
%   xi_mesh / eta_mesh               - Returns 1D mesh vectors
%   xi_eta_mesh                      - Returns 2D meshgrid (XI, ETA)
%   xy_mesh                          - Returns 2D meshgrid in (x,y) space
%   boundary_mesh / boundary_mesh_xy - Closed polygon around patch boundary
%   convert_to_XY                    - Evaluates M_p on a mesh
%   in_patch                         - Bounding-box membership test
%   round_boundary_points            - Snaps near-boundary (xi,eta) to exact boundary
%   inverse_M_p                      - Newton inversion of M_p at a single (x,y)
%   locally_compute                  - Two-pass 1D polynomial interpolation at (xi,eta)
%   apply_w_normalization_xi_right/left  - POU normalization (xi direction)
%   apply_w_normalization_eta_up/down    - POU normalization (eta direction)
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef Q_patch_obj < handle
    properties
        M_p
        J
        n_xi
        n_eta
        xi_start
        xi_end
        eta_start
        eta_end
        f_XY
        x_min
        x_max
        y_min
        y_max
        w_1D

        eps_xi_eta
        eps_xy

        % mex_spec: [] for a generic patch (arbitrary M_p/J, always uses the
        % pure-MATLAB path below), or a mex_curve_spec struct (see
        % Curve_obj.get_mex_samples) when this patch was built by Curve_obj
        % from one of the two closed-form S-type/C-type templates -- in
        % that case inverse_M_p (and, via R_cartesian_mesh_obj,
        % locally_compute) dispatch to the mex-accelerated kernels, which
        % evaluate M_p/J entirely in C with no MATLAB callbacks. See
        % csrc/include/patch_kernels.h for why only these two templates are
        % (and need to be) accelerated.
        mex_spec = []
    end

    methods
        function obj = Q_patch_obj(M_p, J, eps_xi_eta, eps_xy, n_xi, n_eta, xi_start, xi_end, eta_start, eta_end, f_XY, mex_spec)
            % Q_PATCH_OBJ  Constructor.
            %
            % Inputs:
            %   M_p       - Vectorized parametrization: (xi_col, eta_col) -> (n x 2) [x, y]
            %   J         - Jacobian: ([xi; eta]) -> (2 x 2)
            %   eps_xi_eta - Newton convergence tolerance in (xi,eta) space
            %   eps_xy    - Newton convergence tolerance in (x,y) space
            %   n_xi      - Number of mesh points along xi
            %   n_eta     - Number of mesh points along eta
            %   xi_start  - Left bound of xi range
            %   xi_end    - Right bound of xi range
            %   eta_start - Bottom bound of eta range
            %   eta_end   - Top bound of eta range
            %   f_XY      - (n_eta x n_xi) function values, or nan to defer
            %   mex_spec  - (optional) mex_curve_spec struct enabling the
            %               mex-accelerated inverse_M_p path; omit or pass
            %               [] for a fully generic (pure-MATLAB) patch

            if nargin < 12
                mex_spec = [];
            end
            obj.mex_spec = mex_spec;

            obj.M_p = M_p;
            obj.J   = J;

            obj.eps_xi_eta = eps_xi_eta;
            obj.eps_xy     = eps_xy;

            obj.n_eta     = n_eta;
            obj.n_xi      = n_xi;
            obj.xi_start  = xi_start;
            obj.xi_end    = xi_end;
            obj.eta_start = eta_start;
            obj.eta_end   = eta_end;
            obj.f_XY      = f_XY;

            % Compute physical bounding box
            [XI, ETA] = obj.xi_eta_mesh();
            XY = M_p(XI(:), ETA(:));
            obj.x_min = min(XY(:, 1));
            obj.x_max = max(XY(:, 1));
            obj.y_min = min(XY(:, 2));
            obj.y_max = max(XY(:, 2));

            obj.w_1D = @(x) erfc(6*(-2*x + 1)) / 2;
        end

        function [h_xi, h_eta] = h_mesh(obj)
            % H_MESH  Returns the mesh spacings in xi and eta.
            h_xi  = (obj.xi_end  - obj.xi_start)  / (obj.n_xi  - 1);
            h_eta = (obj.eta_end - obj.eta_start) / (obj.n_eta - 1);
        end

        function mesh = xi_mesh(obj)
            % XI_MESH  Returns the n_xi-point uniform mesh over [xi_start, xi_end].
            mesh = transpose(linspace(obj.xi_start, obj.xi_end, obj.n_xi));
        end

        function mesh = eta_mesh(obj)
            % ETA_MESH  Returns the n_eta-point uniform mesh over [eta_start, eta_end].
            mesh = transpose(linspace(obj.eta_start, obj.eta_end, obj.n_eta));
        end

        function [XI, ETA] = xi_eta_mesh(obj)
            % XI_ETA_MESH  Returns the 2D meshgrid (XI, ETA) for the patch.
            [XI, ETA] = meshgrid(obj.xi_mesh(), obj.eta_mesh());
        end

        function [X, Y] = xy_mesh(obj)
            % XY_MESH  Returns the 2D physical mesh M_p(XI, ETA).
            [XI, ETA] = obj.xi_eta_mesh();
            [X, Y]    = obj.convert_to_XY(XI, ETA);
        end

        function [boundary_mesh_xi, boundary_mesh_eta] = boundary_mesh(obj, n_r, pad_boundary)
            % BOUNDARY_MESH  Returns a closed polygon of the patch boundary in (xi,eta).
            %
            % Traverses the four sides of the rectangular domain counter-clockwise.
            %
            % Inputs:
            %   n_r          - Refinement factor for boundary sampling
            %   pad_boundary - If true, expand by one mesh step to capture
            %                  boundary-adjacent Cartesian points

            if pad_boundary
                [h_xi, h_eta] = obj.h_mesh();
            else
                h_xi  = 0;
                h_eta = 0;
            end

            n_r_xi_mesh  = transpose(linspace(obj.xi_start,  obj.xi_end,  obj.n_xi  * n_r));
            n_r_eta_mesh = transpose(linspace(obj.eta_start, obj.eta_end, obj.n_eta * n_r));

            boundary_mesh_xi  = [ones(obj.n_eta*n_r, 1)*(obj.xi_start - h_xi); ...
                                  n_r_xi_mesh; ...
                                  ones(obj.n_eta*n_r, 1)*(obj.xi_end + h_xi); ...
                                  flip(n_r_xi_mesh); ...
                                  obj.xi_start - h_xi];
            boundary_mesh_eta = [n_r_eta_mesh; ...
                                  ones(obj.n_xi*n_r, 1)*(obj.eta_end + h_eta); ...
                                  flip(n_r_eta_mesh); ...
                                  ones(obj.n_xi*n_r, 1)*(obj.eta_start - h_eta); ...
                                  obj.eta_start - h_eta];
        end

        function [boundary_mesh_x, boundary_mesh_y] = boundary_mesh_xy(obj, n_r, pad_boundary)
            % BOUNDARY_MESH_XY  Returns the boundary polygon in physical (x,y) space.
            [bxi, beta] = obj.boundary_mesh(n_r, pad_boundary);
            [boundary_mesh_x, boundary_mesh_y] = obj.convert_to_XY(bxi, beta);
        end

        function [X, Y] = convert_to_XY(obj, XI, ETA)
            % CONVERT_TO_XY  Evaluates M_p on XI, ETA arrays of the same size.
            %
            % Inputs:
            %   XI  - Matrix (or vector) of xi values
            %   ETA - Matrix (or vector) of eta values (same shape as XI)
            %
            % Outputs:
            %   X, Y - Physical coordinates, same shape as XI and ETA
            XY = obj.M_p(XI(:), ETA(:));
            X  = reshape(XY(:, 1), size(XI));
            Y  = reshape(XY(:, 2), size(ETA));
        end

        function patch_msk = in_patch(obj, xi, eta)
            % IN_PATCH  Returns a logical mask: true where (xi,eta) is inside the patch.
            patch_msk = xi >= obj.xi_start & xi <= obj.xi_end & ...
                        eta >= obj.eta_start & eta <= obj.eta_end;
        end

        function [xi, eta] = round_boundary_points(obj, xi, eta)
            % ROUND_BOUNDARY_POINTS  Snaps (xi,eta) within eps_xi_eta of a boundary
            % to the exact boundary value to avoid floating-point misclassification.
            xi( abs(xi  - obj.xi_start)  < obj.eps_xi_eta) = obj.xi_start;
            xi( abs(xi  - obj.xi_end)    < obj.eps_xi_eta) = obj.xi_end;
            eta(abs(eta - obj.eta_start) < obj.eps_xi_eta) = obj.eta_start;
            eta(abs(eta - obj.eta_end)   < obj.eps_xi_eta) = obj.eta_end;
        end

        function [xi, eta, converged] = inverse_M_p(obj, x, y, initial_guesses)
            % INVERSE_M_P  Inverts M_p at a single physical point (x,y) via Newton's method.
            %
            % Inputs:
            %   x               - Scalar x coordinate
            %   y               - Scalar y coordinate
            %   initial_guesses - (2 x k) matrix of initial guesses, or nan to use
            %                     default boundary-mesh guesses
            %
            % Outputs:
            %   xi, eta   - Parameter-space coordinates of (x,y)
            %   converged - true if a converged, in-patch solution was found

            if use_mex_2dfc() && ~isempty(obj.mex_spec)
                [xi, eta, converged] = q_patch_inverse_M_p_mex(obj.mex_spec, ...
                    obj.xi_start, obj.xi_end, obj.eta_start, obj.eta_end, ...
                    obj.eps_xi_eta, obj.eps_xy, x, y, initial_guesses);
                return
            end

            if isnan(initial_guesses)
                initial_guesses = obj.default_initial_guesses(20);
            end

            err_guess = @(x, y, v) transpose(obj.M_p(v(1), v(2))) - [x; y];

            for k = 1:size(initial_guesses, 2)
                initial_guess = initial_guesses(:, k);
                [v_guess, converged] = newton_solve( ...
                    @(v) err_guess(x, y, v), obj.J, initial_guess, obj.eps_xy, 1000);

                [xi, eta] = obj.round_boundary_points(v_guess(1), v_guess(2));
                if converged && obj.in_patch(xi, eta)
                    return
                end
            end
        end

        function initial_guesses = default_initial_guesses(obj, N)
            % DEFAULT_INITIAL_GUESSES  Returns N uniformly-spaced guesses on the
            % four boundary edges for use in inverse_M_p.
            N_segment = ceil(N / 4);

            xi_mesh_vec  = transpose(linspace(obj.xi_start,  obj.xi_end,  N_segment + 1));
            eta_mesh_vec = transpose(linspace(obj.eta_start, obj.eta_end, N_segment + 1));

            xi_initial  = [xi_mesh_vec(1:end-1); xi_mesh_vec(1:end-1); ...
                           ones(N_segment, 1)*obj.xi_start; ones(N_segment, 1)*obj.xi_end];
            eta_initial = [ones(N_segment, 1)*obj.eta_start; ones(N_segment, 1)*obj.eta_end; ...
                           eta_mesh_vec(1:end-1); eta_mesh_vec(1:end-1)];

            initial_guesses = [xi_initial.'; eta_initial.'];
        end

        function [f_xy, in_range] = locally_compute(obj, xi, eta, M)
            % LOCALLY_COMPUTE  Estimates f at (xi,eta) via two-pass 1D Lagrange interpolation.
            %
            % Interpolates first along the xi direction for M adjacent eta rows, then
            % along the eta direction through the resulting M intermediate values.
            % The M-point stencil is shifted away from boundaries via shift_idx_mesh.
            %
            % Inputs:
            %   xi, eta - Scalar parameter-space coordinates (must be in-patch)
            %   M       - Number of interpolation points per 1D pass
            %
            % Outputs:
            %   f_xy     - Interpolated function value at (xi, eta)
            %   in_range - true if (xi, eta) is within the patch bounds

            if ~obj.in_patch(xi, eta)
                f_xy     = nan;
                in_range = false;
                warning('locally computing for point not in patch');
                return
            end

            in_range     = true;
            [h_xi, h_eta] = obj.h_mesh();

            xi_j  = floor((xi  - obj.xi_start)  / h_xi);
            eta_j = floor((eta - obj.eta_start) / h_eta);

            half_M = floor(M / 2);
            if mod(M, 2) ~= 0
                interpol_xi_j_mesh  = transpose(xi_j  - half_M : xi_j  + half_M);
                interpol_eta_j_mesh = transpose(eta_j - half_M : eta_j + half_M);
            else
                interpol_xi_j_mesh  = transpose(xi_j  - half_M + 1 : xi_j  + half_M);
                interpol_eta_j_mesh = transpose(eta_j - half_M + 1 : eta_j + half_M);
            end

            interpol_xi_j_mesh  = shift_idx_mesh(interpol_xi_j_mesh,  0, obj.n_xi  - 1);
            interpol_eta_j_mesh = shift_idx_mesh(interpol_eta_j_mesh, 0, obj.n_eta - 1);

            interpol_xi_mesh  = h_xi  * interpol_xi_j_mesh  + obj.xi_start;
            interpol_eta_mesh = h_eta * interpol_eta_j_mesh + obj.eta_start;

            % First pass: interpolate along xi for each of the M eta rows
            interpol_xi_exact = zeros(M, 1);
            for horz_idx = 1:M
                interpol_val = obj.f_XY(interpol_eta_j_mesh(horz_idx) + 1, interpol_xi_j_mesh + 1).';
                interpol_xi_exact(horz_idx) = barylag([interpol_xi_mesh, interpol_val], xi);
            end

            % Second pass: interpolate along eta through the M intermediate values
            f_xy = barylag([interpol_eta_mesh, interpol_xi_exact], eta);
        end

        function apply_w_normalization_xi_right(obj, window_patch)
            % APPLY_W_NORMALIZATION_XI_RIGHT  Applies POU normalization when the
            % window patch lies to the right (larger xi) of this patch.
            %
            % Computes the overlap region between this patch and window_patch, then
            % scales obj.f_XY in the overlap by w_main / (w_main + w_window) to
            % enforce the partition of unity.
            %
            % Input:
            %   window_patch - Q_patch_obj overlapping from the right

            main_xi_corner   = compute_xi_corner(obj, window_patch, true,  window_patch.xi_end, true);
            window_xi_corner = compute_xi_corner(window_patch, obj, true,  obj.xi_end, false);

            main_R_xi = main_xi_corner - obj.xi_end;
            main_xi_0 = obj.xi_end;
            main_w    = @(xi, eta) obj.w_1D((xi - main_xi_0) / main_R_xi);

            window_w = compute_w_normalization_window(obj, main_w, window_patch, window_xi_corner, false);

            [XI_overlap, ETA_overlap, XI_j, ETA_j] = compute_xi_overlap_mesh(obj, main_xi_corner, true);
            [X_overlap, Y_overlap] = obj.convert_to_XY(XI_overlap, ETA_overlap);

            w_unnormalized  = main_w(XI_overlap, ETA_overlap);
            initial_guesses = [linspace(window_xi_corner, window_patch.xi_end, 20); ...
                                window_patch.eta_start * ones(1, 20)];
            obj.apply_w(w_unnormalized, window_patch, true, window_w, ...
                X_overlap, Y_overlap, XI_j, ETA_j, initial_guesses);
        end

        function apply_w_normalization_xi_left(obj, window_patch)
            % APPLY_W_NORMALIZATION_XI_LEFT  Applies POU normalization when the
            % window patch lies to the left (smaller xi) of this patch.
            %
            % Input:
            %   window_patch - Q_patch_obj overlapping from the left

            main_xi_corner   = compute_xi_corner(obj, window_patch, true,  window_patch.xi_end, false);
            window_xi_corner = compute_xi_corner(window_patch, obj, true,  obj.xi_start, false);

            main_R_xi = main_xi_corner - obj.xi_start;
            main_xi_0 = obj.xi_start;
            main_w    = @(xi, eta) obj.w_1D((xi - main_xi_0) / main_R_xi);

            window_w = compute_w_normalization_window(obj, main_w, window_patch, window_xi_corner, false);

            [XI_overlap, ETA_overlap, XI_j, ETA_j] = compute_xi_overlap_mesh(obj, main_xi_corner, false);
            [X_overlap, Y_overlap] = obj.convert_to_XY(XI_overlap, ETA_overlap);

            w_unnormalized  = main_w(XI_overlap, ETA_overlap);
            initial_guesses = [linspace(window_xi_corner, window_patch.xi_end, 20); ...
                                window_patch.eta_start * ones(1, 20)];
            obj.apply_w(w_unnormalized, window_patch, true, window_w, ...
                X_overlap, Y_overlap, XI_j, ETA_j, initial_guesses);
        end

        function apply_w_normalization_eta_up(obj, window_patch)
            % APPLY_W_NORMALIZATION_ETA_UP  Applies POU normalization when the
            % window patch lies above (larger eta) this patch.
            %
            % Input:
            %   window_patch - Q_patch_obj overlapping from above

            main_eta_corner  = compute_eta_corner(obj, window_patch, true,  window_patch.xi_start, true);
            window_xi_corner = compute_xi_corner(window_patch, obj, false, obj.eta_end, false);

            main_R_eta = main_eta_corner - obj.eta_end;
            main_eta_0 = obj.eta_end;
            main_w     = @(xi, eta) obj.w_1D((eta - main_eta_0) / main_R_eta);

            window_w = compute_w_normalization_window(obj, main_w, window_patch, window_xi_corner, true);

            [XI_overlap, ETA_overlap, XI_j, ETA_j] = compute_eta_overlap_mesh(obj, main_eta_corner, true);
            [X_overlap, Y_overlap] = obj.convert_to_XY(XI_overlap, ETA_overlap);

            w_unnormalized  = main_w(XI_overlap, ETA_overlap);
            initial_guesses = [linspace(window_patch.xi_start, window_xi_corner, 20); ...
                                window_patch.eta_start * ones(1, 20)];
            obj.apply_w(w_unnormalized, window_patch, true, window_w, ...
                X_overlap, Y_overlap, XI_j, ETA_j, initial_guesses);
        end

        function apply_w_normalization_eta_down(obj, window_patch)
            % APPLY_W_NORMALIZATION_ETA_DOWN  Applies POU normalization when the
            % window patch lies below (smaller eta) this patch.
            %
            % Input:
            %   window_patch - Q_patch_obj overlapping from below

            main_eta_corner  = compute_eta_corner(obj, window_patch, true,  window_patch.xi_start, false);
            window_xi_corner = compute_xi_corner(window_patch, obj, false, obj.eta_start, false);

            main_R_eta = main_eta_corner - obj.eta_start;
            main_eta_0 = obj.eta_start;
            main_w     = @(xi, eta) obj.w_1D((eta - main_eta_0) / main_R_eta);

            window_w = compute_w_normalization_window(obj, main_w, window_patch, window_xi_corner, true);

            [XI_overlap, ETA_overlap, XI_j, ETA_j] = compute_eta_overlap_mesh(obj, main_eta_corner, false);
            [X_overlap, Y_overlap] = obj.convert_to_XY(XI_overlap, ETA_overlap);

            w_unnormalized  = main_w(XI_overlap, ETA_overlap);
            initial_guesses = [linspace(window_patch.xi_start, window_xi_corner, 20); ...
                                window_patch.eta_start * ones(1, 20)];
            obj.apply_w(w_unnormalized, window_patch, true, window_w, ...
                X_overlap, Y_overlap, XI_j, ETA_j, initial_guesses);
        end

        function apply_w(obj, w_unnormalized, window_patch, window_patch_bound_xi, window_w, overlap_X, overlap_Y, overlap_XI_j, overlap_ETA_j, initial_guesses)
            % APPLY_W  Core POU normalization loop over the overlap region.
            %
            % For each point in the overlap, inverts the window patch parametrization
            % to find the corresponding (xi,eta) in the window patch's parameter space,
            % then scales obj.f_XY(eta_j, xi_j) by:
            %
            %   f_new = f_old * w_main / (w_main + w_window)
            %
            % Points are traversed in a boustrophedon (snake) order to chain Newton
            % initial guesses from one converged solution to the next.
            %
            % Inputs:
            %   w_unnormalized       - w_main evaluated on the overlap mesh
            %   window_patch         - Q_patch_obj providing w_window
            %   window_patch_bound_xi - true if the window patch is bounded in xi
            %   window_w             - w_window function handle
            %   overlap_X/Y          - Physical coordinates of the overlap region
            %   overlap_XI_j/ETA_j   - Integer (i,j) indices into obj.f_XY
            %   initial_guesses      - Initial guesses for Newton inversion

            for i = 1:size(overlap_X, 1)
                if mod(i, 2) == 1
                    j_lst = 1:size(overlap_X, 2);
                else
                    j_lst = size(overlap_X, 2):-1:1;
                end

                for j = j_lst
                    if isnan(initial_guesses)
                        initial_guesses = obj.default_initial_guesses(20);
                    end

                    for initial_guess = initial_guesses   %#ok<FXUP> loop kept intentionally
                        [window_patch_xi, window_patch_eta, converged] = ...
                            window_patch.inverse_M_p(overlap_X(i,j), overlap_Y(i,j), initial_guesses);

                        if i == 1 && j == 1
                            in_VpR = window_patch.in_patch(window_patch_xi, window_patch_eta);
                            if ~in_VpR
                                warning('First overlap point should be in window patch; check domain geometry.');
                            end
                        else
                            if window_patch_bound_xi
                                in_VpR = window_patch_xi >= window_patch.xi_start && ...
                                         window_patch_xi <= window_patch.xi_end;
                            else
                                in_VpR = window_patch_eta >= window_patch.eta_start && ...
                                         window_patch_eta <= window_patch.eta_end;
                            end
                        end

                        if converged && in_VpR
                            break;
                        end
                    end

                    if converged && in_VpR
                        xi_j  = overlap_XI_j( i, j) + 1;
                        eta_j = overlap_ETA_j(i, j) + 1;
                        obj.f_XY(eta_j, xi_j) = obj.f_XY(eta_j, xi_j) .* w_unnormalized(i,j) ...
                            ./ (window_w(window_patch_xi, window_patch_eta) + w_unnormalized(i,j));
                    elseif ~converged
                        warning('Nonconvergence in computing POU normalization.');
                    end

                    % Use last converged point as initial guess for the next point
                    if converged && in_VpR
                        initial_guesses = [window_patch_xi; window_patch_eta];
                    end
                end
            end
        end
    end
end

% -------------------------------------------------------------------------
% File-private helper functions
% -------------------------------------------------------------------------

function [main_xi_corner] = compute_xi_corner(main_patch, window_patch, window_fix_xi, window_fixed_edge, window_patch_right)
% COMPUTE_XI_CORNER  Finds the xi extent of the overlap between main_patch and
% window_patch by walking along the window patch boundary edge and inverting M_p.
%
% Scans the edge of window_patch that borders main_patch, maps each sample
% point into main_patch's parameter space, and tracks the extremal xi value.

    [h_xi_window, h_eta_window] = window_patch.h_mesh();

    if window_fix_xi
        window_xi_edge  = window_fixed_edge;
        window_eta_edge = window_patch.eta_start;
    else
        window_xi_edge  = window_patch.xi_start;
        window_eta_edge = window_fixed_edge;
    end

    if window_patch_right
        main_xi_corner = main_patch.xi_end;
    else
        main_xi_corner = main_patch.xi_start;
    end

    first_iter = true;
    while true
        window_xy_edge = window_patch.M_p(window_xi_edge, window_eta_edge);
        if first_iter
            [main_xi_edge, main_eta_edge, converged] = ...
                main_patch.inverse_M_p(window_xy_edge(1,1), window_xy_edge(1,2), nan);
        else
            [main_xi_edge, main_eta_edge, converged] = ...
                main_patch.inverse_M_p(window_xy_edge(1,1), window_xy_edge(1,2), [main_xi_edge; main_eta_edge]);
        end
        first_iter = false;

        if ~converged
            warning('Nonconvergence in computing boundary mesh values.');
            break;
        end

        if (main_xi_edge < main_xi_corner && window_patch_right) || ...
           (main_xi_edge > main_xi_corner && ~window_patch_right)
            main_xi_corner = main_xi_edge;
        end

        if main_eta_edge > main_patch.eta_end || main_eta_edge < main_patch.eta_start
            break;
        end

        if window_fix_xi
            window_eta_edge = window_eta_edge + h_eta_window;
        else
            window_xi_edge  = window_xi_edge  + h_xi_window;
        end
    end
end

function [XI_overlap, ETA_overlap, XI_j, ETA_j] = compute_xi_overlap_mesh(main_patch, xi_corner, window_patch_right)
% COMPUTE_XI_OVERLAP_MESH  Returns the (xi,eta) mesh of points in the xi overlap region.

    [h_xi, h_eta] = main_patch.h_mesh();

    if window_patch_right
        xi_corner_j = ceil((xi_corner - main_patch.xi_start) / h_xi);
        [XI_j, ETA_j] = meshgrid((main_patch.n_xi-1):-1:xi_corner_j, 0:(main_patch.n_eta-1));
    else
        xi_corner_j = floor((xi_corner - main_patch.xi_start) / h_xi);
        [XI_j, ETA_j] = meshgrid(0:xi_corner_j, 0:(main_patch.n_eta-1));
    end

    XI_overlap  = XI_j  * h_xi  + main_patch.xi_start;
    ETA_overlap = ETA_j * h_eta + main_patch.eta_start;
end

function [main_eta_corner] = compute_eta_corner(main_patch, window_patch, window_fix_xi, window_fixed_edge, window_patch_up)
% COMPUTE_ETA_CORNER  Finds the eta extent of the overlap between main_patch and
% window_patch by walking along the window patch boundary edge and inverting M_p.

    [h_xi_window, h_eta_window] = window_patch.h_mesh();

    if window_fix_xi
        window_xi_edge  = window_fixed_edge;
        window_eta_edge = window_patch.eta_start;
    else
        window_xi_edge  = window_patch.xi_start;
        window_eta_edge = window_fixed_edge;
    end

    if window_patch_up
        main_eta_corner = main_patch.eta_end;
    else
        main_eta_corner = main_patch.eta_start;
    end

    first_iter = true;
    while true
        window_xy_edge = window_patch.M_p(window_xi_edge, window_eta_edge);
        if first_iter
            [main_xi_edge, main_eta_edge, converged] = ...
                main_patch.inverse_M_p(window_xy_edge(1,1), window_xy_edge(1,2), nan);
        else
            [main_xi_edge, main_eta_edge, converged] = ...
                main_patch.inverse_M_p(window_xy_edge(1,1), window_xy_edge(1,2), [main_xi_edge; main_eta_edge]);
        end
        first_iter = false;

        if ~converged
            warning('Nonconvergence in computing boundary mesh values.');
            break;
        end

        if (main_eta_edge < main_eta_corner && window_patch_up) || ...
           (main_eta_edge > main_eta_corner && ~window_patch_up)
            main_eta_corner = main_eta_edge;
        end

        if main_xi_edge > main_patch.xi_end || main_xi_edge < main_patch.xi_start
            break;
        end

        if window_fix_xi
            window_eta_edge = window_eta_edge + h_eta_window;
        else
            window_xi_edge  = window_xi_edge  + h_xi_window;
        end
    end
end

function [XI_overlap, ETA_overlap, XI_j, ETA_j] = compute_eta_overlap_mesh(main_patch, eta_corner, window_patch_up)
% COMPUTE_ETA_OVERLAP_MESH  Returns the (xi,eta) mesh of points in the eta overlap region.

    [h_xi, h_eta] = main_patch.h_mesh();

    if window_patch_up
        eta_corner_j = ceil((eta_corner - main_patch.eta_start) / h_eta);
        [XI_j, ETA_j] = meshgrid(0:(main_patch.n_xi-1), (main_patch.n_eta-1):-1:eta_corner_j);
    else
        eta_corner_j = floor((eta_corner - main_patch.eta_start) / h_eta);
        [XI_j, ETA_j] = meshgrid(0:(main_patch.n_xi-1), 0:eta_corner_j);
    end

    XI_overlap  = XI_j  * h_xi  + main_patch.xi_start;
    ETA_overlap = ETA_j * h_eta + main_patch.eta_start;
end

function [window_w] = compute_w_normalization_window(main_patch, main_w, window_patch, window_xi_corner, up_down)
% COMPUTE_W_NORMALIZATION_WINDOW  Constructs the window patch's w function and
% applies the POU normalization to the window patch's function values in the
% overlap region.
%
% The window patch's w is a 1D function of xi (for left/right overlaps) or
% of xi evaluated along the window patch (for up/down overlaps).

    if up_down
        window_R_xi = window_xi_corner - window_patch.xi_start;
        window_xi_0 = window_patch.xi_start;
    else
        window_R_xi = window_xi_corner - window_patch.xi_end;
        window_xi_0 = window_patch.xi_end;
    end
    window_w = @(xi, eta) window_patch.w_1D((xi - window_xi_0) / window_R_xi);

    [XI_overlap_window, ETA_overlap_window, XI_j_window, ETA_j_window] = ...
        compute_xi_overlap_mesh(window_patch, window_xi_corner, ~up_down);
    [X_overlap_window, Y_overlap_window] = ...
        window_patch.convert_to_XY(XI_overlap_window, ETA_overlap_window);

    w_unnormalized = window_w(XI_overlap_window, ETA_overlap_window);
    window_patch.apply_w(w_unnormalized, main_patch, ~up_down, main_w, ...
        X_overlap_window, Y_overlap_window, XI_j_window, ETA_j_window, nan);
end
