% R_CARTESIAN_MESH_OBJ  Uniform Cartesian mesh for the 2DFC algorithm.
%
% Stores the bounding Cartesian grid on which the periodic FC extension of f
% is assembled. Provides methods to:
%   - Interpolate FC extension patch values onto the grid
%   - Fill interior points with exact function values
%   - Compute and store 2D Fourier coefficients (fc_coeffs)
%   - Evaluate spectral derivatives (gradient, divergence, Laplacian)
%   - Solve the Poisson particular-solution step (inverse Laplacian)
%   - Evaluate the Fourier series on a finer grid for error estimation
%
% The grid spans [x_start, x_end] x [y_start, y_end] with uniform step h.
% Grid dimensions are rounded up so that x_end and y_end are exact multiples
% of h above x_start and y_start.
%
% Properties:
%   x_start, x_end, y_start, y_end - Physical extent of the grid
%   h          - Uniform mesh spacing (same in both directions)
%   n_x, n_y   - Number of grid points in x and y
%   x_mesh, y_mesh - 1D mesh vectors
%   R_X, R_Y   - Meshgrid arrays of physical coordinates
%   R_idxs     - Linear index array with same shape as R_X
%   boundary_X, boundary_Y - Closed polygon of the domain boundary
%   in_interior  - (n_y x n_x) logical mask: true at interior grid points
%   interior_idxs - Linear indices of interior grid points
%   f_R          - (n_y x n_x) assembled function/continuation values
%   fc_coeffs    - (n_y x n_x) fftshift'd 2D Fourier coefficients of f_R
%
% Methods:
%   R_cartesian_mesh_obj - Constructor
%   interpolate_patch    - Interpolates an FC extension patch onto the grid
%   fill_interior        - Fills interior grid points with exact f values
%   locally_compute      - 2D polynomial interpolation at a single (x,y)
%   compute_fc_coeffs    - Computes 2D FFT of f_R and stores in fc_coeffs
%   ifft_interpolation   - Evaluates Fourier series on a rho_err-times finer grid
%   grad                 - Spectral gradient of a given mesh function
%   div                  - Spectral divergence of a 2D vector field
%   lap                  - Spectral Laplacian of a given mesh function
%   lap_coeff            - Laplacian from pre-computed Fourier coefficients
%   inv_lap              - Spectral inverse Laplacian of fc_coeffs
%   fft_filter           - Builds a Gaussian-type spectral filter
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef R_cartesian_mesh_obj < handle
    properties
        x_start
        x_end
        y_start
        y_end
        h
        n_x
        n_y
        x_mesh
        y_mesh
        R_X
        R_Y
        R_idxs

        boundary_X
        boundary_Y
        in_interior
        interior_idxs
        f_R

        fc_coeffs
    end

    methods
        function obj = R_cartesian_mesh_obj(x_start, x_end, y_start, y_end, h, boundary_X, boundary_Y, n_x_padded, n_y_padded)
            % R_CARTESIAN_MESH_OBJ  Constructor.
            %
            % The x_end and y_end are rounded up to the nearest multiple of h so
            % that the grid has an integer number of intervals.
            %
            % Inputs:
            %   x_start, x_end - Requested x range (x_end is rounded up)
            %   y_start, y_end - Requested y range (y_end is rounded up)
            %   h              - Uniform mesh spacing
            %   boundary_X/Y   - Closed polygon defining the domain interior
            %   n_x_padded     - (optional) If provided, overrides n_x and extends
            %                    x_end to x_start + (n_x_padded-1)*h
            %   n_y_padded     - (optional) If provided, overrides n_y and extends
            %                    y_end to y_start + (n_y_padded-1)*h

            obj.x_start = x_start;
            obj.y_start = y_start;
            obj.h       = h;

            x_end_std = ceil((x_end - x_start) / h) * h + x_start;
            y_end_std = ceil((y_end - y_start) / h) * h + y_start;
            n_x_std   = round((x_end_std - x_start) / h) + 1;
            n_y_std   = round((y_end_std - y_start) / h) + 1;

            if nargin >= 9
                if n_x_padded < n_x_std || n_y_padded < n_y_std
                    error('n_x_padded (%d) and n_y_padded (%d) must be >= the standard grid size n_x=%d, n_y=%d.', ...
                        n_x_padded, n_y_padded, n_x_std, n_y_std);
                end
                obj.n_x        = n_x_padded;
                obj.n_y        = n_y_padded;
                obj.x_end      = x_start + (n_x_padded - 1) * h;
                obj.y_end      = y_start + (n_y_padded - 1) * h;
            else
                obj.x_end = x_end_std;
                obj.y_end = y_end_std;
                obj.n_x   = n_x_std;
                obj.n_y   = n_y_std;
            end

            % Round mesh points to reduce floating-point accumulation errors
            obj.x_mesh = transpose( ...
                round((linspace(obj.x_start, obj.x_end, obj.n_x) - x_start) / h) * h + x_start);
            obj.y_mesh = transpose( ...
                round((linspace(obj.y_start, obj.y_end, obj.n_y) - y_start) / h) * h + y_start);

            [obj.R_X, obj.R_Y] = meshgrid(obj.x_mesh, obj.y_mesh);
            obj.R_idxs = reshape(1:numel(obj.R_X), size(obj.R_X));

            obj.boundary_X = boundary_X;
            obj.boundary_Y = boundary_Y;

            obj.in_interior  = inpolygon_mesh(obj.R_X, obj.R_Y, boundary_X, boundary_Y);
            obj.interior_idxs = obj.R_idxs(obj.in_interior);
            obj.f_R = zeros(obj.n_y, obj.n_x);
        end

        function [R_patch_idxs] = interpolate_patch(obj, patch, n_r, M)
            % INTERPOLATE_PATCH  Accumulates FC extension values from a patch onto the grid.
            %
            % Identifies Cartesian grid points that lie inside the patch (via
            % inpolygon_mesh on the patch boundary) but not already in the interior,
            % inverts the patch parametrization (R_xi_eta_inversion) to get (xi,eta)
            % coordinates, and evaluates the patch function (locally_compute) at each.
            %
            % Inputs:
            %   patch - Q_patch_obj whose extension values to interpolate
            %   n_r   - Refinement factor used to sample the patch boundary polygon
            %   M     - Polynomial interpolation degree for locally_compute
            %
            % Outputs:
            %   R_patch_idxs - Linear indices of grid points filled by this call

            if use_mex_2dfc() && ~isempty(patch.mex_spec)
                [R_patch_idxs, f_R_patch] = r_cartesian_mesh_interpolate_patch_mex( ...
                    patch.mex_spec, patch.xi_start, patch.xi_end, patch.eta_start, patch.eta_end, ...
                    patch.f_XY, obj.R_X, obj.R_Y, obj.in_interior, obj.x_start, obj.y_start, obj.h, ...
                    n_r, M, patch.eps_xi_eta, patch.eps_xy);
                obj.f_R(R_patch_idxs) = obj.f_R(R_patch_idxs) + f_R_patch;
                return
            end

            [bound_X, bound_Y] = patch.boundary_mesh_xy(n_r, false);
            in_patch      = inpolygon_mesh(obj.R_X, obj.R_Y, bound_X, bound_Y) & ~obj.in_interior;
            R_patch_idxs  = obj.R_idxs(in_patch);

            [P_xi, P_eta] = R_xi_eta_inversion(obj, patch, in_patch);

            f_R_patch = zeros(size(R_patch_idxs));
            for i = 1:length(R_patch_idxs)
                [interior_val, in_range] = patch.locally_compute(P_xi(i), P_eta(i), M);
                if in_range
                    f_R_patch(i) = interior_val;
                end
            end
            obj.f_R(R_patch_idxs) = obj.f_R(R_patch_idxs) + f_R_patch;
        end

        function fill_interior(obj, f)
            % FILL_INTERIOR  Overwrites interior grid points with exact f values.
            obj.f_R(obj.interior_idxs) = f(obj.R_X(obj.interior_idxs), obj.R_Y(obj.interior_idxs));
        end

        function [f_xy, in_range] = locally_compute(obj, x, y, M)
            % LOCALLY_COMPUTE  Two-pass 1D Lagrange interpolation at a single (x,y).
            %
            % See Q_patch_obj.locally_compute for the analogous method; this version
            % operates on the Cartesian grid f_R.
            %
            % Inputs:
            %   x, y - Physical coordinates (must be within grid bounds)
            %   M    - Number of interpolation points per 1D pass
            %
            % Outputs:
            %   f_xy     - Interpolated function value
            %   in_range - true if (x,y) is within the grid bounds

            if x > obj.x_end || x < obj.x_start || y > obj.y_end || y < obj.y_start
                f_xy     = nan;
                in_range = false;
                warning('(x, y) not in range');
                return
            end

            in_range = true;
            x_j = floor((x - obj.x_start) / obj.h);
            y_j = floor((y - obj.y_start) / obj.h);

            half_M = floor(M / 2);
            if mod(M, 2) ~= 0
                interpol_x_j_mesh = transpose(x_j - half_M : x_j + half_M);
                interpol_y_j_mesh = transpose(y_j - half_M : y_j + half_M);
            else
                interpol_x_j_mesh = transpose(x_j - half_M + 1 : x_j + half_M);
                interpol_y_j_mesh = transpose(y_j - half_M + 1 : y_j + half_M);
            end

            interpol_x_j_mesh = shift_idx_mesh(interpol_x_j_mesh, 0, obj.n_x - 1);
            interpol_y_j_mesh = shift_idx_mesh(interpol_y_j_mesh, 0, obj.n_y - 1);

            interpol_x_mesh = obj.h * interpol_x_j_mesh + obj.x_start;
            interpol_y_mesh = obj.h * interpol_y_j_mesh + obj.y_start;

            % First pass: interpolate along x for each of the M y-rows
            interpol_x_exact = zeros(M, 1);
            for horz_idx = 1:M
                interpol_val = obj.f_R(interpol_y_j_mesh(horz_idx) + 1, interpol_x_j_mesh + 1).';
                interpol_x_exact(horz_idx) = barylag([interpol_x_mesh, interpol_val], x);
            end

            % Second pass: interpolate along y
            f_xy = barylag([interpol_y_mesh, interpol_x_exact], y);
        end

        function compute_fc_coeffs(obj)
            % COMPUTE_FC_COEFFS  Computes the 2D Fourier coefficients of f_R.
            %
            % Stores coefficients in fftshift order (DC at center) normalized by the
            % total number of grid points.
            obj.fc_coeffs = fftshift(fft2(obj.f_R) / numel(obj.f_R));
        end

        function [R_X_err, R_Y_err, f_interpolation, interior_idx] = ifft_interpolation(obj, rho_err)
            % IFFT_INTERPOLATION  Evaluates the Fourier series on a rho_err-times finer grid.
            %
            % Zero-pads fc_coeffs symmetrically and takes an inverse FFT to obtain
            % the Fourier series values on a grid with step size h/rho_err. Used for
            % error estimation in FC2D.
            %
            % Input:
            %   rho_err - Refinement factor (e.g., 2 for twice as fine)
            %
            % Outputs:
            %   R_X_err, R_Y_err - Meshgrid of the finer evaluation grid
            %   f_interpolation  - Fourier series values on the finer grid
            %   interior_idx     - Linear indices of interior points in the finer grid

            h_err  = obj.h / rho_err;
            n_x_err = round((obj.x_end + obj.h - h_err - obj.x_start) / h_err) + 1;
            n_y_err = round((obj.y_end + obj.h - h_err - obj.y_start) / h_err) + 1;

            x_err_mesh = transpose( ...
                round((linspace(obj.x_start, obj.x_end + obj.h - h_err, n_x_err) - obj.x_start) / h_err) ...
                * h_err + obj.x_start);
            y_err_mesh = transpose( ...
                round((linspace(obj.y_start, obj.y_end + obj.h - h_err, n_y_err) - obj.y_start) / h_err) ...
                * h_err + obj.y_start);

            [R_X_err, R_Y_err] = meshgrid(x_err_mesh, y_err_mesh);

            n_x_diff = n_x_err - obj.n_x;
            n_y_diff = n_y_err - obj.n_y;

            padded_fc_coeffs = [ ...
                zeros(ceil(n_y_diff/2), n_x_err); ...
                zeros(size(obj.fc_coeffs, 1), ceil(n_x_diff/2)), obj.fc_coeffs, zeros(size(obj.fc_coeffs, 1), floor(n_x_diff/2)); ...
                zeros(floor(n_y_diff/2), n_x_err)];

            f_interpolation = real((n_x_err * n_y_err) * ifft2(ifftshift(padded_fc_coeffs)));

            idxs         = reshape(1:numel(R_X_err), size(R_X_err));
            interior_idx = idxs(inpolygon_mesh(R_X_err, R_Y_err, obj.boundary_X, obj.boundary_Y));
        end

        function [grad_X, grad_Y, f_hat_x, f_hat_y] = grad(obj, f_mesh)
            % GRAD  Computes the spectral gradient of f_mesh.
            %
            % Computes partial derivatives by multiplying Fourier coefficients by
            % i*kx and i*ky respectively, then inverting.
            %
            % Input:
            %   f_mesh - (n_y x n_x) mesh of function values
            %
            % Outputs:
            %   grad_X, grad_Y - Physical-space partial derivatives df/dx, df/dy
            %   f_hat_x, f_hat_y - Corresponding Fourier coefficient arrays

            [KX, KY]   = obj.wavenumber_grids();
            fft_coeffs = fftshift(fft2(f_mesh) / numel(obj.f_R));

            f_hat_x = 1i * KX .* fft_coeffs;
            f_hat_y = 1i * KY .* fft_coeffs;

            grad_X = numel(obj.f_R) * real(ifft2(ifftshift(f_hat_x)));
            grad_Y = numel(obj.f_R) * real(ifft2(ifftshift(f_hat_y)));
        end

        function [f_div] = div(obj, f_1_mesh, f_2_mesh)
            % DIV  Computes the spectral divergence of a 2D vector field (f_1, f_2).
            %
            % Inputs:
            %   f_1_mesh - (n_y x n_x) x-component of the vector field
            %   f_2_mesh - (n_y x n_x) y-component of the vector field
            %
            % Outputs:
            %   f_div - (n_y x n_x) divergence df_1/dx + df_2/dy

            [KX, KY]       = obj.wavenumber_grids();
            fft_coeffs_1   = fftshift(fft2(f_1_mesh) / numel(obj.f_R));
            fft_coeffs_2   = fftshift(fft2(f_2_mesh) / numel(obj.f_R));

            f_hat = 1i * KX .* fft_coeffs_1 + 1i * KY .* fft_coeffs_2;
            f_div = numel(obj.f_R) * real(ifft2(ifftshift(f_hat)));
        end

        function [f_lap, f_hat] = lap(obj, f_mesh)
            % LAP  Computes the spectral Laplacian of f_mesh.
            %
            % Input:
            %   f_mesh - (n_y x n_x) mesh of function values
            %
            % Outputs:
            %   f_lap - (n_y x n_x) Laplacian d^2f/dx^2 + d^2f/dy^2
            %   f_hat - Fourier coefficients of f_lap

            [KX, KY]   = obj.wavenumber_grids();
            fft_coeffs = fftshift(fft2(f_mesh) / numel(obj.f_R));

            f_hat = -(KX.^2 + KY.^2) .* fft_coeffs;
            f_lap = numel(obj.f_R) * real(ifft2(ifftshift(f_hat)));
        end

        function [f_lap, f_hat] = lap_coeff(obj, fft_coeffs)
            % LAP_COEFF  Computes the spectral Laplacian from pre-computed coefficients.
            %
            % Equivalent to lap() but takes Fourier coefficients directly, avoiding
            % a redundant FFT when coefficients are already available.
            %
            % Input:
            %   fft_coeffs - (n_y x n_x) fftshift'd Fourier coefficients
            %
            % Outputs:
            %   f_lap - (n_y x n_x) Laplacian in physical space
            %   f_hat - Fourier coefficients of the Laplacian

            [KX, KY] = obj.wavenumber_grids();
            f_hat    = -(KX.^2 + KY.^2) .* fft_coeffs;
            f_lap    = numel(obj.f_R) * real(ifft2(ifftshift(f_hat)));
        end

        function [f_inv_lap] = inv_lap(obj)
            % INV_LAP  Computes the spectral inverse Laplacian of fc_coeffs.
            %
            % Solves Delta(u) = f spectrally by dividing Fourier coefficients by
            % -(kx^2 + ky^2). The zero-frequency mode (DC component) corresponds to
            % an arbitrary constant in the inverse Laplacian; here it is treated by
            % adding a particular solution (R_X^2 + R_Y^2)/4 * fc_coeff_0_0.
            %
            % Outputs:
            %   f_inv_lap - (n_y x n_x) particular solution u_p satisfying Delta(u_p) = f

            [KX, KY] = obj.wavenumber_grids();

            fc_coeffs_modified = ifftshift(obj.fc_coeffs);
            fc_coeff_0_0       = fc_coeffs_modified(1, 1);

            fc_coeffs_modified = -fc_coeffs_modified ./ ...
                (ifftshift(KX.^2) + ifftshift(KY.^2));
            fc_coeffs_modified(1, 1) = 0;

            f_inv_lap = numel(obj.f_R) * ifft2(fc_coeffs_modified) + ...
                fc_coeff_0_0 * (obj.R_X.^2 + obj.R_Y.^2) / 4;
        end

        function filter = fft_filter(obj)
            % FFT_FILTER  Builds a Gaussian-type spectral filter for dealiasing.
            %
            % Returns the filter exp(-alpha*(2*kx/n_x)^(2p)) * exp(-alpha*(2*ky/n_y)^(2p))
            % with alpha = -log(1e-16) and p = 18, which suppresses modes near the
            % Nyquist frequency while leaving low modes essentially unchanged.
            %
            % Outputs:
            %   filter - (n_y x n_x) spectral filter array in fftshift order

            [KX, KY] = obj.wavenumber_grids();
            alpha    = -log(1e-16);
            p        = 18;
            filter   = exp(-alpha * (2*KX/obj.n_x).^(2*p)) .* ...
                       exp(-alpha * (2*KY/obj.n_y).^(2*p));
        end
    end

    methods (Access = private)
        function [KX, KY] = wavenumber_grids(obj)
            % WAVENUMBER_GRIDS  Returns 2D meshgrid of wavenumbers in fftshift order.
            %
            % The period of the Cartesian grid is (x_end - x_start + h) in x and
            % (y_end - y_start + h) in y.

            Lx = obj.x_end - obj.x_start + obj.h;
            Ly = obj.y_end - obj.y_start + obj.h;

            if mod(obj.n_x, 2) == 0
                kx = (2*pi/Lx) .* (-obj.n_x/2 : obj.n_x/2 - 1);
            else
                kx = (2*pi/Lx) .* (-(obj.n_x-1)/2 : (obj.n_x-1)/2);
            end

            if mod(obj.n_y, 2) == 0
                ky = (2*pi/Ly) .* (-obj.n_y/2 : obj.n_y/2 - 1);
            else
                ky = (2*pi/Ly) .* (-(obj.n_y-1)/2 : (obj.n_y-1)/2);
            end

            [KX, KY] = meshgrid(kx, ky);
        end
    end
end
