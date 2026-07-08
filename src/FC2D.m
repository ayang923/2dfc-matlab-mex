function [R, interior_patches, FC_patches, fc_err] = FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C_S, n_r, A_S, Q_S, C_C, A_C, Q_C, M, n_x_padded, n_y_padded, perturb)
% FC2D  Computes a smooth Fourier continuation of f on an arbitrary 2D domain.
%
% The domain is described by a sequence of C^2 curves (curve_seq). The
% algorithm constructs smooth (S-type) and corner (C1/C2-type) boundary
% patches, applies 1D blending-to-zero FC extensions to each patch, then
% interpolates all extension values onto a bounding Cartesian mesh. The
% interior is filled with exact function values, and a 2D FFT is taken to
% produce Fourier coefficients for the periodic continuation function.
%
% Inputs:
%   f          - Function handle f(x,y) to be approximated
%   h          - Cartesian mesh step size
%   curve_seq  - Curve_seq_obj describing the domain boundary
%   eps_xi_eta - Newton inversion tolerance in parameter (xi-eta) space
%   eps_xy     - Newton inversion tolerance in physical (x-y) space
%   d          - Number of Gram matching points for 1D FC
%   C_S        - Number of continuation points for smooth (S-type) patches
%   n_r        - Refinement factor for the FC continuation grid
%   A_S        - Precomputed FC matrix A for S-type patches  (n_r*C_S x ...)
%   Q_S        - Precomputed FC matrix Q for S-type patches  (d x d)
%   C_C        - Number of continuation points for corner patches
%   A_C        - Precomputed FC matrix A for corner patches  (n_r*C_C x ...)
%   Q_C        - Precomputed FC matrix Q for corner patches  (d x d)
%   M          - Degree of polynomial interpolation onto Cartesian mesh
%   n_x_padded - (optional) If provided (non-empty), overrides the computed n_x of R;
%                must be >= the natural grid size (see R_cartesian_mesh_obj).
%                Pass [] to skip padding while still supplying perturb.
%   n_y_padded - (optional) If provided (non-empty), overrides the computed n_y of R;
%                must be >= the natural grid size.
%                Pass [] to skip padding while still supplying perturb.
%   perturb    - (optional) Logical flag. If true, expand the bounding box by
%                rand(1)*h on each side (random perturbation). If false or omitted,
%                expand by the fixed amount h.
%
% Outputs:
%   R               - R_cartesian_mesh_obj containing Cartesian mesh,
%                     continuation function values, and Fourier coefficients
%   interior_patches - Cell array of {S_patch, C_patch} objects per curve
%   FC_patches       - Cell array of 4 FC extension Q_patch_obj objects per
%                      curve (1 from S, 3 from C)
%   fc_err           - Relative L2 error of the Fourier approximation

    interior_patches = curve_seq.construct_patches(f, d, eps_xi_eta, eps_xy);

    x_min = curve_seq.first_curve.l_1(0);
    x_max = curve_seq.first_curve.l_1(0);
    y_min = curve_seq.first_curve.l_2(0);
    y_max = curve_seq.first_curve.l_2(0);

    FC_patches = cell(4 * curve_seq.n_curves, 1);
    for i = 1:curve_seq.n_curves
        FC_patches{4*i-3} = interior_patches{2*i-1}.FC(C_S, n_r, d, A_S, Q_S);
        [FC_patches{4*i-2}, FC_patches{4*i-1}, FC_patches{4*i}] = interior_patches{2*i}.FC(C_C, n_r, d, A_C, Q_C, M);

        for j = 0:3
            x_min = min(FC_patches{4*i-j}.x_min, x_min);
            x_max = max(FC_patches{4*i-j}.x_max, x_max);
            y_min = min(FC_patches{4*i-j}.y_min, y_min);
            y_max = max(FC_patches{4*i-j}.y_max, y_max);
        end
    end

    [boundary_X, boundary_Y] = curve_seq.construct_boundary_mesh(n_r * 20);

    % Expand the bounding box to prevent grid points from landing exactly on
    % the domain boundary (inpolygon_mesh ambiguity). With perturb=true, each
    % side is expanded by an independent rand(1)*h; otherwise by the fixed h.
    if nargin >= 17 && perturb
        dx = rand(1)*h;  dy = rand(1)*h;
    else
        dx = h;  dy = h;
    end

    use_padding = nargin >= 16 && ~isempty(n_x_padded);
    if use_padding
        R = R_cartesian_mesh_obj( ...
            x_min - dx, x_max + dx, ...
            y_min - dy, y_max + dy, ...
            h, boundary_X, boundary_Y, n_x_padded, n_y_padded);
    else
        R = R_cartesian_mesh_obj( ...
            x_min - dx, x_max + dx, ...
            y_min - dy, y_max + dy, ...
            h, boundary_X, boundary_Y);
    end

    for i = 1:length(FC_patches)
        R.interpolate_patch(FC_patches{i}, n_r, M);
    end

    R.fill_interior(f);
    R.compute_fc_coeffs();

    % Evaluate error on a grid that is 2x finer than R.
    [R_X_err, R_Y_err, f_interpolation, interior_idx] = R.ifft_interpolation(2);
    f_exact = f(R_X_err, R_Y_err);

    abs_err = max(abs(f_exact(interior_idx) - f_interpolation(interior_idx)));
    disp(['abs max error: ', num2str(abs_err)]);

    rel_max_err = abs_err / max(abs(f_exact(interior_idx)));
    disp(['rel max error: ', num2str(rel_max_err)]);

    fc_err = sqrt(sum((f_exact(interior_idx) - f_interpolation(interior_idx)).^2) ...
                  ./ sum(f_exact(interior_idx).^2));
    disp(['rel L2 error:  ', num2str(fc_err)]);
end
