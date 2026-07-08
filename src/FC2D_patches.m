function FC2D_patches(interior_patches, R, f_interior, curve_seq, d, C_S, n_r, A_S, Q_S, C_C, A_C, Q_C, M)
% FC2D_PATCHES  Fills a pre-allocated Cartesian mesh using pre-built patches.
%
% This is a variant of FC2D intended for cases where the interior patches and
% Cartesian mesh R have already been constructed (e.g., when reusing the same
% geometry with different function values). It applies the 1D FC extensions,
% interpolates them onto R, fills the interior, and computes the FFT.
%
% Inputs:
%   interior_patches - Cell array of pre-built patch objects
%                      (2*n_curves entries, alternating S and C patches)
%   R                - Pre-constructed R_cartesian_mesh_obj to fill
%   f_interior       - Vector of exact f values at R's interior grid points
%   curve_seq        - Curve_seq_obj describing the domain boundary
%   d                - Number of Gram matching points for 1D FC
%   C_S              - Number of continuation points for smooth (S-type) patches
%   n_r              - Refinement factor for the FC continuation grid
%   A_S              - Precomputed FC matrix A for S-type patches
%   Q_S              - Precomputed FC matrix Q for S-type patches
%   C_C              - Number of continuation points for corner patches
%   A_C              - Precomputed FC matrix A for corner patches
%   Q_C              - Precomputed FC matrix Q for corner patches
%   M                - Degree of polynomial interpolation onto Cartesian mesh

    FC_patches = cell(4 * curve_seq.n_curves, 1);
    for i = 1:curve_seq.n_curves
        FC_patches{4*i-3} = interior_patches{2*i-1}.FC(C_S, n_r, d, A_S, Q_S);
        [FC_patches{4*i-2}, FC_patches{4*i-1}, FC_patches{4*i}] = ...
            interior_patches{2*i}.FC(C_C, n_r, d, A_C, Q_C, M);
    end

    for i = 1:length(FC_patches)
        R.interpolate_patch(FC_patches{i}, n_r, M);
    end

    R.f_R(R.in_interior) = f_interior;
    R.compute_fc_coeffs();
end
