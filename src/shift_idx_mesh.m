function idx_mesh = shift_idx_mesh(idx_mesh, min_bound, max_bound)
% SHIFT_IDX_MESH  Shifts an integer index window to stay within [min_bound, max_bound].
%
% When a symmetric M-point interpolation stencil is centered near the edge of
% a patch, some indices fall out of range. This function translates the entire
% stencil so that the lowest index equals min_bound (if it underflows) or the
% highest index equals max_bound (if it overflows). Only one end can be out of
% range at a time, since the stencil width is assumed smaller than the domain.
%
% Inputs:
%   idx_mesh  - Integer index vector representing the stencil
%   min_bound - Minimum allowed index value
%   max_bound - Maximum allowed index value
%
% Outputs:
%   idx_mesh  - Shifted index vector within [min_bound, max_bound]

    if idx_mesh(1) < min_bound
        idx_mesh = idx_mesh + (min_bound - idx_mesh(1));
    end
    if idx_mesh(end) > max_bound
        idx_mesh = idx_mesh + (max_bound - idx_mesh(end));
    end
end
