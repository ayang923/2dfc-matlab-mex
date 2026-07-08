function [in] = inpolygon_mesh(R_X, R_Y, boundary_x, boundary_y)
% INPOLYGON_MESH  Fast interior test for points on a uniform Cartesian mesh.
%
% Determines which grid points of the meshgrid (R_X, R_Y) lie inside the
% polygon defined by (boundary_x, boundary_y). Exploits the uniform spacing
% of the Cartesian mesh with a scanline / ray-casting algorithm:
%
%   1. For each boundary edge that crosses a horizontal grid line, record the
%      column index of the crossing ("toggle" cell).
%   2. Scan each row left-to-right, toggling an in/out flag at each recorded
%      crossing to fill the interior.
%
% This is substantially faster than MATLAB's built-in inpolygon for large
% uniform grids.
%
% Inputs:
%   R_X        - (n_y x n_x) meshgrid of x-coordinates
%   R_Y        - (n_y x n_x) meshgrid of y-coordinates
%   boundary_x - (n_edges+1 x 1) x-coordinates of polygon vertices (closed)
%   boundary_y - (n_edges+1 x 1) y-coordinates of polygon vertices (closed)
%
% Outputs:
%   in - (n_y x n_x) logical array; true for interior grid points

    if use_mex_2dfc()
        in = inpolygon_mesh_mex(R_X, R_Y, boundary_x, boundary_y);
        return
    end

    boundary_x_edge_1 = boundary_x(1:end-1);
    boundary_x_edge_2 = boundary_x(2:end);
    boundary_y_edge_1 = boundary_y(1:end-1);
    boundary_y_edge_2 = boundary_y(2:end);

    boundary_idxs = transpose(1:length(boundary_x_edge_1));

    x_start = R_X(1, 1);
    y_start = R_Y(1, 1);
    h_x     = R_X(1, 2) - R_X(1, 1);
    h_y     = R_Y(2, 1) - R_Y(1, 1);

    % Convert boundary y-coordinates to fractional row indices
    boundary_y_j         = (boundary_y - y_start) / h_y;
    boundary_y_edge_1_j  = boundary_y_j(1:end-1);
    boundary_y_edge_2_j  = boundary_y_j(2:end);

    % Snap values that are within floating-point noise of a grid line
    snap_1 = abs(boundary_y_edge_1_j - round(boundary_y_edge_1_j)) < eps;
    snap_2 = abs(boundary_y_edge_2_j - round(boundary_y_edge_2_j)) < eps;
    boundary_y_edge_1_j(snap_1) = round(boundary_y_edge_1_j(snap_1));
    boundary_y_edge_2_j(snap_2) = round(boundary_y_edge_2_j(snap_2));

    % Keep only edges that cross at least one horizontal grid line
    intersection_idxs = boundary_idxs(floor(boundary_y_edge_1_j) ~= floor(boundary_y_edge_2_j));

    % Phase 1: mark the first grid cell to the left of each edge-crossing
    in = false(size(R_X));
    for intersection_idx = intersection_idxs'
        x_edge_1   = boundary_x_edge_1(intersection_idx);
        x_edge_2   = boundary_x_edge_2(intersection_idx);
        y_edge_1   = boundary_y_edge_1(intersection_idx);
        y_edge_2   = boundary_y_edge_2(intersection_idx);
        y_edge_1_j = floor(boundary_y_edge_1_j(intersection_idx));
        y_edge_2_j = floor(boundary_y_edge_2_j(intersection_idx));

        % y-coordinates of the horizontal grid lines crossed by this edge
        intersection_mesh_y = ((min(y_edge_1_j, y_edge_2_j)+1) : max(y_edge_1_j, y_edge_2_j)) * h_y + y_start;

        % x-coordinate where the edge crosses each horizontal grid line
        intersection_x = x_edge_1 + (x_edge_2 - x_edge_1) .* ...
            (intersection_mesh_y - y_edge_1) ./ (y_edge_2 - y_edge_1);

        mesh_intersection_idxs = sub2ind(size(in), ...
            round((intersection_mesh_y - y_start) / h_y) + 1, ...
            floor((intersection_x - x_start) / h_x) + 1);

        in(mesh_intersection_idxs) = ~in(mesh_intersection_idxs);
    end

    % Phase 2: scanline fill — convert toggle markers to filled interior
    for vert_idx = 1:size(in, 1)
        in_interior = false;
        for horz_idx = 1:size(in, 2)
            if in(vert_idx, horz_idx) && ~in_interior
                in_interior = true;
                in(vert_idx, horz_idx) = false;
            elseif in(vert_idx, horz_idx) && in_interior
                in_interior = false;
            elseif in_interior
                in(vert_idx, horz_idx) = true;
            end
        end
    end
end
