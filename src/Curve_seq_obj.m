% CURVE_SEQ_OBJ  Circular linked list of Curve_obj objects describing a closed domain.
%
% A Curve_seq_obj stores a sequence of k C^2 curves (c_1, c_2, ..., c_k) that
% together form the closed boundary of a 2D domain. The curves must be ordered
% so that the end of each curve connects to the start of the next, and the last
% curve connects back to the first (counter-clockwise orientation).
%
% Properties:
%   first_curve - Handle to the first Curve_obj in the sequence
%   last_curve  - Handle to the most recently added Curve_obj
%   n_curves    - Total number of curves in the sequence
%
% Methods:
%   Curve_seq_obj          - Constructor (creates an empty sequence)
%   add_curve              - Appends a new curve to the sequence
%   construct_patches      - Builds all boundary patches and applies POU normalization
%   plot_geometry          - Plots all patch boundaries for visual inspection
%   construct_boundary_mesh - Returns a discretized polygon of the full boundary
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef Curve_seq_obj < handle
    properties
        first_curve   % First Curve_obj in the circular linked list
        last_curve    % Most recently added Curve_obj
        n_curves      % Number of curves in the sequence
    end

    methods
        function obj = Curve_seq_obj()
            % CURVE_SEQ_OBJ  Constructor. Creates an empty curve sequence.
            obj.first_curve = [];
            obj.last_curve  = [];
            obj.n_curves    = 0;
        end

        function add_curve(obj, l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
                n, frac_n_C_0, frac_n_C_1, frac_n_S_0, frac_n_S_1, h_norm)
            % ADD_CURVE  Appends a new C^2 curve to the end of the sequence.
            %
            % The new curve is always linked to first_curve as its next_curve to
            % maintain the circular list structure.
            %
            % Inputs: see Curve_obj constructor for parameter descriptions.

            new_curve = Curve_obj(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
                n, frac_n_C_0, frac_n_C_1, frac_n_S_0, frac_n_S_1, h_norm, obj.first_curve);

            if isempty(obj.first_curve)
                obj.first_curve = new_curve;
                obj.last_curve  = new_curve;
            end

            obj.last_curve.next_curve = new_curve;
            obj.last_curve = new_curve;
            obj.n_curves   = obj.n_curves + 1;
        end

        function patches = construct_patches(obj, f, d, eps_xi_eta, eps_xy)
            % CONSTRUCT_PATCHES  Builds all boundary patches and normalizes the POU.
            %
            % For each curve, constructs one S-type (smooth) patch and one C-type
            % (corner) patch. Then applies partition-of-unity (POU) normalization
            % between adjacent patches so that the sum of all window functions equals
            % 1 everywhere in the overlap regions.
            %
            % POU normalization order (applied between consecutive patches):
            %   C_patch_{i-1}.apply_w_W(S_patch_{i-1})  - normalize W against its S-patch
            %   C_patch_{i-1}.apply_w_L(S_patch_i)       - normalize L against next S-patch
            % The loop applies these after the second curve, then the final pair is
            % handled outside the loop.
            %
            % Inputs:
            %   f          - Function handle f(x,y) to sample on each patch
            %   d          - Number of Gram matching points
            %   eps_xi_eta - Newton inversion tolerance in parameter space
            %   eps_xy     - Newton inversion tolerance in physical space
            %
            % Outputs:
            %   patches - (2*n_curves x 1) cell array alternating {S_patch, C_patch}
            %             for each curve

            curr         = obj.first_curve;
            prev_S_patch = nan;
            prev_C_patch = nan;
            patches      = cell(obj.n_curves * 2, 1);

            for i = 1:obj.n_curves
                S_patch = curr.construct_S_patch(f, d, eps_xi_eta, eps_xy);
                C_patch = curr.construct_C_patch(f, d, eps_xi_eta, eps_xy);

                if i ~= 1
                    prev_C_patch.apply_w_W(prev_S_patch);
                    prev_C_patch.apply_w_L(S_patch);
                end

                patches{2*i-1} = S_patch;
                patches{2*i}   = C_patch;

                prev_S_patch = S_patch;
                prev_C_patch = C_patch;
                curr = curr.next_curve;
            end

            % Close the POU normalization loop for the last curve
            C_patch.apply_w_W(S_patch);
            C_patch.apply_w_L(patches{1});
        end

        function plot_geometry(obj, d)
            % PLOT_GEOMETRY  Plots all patch boundary outlines for debugging.
            %
            % Constructs each S and C patch with f = 0 (no function values needed)
            % and plots their boundary meshes in the current figure.
            %
            % Input:
            %   d - Number of Gram matching points (needed to size patch meshes)

            curr = obj.first_curve;
            figure;

            for i = 1:obj.n_curves
                S_patch = curr.construct_S_patch(@(x, y) zeros(size(x)), d, nan, nan);
                C_patch = curr.construct_C_patch(@(x, y) zeros(size(x)), d, nan, nan);

                [boundary_X, boundary_Y] = S_patch.Q.boundary_mesh_xy(1, false);
                plot(boundary_X(:), boundary_Y(:));
                hold on;

                [boundary_X, boundary_Y] = C_patch.L.boundary_mesh_xy(1, false);
                plot(boundary_X(:), boundary_Y(:));

                [boundary_X, boundary_Y] = C_patch.W.boundary_mesh_xy(1, false);
                plot(boundary_X(:), boundary_Y(:));

                curr = curr.next_curve;
            end

            [boundary_X, boundary_Y] = obj.construct_boundary_mesh(1);
            plot(boundary_X, boundary_Y);
        end

        function [boundary_X, boundary_Y] = construct_boundary_mesh(obj, n_r)
            % CONSTRUCT_BOUNDARY_MESH  Returns a discretized polygon of the full boundary.
            %
            % Concatenates the parametrized boundary points from all curves into a
            % single closed polygon. Used for inpolygon_mesh tests and for plotting.
            %
            % Input:
            %   n_r - Refinement factor: each curve segment uses (n-1)*n_r + 1 points
            %
            % Outputs:
            %   boundary_X - ((sum (n_i-1)*n_r + 1) x 1) x-coordinates
            %   boundary_Y - ((sum (n_i-1)*n_r + 1) x 1) y-coordinates

            curr     = obj.first_curve;
            n_points = 0;
            for i = 1:obj.n_curves
                n_points = n_points + (curr.n - 1)*n_r + 1;
                curr = curr.next_curve;
            end

            boundary_X = zeros(n_points, 1);
            boundary_Y = zeros(n_points, 1);
            curr_idx   = 1;
            curr       = obj.first_curve;

            for i = 1:obj.n_curves
                n_seg = (curr.n - 1)*n_r;
                boundary_X(curr_idx : curr_idx + n_seg) = ...
                    curr.l_1(linspace(0, 1, n_seg + 1).');
                boundary_Y(curr_idx : curr_idx + n_seg) = ...
                    curr.l_2(linspace(0, 1, n_seg + 1).');

                curr_idx = curr_idx + n_seg + 1;
                curr     = curr.next_curve;
            end
        end
    end
end
