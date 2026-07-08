% C1_PATCH_OBJ  Concave corner (C1-type) boundary patch for the 2DFC algorithm.
%
% A C1-type patch covers a concave corner of the domain (interior angle > 180
% degrees, i.e. a reflex corner). The parametrization is:
%
%   M_p(xi, eta) = l_curr(xi_tilde(xi)) + l_next(eta_tilde(eta)) - corner_point
%
% The patch is split into two overlapping sub-patches:
%
%   L ("long") - xi in [1/2, 1/2+(d-1)*h_xi], eta in [0, 1/2+(d-1)*h_eta]
%   W ("wide") - xi in [0,   1/2],             eta in [1/2, 1/2+(d-1)*h_eta]
%
% Note: n_xi and n_eta must be odd so that xi=1/2 and eta=1/2 fall exactly
% on mesh points, ensuring clean overlap between L and W.
%
% The FC extension produces three Q_patch_obj regions:
%   fcont_L         - extension of L leftward (xi < 1/2)
%   fcont_W_refined   - extension of W in the region that overlaps with fcont_L
%   fcont_W_unrefined - extension of W in the non-overlapping region
%
% Properties:
%   L - Q_patch_obj for the long sub-patch
%   W - Q_patch_obj for the wide sub-patch
%
% Methods:
%   C1_patch_obj - Constructor
%   FC           - Computes the blending-to-zero FC extension
%   refine_W     - Upsample W function values by n_r using Barycentric interpolation
%   apply_w_W    - Applies POU normalization to W against a window patch
%   apply_w_L    - Applies POU normalization to L against a window patch
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef C1_patch_obj < handle
    properties
        L   % Q_patch_obj: long sub-patch (right arm of corner)
        W   % Q_patch_obj: wide sub-patch (top arm of corner)
    end

    methods
        function obj = C1_patch_obj(M_p, J, eps_xi_eta, eps_xy, n_xi, n_eta, d, f_L, f_W, mex_spec)
            % C1_PATCH_OBJ  Constructor.
            %
            % Inputs:
            %   M_p       - Parametrization handle (xi,eta) -> (x,y)
            %   J         - Jacobian handle of M_p
            %   eps_xi_eta - Newton inversion tolerance in (xi,eta) space
            %   eps_xy    - Newton inversion tolerance in (x,y) space
            %   n_xi      - Number of discretization points along full xi axis
            %               (must be odd so that xi=1/2 is a mesh point)
            %   n_eta     - Number of discretization points along full eta axis
            %               (must be odd so that eta=1/2 is a mesh point)
            %   d         - Number of Gram matching points
            %   f_L       - Function values for sub-patch L, or nan to defer
            %   f_W       - Function values for sub-patch W, or nan to defer
            %   mex_spec  - (optional) mex_curve_spec shared by L and W (both
            %               sub-patches use the same corner M_p/J); see
            %               Curve_obj.construct_C_patch

            if nargin < 10
                mex_spec = [];
            end

            assert(mod(n_xi, 2) == 1 && mod(n_eta, 2) == 1, ...
                'n_xi and n_eta must be odd so that xi=1/2 and eta=1/2 are mesh points');

            h_xi  = 1 / (n_xi - 1);
            h_eta = 1 / (n_eta - 1);

            obj.L = Q_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                d, (n_eta+1)/2 + (d-1), ...
                1/2, 1/2 + (d-1)*h_xi, ...
                0,   1/2 + (d-1)*h_eta, f_L, mex_spec);

            obj.W = Q_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                (n_xi+1)/2, d, ...
                0,   1/2, ...
                1/2, 1/2 + (d-1)*h_eta, f_W, mex_spec);
        end

        function [C1_fcont_patch_L, C1_fcont_patch_W_refined, C1_fcont_patch_W_unrefined] = FC(obj, C, n_r, d, A, Q, M)
            % FC  Computes the blending-to-zero FC extension for this C1 patch.
            %
            % Procedure:
            %   1. Extend L leftward (along xi) via fcont_gram_blend_S on rows.
            %   2. Refine W function values onto the n_r-finer mesh in the region
            %      that will overlap with the L extension (using refine_W).
            %   3. Subtract the L extension values from the refined W values in the
            %      overlap region (so the combined L+W extension is consistent).
            %   4. Extend the overlap-subtracted W downward (along eta).
            %   5. Also extend the non-overlapping part of W downward.
            %
            % Inputs:
            %   C   - Number of unrefined continuation points
            %   n_r - Refinement factor for the continuation grid
            %   d   - Number of Gram matching points
            %   A   - (n_r*C x d) FC continuation matrix
            %   Q   - (d x d) Gram polynomial matrix
            %   M   - Polynomial interpolation degree used in refine_W
            %
            % Outputs:
            %   C1_fcont_patch_L         - Q_patch_obj: leftward extension of L
            %   C1_fcont_patch_W_refined   - Q_patch_obj: downward extension of the
            %                               overlap region of W (xi near 1/2)
            %   C1_fcont_patch_W_unrefined - Q_patch_obj: downward extension of the
            %                               non-overlap region of W (xi near 0)

            L_f_XY = obj.L.f_XY;
            W_f_XY = obj.W.f_XY;

            [h_xi, ~] = obj.L.h_mesh();

            % Step 1: extend L leftward
            L_fcont = transpose(fcont_gram_blend_S(L_f_XY.', d, A, Q));
            C1_fcont_patch_L = Q_patch_obj(obj.L.M_p, obj.L.J, ...
                obj.L.eps_xi_eta, obj.L.eps_xy, ...
                C*n_r + 1, obj.L.n_eta - (d-1), ...
                1/2 - C*h_xi, 1/2, ...
                0, 1/2, ...
                L_fcont(1:(obj.L.n_eta - (d-1)), :), obj.L.mex_spec);

            if 1/2 - C*h_xi < 0
                error('C1 patch extension overflows xi < 0; please refine the mesh or reduce C.');
            end

            % Steps 2-5: extend W
            [W_unrefined_f_XY, W_refined_f_XY] = obj.refine_W(W_f_XY, C, n_r, M);

            % Subtract L extension from W in the overlap region to avoid double-counting
            W_minus_fcont = W_refined_f_XY - L_fcont(end-(d-1):end, :);

            W_fcont_refined   = fcont_gram_blend_S(W_minus_fcont, d, A, Q);
            W_fcont_unrefined = fcont_gram_blend_S(W_unrefined_f_XY, d, A, Q);

            [h_xi_W, h_eta_W] = obj.W.h_mesh();

            C1_fcont_patch_W_refined = Q_patch_obj(obj.W.M_p, obj.W.J, ...
                obj.W.eps_xi_eta, obj.W.eps_xy, ...
                C*n_r + 1, C*n_r + 1, ...
                1/2 - C*h_xi,  1/2, ...
                1/2 - C*h_eta_W, 1/2, ...
                W_fcont_refined, obj.W.mex_spec);

            C1_fcont_patch_W_unrefined = Q_patch_obj(obj.W.M_p, obj.W.J, ...
                obj.W.eps_xi_eta, obj.W.eps_xy, ...
                obj.W.n_xi - C, C*n_r + 1, ...
                0, 1/2 - C*h_xi, ...
                1/2 - C*h_eta_W, 1/2, ...
                W_fcont_unrefined, obj.W.mex_spec);
        end

        function [W_unrefined_f_XY, W_refined_f_XY] = refine_W(obj, W_f_XY, C, n_r, M)
            % REFINE_W  Upsample W function values by n_r using Barycentric interpolation.
            %
            % In the C columns of W nearest to xi=1/2 (the overlap with L), the W
            % mesh is too coarse to match the n_r-refined L extension. This method
            % refines those C columns to a grid n_r times finer using row-by-row
            % Barycentric Lagrange interpolation.
            %
            % Inputs:
            %   W_f_XY - (n_eta x n_xi_W) function values on the W mesh
            %   C      - Number of columns to refine (nearest to xi=1/2)
            %   n_r    - Upsampling factor
            %   M      - Number of interpolation points per Barycentric stencil
            %
            % Outputs:
            %   W_unrefined_f_XY - (n_eta x (n_xi_W - C)) left part of W (unchanged)
            %   W_refined_f_XY   - (n_eta x (C*n_r + 1)) refined right part of W

            [h_xi, ~] = obj.W.h_mesh();

            W_unrefined_f_XY = W_f_XY(:, 1:(obj.W.n_xi - C));

            W_refined_xi_mesh = transpose((1/2 - C*h_xi) : (h_xi/n_r) : 1/2);

            W_refined_f_XY = zeros(obj.W.n_eta, length(W_refined_xi_mesh));
            half_M = floor(M / 2);

            for eta_j = 0:obj.W.n_eta - 1
                for xi_j = (obj.W.n_xi - C - 1) : (obj.W.n_xi - 2)
                    if mod(M, 2) ~= 0
                        interpol_xi_j_mesh = transpose(xi_j - half_M : xi_j + half_M);
                    else
                        interpol_xi_j_mesh = transpose(xi_j - half_M + 1 : xi_j + half_M);
                    end

                    interpol_xi_j_mesh = shift_idx_mesh(interpol_xi_j_mesh, 0, obj.W.n_xi - 1);
                    interpol_xi_mesh   = h_xi * interpol_xi_j_mesh;
                    interpol_val       = W_f_XY(eta_j + 1, interpol_xi_j_mesh + 1).';

                    xi_eval_j = ((xi_j - (obj.W.n_xi - C - 1))*n_r) : ...
                                ((xi_j - (obj.W.n_xi - C - 1) + 1)*n_r);

                    W_refined_f_XY(eta_j + 1, xi_eval_j + 1) = ...
                        barylag([interpol_xi_mesh, interpol_val], W_refined_xi_mesh(xi_eval_j + 1));
                end
            end
        end

        function apply_w_W(obj, window_patch_W)
            % APPLY_W_W  Applies POU normalization to W against a window S-patch.
            %
            % The window patch is the S-type patch from the previous curve; it
            % overlaps W from the right (larger xi side, near xi=1/2).
            %
            % Input:
            %   window_patch_W - S_patch_obj acting as the POU window for W
            obj.W.apply_w_normalization_xi_left(window_patch_W.Q);
        end

        function apply_w_L(obj, window_patch_L)
            % APPLY_W_L  Applies POU normalization to L against a window S-patch.
            %
            % The window patch is the S-type patch from the next curve; it
            % overlaps L from below (smaller eta side, near eta=1/2).
            %
            % Input:
            %   window_patch_L - S_patch_obj acting as the POU window for L
            obj.L.apply_w_normalization_eta_down(window_patch_L.Q);
        end
    end
end
