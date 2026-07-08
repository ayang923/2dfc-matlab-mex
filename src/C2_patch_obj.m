% C2_PATCH_OBJ  Convex corner (C2-type) boundary patch for the 2DFC algorithm.
%
% A C2-type patch covers a convex corner of the domain (interior angle < 180
% degrees). The C2-type corner patch parametrization is:
%
%   M_p(xi, eta) = l_curr(xi_tilde(xi)) + l_next(eta_tilde(eta)) - corner_point
%
% where xi runs along the current curve near theta=1 and eta runs along the
% next curve near theta=0. The patch is split into two overlapping sub-patches:
%
%   W ("wide")  - xi in [0,1],          eta in [0, (d-1)*h_eta]  (bottom strip)
%   L ("long")  - xi in [0, (d-1)*h_xi], eta in [(d-1)*h_eta, 1] (left strip)
%
% The FC extension produces three Q_patch_obj regions:
%   fcont_W      - extension of W downward (eta < 0)
%   fcont_L      - extension of L leftward (xi < 0)
%   fcont_corner - 2D corner extension (xi < 0 and eta < 0)
%
% Properties:
%   L - Q_patch_obj for the long sub-patch
%   W - Q_patch_obj for the wide sub-patch
%
% Methods:
%   C2_patch_obj        - Constructor
%   FC                  - Computes the blending-to-zero FC extension
%   apply_w_W           - Applies POU normalization to W against a window patch
%   apply_w_L           - Applies POU normalization to L against a window patch
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef C2_patch_obj < handle
    properties
        L   % Q_patch_obj: long sub-patch (left strip near corner)
        W   % Q_patch_obj: wide sub-patch (bottom strip near corner)
    end

    methods
        function obj = C2_patch_obj(M_p, J, eps_xi_eta, eps_xy, n_xi, n_eta, d, f_L, f_W, mex_spec)
            % C2_PATCH_OBJ  Constructor.
            %
            % Inputs:
            %   M_p       - Parametrization handle (xi,eta) -> (x,y)
            %   J         - Jacobian handle of M_p
            %   eps_xi_eta - Newton inversion tolerance in (xi,eta) space
            %   eps_xy    - Newton inversion tolerance in (x,y) space
            %   n_xi      - Number of discretization points along xi (for W)
            %   n_eta     - Number of discretization points along eta (for L+W combined)
            %   d         - Number of Gram matching points
            %   f_L       - Function values for sub-patch L, or nan to defer
            %   f_W       - Function values for sub-patch W, or nan to defer
            %   mex_spec  - (optional) mex_curve_spec shared by L and W (both
            %               sub-patches use the same corner M_p/J); see
            %               Curve_obj.construct_C_patch

            if nargin < 10
                mex_spec = [];
            end

            h_xi  = 1 / (n_xi - 1);
            h_eta = 1 / (n_eta - 1);

            obj.W = Q_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                n_xi, d, 0, 1, 0, (d-1)*h_eta, f_W, mex_spec);
            obj.L = Q_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                d, n_eta-d+1, 0, (d-1)*h_xi, (d-1)*h_eta, 1, f_L, mex_spec);
        end

        function [C2_fcont_patch_L, C2_fcont_patch_W, C2_fcont_patch_corner] = FC(obj, C, n_r, d, A, Q, M)
            % FC  Computes the blending-to-zero FC extension for this C2 patch.
            %
            % Three-step procedure:
            %   1. Apply fcont_gram_blend_S to W row-wise -> fcont_W (extend downward)
            %   2. Apply fcont_gram_blend_S to fcont_W column-wise -> fcont_corner (2D corner)
            %   3. Apply fcont_gram_blend_S column-wise to [W(:,1:d); L(2:end,:)] -> fcont_L
            %
            % Inputs:
            %   C   - Number of unrefined continuation points
            %   n_r - Refinement factor for the continuation grid
            %   d   - Number of Gram matching points
            %   A   - (n_r*C x d) FC continuation matrix
            %   Q   - (d x d) Gram polynomial matrix
            %   M   - Polynomial interpolation degree (unused here; kept for interface parity)
            %
            % Outputs:
            %   C2_fcont_patch_L      - Q_patch_obj: leftward extension of L
            %   C2_fcont_patch_W      - Q_patch_obj: downward extension of W
            %   C2_fcont_patch_corner - Q_patch_obj: 2D corner extension

            [h_xi, h_eta] = obj.W.h_mesh();   % h_xi and h_eta are identical for W and L
            f_XY_W = obj.W.f_XY;
            f_XY_L = obj.L.f_XY;

            % Step 1: extend W downward (along eta)
            fcont_W = fcont_gram_blend_S(f_XY_W, d, A, Q);

            % Step 2: extend the W extension leftward (along xi) -> 2D corner
            fcont_corner = transpose(fcont_gram_blend_S(fcont_W.', d, A, Q));

            % Step 3: extend [bottom rows of W stacked above L] leftward -> L extension
            fcont_L = transpose(fcont_gram_blend_S( ...
                [f_XY_W(:, 1:d); f_XY_L(2:end, :)].', d, A, Q));

            C2_fcont_patch_W = Q_patch_obj(obj.W.M_p, obj.W.J, ...
                obj.W.eps_xi_eta, obj.W.eps_xy, ...
                obj.W.n_xi, C*n_r + 1, ...
                0, 1, -C*h_eta, 0, fcont_W, obj.W.mex_spec);

            C2_fcont_patch_L = Q_patch_obj(obj.L.M_p, obj.L.J, ...
                obj.L.eps_xi_eta, obj.L.eps_xy, ...
                C*n_r + 1, obj.L.n_eta + d - 1, ...
                -C*h_xi, 0, 0, 1, fcont_L, obj.L.mex_spec);

            C2_fcont_patch_corner = Q_patch_obj(obj.W.M_p, obj.W.J, ...
                obj.W.eps_xi_eta, obj.W.eps_xy, ...
                C*n_r + 1, C*n_r + 1, ...
                -C*h_xi, 0, -C*h_eta, 0, fcont_corner, obj.W.mex_spec);
        end

        function apply_w_W(obj, window_patch_W)
            % APPLY_W_W  Applies POU normalization to W against a window S-patch.
            %
            % The window patch is the S-type patch from the current curve; it
            % overlaps W from the right (larger xi side).
            %
            % Input:
            %   window_patch_W - S_patch_obj acting as the POU window for W
            obj.W.apply_w_normalization_xi_right(window_patch_W.Q);
        end

        function apply_w_L(obj, window_patch_L)
            % APPLY_W_L  Applies POU normalization to L against a window S-patch.
            %
            % The window patch is the S-type patch from the next curve; it
            % overlaps L from below (smaller eta side).
            %
            % Input:
            %   window_patch_L - S_patch_obj acting as the POU window for L
            obj.L.apply_w_normalization_eta_up(window_patch_L.Q);
        end
    end
end
