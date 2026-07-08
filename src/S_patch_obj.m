% S_PATCH_OBJ  Smooth (S-type) boundary patch for the 2DFC algorithm.
%
% An S-type patch covers a thin rectangular strip along a smooth portion of
% the domain boundary. The strip extends inward from the boundary in the
% normal direction (eta) by (d-1)*h_norm, and along the boundary in the
% tangential direction (xi) over [0,1], trimmed to avoid overlap with the
% corner patches on each end.
%
% The parametrization M_p maps:
%   xi  in [0,1]           -> along the boundary (tangential)
%   eta in [0, (d-1)*h]    -> inward from the boundary (normal)
%
% Properties:
%   Q - Underlying Q_patch_obj storing the mesh, parametrization, and
%       function values for this patch
%   h - Normal step size (mesh spacing in the eta direction)
%
% Methods:
%   S_patch_obj  - Constructor
%   FC           - Applies 1D blending-to-zero FC in the eta direction,
%                  returning a Q_patch_obj covering the extension region
%
% Author: Allen Yang
% Email:  aryang@caltech.edu

classdef S_patch_obj < handle
    properties
        Q   % Q_patch_obj storing the patch mesh and function values
        h   % Normal step size (eta mesh spacing in physical space)
    end

    methods
        function obj = S_patch_obj(M_p, J, h, eps_xi_eta, eps_xy, n_xi, d, f_XY, mex_spec)
            % S_PATCH_OBJ  Constructor.
            %
            % Creates an S-type patch with xi in [0,1] and eta in [0, (d-1)*h].
            %
            % Inputs:
            %   M_p       - Parametrization handle (xi,eta) -> (x,y)
            %   J         - Jacobian handle of M_p
            %   h         - Normal step size in physical space
            %   eps_xi_eta - Newton inversion tolerance in (xi,eta) space
            %   eps_xy    - Newton inversion tolerance in (x,y) space
            %   n_xi      - Number of discretization points along xi
            %   d         - Number of normal layers (Gram matching points)
            %   f_XY      - (d x n_xi) function values, or nan to defer
            %   mex_spec  - (optional) mex_curve_spec enabling mex-accelerated
            %               inverse_M_p for this patch and its FC extension;
            %               see Curve_obj.construct_S_patch

            if nargin < 9
                mex_spec = [];
            end

            obj.Q = Q_patch_obj(M_p, J, eps_xi_eta, eps_xy, ...
                n_xi, d, 0, 1, 0, h*(d-1), f_XY, mex_spec);
            obj.h = h;
        end

        function S_fcont_patch = FC(obj, C, n_r, d, A, Q)
            % FC  Applies 1D blending-to-zero FC in the normal (eta) direction.
            %
            % Applies fcont_gram_blend_S row-wise to the patch's function values
            % to produce C*n_r continuation values that decay smoothly to zero
            % below eta = 0 (i.e., outside the domain).
            %
            % Inputs:
            %   C   - Number of unrefined continuation points
            %   n_r - Refinement factor for the continuation grid
            %   d   - Number of Gram matching points
            %   A   - (n_r*C x d) FC continuation matrix
            %   Q   - (d x d) Gram polynomial matrix
            %
            % Outputs:
            %   S_fcont_patch - Q_patch_obj covering the extension region
            %                   eta in [-(C)*h_eta, 0], xi in [0,1]

            h_eta  = (obj.Q.eta_end - obj.Q.eta_start) / (obj.Q.n_eta - 1);
            fcont  = fcont_gram_blend_S(obj.Q.f_XY, d, A, Q);

            S_fcont_patch = Q_patch_obj(obj.Q.M_p, obj.Q.J, ...
                obj.Q.eps_xi_eta, obj.Q.eps_xy, ...
                obj.Q.n_xi, C*n_r + 1, ...
                obj.Q.xi_start, obj.Q.xi_end, ...
                obj.Q.eta_start - C*h_eta, obj.Q.eta_start, ...
                fcont, obj.Q.mex_spec);
        end
    end
end
