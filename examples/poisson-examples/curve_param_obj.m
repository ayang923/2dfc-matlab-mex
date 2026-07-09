classdef curve_param_obj < handle
% CURVE_PARAM_OBJ  Struct-like container for boundary curve discretization parameters.
%
% Stores the per-curve quadrature point counts together with derived index
% ranges used to assemble the global IE linear system.
%
% Properties:
%   curve_n           - (n_curves x 1) integer vector of quadrature points per curve
%   n_total           - Total number of quadrature points across all curves
%   start_idx         - (n_curves x 1) first global index for each curve
%   end_idx           - (n_curves x 1) last  global index for each curve
%   U_c_0_intervals   - (n_curves x 2) parameter intervals for concave corners (C0)
%   U_c_1_intervals   - (n_curves x 2) parameter intervals for convex  corners (C1)

    properties
        curve_n
        n_total
        start_idx
        end_idx

        U_c_0_intervals
        U_c_1_intervals
    end

    methods
        function obj = curve_param_obj(curve_n)
        % CURVE_PARAM_OBJ  Constructor.
        %
        % Input:
        %   curve_n - (n_curves x 1) vector of quadrature point counts per curve
            obj.curve_n  = curve_n;
            obj.n_total  = sum(curve_n);
            obj.start_idx = cumsum([1, curve_n(1:end-1)'])';
            obj.end_idx   = obj.start_idx + curve_n - 1;

            obj.U_c_0_intervals = zeros(length(curve_n), 2);
            obj.U_c_1_intervals = zeros(length(curve_n), 2);
        end
    end
end
