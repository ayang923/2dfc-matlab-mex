% fcont_gram_blend_C2 : This routine computes the Fourier continuation in
% parameter space for a C2 type patch, for a function over the unit square
% domain. This consists blending upwards before blending to the left (along
% the x axis, and then the y axis)
%
% Inputs: 
%   f_xy : (real) values of the functions f over the unit square domain given
%   a 2D matrix
%   d : number of Gram interpolation points 
%   A : matrix containing the values of the extended Gram polynomials at 
%   the continuation points
%   Q : matrix containing the values of the Gram polynomials at the first 
%        d points of the interval. 

%
% Outputs:
%   fcont : (real) vector of size lenfgth(fx) + C containing the values of
%   the fx and its continuation on the extended grid

function [fcont] = fcont_gram_blend_C2(f_xy, d, A, Q)
% 1st step FC (along x axis)
fcont1 = fcont_gram_blend_S(f_xy, d, A, Q); % 

% 2nd step FC(along y axis)
fcont = transpose(fcont_gram_blend_S(fcont1', d, A, Q));
end

