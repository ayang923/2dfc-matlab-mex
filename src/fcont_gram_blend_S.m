function [fcont] = fcont_gram_blend_S(fx, d, A, Q)
% FCONT_GRAM_BLEND_S  1D blending-to-zero Fourier continuation for S-type patches.
%
% Projects the first d rows of fx onto the Gram polynomial basis Q, then
% evaluates the extended Gram polynomials at the C*n_r continuation points
% via A. The result is a smooth, high-order extension of fx that decays to
% zero, prepended to the boundary row fx(1,:).
%
% This routine operates on 2D arrays: each column is an independent 1D signal,
% and the FC extension is applied row-wise (i.e., along the eta direction).
%
% Inputs:
%   fx - (n_eta x n_xi) matrix of function values; rows are indexed in
%        increasing eta, so fx(1,:) is the boundary row (eta = 0)
%   d  - Number of Gram matching points (rows of fx used for projection)
%   A  - (n_r*C x d) matrix mapping Gram coefficients to continuation values
%   Q  - (d x d) orthonormal Gram polynomial matrix
%
% Outputs:
%   fcont - ((n_r*C + 1) x n_xi) matrix: the C*n_r extension rows followed
%           by the boundary row fx(1,:)

    if use_mex_2dfc() && isa(fx, 'double') && isa(A, 'double') && isa(Q, 'double')
        fcont = fcont_gram_blend_S_mex(fx, d, A, Q);
        return
    end

    fl = fx(1:d, :);
    fc = flipud(A * (Q.' * flipud(fl)));
    fc = double(fc);
    fcont = [fc; fx(1, :)];
end
