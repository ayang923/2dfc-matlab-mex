% TEARDROP_SHARP_2DFC  2DFC example: extremely sharp teardrop domain.
%
% Same teardrop curve as teardrop_2DFC.m but with alpha = 0.01, giving an
% extremely narrow tip (beta ~ 0.016). The near-cusp geometry requires a
% very fine mesh (h = 2.5e-5) and small corner patch fraction.
%
%   l_1(theta) =  2 * sin(pi*theta)
%   l_2(theta) = -beta * sin(2*pi*theta),  beta = tan(alpha*pi/2)
%
% The test function is:
%   f(x,y) = -((x+1).^2 + (y+1).^2) .* sin(pi*(x-0.1)) .* cos(pi*y);

clc; clear; close all;

d   = 7;
h   = 0.000025;
n_x = 82944;
n_y = 1458;

f = @(x, y) -((x+1).^2+(y+1).^2).*sin(pi*(x-0.1)).*cos(pi*y);

alpha = 0.01;
beta  = tan(alpha * pi / 2);

l_1       = @(theta)  2 * sin(theta * pi);
l_2       = @(theta) -beta * sin(theta * 2*pi);
l_1_prime = @(theta)  2*pi * cos(theta * pi);
l_2_prime = @(theta) -2*beta*pi * cos(theta * 2*pi);
l_1_dprime = @(theta) -2*pi^2 * sin(theta * pi);
l_2_dprime = @(theta)  4*beta*pi^2 * sin(theta * 2*pi);

C   = 27;
n_r = 6;
M   = d + 3;

h_norm   = 1.5 * h;
h_tan    = 16 * h_norm;
n_frac_C = 0.025;
n_frac_S = 0.85;
n_curve  = ceil(4 / h_tan);

% Load or generate FC precomputation matrices
if exist(['FC_data/A_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']) == 0 || ...
   exist(['FC_data/Q_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']) == 0
    disp('FC data not found. Generating FC operators...');
    generate_bdry_continuations(d, C, C, 12, 20, 4, 256, n_r);
end

load(['FC_data/A_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']);
load(['FC_data/Q_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']);
A = double(A);
Q = double(Q);

curve_seq = Curve_seq_obj();
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C, n_frac_C, n_frac_S, n_frac_S, h_norm);

curve_seq.plot_geometry(d);
R = FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, C, A, Q, M, n_x, n_y);
