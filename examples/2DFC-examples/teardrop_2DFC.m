% TEARDROP_2DFC  2DFC example: teardrop-shaped domain.
%
% Demonstrates the 2DFC algorithm on a single smooth closed curve tracing
% a teardrop shape. The curve is parameterized as:
%
%   l_1(theta) =  2 * sin(pi*theta)
%   l_2(theta) = -beta * sin(2*pi*theta)
%
% where beta = tan(alpha*pi/2) with alpha = 0.5.
% The two tips of the teardrop lie at theta=0 (theta=1) and theta=0.5.
%
% The test function is:
%   f(x,y) = -exp(0.5*(x^2+y^2)) * (sin(3*pi*x) + cos(2.5*pi*y))

clc; clear; close all;

d   = 4;
h   = 0.01;
n_x = 384;
n_y = 288;

f = @(x, y) exp(0.5*(x.^2 + y.^2)) .* (sin(10*pi*x) + cos(10*pi*y));

alpha = 0.5;
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

h_norm  = 1.5 * h;
n_frac_C = 0.1;
n_frac_S = 0.6;
n_curve  = 0;   % auto-compute from arc length

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

tic;
R = FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, C, A, Q, M, n_x, n_y);
toc
