% HEART_SHARP_2DFC  2DFC example: near-cusp heart-shaped domain.
%
% Demonstrates the 2DFC algorithm on a heart-shaped curve with alpha = 1.99,
% which produces a near-cusp indentation at the top of the heart. The curve
% is parameterized as a composition of circular arcs:
%
%   l_1(theta) = beta*cos((1+alpha)*pi*theta) - sin((1+alpha)*pi*theta) - beta
%   l_2(theta) = beta*sin((1+alpha)*pi*theta) + cos((1+alpha)*pi*theta) - cos(pi*theta)
%
% where beta = tan(alpha*pi/2).
%
% The test function is:
%   f(x,y) = 4 + (1 + x^2 + y^2) * (sin(2.5*pi*x - 0.5) + cos(2*pi*y - 0.5))
%
% Parameters: d=7, C=27, n_r=6, h=2e-4

clc; clear; close all;

d   = 7;
h   = 0.0002;
n_x = 10368;
n_y = 16384;

f = @(x, y) (y - 1.0).^2 .* cos(4*pi*x);

alpha = 1.99;
beta  = tan(alpha * pi / 2);

l_1 = @(theta)  beta*cos((1+alpha)*pi*theta) - sin((1+alpha)*pi*theta) - beta;
l_2 = @(theta)  beta*sin((1+alpha)*pi*theta) + cos((1+alpha)*pi*theta) - cos(pi*theta);

l_1_prime = @(theta) -beta*(1+alpha)*pi*sin((1+alpha)*pi*theta) ...
                     - (1+alpha)*pi*cos((1+alpha)*pi*theta);
l_2_prime = @(theta)  beta*(1+alpha)*pi*cos((1+alpha)*pi*theta) ...
                     - (1+alpha)*pi*sin((1+alpha)*pi*theta) ...
                     + pi*sin(pi*theta);

l_1_dprime = @(theta) -beta*(1+alpha)^2*pi^2*cos((1+alpha)*pi*theta) ...
                      + (1+alpha)^2*pi^2*sin((1+alpha)*pi*theta);
l_2_dprime = @(theta) -beta*(1+alpha)^2*pi^2*sin((1+alpha)*pi*theta) ...
                      - (1+alpha)^2*pi^2*cos((1+alpha)*pi*theta) ...
                      + pi^2*cos(pi*theta);


C   = 27;
n_r = 6;
M   = d + 3;

h_norm = 5.5 * h;
h_tan   = h_norm;
n_frac_C = 0.1;
n_frac_S = 0.6;
n_curve  = ceil(3.5 / h_norm);

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
