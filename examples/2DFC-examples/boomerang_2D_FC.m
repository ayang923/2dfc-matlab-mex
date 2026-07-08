% BOOMERANG_2D_FC  2DFC example: boomerang-shaped domain.
%
% Demonstrates the 2DFC algorithm on a single smooth closed curve that traces
% out a boomerang shape. The curve is parameterized as:
%
%   l_1(theta) = -2/3 * sin(3*pi*theta)
%   l_2(theta) = beta * sin(2*pi*theta)
%
% where beta = tan(alpha*pi/2) with alpha = 3/2.
%
% The test function is:
%   f(x,y) = exp(0.5*(x.^2+y.^2)).*(sin(10*pi*x)+cos(10*pi*y))

clc; clear; close all;

d   = 5;
h   = 2.5e-3;
n_x = 972;
n_y = 972;


f = @(x, y) exp(0.5*(x.^2+y.^2)).*(sin(10*pi*x)+cos(10*pi*y));

C   = 27;
n_r = 6;
M   = d + 3;

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

% Boomerang curve parametrization
alpha = 3/2;
beta  = tan(alpha * pi / 2);

l_1       = @(theta)  -2/3 * sin(theta * 3*pi);
l_2       = @(theta)   beta * sin(theta * 2*pi);
l_1_prime = @(theta)  -2*pi * cos(theta * 3*pi);
l_2_prime = @(theta)   2*pi*beta * cos(theta * 2*pi);
l_1_dprime = @(theta)  6*pi^2 * sin(theta * 3*pi);
l_2_dprime = @(theta) -4*pi^2*beta * sin(theta * 2*pi);

curve_seq = Curve_seq_obj();
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    0, 0.1, 0.1, 0.67, 0.67, h);

[R, ~, FC_patches, ~] = FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, C, A, Q, M, n_x, n_y);