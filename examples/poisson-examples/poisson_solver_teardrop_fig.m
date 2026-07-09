% POISSON_SOLVER_TEARDROP_FIG  Poisson BVP on a teardrop domain — publication figure.
%
% Same domain and problem as poisson_solver_teardrop.m, solved using the
% coarse-evaluation variant (h = 0.00125 fine, h_eval = 0.01) and rendered
% as a smooth surface plot.
%
% Domain:
%   Single closed teardrop curve
%   (l_1, l_2)(theta) = (2*sin(pi*theta), -sin(2*pi*theta)), theta in [0,1]
%
% Problem:
%   Delta(u) = f  in Omega,   u = u_boundary  on d(Omega)
%   f(x,y)          = (40*pi)^2 * sin(40*pi*x - 1) * sin(40*pi*y - 1)
%   u_boundary(x,y) = -1/2     * sin(40*pi*x - 1) * sin(40*pi*y - 1)
%
% Output:
%   Smooth surface (shading interp) of u on R_eval.
%
% Parameters:
%   h = 0.00125 (finest), h_eval = 0.01, d = 10

clc; clear; close all;

f          = @(x, y) (40*pi)^2 * sin(40*pi*x - 1) .* sin(40*pi*y - 1);
u_boundary = @(x, y)  -1/2    * sin(40*pi*x - 1) .* sin(40*pi*y - 1);

l_1 = @(theta)  2 * sin(theta * pi);
l_2 = @(theta) -sin(theta * 2 * pi);
l_1_prime  = @(theta)  2 * pi * cos(theta * pi);
l_2_prime  = @(theta) -2 * pi * cos(theta * 2 * pi);
l_1_dprime = @(theta) -2 * pi^2 * sin(theta * pi);
l_2_dprime = @(theta)  4 * pi^2 * sin(theta * 2 * pi);

d   = 10;
C   = 27;
n_r = 6;
M   = d + 3;
p   = M;

h        = 0.00125;
n_frac_C = 0.1;
n_frac_S = 0.6;

if exist(['FC_data/A_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat'], 'file') == 0 || ...
   exist(['FC_data/Q_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat'], 'file') == 0
    fprintf('FC data not found. Generating FC operators...\n');
    generate_bdry_continuations(d, C, C, 12, 20, 4, 256, n_r);
end

load(['FC_data/A_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']);
load(['FC_data/Q_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']);
A = double(A);
Q = double(Q);

curve_seq = Curve_seq_obj();
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    0, n_frac_C, n_frac_C, n_frac_S, n_frac_S, h);

h_eval = 0.01;
[u_num_mat, R, R_eval] = poisson_solver_coarse(curve_seq, f, u_boundary, h, 1, p, ...
    1e-13, 1e-13, 1e-13, d, C, n_r, A, Q, M, h_eval);

figure;
surf(R_eval.R_X, R_eval.R_Y, u_num_mat, 'EdgeColor', 'none');
shading interp;
