% POISSON_SOLVER_TEARDROP  Poisson BVP on a teardrop domain.
%
% Domain:
%   Single closed curve  (l_1(theta), l_2(theta)) = (2*sin(pi*theta), -sin(2*pi*theta))
%   for theta in [0,1]. The curve has a single corner at theta=0 (theta=1).
%
% Problem:
%   Delta(u) = f  in Omega,   u = u_boundary  on d(Omega)
%   f(x,y)          = (40*pi)^2 * sin(40*pi*x - 1) * sin(40*pi*y - 1)
%   u_boundary(x,y) = -1/2     * sin(40*pi*x - 1) * sin(40*pi*y - 1)
%   (manufactured solution; high-frequency to stress accuracy)

clc; clear; close all;

d       = 10;
h       = 0.005;
n_x     = 486;
n_y     = 486;
int_eps = 1e-7;

f          = @(x, y) (40*pi)^2 * sin(40*pi*x - 1) .* sin(40*pi*y - 1);
u_boundary = @(x, y)  -1/2    * sin(40*pi*x - 1) .* sin(40*pi*y - 1);

l_1 = @(theta)  2 * sin(theta * pi);
l_2 = @(theta) -sin(theta * 2 * pi);
l_1_prime  = @(theta)  2 * pi * cos(theta * pi);
l_2_prime  = @(theta) -2 * pi * cos(theta * 2 * pi);
l_1_dprime = @(theta) -2 * pi^2 * sin(theta * pi);
l_2_dprime = @(theta)  4 * pi^2 * sin(theta * 2 * pi);

C   = 27;
n_r = 6;
M   = d + 3;
p   = M;

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

tic;
rng('default')
[u_num_mat, R] = poisson_solver(curve_seq, f, u_boundary, h, 1, p, 1e-7, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x, n_y, true);
u_exact = u_boundary(R.R_X, R.R_Y);
toc;

abs_max_err = max(abs(u_num_mat(R.in_interior) - u_exact(R.in_interior)), [], 'all')
rel_2_err   = sqrt(sum((u_num_mat(R.in_interior) - u_exact(R.in_interior)).^2, 'all') / ...
                   sum(u_exact(R.in_interior).^2, 'all'))
