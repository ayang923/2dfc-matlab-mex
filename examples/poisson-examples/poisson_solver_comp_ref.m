clc; clear; close all;

d       = 10;
h       = 0.00025;
n_x     = 1057;
n_y     = 2102;
int_eps = 1e-14;

f          = @(x, y) (10*pi)^2 * sin(10*pi*x - 1) .* sin(10*pi*y - 1);
u_boundary = @(x, y) (2*x).^2+(2*y).^2;

l_1 = @(theta)  -1/4*sin(theta * 2*pi);
l_2 = @(theta) -sin(theta * pi);
l_1_prime  = @(theta) -(pi/2) * cos(theta * 2*pi);
l_2_prime  = @(theta) -pi * cos(theta * pi);
l_1_dprime = @(theta) pi^2 * sin(theta * 2*pi);
l_2_dprime = @(theta) pi^2 * sin(theta * pi);

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

load("poisson_solver_comp.mat")
rng('default')
[u_exact_mat, ~, R_eval_ref] = poisson_solver_coarse(curve_seq, f, u_boundary, h, 1, p, 1e-7, 1e-13, 1e-13, d, C, n_r, A, Q, M, R_eval, n_x, n_y, true);

%% Fully adaptive poisson solver
addpath('/Users/allenyang/Documents/2dfc-paper/fully-adaptive-poisson');
setup;

%% Geometry: z(t) = -1/4*sin(t) - i*sqrt(sin(t/2)^2 + 1e-6)
eps_reg = 1e-6;
g   = @(t) sqrt(sin(t/2).^2 + eps_reg);
z   = @(t) -1/4*sin(t) - 1i*g(t);
dz  = @(t) -1/4*cos(t) - 1i*sin(t)./(4*g(t));
dzz = @(t) 1/4*sin(t) + 1i*(sin(t).^2 - 4*cos(t).*g(t).^2)./(16*g(t).^3);

n = 16;
Gamma = Boundary(n, z, dz, dzz, [], quadrature='panel', tol=1e-12);
Gamma = kdrefine(Gamma);
Gamma = levelrestrict(Gamma);

%% Problem data
f          = @(x, y) (10*pi)^2 * sin(10*pi*x - 1) .* sin(10*pi*y - 1);
u_boundary = @(x, y) (2*x).^2 + (2*y).^2;

%% Solve
opts = [];
opts.tol = 1e-10;
S = AdaptivePoissonSolver(Gamma, f, opts);
u = S.solve(u_boundary);

%% Evaluate and plot
[uu, inGamma, inGamma1, inStrip] = S.sample(u, R_eval.R_X, R_eval.R_Y);
uu(~inGamma) = NaN;

figure(1); clf
surf(R_eval.R_X, R_eval.R_Y, uu, 'EdgeColor', 'none');
title('Solution u')

figure(2); 
surf(R_eval.R_X, R_eval.R_Y, uu, 'EdgeColor', 'none')
