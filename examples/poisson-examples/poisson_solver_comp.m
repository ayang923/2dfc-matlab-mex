clc; clear; close all;

%% 2D-FC Corner Solver
d       = 10;
h       = 0.002;
n_x     = 375;
n_y     = 720;
int_eps = 1e-6;

u_boundary = @(x, y) sin(40*pi*x - 1) .* sin(40*pi*y - 1);
f          = @(x, y) -2*(40*pi)^2 * sin(40*pi*x - 1) .* sin(40*pi*y - 1);

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

% Warm-up call (untimed/discarded) to absorb first-call JIT/dispatch
% overhead before the timed comparison run below.
curve_seq = Curve_seq_obj();
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    0, n_frac_C, n_frac_C, n_frac_S, n_frac_S, h*2);
poisson_solver(curve_seq, f, u_boundary, h, 2, p, int_eps, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x, n_y, true, M*h, false);

curve_seq = Curve_seq_obj();
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    0, n_frac_C, n_frac_C, n_frac_S, n_frac_S, h*2);

rng('default')
[u_2dfc, R_eval] = poisson_solver(curve_seq, f, u_boundary, h, 2, p, int_eps, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x, n_y, true, M*h, true);
u_exact = u_boundary(R_eval.R_X, R_eval.R_Y); u_exact(~R_eval.in_interior) = nan;


%% Fully adaptive rounded solver
addpath('/Users/allenyang/Documents/2dfc-paper/fully-adaptive-poisson');
setup;

% rounded geometry
eps_reg = 1e-6;
g   = @(t) sqrt(sin(t/2).^2 + eps_reg^2);
z   = @(t) -1/4*sin(t) - 1i*g(t);
dz  = @(t) -1/4*cos(t) - 1i*sin(t)./(4*g(t));
dzz = @(t) 1/4*sin(t) + 1i*(sin(t).^2 - 4*cos(t).*g(t).^2)./(16*g(t).^3);


n = 16;

tol = 1e-7;
geom_tol = tol;

opts = [];
opts.debug = true;
opts.n_re  = 2*n;
opts.n_sem = [n 2*n];
opts.n_box = n;
opts.beta = 1;
opts.stripWidth = 0.5;
opts.boxToStripRatio = 0.5;
opts.tol = tol;
solve_opts = [];
solve_opts.tol = tol;

% Warm-up call (untimed/discarded) to absorb first-call JIT/dispatch
% overhead before the timed comparison run below.
Gamma_warmup = Boundary(n, z, dz, dzz, [], quadrature='panel', tol=geom_tol);
Gamma_warmup = kdrefine(Gamma_warmup);
Gamma_warmup = levelrestrict(Gamma_warmup);
S_warmup = AdaptivePoissonSolver(Gamma_warmup, f, opts);
u_warmup = S_warmup.solve(u_boundary);
S_warmup.sample(u_warmup, R_eval.R_X, R_eval.R_Y);
clear Gamma_warmup S_warmup u_warmup

Gamma = Boundary(n, z, dz, dzz, [], ...
                 quadrature='panel', tol=geom_tol);
Gamma = kdrefine(Gamma);
Gamma = levelrestrict(Gamma);

Timer.reset()
S = AdaptivePoissonSolver(Gamma, f, opts);
u = S.solve(u_boundary);

[u_fullap, inGamma, ~, ~] = S.sample(u, R_eval.R_X, R_eval.R_Y);
u_fullap(~inGamma) = NaN;

n_dof = numel(S.tf) + ...                          % Tree DOFs
               prod(opts.n_sem)*length(S.strip_dom) + ... % Strip DOFs
               2*opts.n_re*length(S.Gamma1) + ...         % Gamma' DOFs
               opts.n_re*length(S.Gamma);                 % Gamma DOFs


%% 
n_dof

err_2dfc = abs(u_2dfc - u_exact);
err_fullap = abs(u_fullap - u_exact);

max(err_2dfc, [], 'all')
max(err_fullap(inGamma), [], 'all')

sqrt(sum(err_2dfc(R_eval.in_interior).^2)) / sqrt(sum(u_exact(R_eval.in_interior).^2))
sqrt(sum(err_fullap(~isnan(err_fullap)).^2)) / sqrt(sum(u_exact(~isnan(err_fullap)).^2))


