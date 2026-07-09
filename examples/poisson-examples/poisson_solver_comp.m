clc; clear; close all;

%% 2D-FC Corner Solver
d       = 10;
h       = 0.0005;
n_x     = 1111;
n_y     = 2200;
int_eps = 1e-11;

u_boundary = @(x, y) sin(80*pi*x - 1) .* sin(80*pi*y - 1);
f          = @(x, y) -2*(80*pi)^2 * sin(80*pi*x - 1) .* sin(80*pi*y - 1);

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
    0, n_frac_C, n_frac_C, n_frac_S, n_frac_S, h*2);

rng('default')
tic
[u_2dfc, R, R_eval] = poisson_solver_coarse(curve_seq, f, u_boundary, h, 1, p, int_eps, 1e-13, 1e-13, d, C, n_r, A, Q, M, 0.01, n_x, n_y, true);
toc
u_exact = u_boundary(R_eval.R_X, R_eval.R_Y); u_exact(~R_eval.in_interior) = nan;


%% Fully adaptive rounded solver
addpath('/Users/allenyang/Documents/2dfc-paper/fully-adaptive-poisson');
setup;

% rounded geometry
eps_reg = 1e-3;
g   = @(t) sqrt(sin(t/2).^2 + eps_reg^2);
z   = @(t) -1/4*sin(t) - 1i*g(t);
dz  = @(t) -1/4*cos(t) - 1i*sin(t)./(4*g(t));
dzz = @(t) 1/4*sin(t) + 1i*(sin(t).^2 - 4*cos(t).*g(t).^2)./(16*g(t).^3);

tic
n = 16;
Gamma = Boundary(n, z, dz, dzz, [], quadrature='panel', tol=1e-12);
Gamma = kdrefine(Gamma);
Gamma = levelrestrict(Gamma);

tol = 1e-14;

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

S = AdaptivePoissonSolver(Gamma, f, opts);
u = S.solve(u_boundary);

[u_fullap, ~, ~, ~] = S.sample(u, R_eval.R_X, R_eval.R_Y);
u_fullap(~R_eval.in_interior) = NaN;
toc

max(abs(u_2dfc - u_exact), [], 'all')
max(abs(u_fullap - u_exact), [], 'all')

figure

ids = leaves(S.tf);
keep = [];
for id = ids(:).'
    if ( ~all(S.tf.coeffs{id} == 0, 'all') )
        keep = [keep id];
    end
end

plotOnly(S.tf, keep, boxes=true, values=false)
axis equal off

