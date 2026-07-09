% POISSON_SOLVER_GUITARBASE  Poisson BVP on a guitar-body ("guitarbase") domain.
%
% Domain:
%   Four smoothly connected curves form a closed "guitar body" outline built
%   from arclength-reparametrized sinusoidal and quadratic Bezier segments.
%   The boundary has concave corners where the pieces meet.
%
% Problem:
%   Delta(u) = f  in Omega,   u = u_boundary  on d(Omega)
%   f(x,y)          = (3*pi)^2 * sin(3*pi*x - 1) * sin(3*pi*y - 1)
%   u_boundary(x,y) = -1/2    * sin(3*pi*x - 1) * sin(3*pi*y - 1)
%   (manufactured solution with exact solution u = u_boundary)

clc; clear; close all;

h       = 0.004;
d       = 5;
n_x     = 576;
n_y     = 648;
int_eps = 1e-8;

f          = @(x, y) (3*pi)^2 * sin(3*pi*x - 1) .* sin(3*pi*y - 1);
u_boundary = @(x, y)  -1/2   * sin(3*pi*x - 1) .* sin(3*pi*y - 1);

alph = 1/2;
bet  = tan(alph * pi / 2);

curve_seq = Curve_seq_obj();

% -------------------------------------------------------------------------
% Curve 1: lower-left arc (sinusoidal, theta in [0,1])
% -------------------------------------------------------------------------
l_1 = @(theta) 2 * sin(0.25 * theta * pi);
l_2 = @(theta) -bet * sin(0.25 * theta * 2 * pi);
l_1_prime  = @(theta)  2 * 0.25 * pi   * cos(0.25 * theta * pi);
l_2_prime  = @(theta) -2 * 0.25 * bet  * pi * cos(0.25 * theta * 2 * pi);
l_1_dprime = @(theta) -2 * 0.25^2 * pi^2  * sin(0.25 * theta * pi);
l_2_dprime = @(theta)  4 * bet * 0.25^2 * pi^2 * sin(0.25 * theta * 2 * pi);

h_tan = 2 * h;   h_norm = h_tan;   n_curve = 0;
n_frac_C_0 = 0.1;   n_frac_C_1 = 0.2;
n_frac_S_0 = 0.6;   n_frac_S_1 = 0.7;

curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% -------------------------------------------------------------------------
% Curve 2: right connector (quadratic Bezier, theta in [0,1])
% -------------------------------------------------------------------------
x_1 = [l_1(1); l_2(1)];
x_2 = [1; 0];
n   = [x_2(2) - x_1(2); x_1(1) - x_2(1)];  n = n ./ norm(n);
c   = 1/2 * (x_1 + x_2) + 0.1 * n;

l_1 = @(theta) (1-theta).^2 .* x_1(1) + 2*(1-theta).*theta .* c(1) + theta.^2 .* x_2(1);
l_2 = @(theta) (1-theta).^2 .* x_1(2) + 2*(1-theta).*theta .* c(2) + theta.^2 .* x_2(2);
l_1_prime  = @(theta)  2*((theta-1)*x_1(1) + (1-2*theta)*c(1) + theta*x_2(1));
l_2_prime  = @(theta)  2*((theta-1)*x_1(2) + (1-2*theta)*c(2) + theta*x_2(2));
l_1_dprime = @(theta)  2*x_1(1) - 4*c(1) + 2*x_2(1);
l_2_dprime = @(theta)  2*x_1(2) - 4*c(2) + 2*x_2(2);

h_tan = 2 * h;   h_norm = h_tan;   n_curve = 0;
n_frac_C_0 = 0.3;   n_frac_C_1 = 0.3;
n_frac_S_0 = 0.7;   n_frac_S_1 = 0.7;

curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% -------------------------------------------------------------------------
% Curve 3: upper connector (quadratic Bezier, theta in [0,1])
% -------------------------------------------------------------------------
l_1_full = @(theta) 2 * sin(theta * pi);
l_2_full = @(theta) -bet * sin(theta * 2 * pi);

x_1 = [1; 0];
x_2 = [l_1_full(0.75); l_2_full(0.75)];
n   = [x_2(2) - x_1(2); x_1(1) - x_2(1)];  n = n ./ norm(n);
c   = 1/2 * (x_1 + x_2) - 0.1 * n;

l_1 = @(theta) (1-theta).^2 .* x_1(1) + 2*(1-theta).*theta .* c(1) + theta.^2 .* x_2(1);
l_2 = @(theta) (1-theta).^2 .* x_1(2) + 2*(1-theta).*theta .* c(2) + theta.^2 .* x_2(2);
l_1_prime  = @(theta)  2*((theta-1)*x_1(1) + (1-2*theta)*c(1) + theta*x_2(1));
l_2_prime  = @(theta)  2*((theta-1)*x_1(2) + (1-2*theta)*c(2) + theta*x_2(2));
l_1_dprime = @(theta)  2*x_1(1) - 4*c(1) + 2*x_2(1);
l_2_dprime = @(theta)  2*x_1(2) - 4*c(2) + 2*x_2(2);

h_tan = 2 * h;   h_norm = h_tan;   n_curve = 0;
n_frac_C_0 = 0.3;   n_frac_C_1 = 0.3;
n_frac_S_0 = 0.7;   n_frac_S_1 = 0.7;

curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% -------------------------------------------------------------------------
% Curve 4: upper-right arc (sinusoidal, theta in [0,1])
% -------------------------------------------------------------------------
l_1 = @(theta) 2 * sin((0.25*theta + 0.75) * pi);
l_2 = @(theta) -bet * sin((0.25*theta + 0.75) * 2 * pi);
l_1_prime  = @(theta)  2 * 0.25 * pi   * cos((0.25*theta + 0.75) * pi);
l_2_prime  = @(theta) -2 * 0.25 * bet  * pi * cos((0.25*theta + 0.75) * 2 * pi);
l_1_dprime = @(theta) -2 * 0.25^2 * pi^2  * sin((0.25*theta + 0.75) * pi);
l_2_dprime = @(theta)  4 * bet * 0.25^2 * pi^2 * sin((0.25*theta + 0.75) * 2 * pi);

h_tan = 2 * h;   h_norm = h_tan;   n_curve = 0;
n_frac_C_0 = 0.3;   n_frac_C_1 = 0.1;
n_frac_S_0 = 0.7;   n_frac_S_1 = 0.7;

curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% -------------------------------------------------------------------------
% FC and solver parameters
% -------------------------------------------------------------------------
C   = 27;
n_r = 6;
M   = d + 3;
p   = M;

if exist(['FC_data/A_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat'], 'file') == 0 || ...
   exist(['FC_data/Q_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat'], 'file') == 0
    fprintf('FC data not found. Generating FC operators...\n');
    generate_bdry_continuations(d, C, C, 12, 20, 4, 256, n_r);
end

load(['FC_data/A_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']);
load(['FC_data/Q_d', num2str(d), '_C', num2str(C), '_r', num2str(n_r), '.mat']);
A = double(A);
Q = double(Q);

tic;
rng('default')
[u_num_mat, R] = poisson_solver(curve_seq, f, u_boundary, h, 1, p, int_eps, 1e-13, 1e-13, d, C, n_r, A, Q, M, n_x, n_y, true);
u_exact = u_boundary(R.R_X, R.R_Y);
toc;

abs_max_err = max(abs(u_num_mat(R.in_interior) - u_exact(R.in_interior)), [], 'all')
rel_2_err   = sqrt(sum((u_num_mat(R.in_interior) - u_exact(R.in_interior)).^2, 'all') / ...
                   sum(u_exact(R.in_interior).^2, 'all'))
