% POISSON_SOLVER_GUITARBASE_FIG  Poisson BVP on a guitarbase domain — publication figure.
%
% Same domain and problem as poisson_solver_guitarbase.m, solved using the
% coarse-evaluation variant (h = 0.001 fine, h_eval = 0.02) and rendered as a
% triangulated surface plot with the domain boundary overlaid.
%
% Domain:
%   Four smoothly connected curves (sinusoidal arcs + quadratic Bezier connectors)
%   forming a closed guitar-body outline.
%
% Problem:
%   Delta(u) = f  in Omega,   u = u_boundary  on d(Omega)
%   f(x,y)          = (3*pi)^2 * sin(3*pi*x - 1) * sin(3*pi*y - 1)
%   u_boundary(x,y) = -1/2    * sin(3*pi*x - 1) * sin(3*pi*y - 1)
%
% Output:
%   Triangulated surface of u on the domain interior + boundary curve.
%
% Parameters:
%   h = 0.001 (finest), h_eval = 0.02, d = 7

clc; clear; close all;

f          = @(x, y) (3*pi)^2 * sin(3*pi*x - 1) .* sin(3*pi*y - 1);
u_boundary = @(x, y)  -1/2   * sin(3*pi*x - 1) .* sin(3*pi*y - 1);

alph = 1/2;
bet  = tan(alph * pi / 2);

h = 0.001;
curve_seq = Curve_seq_obj();

% -------------------------------------------------------------------------
% Curve 1: lower-left arc
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
% Curve 2: right connector (quadratic Bezier)
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
% Curve 3: upper connector (quadratic Bezier)
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
% Curve 4: upper-right arc
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
d   = 7;
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

h_eval = 0.02;
[u_num_mat, R, R_eval] = poisson_solver_coarse(curve_seq, f, u_boundary, h, 1, p, ...
    1e-14, 1e-13, 1e-13, d, C, n_r, A, Q, M, h_eval);

% -------------------------------------------------------------------------
% Plot: triangulated surface with constrained Delaunay triangulation
% -------------------------------------------------------------------------

% Interior evaluation-grid vertices
Xint = R_eval.R_X(:);
Yint = R_eval.R_Y(:);
Zint = u_num_mat(:);

% Boundary vertices (dense boundary mesh for the constraint)
[Xb, Yb] = curve_seq.construct_boundary_mesh(1);
Zb = u_boundary(Xb, Yb);

% Ensure the boundary polygon is closed
if ~(Xb(1) == Xb(end) && Yb(1) == Yb(end))
    Xb = [Xb; Xb(1)];
    Yb = [Yb; Yb(1)];
    Zb = [Zb; Zb(1)];
end

% Build inside/outside test polygon
P = polyshape(Xb, Yb);

% Merge interior and boundary point sets
X = [Xint; Xb];
Y = [Yint; Yb];
Z = [Zint; Zb];

% Constrained edges connecting consecutive boundary vertices
nint = numel(Xint);
nb   = numel(Xb);
bidx = (nint+1):(nint+nb);
E    = [bidx(1:end-1)' bidx(2:end)'];

% Constrained Delaunay triangulation; keep triangles inside the domain
DT     = delaunayTriangulation(X, Y, E);
T      = DT.ConnectivityList;
xc     = mean(X(T), 2);
yc     = mean(Y(T), 2);
inside = isinterior(P, xc, yc);
T_in   = T(inside, :);

figure;
trisurf(T_in, X, Y, Z, 'EdgeColor', 'k');
shading interp;
hold on;
plot(R_eval.boundary_X, R_eval.boundary_Y, 'LineWidth', 2);
