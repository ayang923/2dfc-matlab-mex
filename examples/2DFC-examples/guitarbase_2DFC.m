% GUITARBASE_2DFC  2DFC example: guitar-body-shaped domain with 4 curves.
%
% Demonstrates the 2DFC algorithm on a multi-curve domain shaped like the base
% of a guitar body. The domain is formed by four connected curves:
%
%   Curve 1: Bottom-right arc   - lower-right quarter of teardrop curve
%   Curve 2: Right Bezier arc   - quadratic Bezier connecting right tip to top-right
%   Curve 3: Left Bezier arc    - quadratic Bezier connecting top-right to upper-left
%   Curve 4: Top-left arc       - upper-left quarter of teardrop curve
%
% Curves 2 and 3 are quadratic Bezier curves that create the characteristic
% waist indentation of a guitar body.
%
% The test function is:
%   f(x,y) = 4 + (1 + x^2 + y^2) * (sin(10.5*pi*x - 0.5) + cos(pi*y - 0.5))
%

clc; clear; close all;

h   = 0.0005;
d   = 6;
n_x = 3072;
n_y = 4374;


f = @(x, y) 4 + (1 + x.^2 + y.^2) .* (sin(10.5*pi*x - 0.5) + cos(1*pi*y - 0.5));

alpha = 1/2;
beta  = tan(alpha * pi / 2);

h_tan     = 2 * h;
h_norm    = h_tan;
n_curve   = 0;   % auto-compute from arc length

curve_seq = Curve_seq_obj();

% --- Curve 1: lower-right arc of teardrop (theta from 0 to 0.25) ---
l_1       = @(theta)  2 * sin(0.25*theta*pi);
l_2       = @(theta) -beta * sin(0.25*theta*2*pi);
l_1_prime = @(theta)  2*0.25*pi * cos(0.25*theta*pi);
l_2_prime = @(theta) -2*0.25*beta*pi * cos(0.25*theta*2*pi);
l_1_dprime = @(theta) -2*0.25^2*pi^2 * sin(0.25*theta*pi);
l_2_dprime = @(theta)  4*beta*0.25^2*pi^2 * sin(0.25*theta*2*pi);

n_frac_C_0 = 0.1;  n_frac_C_1 = 0.2;
n_frac_S_0 = 0.6;  n_frac_S_1 = 0.7;
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% --- Curve 2: right Bezier connecting curve 1 end to (1, 0) ---
% Control point offset inward from the chord midpoint by 0.1 * outward normal
x_1 = [l_1(1); l_2(1)];
x_2 = [1; 0];
n   = [x_2(2) - x_1(2); x_1(1) - x_2(1)];
n   = n / norm(n);
c   = 0.5*(x_1 + x_2) + 0.1*n;

l_1       = @(theta) (1-theta).^2.*x_1(1) + 2*(1-theta).*theta.*c(1) + theta.^2.*x_2(1);
l_2       = @(theta) (1-theta).^2.*x_1(2) + 2*(1-theta).*theta.*c(2) + theta.^2.*x_2(2);
l_1_prime = @(theta) 2*((theta-1)*x_1(1) + (1-2*theta)*c(1) + theta*x_2(1));
l_2_prime = @(theta) 2*((theta-1)*x_1(2) + (1-2*theta)*c(2) + theta*x_2(2));
l_1_dprime = @(theta) 2*x_1(1) - 4*c(1) + 2*x_2(1);
l_2_dprime = @(theta) 2*x_1(2) - 4*c(2) + 2*x_2(2);

n_frac_C_0 = 0.3;  n_frac_C_1 = 0.3;
n_frac_S_0 = 0.7;  n_frac_S_1 = 0.7;
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% --- Curve 3: left Bezier connecting (1,0) to upper-left point on teardrop ---
% Evaluate teardrop at theta=0.75 to get the upper-left endpoint
l_1_td = @(theta) 2 * sin(theta * pi);
l_2_td = @(theta) -beta * sin(theta * 2*pi);

x_1 = [1; 0];
x_2 = [l_1_td(0.75); l_2_td(0.75)];
n   = [x_2(2) - x_1(2); x_1(1) - x_2(1)];
n   = n / norm(n);
c   = 0.5*(x_1 + x_2) - 0.1*n;

l_1       = @(theta) (1-theta).^2.*x_1(1) + 2*(1-theta).*theta.*c(1) + theta.^2.*x_2(1);
l_2       = @(theta) (1-theta).^2.*x_1(2) + 2*(1-theta).*theta.*c(2) + theta.^2.*x_2(2);
l_1_prime = @(theta) 2*((theta-1)*x_1(1) + (1-2*theta)*c(1) + theta*x_2(1));
l_2_prime = @(theta) 2*((theta-1)*x_1(2) + (1-2*theta)*c(2) + theta*x_2(2));
l_1_dprime = @(theta) 2*x_1(1) - 4*c(1) + 2*x_2(1);
l_2_dprime = @(theta) 2*x_1(2) - 4*c(2) + 2*x_2(2);

n_frac_C_0 = 0.3;  n_frac_C_1 = 0.3;
n_frac_S_0 = 0.7;  n_frac_S_1 = 0.7;
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% --- Curve 4: upper-left arc of teardrop (theta from 0.75 to 1) ---
l_1       = @(theta)  2 * sin((0.25*theta + 0.75)*pi);
l_2       = @(theta) -beta * sin((0.25*theta + 0.75)*2*pi);
l_1_prime = @(theta)  2*0.25*pi * cos((0.25*theta + 0.75)*pi);
l_2_prime = @(theta) -2*0.25*beta*pi * cos((0.25*theta + 0.75)*2*pi);
l_1_dprime = @(theta) -2*0.25^2*pi^2 * sin((0.25*theta + 0.75)*pi);
l_2_dprime = @(theta)  4*beta*0.25^2*pi^2 * sin((0.25*theta + 0.75)*2*pi);

n_frac_C_0 = 0.3;  n_frac_C_1 = 0.1;
n_frac_S_0 = 0.7;  n_frac_S_1 = 0.7;
curve_seq.add_curve(l_1, l_2, l_1_prime, l_2_prime, l_1_dprime, l_2_dprime, ...
    n_curve, n_frac_C_0, n_frac_C_1, n_frac_S_0, n_frac_S_1, h_norm);

% FC parameters
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

curve_seq.plot_geometry(d);
R = FC2D(f, h, curve_seq, 1e-13, 1e-13, d, C, n_r, A, Q, C, A, Q, M, n_x, n_y);
