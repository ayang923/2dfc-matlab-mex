% POISSON_SOLVER_GUITARBASE_DENSITY_FIG  Visualize the true (ungraded)
% boundary density of the homogeneous (Laplace) BIE solve for the guitarbase
% example, for both the non-singular and singular (sharp-corner) variants.
%
% IE_curve_seq_obj.construct_A_b solves the graded-mesh system
%
%   what/2 - w'(s) sum_j int_0^1 K(r_i(w(s)), r_j(w(s'))) what(r_j(w(s'))) J_j(w(s')) ds'
%          = -w'(s) g_hom(r_i(w(s)))                                    (rescaled-ie)
%
% whose solution what = w'(s)*phi is the density already scaled by the
% graded-mesh derivative w'(s) -- it is NOT the true density phi, and it is
% this what (not phi) that laplace_solver.m stores as gr_phi_rho1
% (laplace_solver.m:107) and later windows further by the geometric Jacobian
% sqrt(l_1_prime.^2+l_2_prime.^2) at potential-evaluation time (see
% IE_curve_obj.u_num_curve / IE_curve_seq_obj.build_flat_nodes's l1p/l2p).
%
% The true density phi instead solves
%
%   phi/2 - sum_j int_0^1 K(r_i(w(s)), r_j(w(s'))) w'(s') phi(r_j(w(s'))) J_j(w(s')) ds'
%         = -g_hom(r_i(w(s)))                                           (graded-integral)
%
% i.e. the w'(s') factor moves from an overall target-side scaling to a
% source-side factor inside the integral. This script's construct_A_b_phi
% assembles that system directly (modeled on IE_curve_seq_obj.construct_A_b),
% and solve_boundary_density calls it instead, so the plotted quantity here
% is the true phi -- with no graded-mesh window and no geometric Jacobian
% window (the Jacobian is applied only later, at potential-evaluation time,
% and is not applied here either).
%
% This script recomputes just that density (skipping the expensive
% well-interior/patch/corner adaptive grid-evaluation passes entirely, since
% they aren't needed to see phi itself) for both:
%   - the "non-sing" example (poisson_solver_guitarbase.m's parameters)
%   - the "sing" example (poisson_solver_guitarbase_sing.m's parameters)
% and plots each as a function of normalized boundary arclength, colored by
% curve segment, with corner locations marked.
%
% It does not modify laplace_solver.m/poisson_solver.m/IE_curve_seq_obj.m
% (used by other examples); instead it mirrors their setup up through the
% BIE solve, swapping in construct_A_b_phi in place of construct_A_b.

clc; clear; close all;

% -------------------------------------------------------------------------
% Shared FC / domain parameters (identical in both original examples)
% -------------------------------------------------------------------------
d   = 5;
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

f = @(x, y) (3*pi)^2 * sin(3*pi*x - 1) .* sin(3*pi*y - 1);

% -------------------------------------------------------------------------
% Per-example parameters (mirrors poisson_solver_guitarbase.m / _sing.m)
% -------------------------------------------------------------------------
cfgs(1).name       = 'non-sing';
cfgs(1).h          = 0.001;
cfgs(1).n_x        = 1728;
cfgs(1).n_y        = 2187;
cfgs(1).u_boundary = @(x, y) -1/2 * sin(3*pi*x - 1) .* sin(3*pi*y - 1);

cfgs(2).name       = 'sing';
cfgs(2).h          = 0.001;
cfgs(2).n_x        = 1728;
cfgs(2).n_y        = 2187;
cfgs(2).u_boundary = @(x, y) (x/2).^2 + (y/2).^2;

for k = 1:numel(cfgs)
    cfg = cfgs(k);

    rng('default')
    curve_seq = build_guitarbase_curve_seq(cfg.h);

    [phi, curve_param, l1, l2, l1p, l2p, ds_per_node] = solve_boundary_density( ...
        curve_seq, f, cfg.u_boundary, cfg.h, 1, p, 1e-13, 1e-13, ...
        d, C, n_r, A, Q, M, cfg.n_x, cfg.n_y);

    % Normalized arclength x-axis (rectangle-rule approximation of
    % cumulative boundary length; used only to lay out the plot, never
    % applied to phi itself).
    seg_len = sqrt(l1p.^2 + l2p.^2) .* ds_per_node;
    s_left  = [0; cumsum(seg_len(1:end-1))];
    s_norm  = s_left / (s_left(end) + seg_len(end));

    plot_boundary_density(cfg.name, cfg.h, phi, s_norm, curve_param);
    plot_boundary_density_3d(cfg.name, cfg.h, phi, l1, l2, curve_param);
end

% =========================================================================

function curve_seq = build_guitarbase_curve_seq(h)
% BUILD_GUITARBASE_CURVE_SEQ  4-curve "guitar body" domain shared by
% poisson_solver_guitarbase.m and poisson_solver_guitarbase_sing.m
% (identical geometry in both; only h differs between the two examples).

    alph = 1/2;
    bet  = tan(alph * pi / 2);

    curve_seq = Curve_seq_obj();

    % ---------------------------------------------------------------- %
    % Curve 1: lower-left arc (sinusoidal, theta in [0,1])
    % ---------------------------------------------------------------- %
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

    % ---------------------------------------------------------------- %
    % Curve 2: right connector (quadratic Bezier, theta in [0,1])
    % ---------------------------------------------------------------- %
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

    % ---------------------------------------------------------------- %
    % Curve 3: upper connector (quadratic Bezier, theta in [0,1])
    % ---------------------------------------------------------------- %
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

    % ---------------------------------------------------------------- %
    % Curve 4: upper-right arc (sinusoidal, theta in [0,1])
    % ---------------------------------------------------------------- %
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
end

% =========================================================================

function [phi, curve_param, l1, l2, l1p, l2p, ds_per_node] = solve_boundary_density( ...
    curve_seq, f, u_boundary, h, G_cf, p, eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x_padded, n_y_padded)
% SOLVE_BOUNDARY_DENSITY  Solves for the true (ungraded) double-layer density
% phi of the homogeneous BVP u_h = u_boundary - u_p on the boundary,
% mirroring poisson_solver.m's particular-solution setup (FC2D + inv_lap)
% and laplace_solver.m's BIE assembly/solve (curve_param_obj,
% IE_curve_seq_obj) -- but using construct_A_b_phi (below) in place of
% IE_curve_seq_obj.construct_A_b, and stopping after the backslash solve,
% skipping the adaptive grid-evaluation passes laplace_solver performs
% afterward, since only the density itself is needed here.
%
% Outputs:
%   phi         - (n_total x 1) true double-layer density (no graded-mesh
%                 window, no geometric-Jacobian window)
%   curve_param - curve_param_obj indexing phi per curve (start_idx/end_idx)
%   l1, l2      - physical (x,y) location of each density DOF
%   l1p, l2p    - boundary derivative components at each DOF (the Jacobian
%                 sqrt(l1p.^2+l2p.^2) is what would additionally window phi
%                 at potential-evaluation time; it is NOT applied here)
%   ds_per_node - per-curve graded-mesh quadrature step

    [R, ~, ~, ~] = FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, C, A, Q, M, ...
        n_x_padded, n_y_padded, true, false);
    R.f_R = R.inv_lap();

    u_G = @(x, y) u_boundary(x, y) - R.locally_compute_vec(x, y, M);

    curve_n = zeros(curve_seq.n_curves, 1);
    curr = curve_seq.first_curve;
    for i = 1:curve_seq.n_curves
        curve_n(i) = ceil((curr.n - 1) * G_cf);
        curr = curr.next_curve;
    end
    curve_param = curve_param_obj(curve_n);

    IE_curve_seq = IE_curve_seq_obj(curve_seq, p);
    [A_bie, b_bie] = construct_A_b_phi(IE_curve_seq, curve_param, u_G);
    phi = A_bie \ b_bie;

    [l1, l2, l1p, l2p, ds_per_node] = IE_curve_seq.build_flat_nodes(curve_param);
end

% =========================================================================

function [A, b] = construct_A_b_phi(IE_curve_seq, curve_param, f)
% CONSTRUCT_A_B_PHI  Assembles the 2nd-kind BIE system for the true density
% phi (the "graded-integral" equation), i.e. the same discretization as
% IE_curve_seq_obj.construct_A_b (same graded mesh, same quadrature, same
% kernel calls) except for where the graded-mesh derivative w'(s) lands:
% construct_A_b solves for what = w'(s)*phi, so it puts the TARGET curve's
% w'(s) as an overall row-block scaling (and on the RHS) and omits any
% source-side w'. Here, to solve for phi itself, the target-side w'(s)
% scaling is removed from the off-diagonal blocks and the RHS, and a
% SOURCE-side w'(s') factor is added inside the integral instead (it
% multiplies along columns, since columns are the source/integration index
% -- see the curve-pair loop below). The same-point (diagonal) entries are
% unchanged, since target and source coincide there (w'(s) = w'(s')
% pointwise), so the diagonal expression is identical either way.
    curve_n   = curve_param.curve_n;
    n_total   = curve_param.n_total;
    start_idx = curve_param.start_idx;
    end_idx   = curve_param.end_idx;

    A = zeros(n_total, n_total);
    b = zeros(n_total, 1);

    curr_y = IE_curve_seq.first_curve;
    for i = 1:IE_curve_seq.n_curves
        [s_mesh_y, ds_y] = curr_y.s_mesh(curve_n(i));
        theta_mesh_y = curr_y.w(s_mesh_y);
        b(start_idx(i):end_idx(i)) = f(curr_y.l_1(theta_mesh_y), curr_y.l_2(theta_mesh_y));

        curr_x = IE_curve_seq.first_curve;
        for j = 1:IE_curve_seq.n_curves
            [s_mesh_x, ds_x] = curr_x.s_mesh(curve_n(j));
            theta_mesh_x = curr_x.w(s_mesh_x);
            [theta_2, theta_1] = meshgrid(theta_mesh_y, theta_mesh_x);

            A_local = curr_x.K_boundary(theta_1, theta_2, curr_y).*curr_y.w_prime(s_mesh_y').*sqrt(curr_y.l_1_prime(theta_mesh_y').^2+curr_y.l_2_prime(theta_mesh_y').^2)*ds_y;

            if i == j
                msk_diagonals = logical(diag(ones(curve_n(j), 1)));
                A_local(msk_diagonals) = curr_x.w_prime(s_mesh_x).*curr_x.K_boundary_same_point(theta_mesh_y).*sqrt(curr_x.l_1_prime(theta_mesh_x).^2+curr_x.l_2_prime(theta_mesh_x).^2)*ds_x + 1/2;
            end

            A(start_idx(j):end_idx(j), start_idx(i):end_idx(i)) = A_local;

            curr_x = curr_x.next_curve;
        end

        curr_y = curr_y.next_curve;
    end
    A(isnan(A)) = 0;
end

% =========================================================================

function plot_boundary_density(name_str, h, phi, s_norm, curve_param)
% PLOT_BOUNDARY_DENSITY  Plots phi vs. normalized boundary arclength,
% colored per curve, with corner locations marked.

    n_curves = numel(curve_param.curve_n);
    colors = lines(n_curves);

    figure('Name', sprintf('Boundary density (%s, h=%.4g)', name_str, h));
    hold on;
    for i = 1:n_curves
        idx = curve_param.start_idx(i):curve_param.end_idx(i);
        next_i = mod(i, n_curves) + 1;

        % Append the next curve's first node so each curve's segment is
        % visually connected across the corner instead of leaving a gap
        % (each curve's own graded mesh samples s in [0,1), never reaching
        % its own endpoint exactly). The last curve wraps back to curve 1,
        % which sits at normalized arclength 0, not 1 -- plot it at s=1
        % (the other end of this axis) so the wrap-around segment doesn't
        % jump backward across the whole plot.
        x_plot = s_norm(idx);
        y_plot = phi(idx);
        if i == n_curves
            x_plot(end+1) = 1;
        else
            x_plot(end+1) = s_norm(curve_param.start_idx(next_i));
        end
        y_plot(end+1) = phi(curve_param.start_idx(next_i));

        plot(x_plot, y_plot, 'Color', colors(i, :), ...
            'DisplayName', sprintf('curve %d', i));
    end
    for i = 1:n_curves
        xline(s_norm(curve_param.start_idx(i)), '--', sprintf('corner %d', i), ...
            'Color', [0.5 0.5 0.5], 'LabelVerticalAlignment', 'bottom', ...
            'HandleVisibility', 'off');
    end
    hold off;

    xlabel('normalized arclength s / total length');
    ylabel('boundary density \phi (unwindowed)');
    title(sprintf('%s example boundary density (h = %.4g)', name_str, h));
    legend('show', 'Location', 'best');
end

% =========================================================================

function plot_boundary_density_3d(name_str, h, phi, l1, l2, curve_param)
% PLOT_BOUNDARY_DENSITY_3D  Plots phi as a curve raised above the domain
% boundary in 3D: (x,y) = the boundary location of each density DOF,
% z = phi at that DOF. The same boundary is also drawn flat (z=0) for
% context, and colored per curve to match plot_boundary_density.

    n_curves = numel(curve_param.curve_n);
    colors = lines(n_curves);

    figure('Name', sprintf('Boundary density 3D (%s, h=%.4g)', name_str, h));
    hold on;
    % Flat reference outline, closed back to the first node.
    plot3([l1; l1(1)], [l2; l2(1)], zeros(numel(l1) + 1, 1), ...
        'Color', [0.6 0.6 0.6], 'LineWidth', 1, 'HandleVisibility', 'off');
    for i = 1:n_curves
        idx = curve_param.start_idx(i):curve_param.end_idx(i);
        next_i = mod(i, n_curves) + 1;

        % Append the next curve's (physical) first node -- unlike the
        % normalized-arclength 2D plot, no wrap-around special case is
        % needed here since l1/l2 are real (x,y) coordinates.
        idx_ext = [idx, curve_param.start_idx(next_i)];

        plot3(l1(idx_ext), l2(idx_ext), phi(idx_ext), 'Color', colors(i, :), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('curve %d', i));
    end
    hold off;

    grid on;
    view(-40, 30);
    xlabel('x'); ylabel('y'); zlabel('boundary density \phi (unwindowed)');
    title(sprintf('%s example boundary density over domain boundary (h = %.4g)', name_str, h));
    legend('show', 'Location', 'best');
end
