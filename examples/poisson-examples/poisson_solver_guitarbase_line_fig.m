% POISSON_SOLVER_GUITARBASE_LINE_FIG  Plots the guitarbase Poisson solution
% along the line y=0, near the domain's concave ("waist") corner, for both
% the non-singular (poisson_solver_guitarbase.m) and singular/sharp-corner
% (poisson_solver_guitarbase_sing.m) boundary data.
%
% The guitarbase boundary has 4 corners (one between each pair of adjacent
% curves); measuring the turning angle between consecutive tangents at each
% (turn > 0 => convex, turn < 0 => concave/reflex) finds exactly one
% concave corner, at the junction of curve 2 (right connector) and curve 3
% (upper connector), located at (x,y) = (1, 0) -- turn angle -45 deg (a
% 225 deg interior angle). The other 3 corners (curve1-curve2 at
% (1.41421,-1), curve3-curve4 at (1.41421,1), curve4-curve1 at the origin)
% are all convex (turn +102, +123, +90 deg respectively).
%
% Checking inpolygon along y=0 confirms the domain's y=0 cross-section is
% exactly the open interval x in (0,1): the origin (convex corner) and
% (1,0) (this concave corner) are its two endpoints, and x > 1 along y=0 is
% exterior all the way out (the concave corner's 225 deg interior angle is
% on the x<1 side; the local 135 deg exterior wedge straddles x>1). So,
% unlike the two-sided intuition a generic reflex corner might suggest,
% (1,0) is only reachable from the interior along y=0 from the x<1 side --
% this line mesh approaches it from there only, analogous to how the origin
% corner is approached from its own interior side (x>0).
%
% Evaluation mirrors poisson_solver.m/laplace_solver.m's own two-part
% solution u = u_p + u_h, but evaluates both parts directly at the line
% nodes instead of over R's whole Cartesian grid:
%
%   - Particular solution u_p: FC2D + inv_lap (exactly as poisson_solver.m),
%     then interpolated onto the line nodes via R.locally_compute_vec(x,y,M)
%     -- the same interpolation poisson_solver.m uses to evaluate u_p at
%     the boundary when building the homogeneous BVP's Dirichlet data.
%   - Homogeneous solution u_h: the double-layer density gr_phi is solved
%     for once (IE_curve_seq_obj.construct_A_b, the same system
%     laplace_solver.m solves), then evaluated directly at the line nodes
%     via adaptive quadrature refinement (IE_curve_seq_obj.u_num_batch at
%     successively finer resolutions -- refine_gr_phi/next_rho_IE, copied
%     verbatim from laplace_solver.m -- until two successive resolutions
%     agree to within int_eps). This is exactly laplace_solver.m's
%     "well-interior" evaluation procedure, applied uniformly to every line
%     node regardless of distance to the corner: laplace_solver.m's
%     smooth-patch polynomial interpolation and well-interior/patch/corner
%     point classification are both skipped entirely, since the line nodes
%     here specifically run up to the domain's sharp corner, which is
%     exactly where that interpolation wouldn't apply anyway (laplace_solver
%     itself falls back to this same direct refinement for corner-region
%     points).
%
% Points very close to the corner can need many refinement rounds to reach
% int_eps=1e-13 (see laplace_solver.m's MAX_RHO_IE comment); a warning is
% raised (not an error) if the cap is hit, and the best available value is
% kept.
%
% du/dx along the same line is also plotted, via a centered finite
% difference of u (step fd_frac*delta, i.e. a step scaled to the local
% distance from the corner, evaluated with the same solve_line_solution
% pipeline -- no analytic derivative of the BIE kernels is needed). This
% matters because for a concave corner of interior angle omega, the
% classical corner-singularity expansion for Laplace's equation gives a
% leading term ~ r^(pi/omega); here omega = 225 deg = 5*pi/4, so
% pi/omega = 4/5 = 0.8 < 1. That means u itself stays continuous and
% bounded right up to the corner (a plot of u alone will never look
% "singular" here, no matter how close x gets to 1) -- but grad(u) behaves
% like r^(0.8-1) = r^(-0.2), which is unbounded as r -> 0. Plotting du/dx
% is what actually shows the corner singularity.
%
% delta_min is pushed down to 1e-4 (rather than the earlier 5e-3), and
% n_line_pts raised to 100, so the mesh gets much closer to the corner --
% r^-0.2 is a mild exponent, so the du/dx plot only really starts to look
% like it's blowing up once x is extremely close to 1. Getting this close
% means more IE refinement rounds per point (see laplace_solver.m's
% MAX_RHO_IE comment) -- expect this script to run noticeably slower than
% the 5e-3 version.

clc; clear; close all;

% -------------------------------------------------------------------------
% Requested solver parameters (shared by both examples)
% -------------------------------------------------------------------------
h       = 1e-3;
int_eps = 1e-13;
n_x     = 1728;
n_y     = 2187;
d       = 5;

eps_xi_eta = 1e-13;
eps_xy     = 1e-13;

C    = 27;
n_r  = 6;
M    = d + 3;
p    = M;
G_cf = 1;   % matches poisson_solver_guitarbase.m/_sing.m's own poisson_solver call

% -------------------------------------------------------------------------
% Refined line mesh along y=0, near the concave corner at (1,0), approached
% from its interior side x<1 only (see header -- x>1 along y=0 is exterior,
% all the way out). Log-spaced in distance from the corner to resolve
% behavior close to it; adjust delta_min/delta_max/n_line_pts as needed
% (smaller delta_min => more IE refinement rounds per laplace_solver's
% adaptive scheme, hence slower).
% -------------------------------------------------------------------------
x_corner   = 1;
delta_min  = 1e-4;
delta_max  = 0.3;
n_line_pts = 100;

deltas = logspace(log10(delta_min), log10(delta_max), n_line_pts)';
x_line = sort(x_corner - deltas);
y_line = zeros(size(x_line));

% Centered finite-difference step for du/dx, scaled to each point's own
% distance from the corner (recomputed post-sort, so it stays correct
% regardless of x_line's ordering) -- keeps the finite-difference step
% small relative to distance-to-corner across the whole delta_min..delta_max
% range, rather than using one fixed absolute step.
fd_frac    = 0.01;
delta_line = x_corner - x_line;
fd_step    = fd_frac * delta_line;

% -------------------------------------------------------------------------
% Shared FC data
% -------------------------------------------------------------------------
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
cfgs(1).name       = 'manufactured';
cfgs(1).u_boundary = @(x, y) -1/2 * sin(3*pi*x - 1) .* sin(3*pi*y - 1);
cfgs(1).exact      = cfgs(1).u_boundary;   % manufactured solution: u == u_boundary everywhere
cfgs(1).exact_dx   = @(x, y) -(3*pi/2) * cos(3*pi*x - 1) .* sin(3*pi*y - 1);   % d/dx of exact

cfgs(2).name       = 'non-manufactured';
cfgs(2).u_boundary = @(x, y) (x/2).^2 + (y/2).^2;
cfgs(2).exact      = [];                   % no closed-form exact solution
cfgs(2).exact_dx   = [];

figure('Name', 'Guitarbase solution along y=0 near the concave corner');
ax_u = axes();
hold(ax_u, 'on');
set(ax_u, 'FontSize', 16);

figure('Name', 'Guitarbase du/dx along y=0 near the concave corner');
ax_du = axes();
hold(ax_du, 'on');
set(ax_du, 'FontSize', 16);

colors = lines(numel(cfgs));

for k = 1:numel(cfgs)
    cfg = cfgs(k);

    rng('default')
    curve_seq = build_guitarbase_curve_seq(h);

    % Evaluate u at the line nodes and at the +/- finite-difference offset
    % nodes in one combined batched solve (same FC2D/BIE solve is reused
    % for all of them; only the u_p interpolation and u_h adaptive
    % refinement depend on which points are queried).
    n_pts  = numel(x_line);
    x_eval = [x_line; x_line - fd_step; x_line + fd_step];
    y_eval = zeros(size(x_eval));

    tic;
    u_eval = solve_line_solution(curve_seq, f, cfg.u_boundary, h, G_cf, p, int_eps, ...
        eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x, n_y, x_eval, y_eval);
    fprintf('%s: line solve took %.3fs\n', cfg.name, toc);

    u_line  = u_eval(1:n_pts);
    u_minus = u_eval(n_pts+1 : 2*n_pts);
    u_plus  = u_eval(2*n_pts+1 : 3*n_pts);
    du_dx   = (u_plus - u_minus) ./ (2 * fd_step);

    plot(ax_u, x_line, u_line, '-', 'Color', colors(k, :), 'LineWidth', 1.5, ...
        'DisplayName', cfg.name);
    plot(ax_du, x_line, du_dx, '-', 'Color', colors(k, :), 'LineWidth', 1.5, ...
        'DisplayName', cfg.name);

    if ~isempty(cfg.exact)
        u_exact = cfg.exact(x_line, y_line);
        fprintf('%s: max|u_num - u_exact| along line = %.6e\n', cfg.name, max(abs(u_line - u_exact)));

        du_dx_exact = cfg.exact_dx(x_line, y_line);
        fprintf('%s: max|du/dx_num - du/dx_exact| along line = %.6e\n', cfg.name, max(abs(du_dx - du_dx_exact)));
    end
end
hold(ax_u, 'off');
hold(ax_du, 'off');

xline(ax_u, x_corner, '--', 'concave corner (1,0)', 'Color', [0.5 0.5 0.5], ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
ylabel(ax_u, 'u(x, 0)', 'FontSize', 16);
title(ax_u, sprintf('Guitarbase solution along y=0 near the concave corner (h=%.4g, int\\_eps=%.1e)', h, int_eps));
legend(ax_u, 'show', 'Location', 'best', 'Orientation', 'horizontal');

xline(ax_du, x_corner, '--', 'concave corner (1,0)', 'Color', [0.5 0.5 0.5], ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xlabel(ax_du, 'x  (y = 0, approaching the concave corner at x=1 from the interior side x<1)', 'FontSize', 16);
ylabel(ax_du, 'du/dx (x, 0)', 'FontSize', 16);
title(ax_du, 'du/dx along the same line (expected to diverge like r^{-0.2} as x -> 1; see header)');
legend(ax_du, 'show', 'Location', 'best', 'Orientation', 'horizontal');

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

function u_line = solve_line_solution(curve_seq, f, u_boundary, h, G_cf, p, int_eps, ...
    eps_xi_eta, eps_xy, d, C, n_r, A, Q, M, n_x_padded, n_y_padded, x_line, y_line)
% SOLVE_LINE_SOLUTION  Solves u = u_p + u_h (poisson_solver.m's decomposition)
% and evaluates it directly at (x_line, y_line), without evaluating u over
% R's whole Cartesian grid.
%
%   u_p (particular solution): FC2D + inv_lap, exactly as poisson_solver.m,
%   then interpolated onto the line nodes via R.locally_compute_vec (the
%   same interpolation poisson_solver.m uses for u_p at the boundary).
%
%   u_h (homogeneous solution): the same BIE system laplace_solver.m solves
%   (IE_curve_seq_obj.construct_A_b at cfac=G_cf), then evaluated directly
%   at the line nodes by adaptive IE refinement (see
%   eval_homogeneous_refined below) instead of laplace_solver's
%   well-interior/smooth-patch/corner-region evaluation passes.

    [R, ~, ~, ~] = FC2D(f, h, curve_seq, eps_xi_eta, eps_xy, d, C, n_r, A, Q, C, A, Q, M, ...
        n_x_padded, n_y_padded, true, false);
    R.f_R = R.inv_lap();

    x_line = x_line(:);
    y_line = y_line(:);

    u_p_line = R.locally_compute_vec(x_line, y_line, M);

    u_G = @(x, y) u_boundary(x, y) - R.locally_compute_vec(x, y, M);

    curve_n_rho1 = zeros(curve_seq.n_curves, 1);
    curr = curve_seq.first_curve;
    for i = 1:curve_seq.n_curves
        curve_n_rho1(i) = ceil((curr.n - 1) * G_cf);
        curr = curr.next_curve;
    end
    curve_param_rho1 = curve_param_obj(curve_n_rho1);

    IE_curve_seq = IE_curve_seq_obj(curve_seq, p);
    [A_rho1, b_rho1] = IE_curve_seq.construct_A_b(curve_param_rho1, u_G);
    gr_phi_rho1     = A_rho1 \ b_rho1;
    gr_phi_fft_rho1 = fftshift(fft(gr_phi_rho1)) / curve_param_rho1.n_total;

    u_h_line = eval_homogeneous_refined(IE_curve_seq, curve_param_rho1, ...
        gr_phi_rho1, gr_phi_fft_rho1, x_line, y_line, int_eps);

    u_line = u_p_line + u_h_line;
end

% =========================================================================

function u_h = eval_homogeneous_refined(IE_curve_seq, curve_param_rho1, gr_phi_rho1, gr_phi_fft_rho1, x_pts, y_pts, int_eps)
% EVAL_HOMOGENEOUS_REFINED  Evaluates the homogeneous double-layer potential
% at x_pts,y_pts by direct adaptive IE refinement -- laplace_solver.m's
% "well-interior" evaluation procedure (refine_gr_phi/next_rho_IE,
% IE_curve_seq_obj.u_num_batch, ratcheting the resolution up until two
% successive levels agree to within int_eps), applied uniformly to every
% point passed in here. Unlike laplace_solver.m, there is no
% well-interior/smooth-patch/corner-region classification and no
% patch-local polynomial interpolation: every point is evaluated by direct
% refinement, which is what laplace_solver.m itself falls back to anyway
% for points too close to a corner for patch interpolation to apply -- and
% the point of this line mesh is specifically to probe right up to such a
% corner.

    MAX_RHO_IE = 10000;   % same cap as laplace_solver.m

    x_pts = x_pts(:);
    y_pts = y_pts(:);

    u_h      = IE_curve_seq.u_num_batch(x_pts, y_pts, curve_param_rho1, gr_phi_rho1);
    u_h_fine = zeros(size(u_h));

    to_update   = true(size(x_pts));
    rho_fine_IE = 2;

    while any(to_update) && rho_fine_IE <= MAX_RHO_IE
        [gr_phi_fine, curve_param_fine] = refine_gr_phi(curve_param_rho1, rho_fine_IE, gr_phi_fft_rho1);

        idxs = find(to_update);
        u_h_fine(idxs) = IE_curve_seq.u_num_batch(x_pts(idxs), y_pts(idxs), curve_param_fine, gr_phi_fine);

        resid = abs(u_h_fine - u_h);
        to_update = to_update & (resid > int_eps);
        u_h(to_update) = u_h_fine(to_update);
        rho_fine_IE = next_rho_IE(rho_fine_IE);
    end

    if any(to_update)
        warning('poisson_solver_guitarbase_line_fig:maxRhoIE', ...
            'Hit max IE refinement level (rho=%d) with %d line point(s) still not converged to int_eps=%.3e (max residual = %.3e); using best available value.', ...
            MAX_RHO_IE, sum(to_update), int_eps, max(resid(to_update)));
    end
end

% =========================================================================

function [gr_phi_rho, curve_param_rho] = refine_gr_phi(curve_param_rho1, rho_IE, gr_phi_fft_rho1)
% REFINE_GR_PHI  Zero-pads FFT coefficients to upsample the density by
% rho_IE. Copied verbatim from laplace_solver.m (a local, not exported,
% function there).
    curve_param_rho = curve_param_obj(curve_param_rho1.curve_n * rho_IE);
    n_base          = curve_param_rho1.n_total;
    n_fine          = n_base * rho_IE;

    padded_fft_coeffs = [ ...
        zeros(ceil((n_fine - n_base) / 2), 1); ...
        gr_phi_fft_rho1; ...
        zeros(floor((n_fine - n_base) / 2), 1)];
    gr_phi_rho = rho_IE * n_base * real(ifft(ifftshift(padded_fft_coeffs)));
end

% =========================================================================

function rho_next = next_rho_IE(rho)
% NEXT_RHO_IE  Advances the IE refinement level (+1 below 10, +10 below
% 100, +100 below 1000, etc). Copied verbatim from laplace_solver.m.
    step = 1;
    while rho >= step * 10
        step = step * 10;
    end
    rho_next = rho + step;
end
