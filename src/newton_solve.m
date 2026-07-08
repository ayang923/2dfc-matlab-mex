function [xn, converged] = newton_solve(f, J, x0, tau, Nmax)
% NEWTON_SOLVE  Solves f(x) = 0 via Newton iteration.
%
% Terminates when both the residual and the step size fall below tau, or
% when Nmax iterations are reached.
%
% Inputs:
%   f    - Function handle returning the residual vector f(x)
%   J    - Function handle returning the Jacobian matrix J(x)
%   x0   - Initial guess vector
%   tau  - Convergence tolerance (applied to both residual and step size)
%   Nmax - Maximum number of iterations
%
% Outputs:
%   xn        - Final iterate (solution estimate)
%   converged - true if convergence criterion was met before Nmax iterations

    xn = x0;

    for i = 2:Nmax
        xprev = xn;
        xn    = xn - J(xn) \ f(xn);

        if max(abs(f(xn))) < tau && max(abs(xprev - xn)) < tau
            break
        end
    end

    converged = (i < Nmax);
end
