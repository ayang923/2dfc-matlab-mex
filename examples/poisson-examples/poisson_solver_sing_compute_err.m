% POISSON_SOLVER_SING_COMPUTE_ERR  Compare a guitarbase Poisson solve against
% a reference solution and report error norms.
%
% Loads:
%   NUM_PATH - saved by poisson_solver_guitarbase_sing.m: the numerical
%              solution u_num_mat on its fine FC grid R (R.in_interior masks
%              the points strictly inside the domain).
%   REF_PATH - saved by poisson_solver_guitarbase_sing_ref.m: a higher-
%              resolution "exact" solution u_exact_mat, evaluated on R_eval,
%              a coarse grid built to share the same spacing/alignment as
%              NUM_PATH's R (so R.in_interior and R_eval.in_interior index
%              the same physical points in the same order).
%
% Reports:
%   Absolute max error - max(|u_num - u_exact|) over interior points.
%   Relative l2 error  - ||u_num - u_exact||_2 / ||u_exact||_2 over interior
%                        points.
%
% Interior points (rather than the full NaN-padded grid) are used because
% sum() does not skip NaNs the way max() does, so the l2 norm would come out
% NaN if computed over the whole padded grid.

clc; clear; close all;

NUM_PATH = 'poisson_solver_guitarbase_sing_h4.mat';
REF_PATH = 'poisson_solver_guitarbase_sing_h4_ref.mat';

% load reference and num solution mats
load(NUM_PATH, 'u_num_mat', 'R')
load(REF_PATH, 'u_exact_mat', 'R_eval')

u_num_int   = u_num_mat(R.in_interior);
u_exact_int = u_exact_mat(R_eval.in_interior);

if numel(u_num_int) ~= numel(u_exact_int)
    error(['Interior point counts of NUM_PATH and REF_PATH disagree (%d vs %d); ' ...
        'R and R_eval are not aligned to the same grid.'], ...
        numel(u_num_int), numel(u_exact_int));
end

% compute absolute max error and rel l2 err
err = u_num_int - u_exact_int;

err_inf = max(abs(err));
err_2   = sqrt(sum(err.^2)) ./ sqrt(sum(u_exact_int.^2));

fprintf('Absolute max error: %.6e\n', err_inf);
fprintf('Relative l2 error: %.6e\n', err_2);
