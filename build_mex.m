function build_mex()
% BUILD_MEX  Compiles the mex-accelerated gateways for 2dfc-matlab-mex.
%
% Compiles each mex/*_mex.c gateway together with the portable C kernels in
% csrc/src against the system CBLAS (Accelerate on macOS, OpenBLAS
% elsewhere -- no Intel MKL/compiler needed anywhere). Each gateway is
% written to the private/ subfolder of whichever directory its calling .m
% file lives in (MATLAB's private-function visibility is scoped to a
% function's own containing directory, not the whole MATLAB path): most
% gateways dispatch from src/*.m (Q_patch_obj.m, R_cartesian_mesh_obj.m,
% fcont_gram_blend_S.m, inpolygon_mesh.m, barylag.m) so they land in
% src/private/; the two IE gateways dispatch from
% examples/poisson-examples/IE_curve_seq_obj.m (which lives alongside the
% rest of the Poisson/IE solver, matching 2dfc-matlab's own layout, not
% under src/) so they land in examples/poisson-examples/private/ instead.
% End users keep calling the same public API either way.
%
% Run this once after cloning the repo (and again after editing any file
% under csrc/ or mex/):
%   >> build_mex

    here = fileparts(mfilename('fullpath'));
    csrc_inc = fullfile(here, 'csrc', 'include');
    csrc_src = fullfile(here, 'csrc', 'src');
    mex_dir  = fullfile(here, 'mex');
    src_out_dir      = fullfile(here, 'src', 'private');
    poisson_out_dir  = fullfile(here, 'examples', 'poisson-examples', 'private');

    if ~exist(src_out_dir, 'dir')
        mkdir(src_out_dir);
    end
    if ~exist(poisson_out_dir, 'dir')
        mkdir(poisson_out_dir);
    end

    common_srcs = { ...
        fullfile(csrc_src, 'num_linalg.c'), ...
        fullfile(csrc_src, 'curve_eval.c'), ...
        fullfile(csrc_src, 'patch_kernels.c'), ...
        fullfile(csrc_src, 'fc_kernels.c'), ...
        fullfile(csrc_src, 'grid_interp.c'), ...
        fullfile(csrc_src, 'cartesian_kernels.c'), ...
        fullfile(csrc_src, 'ie_kernels.c'), ...
        fullfile(mex_dir, 'mex_common.c')};

    gateways = { ...
        'inpolygon_mesh_mex',                     src_out_dir; ...
        'fcont_gram_blend_S_mex',                  src_out_dir; ...
        'barylag_mex',                             src_out_dir; ...
        'q_patch_inverse_M_p_mex',                  src_out_dir; ...
        'r_cartesian_mesh_interpolate_patch_mex',   src_out_dir; ...
        'r_cartesian_mesh_locally_compute_vec_mex', src_out_dir; ...
        'ie_u_num_mex',                             poisson_out_dir; ...
        'ie_u_num_batch_mex',                       poisson_out_dir};

    inc_flags = {['-I' csrc_inc], ['-I' mex_dir]};

    if ismac
        extra_flags = {'LDFLAGS=$LDFLAGS -framework Accelerate'};
    elseif isunix
        extra_flags = {'-lopenblas'};
    else
        extra_flags = {'-lopenblas'}; % Windows: requires an OpenBLAS-compatible cblas.lib on the linker path
    end

    for i = 1:size(gateways, 1)
        name    = gateways{i, 1};
        out_dir = gateways{i, 2};
        src = fullfile(mex_dir, [name '.c']);
        fprintf('Building %s...\n', name);
        args = [{'-outdir', out_dir}, inc_flags, {src}, common_srcs, extra_flags];
        mex(args{:});
    end

    fprintf('Done. Mex binaries written to %s and %s\n', src_out_dir, poisson_out_dir);
end
