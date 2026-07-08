function build_mex()
% BUILD_MEX  Compiles the mex-accelerated gateways for 2dfc-matlab-mex.
%
% Compiles each mex/*_mex.c gateway together with the portable C kernels in
% csrc/src against the system CBLAS (Accelerate on macOS, OpenBLAS
% elsewhere -- no Intel MKL/compiler needed anywhere). Output is written to
% src/private/ so it's visible only to the .m files in src/ that dispatch
% to it (Q_patch_obj.m, R_cartesian_mesh_obj.m, fcont_gram_blend_S.m,
% inpolygon_mesh.m, barylag.m), matching MATLAB's private-function
% convention -- end users keep calling the same public API either way.
%
% Run this once after cloning the repo (and again after editing any file
% under csrc/ or mex/):
%   >> build_mex

    here = fileparts(mfilename('fullpath'));
    csrc_inc = fullfile(here, 'csrc', 'include');
    csrc_src = fullfile(here, 'csrc', 'src');
    mex_dir  = fullfile(here, 'mex');
    out_dir  = fullfile(here, 'src', 'private');

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    common_srcs = { ...
        fullfile(csrc_src, 'num_linalg.c'), ...
        fullfile(csrc_src, 'curve_eval.c'), ...
        fullfile(csrc_src, 'patch_kernels.c'), ...
        fullfile(csrc_src, 'fc_kernels.c'), ...
        fullfile(csrc_src, 'grid_interp.c'), ...
        fullfile(csrc_src, 'cartesian_kernels.c'), ...
        fullfile(mex_dir, 'mex_common.c')};

    gateways = { ...
        'inpolygon_mesh_mex', ...
        'fcont_gram_blend_S_mex', ...
        'barylag_mex', ...
        'q_patch_inverse_M_p_mex', ...
        'r_cartesian_mesh_interpolate_patch_mex'};

    inc_flags = {['-I' csrc_inc], ['-I' mex_dir]};

    if ismac
        extra_flags = {'LDFLAGS=$LDFLAGS -framework Accelerate'};
    elseif isunix
        extra_flags = {'-lopenblas'};
    else
        extra_flags = {'-lopenblas'}; % Windows: requires an OpenBLAS-compatible cblas.lib on the linker path
    end

    for i = 1:numel(gateways)
        name = gateways{i};
        src = fullfile(mex_dir, [name '.c']);
        fprintf('Building %s...\n', name);
        args = [{'-outdir', out_dir}, inc_flags, {src}, common_srcs, extra_flags];
        mex(args{:});
    end

    fprintf('Done. Mex binaries written to %s\n', out_dir);
end
