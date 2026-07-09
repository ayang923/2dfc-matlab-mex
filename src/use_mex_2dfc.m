function tf = use_mex_2dfc()
% USE_MEX_2DFC  True if the compiled mex-accelerated kernels are available.
%
% Every mex dispatch point in this codebase (Q_patch_obj.inverse_M_p,
% Q_patch_obj.locally_compute, R_cartesian_mesh_obj.interpolate_patch,
% R_cartesian_mesh_obj.locally_compute_vec, fcont_gram_blend_S,
% inpolygon_mesh, barylag, IE_curve_seq_obj.u_num, IE_curve_seq_obj.u_num_batch)
% checks this before calling into its mex gateway, falling back to the
% original pure-MATLAB implementation otherwise. This keeps the repository
% fully usable (just slower) on a machine where `build_mex` hasn't been run
% or mex compilation isn't available, and keeps every public function's
% behavior identical either way -- only the implementation backing it changes.
%
% The check result is cached for the session (mex file presence doesn't
% change at runtime) using a persistent variable, so this costs one `exist`
% call total rather than one per patch/grid-point.

    persistent cached
    if isempty(cached)
        cached = exist('inpolygon_mesh_mex', 'file') == 3;
    end
    tf = cached;
end
