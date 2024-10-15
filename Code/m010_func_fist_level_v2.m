function m010_func_fist_level_v2(analysis_dir,data_fn,multicond_fn)

disp(analysis_dir)
disp(data_fn)
disp(multicond_fn)

cd '/mmfs1/data/aglinska/BC-ABCD-denoise/Code'
addpath('~/spm12')

spm_jobman('initcfg');
matlabbatch = {}; 
matlabbatch{1}.spm.stats.fmri_spec.dir = {analysis_dir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

disp(data_fn)
assert(exist(data_fn,'file'),'bad path for data_fn')
n = length(spm_vol(data_fn));

% get multicond info
disp(multicond_fn)
assert(exist(multicond_fn,'file'),'bad path for multicond_fn')

mkdir(analysis_dir)
delete([analysis_dir '/*'])
%% Add scans 
matlabbatch{1}.spm.stats.fmri_spec.sess.scans = arrayfun(@(x) [data_fn ',' num2str(x)],1:n,'UniformOutput',0)';
%%
matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {multicond_fn};
matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'face > place';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 1 1 1 1 1 -3 -3 0 0];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'place > face';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 -1 -1 -1 -1 -1 3 3 0 0];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 1;

%spm_jobman('initcfg');
spm_jobman('run',matlabbatch);

end