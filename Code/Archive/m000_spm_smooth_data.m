function m000_spm_smooth_data(subID,rID)

disp('Args received')
disp(subID)
disp(rID)

cd '/mmfs1/data/aglinska/BC-ABCD-denoise/Code'
addpath('~/spm12')
spm_jobman('initcfg')

%load('subject_list.mat')
%s = 1;
%rID = 1; % run ID: 1 or 2
%subID = subjects{s};

%rID=str2num(rID);

% RAW data
%data_fn = sprintf('/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/%s/ses-baselineYear1Arm1/func/%s_ses-baselineYear1Arm1_task-nback_run-0%i_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii',subID,subID,rID);

% COMPCORR
data_fn = sprintf('/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/%s/ses-baselineYear1Arm1/func/%s_ses-baselineYear1Arm1_task-nback_run-0%i_space-MNI152NLin2009cAsym_res-2_desc-COMPCORR_bold.nii',subID,subID,rID);

% DeepCor
%data_fn = sprintf('/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/%s/ses-baselineYear1Arm1/func/%s_ses-baselineYear1Arm1_task-nback_run-0%i_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR_bold.nii',subID,subID,rID);



disp(data_fn)
assert(exist(data_fn,'file'),'bad path for data_fn')
n = length(spm_vol(data_fn));

matlabbatch{1}.spm.spatial.smooth.data = arrayfun(@(x) [data_fn ',' num2str(x)],1:n,'UniformOutput',0)';
matlabbatch{1}.spm.spatial.smooth.fwhm = [8 8 8];
matlabbatch{1}.spm.spatial.smooth.dtype = 0;
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = 's8_';
spm_jobman('run',matlabbatch)

end

