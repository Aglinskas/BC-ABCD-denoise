cd '/mmfs1/data/aglinska/BC-ABCD-denoise/Code'
addpath('~/spm12')
spm_jobman('initcfg')
load('subject_list.mat')

n = length(subjects);
for s = 1:n
for r = 1:2

disp(sprintf('%i/%i | %i/%i',s,n,r,2))
m000_spm_smooth_data(s,r)

end
end