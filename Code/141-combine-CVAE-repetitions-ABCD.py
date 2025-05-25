import os
import numpy as np
import ants
from tqdm import tqdm

nreps = 20

analysis_dir = '../Data/StudyForrest/ensembles_last_CVAE/'
analysis_name = 'DeepCor-ABCD-v5'
fn_temp = os.path.join(analysis_dir,analysis_name,'signal_S{s}_R{r}_rep_{rep}.nii.gz')

for s in tqdm( np.random.permutation(np.arange(34)) ):
    for r in tqdm( np.random.permutation([1,2]), leave = False ):
        ofn_avg = os.path.join(analysis_dir,analysis_name,f'signal_S{s}_R{r}_avg.nii.gz')
        ofn_med = os.path.join(analysis_dir,analysis_name,f'signal_S{s}_R{r}_med.nii.gz')
        if not all((os.path.exists(ofn_avg),os.path.exists(ofn_med))):
            try:
                ims = [ants.image_read(fn_temp.format(s=s,r=r,rep=rep)) for rep in np.arange(20) if os.path.exists(fn_temp.format(s=s,r=r,rep=rep))]
                arrs = np.array([im.numpy() for im in ims])
                good_sub_vec = (np.isnan(arrs)*1.0).sum(axis=(1, 2, 3, 4))==0
                good_sub_vec*=arrs.max(axis=(1, 2, 3, 4))<1e3
                good_sub_vec*=arrs.max(axis=(1, 2, 3, 4))>1e-3
                arrs = arrs[good_sub_vec]
                if len(ims)>10:
                    arr_avg = np.average(arrs,axis=0)
                    arr_med = np.median(arrs,axis=0)
                    im_avg = ims[0].new_image_like(arr_avg)
                    im_avg.to_filename(ofn_avg)
                    im_med = ims[0].new_image_like(arr_med)
                    im_med.to_filename(ofn_med)
            except:
                print(f'error on S{s}/R{r}')