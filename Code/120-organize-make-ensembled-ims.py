import os
import numpy as np
import ants

nreps = 19
model_names = ['conv_denoise',
 'conv_denoise500',
 'conv_denoise_bn',
 'conv_denoise_large',
 'conv_denoise_large500',
 'conv_smooth',
 'conv_smooth-small',
 'conv_weights_unet_denoise',
 'conv_weights_unet_denoise500']


fn_temp = '../Data/StudyForrest/Stefano_adv_outputs/new-stefano-conv-allSubs-rep-{rep}/{model}-S_{s}-R_{r}-rep_0.nii.gz'


for s in np.random.permutation(np.arange(14)):
    for r in np.random.permutation([1,2,3,4]):
        for m in np.random.permutation(np.arange(len(model_names))):
            ofn_avg = f'../Data/StudyForrest/Stefano_adv_outputs/01-signals_averaged/S{s}-R{r}-{model_names[m]}-avg.nii.gz'
            ofn_med = f'../Data/StudyForrest/Stefano_adv_outputs/01-signals_averaged/S{s}-R{r}-{model_names[m]}-med.nii.gz'
            print(f'S{s}/R{r}/M{m}')
            if not all((os.path.exists(ofn_avg),os.path.exists(ofn_med))):                
                ims = [ants.image_read(fn_temp.format(s=0,r=1,rep=0,model=model_names[m])) for rep in range(nreps)]
                arr_avg = np.average(np.array([im.numpy() for im in ims]),axis=0)
                arr_med = np.median(np.array([im.numpy() for im in ims]),axis=0)
                im_avg = ims[0].new_image_like(arr_avg)
                im_avg.to_filename(ofn_avg)
                im_med = ims[0].new_image_like(arr_med)
                im_med.to_filename(ofn_med)