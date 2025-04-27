import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as nibp
from scipy import signal
from scipy import stats
from itertools import combinations_with_replacement
from numpy import savetxt
import nibabel as nib
import math
from numpy import random
import sklearn.preprocessing  
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn import linear_model
import sys
from IPython import display

import utils


def plot_r_w_reg(track):
    # plots correlation with regressors over time
    corr_w_reg = np.array(track['corr_w_reg'])
    c_ffa_bold_facereg = corr_w_reg[:,0]
    c_ffa_bold_placereg = corr_w_reg[:,1]
    c_ffa_bold_diff = corr_w_reg[:,2]
    c_tg_bold_facereg = corr_w_reg[:,3]
    c_tg_bold_placereg = corr_w_reg[:,4]
    c_tg_bold_diff = corr_w_reg[:,5]
    c_fg_bold_facereg = corr_w_reg[:,6]
    c_fg_bold_placereg = corr_w_reg[:,7]
    c_fg_bold_diff = corr_w_reg[:,8]
    c_bg_bold_facereg = corr_w_reg[:,9]
    c_bg_bold_placereg = corr_w_reg[:,10]
    c_bg_bold_diff = corr_w_reg[:,11]
    
    plt.figure(figsize=(20,5))
    plt.subplot(1,3,1)
    plt.plot(c_ffa_bold_diff)
    plt.plot(c_tg_bold_diff)
    plt.plot(c_fg_bold_diff)
    plt.plot(c_bg_bold_diff)
    plt.legend(['ffa','tg','fg','bg'])
    plt.title(f'Diff {c_fg_bold_diff[-1]:.2f}')
    
    plt.subplot(1,3,2)
    plt.plot(c_ffa_bold_facereg)
    plt.plot(c_tg_bold_facereg)
    plt.plot(c_fg_bold_facereg)
    plt.plot(c_bg_bold_facereg)
    plt.legend(['ffa','tg','fg','bg'])
    plt.title('Face Reg')
    
    plt.subplot(1,3,3)
    plt.plot(c_ffa_bold_placereg)
    plt.plot(c_tg_bold_placereg)
    plt.plot(c_fg_bold_placereg)
    plt.plot(c_bg_bold_placereg)
    plt.legend(['ffa','tg','fg','bg'])
    plt.title('Place Reg')

def show_voxel_recon(inputs_gm,outputs_gm_tg,outputs_gm_fg,outputs_gm_bg,inputs_cf,outputs_cf_tg,outputs_cf_fg,outputs_cf_bg):
    # Shows voxel reconstructions
    plt.figure(figsize=(15,5))
    plt.plot( inputs_gm.detach().cpu().numpy()[:,0,:][0,:] )
    plt.plot( outputs_gm_tg.detach().cpu().numpy()[:,0,:][0,:] )
    plt.plot( outputs_gm_fg.detach().cpu().numpy()[:,0,:][0,:] )
    plt.plot( outputs_gm_bg.detach().cpu().numpy()[:,0,:][0,:] )
    plt.legend(['inputs_gm','outputs_gm_tg','outputs_gm_fg','outputs_gm_bg'])
    plt.title('GM')
    
    
    plt.figure(figsize=(15,5))
    plt.plot( inputs_cf.detach().cpu().numpy()[:,0,:][0,:] )
    plt.plot( outputs_cf_tg.detach().cpu().numpy()[:,0,:][0,:] )
    plt.plot( outputs_cf_fg.detach().cpu().numpy()[:,0,:][0,:] )
    plt.plot( outputs_cf_bg.detach().cpu().numpy()[:,0,:][0,:] )
    plt.legend(['inputs_cf','outputs_cf_tg','outputs_cf_fg','outputs_cf_bg'])
    plt.title('CF')

def show_dashboard(track):
    # Dashboard plotting losses 

    running_loss_L = track['running_loss_L']
    running_recons_L = track['running_recons_L']
    running_recons_roi_L = track['running_recons_roi_L']
    running_recons_roni_L = track['running_recons_roni_L']
    running_KLD_L = track['running_KLD_L']
    running_TC_L = track['running_TC_L']
    running_recons_discourage_L = track['running_recons_discourage_L']
    running_VOL_L = track['running_VOL_L']
    running_recons_FG_L = track['running_recons_FG_L']
    running_frequency_L = track['running_frequency_L']
    
    plt.figure(figsize=(20,10))
    plt.subplot(3,4,1)
    plt.plot(running_loss_L);plt.title(f'running_loss_L: {running_loss_L[-1]:.4f}')
    
    plt.subplot(3,4,2)
    plt.plot(running_recons_L);plt.title(f'running_recons_L: {running_recons_L[-1]:.4f}')
    
    plt.subplot(3,4,3)
    plt.plot(running_recons_roi_L);plt.title(f'running_recons_roi_L: {running_recons_roi_L[-1]:.4f}')
    
    plt.subplot(3,4,4)
    plt.plot(running_recons_roni_L);plt.title(f'running_recons_roni_L: {running_recons_roni_L[-1]:.4f}')
    
    
    plt.subplot(3,4,5)
    plt.plot(running_KLD_L);plt.title(f'running_KLD_L: {running_KLD_L[-1]:.4f}')
    
    plt.subplot(3,4,6)
    plt.plot(running_TC_L);plt.title(f'running_TC_L: {running_TC_L[-1]:.4f}')
    
    plt.subplot(3,4,7)
    plt.plot(running_recons_discourage_L);plt.title(f'running_recons_discourage_L: {running_recons_discourage_L[-1]:.4f}')
    
    plt.subplot(3,4,8)
    plt.plot('TG-BG-SL-RSA');plt.title('RSA: {:.4f}'.format(track['TG-BG-SL-RSA'][-1]))
    
    plt.subplot(3,4,9)
    plt.plot(running_VOL_L);plt.title(f'running_VOL_L: {running_VOL_L[-1]:.4f}')
    
    plt.subplot(3,4,10)
    plt.plot(running_recons_FG_L);plt.title(f'running_recons_FG_L: {running_recons_FG_L[-1]:.4f}')
    
    plt.subplot(3,4,11)
    plt.plot(running_frequency_L);plt.title(f'frequency loss: {running_frequency_L[-1]:.4f}')
    # plt.plot(running_recons_roi_L[-10::]);plt.title(f'running_recons_roi_L (last 10): {running_recons_roi_L[-1]:.4f}')
    
    plt.subplot(3,4,12)
    plt.plot(track['varexp-gm'])
    plt.plot(track['varexp-cf'])
    plt.plot(track['varexp-fg'])
    plt.legend(['varexp-gm','varexp-cf','varexp-fg'])
    plt.title('var exp')

def show_ffa_dashboard(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    ffa_tg = model.forward_tg(ffa_batch)
    ffa_tg = ffa_tg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_fg = model.forward_fg(ffa_batch)
    ffa_fg = ffa_fg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_bg = model.forward_bg(ffa_batch)
    ffa_bg = ffa_bg[0].detach().cpu().numpy()[:,0,:]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,4,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,4,2)
    plt.plot(ffa_tg[:,:].mean(axis=0))
    plt.plot(face_reg*ffa_tg[:,:].mean(axis=0).max(),alpha=.5)
    #plt.plot(place_reg*ffa_tg[:,:].mean(axis=0).max())
    c = np.corrcoef(ffa_tg[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa TG + regs {c:.2f}')
    
    plt.subplot(1,4,3)
    plt.plot(ffa_fg[:,:].mean(axis=0))
    plt.plot(face_reg*ffa_fg[:,:].mean(axis=0).max(),alpha=.5)
    #plt.plot(place_reg*ffa_fg[:,:].mean(axis=0).max())
    c = np.corrcoef(ffa_fg[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa FG + regs {c:.2f}')
    
    plt.subplot(1,4,4)
    plt.plot(ffa_bg[:,:].mean(axis=0))
    plt.plot(face_reg*ffa_bg[:,:].mean(axis=0).max(),alpha=.5)
    #plt.plot(place_reg*ffa_bg[:,:].mean(axis=0).max())
    c = np.corrcoef(ffa_bg[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa BG + regs {c:.2f}')
    plt.tight_layout()
    
def show_contrast_dash(ffa_list,batch_size,device,model,epi_flat,cf_flat,design_matrix):
    contrast_vals = []

    # Plots contrast values over time
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    ffa_tg = model.forward_tg(ffa_batch)
    ffa_tg = ffa_tg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_fg = model.forward_fg(ffa_batch)
    ffa_fg = ffa_fg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_bg = model.forward_bg(ffa_batch)
    ffa_bg = ffa_bg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_batch = ffa_list[0:batch_size,:]

    conf_pcs = PCA(n_components=5).fit_transform(epi_flat[cf_flat==1,:].transpose())
    conf_pcs.shape
    
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(conf_pcs,ffa_batch.transpose());
    ffa_compcorr = ffa_batch.transpose()-lin_reg.predict(conf_pcs)
    ffa_compcorr = ffa_compcorr.transpose()
    
    contrast_vals.append([utils.get_contrast(ffa_batch,design_matrix),
    utils.get_contrast(ffa_compcorr,design_matrix),
    utils.get_contrast(ffa_tg,design_matrix),
    utils.get_contrast(ffa_fg,design_matrix),
    utils.get_contrast(ffa_bg,design_matrix),])
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(utils.get_betas(ffa_batch,design_matrix).mean(axis=0),'b.',markersize=20,alpha=.5);
    plt.plot(utils.get_betas(ffa_compcorr,design_matrix).mean(axis=0),'k.',markersize=20,alpha=.5);
    plt.plot(utils.get_betas(ffa_tg,design_matrix).mean(axis=0),'y.',markersize=20,alpha=.5);
    plt.plot(utils.get_betas(ffa_fg,design_matrix).mean(axis=0),'g.',markersize=20,alpha=.5);
    plt.plot(utils.get_betas(ffa_bg,design_matrix).mean(axis=0),'r.',markersize=20,alpha=.5);
    plt.legend(['BOLD','COMPCOR','TG','FG','BG']);
    plt.xticks(np.arange(design_matrix.shape[-1]),labels=list(design_matrix.columns),rotation=45);
    plt.title('betas')
    
    plt.subplot(1,2,2)
    plt.plot(np.array(contrast_vals)[:,0],'b-')
    plt.plot(np.array(contrast_vals)[:,1],'k-')
    plt.plot(np.array(contrast_vals)[:,2],'y-')
    plt.plot(np.array(contrast_vals)[:,3],'g-')
    plt.plot(np.array(contrast_vals)[:,4],'r-')
    plt.legend(['BOLD','COMPCOR','TG','FG','BG'],loc='lower left');
    plt.title('Contrast')


### For adversarial model

def adversarial_losses(track):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
    
    axes[0].plot(track['loss_L'])
    axes[0].set_title('Total loss')
    
    axes[1].plot(track['loss_signal_L'])
    axes[1].set_title('Signal loss')
    
    axes[2].plot(track['loss_noise_L'])
    axes[2].set_title('Noise loss')
    
    plt.tight_layout()


def adversarial_ffa(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    signal,noise, z = model.forward(ffa_batch)
    signal = signal[0].detach().cpu().numpy()[:,:]
    noise = noise[0].detach().cpu().numpy()[:,:]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,3,2)
    plt.plot(signal[:,:].mean(axis=0))
    plt.plot(face_reg*signal[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(signal[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa signal + regs {c:.2f}')
    
    plt.subplot(1,3,3)
    plt.plot(noise[:,:].mean(axis=0))
    plt.plot(face_reg*noise[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(noise[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa noise + regs {c:.2f}')
    
    plt.tight_layout()



def adversarial_ffa_coords(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg,ffa_coords):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    signal,noise, z = model.forward(ffa_batch,ffa_coords.to(device))
    signal = signal[0].detach().cpu().numpy()[:,:]
    noise = noise[0].detach().cpu().numpy()[:,:]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,3,2)
    plt.plot(signal[:,:].mean(axis=0))
    plt.plot(face_reg*signal[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(signal[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa signal + regs {c:.2f}')
    
    plt.subplot(1,3,3)
    plt.plot(noise[:,:].mean(axis=0))
    plt.plot(face_reg*noise[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(noise[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa noise + regs {c:.2f}')
    
    plt.tight_layout()



def adversarial_ffa_coords_corr(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg,ffa_coords):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    signal,noise, z = model.forward(ffa_batch,ffa_coords.to(device))
    signal = signal[0].detach().cpu().numpy()[:,:]
    noise = noise[0].detach().cpu().numpy()[:,:]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,3,2)
    plt.plot(stats.zscore(signal[:,:].mean(axis=0)))
    plt.plot(stats.zscore(face_reg),alpha=.5)
    c = np.corrcoef(signal[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa signal + regs {c:.2f}')
    
    plt.subplot(1,3,3)
    plt.plot(noise[:,:].mean(axis=0))
    plt.plot(face_reg*noise[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(noise[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa noise + regs {c:.2f}')
    
    plt.tight_layout()

def adversarial_ffa_coords2_corr(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg,ffa_coords):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,:]).to(device)
    
    signal,noise, z = model.forward(torch.cat((ffa_batch,ffa_coords.to(device)),axis=1))
    signal = signal.detach().cpu().numpy()[:,:]
    noise = noise.detach().cpu().numpy()[:,:]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,3,2)
    plt.plot(stats.zscore(signal[:,:].mean(axis=0)))
    plt.plot(stats.zscore(face_reg),alpha=.5)
    c = np.corrcoef(signal[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa signal + regs {c:.2f}')
    
    plt.subplot(1,3,3)
    plt.plot(noise[:,:].mean(axis=0))
    plt.plot(face_reg*noise[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(noise[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa noise + regs {c:.2f}')
    
    plt.tight_layout()


def adversarial_ffa_coords2_conv(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg,ffa_coords):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,:]).to(device)

    ffa_tensor = ffa_batch.unsqueeze(1)
    coords_tensor = ffa_coords.unsqueeze(2)
    coords_tensor = coords_tensor.repeat(1,1,ffa_tensor.shape[2])
    observation = torch.cat((ffa_tensor,coords_tensor.to(device)),axis=1)
    min_val = torch.min(observation[0,:])
    max_val = torch.max(observation[0,:])
    observation[0,:] = (observation[0,:]-min_val)/(max_val-min_val)
    
    signal,noise = model.forward(observation.to(device))
    signal = signal.detach().cpu().numpy()[:,:156]
    noise = noise.detach().cpu().numpy()[:,:156]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,3,2)
    plt.plot(stats.zscore(signal[:,:].mean(axis=0)))
    plt.plot(stats.zscore(face_reg),alpha=.5)
    c = np.corrcoef(signal[:,:].mean(axis=0),face_reg)[0,1]  # remember to fix this back
    plt.title(f'ffa signal + regs {c:.2f}')
    
    plt.subplot(1,3,3)
    plt.plot(noise[:,:].mean(axis=0))
    plt.plot(face_reg*noise[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(noise[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa noise + regs {c:.2f}')
    
    plt.tight_layout()

# For Autoencoder without denoising

def autoencoder_losses(track):
    
    plt.plot(track['loss_L'])
    plt.title('Total loss')
    
    plt.tight_layout()

def autoencoder_ffa(ffa_list,batch_size,device,model,noi_list,face_reg,place_reg):
    # Plots FFA, reconstructions and correlations with regressors
    ffa_batch = ffa_list[0:batch_size,:]
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    signal = model.forward(ffa_batch)
    signal = signal[0].detach().cpu().numpy()[:,:]
    
    plt.figure(figsize=(15,3))
    plt.subplot(1,2,1)
    plt.plot(ffa_list[:,:].mean(axis=0))
    plt.plot(noi_list.mean(axis=0))
    plt.plot(face_reg)
    plt.plot(place_reg)
    
    c = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa activity + regs: {c:.2f}')
    
    plt.subplot(1,2,2)
    plt.plot(signal[:,:].mean(axis=0))
    plt.plot(face_reg*signal[:,:].mean(axis=0).max(),alpha=.5)
    c = np.corrcoef(signal[:,:].mean(axis=0),face_reg)[0,1]
    plt.title(f'ffa signal + regs {c:.2f}')
    
    plt.tight_layout()



    