import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as nibp
from scipy import signal
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
import ants 

def gather_niftis(sub,epi_fn,cf_fn,gm_fn,brain_mask,run):
    r = int(run) # If runID is passed from a terminal call (via papermill), make sure it's converted to int

    ## Gather their niftis
    epi = ants.image_read(epi_fn.format(sub=sub,r=r)) # BOLD data
    gm = ants.image_read(gm_fn) # ROI mask
    cf = ants.image_read(cf_fn) # RONI mask
    brain = ants.image_read(brain_mask.format(sub=sub,r=r)) # Loose brain mask for after training
    
    nt = epi.shape[-1] # Number of timepoints
    ndummy = 0 # how many dummy scans to discard
    
    epi_flat = epi.numpy().reshape(-1,nt).transpose() 
    epi_flat[0:ndummy,:] = epi_flat[ndummy+1::,:].mean(axis=0) # What to do with dummy scans, set to mean
    epi_flat = epi_flat.transpose()
    gm_flat = gm.numpy().flatten().astype(int) # (1082035,)
    cf_flat = cf.numpy().flatten().astype(int) # (1082035,)
    
    assert max(np.unique(cf_flat+gm_flat))!=2, 'overlap' # Assert that voxels in the ROI are NOT in the RONI and vice versa
    return epi,gm,cf,brain,epi_flat,gm_flat,cf_flat

def gather_niftis_coords(sub,epi_fn,cf_fn,gm_fn,brain_mask,run):
    r = int(run) # If runID is passed from a terminal call (via papermill), make sure it's converted to int

    ## Gather their niftis
    epi = ants.image_read(epi_fn.format(sub=sub,r=r)) # BOLD data
    gm = ants.image_read(gm_fn) # ROI mask
    cf = ants.image_read(cf_fn) # RONI mask
    brain = ants.image_read(brain_mask.format(sub=sub,r=r)) # Loose brain mask for after training
    
    nt = epi.shape[-1] # Number of timepoints
    ndummy = 0 # how many dummy scans to discard

    # Create 3D coordinate grids
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(gm.shape[0]),  # x-coordinates
        np.arange(gm.shape[1]),  # y-coordinates
        np.arange(gm.shape[2]),  # z-coordinates
        indexing="ij"  # "ij" for matrix-style indexing
    )
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    z_coords_flat = z_coords.flatten()
    
    epi_flat = epi.numpy().reshape(-1,nt).transpose() 
    epi_flat[0:ndummy,:] = epi_flat[ndummy+1::,:].mean(axis=0) # What to do with dummy scans, set to mean
    epi_flat = epi_flat.transpose()
    gm_flat = gm.numpy().flatten().astype(int) # (1082035,)
    cf_flat = cf.numpy().flatten().astype(int) # (1082035,) 
    
    gm_x_coords = x_coords_flat[gm_flat.astype(bool)]
    gm_y_coords = y_coords_flat[gm_flat.astype(bool)]
    gm_z_coords = z_coords_flat[gm_flat.astype(bool)]
    gm_coords = np.stack((gm_x_coords, gm_y_coords, gm_z_coords), axis=-1)

    cf_x_coords = x_coords_flat[cf_flat.astype(bool)]
    cf_y_coords = y_coords_flat[cf_flat.astype(bool)]
    cf_z_coords = z_coords_flat[cf_flat.astype(bool)]
    cf_coords = np.stack((cf_x_coords, cf_y_coords, cf_z_coords), axis=-1)
    
    assert max(np.unique(cf_flat+gm_flat))!=2, 'overlap' # Assert that voxels in the ROI are NOT in the RONI and vice versa
    return epi,gm,cf,brain,epi_flat,gm_flat,cf_flat,gm_coords,cf_coords

def norm(mat):
    return (mat - mat.min()) / ( mat.max()-mat.min() )

def safe_mkdir(path):
    # Make directory safely
    if not os.path.exists(path):
        os.mkdir(path)

def correlation(x,y):
  x_mean = np.repeat(x.mean(),x.shape,axis=0)
  y_mean = np.repeat(y.mean(),y.shape,axis=0)
  cov = (x-x_mean)*(y-y_mean)
  r = cov.sum()/(x.std()*y.std()*x.shape[0])
  return r

def remove_std0(arr):
    std0 = np.argwhere(np.std(arr, axis=1) == 0.0)
    arr_o = np.delete(arr,std0 ,axis=0) 
    return arr_o

def remove_std0_coords(arr,coords):
    std0 = np.argwhere(np.std(arr, axis=1) == 0.0)
    arr_o = np.delete(arr,std0 ,axis=0) 
    coords_o = np.delete(coords,std0,axis=0)
    return arr_o,coords_o

def correlate_columns(arr1, arr2):
    """
    Computes the Pearson correlation between corresponding columns of two matrices.
    
    Parameters:
    arr1 (np.ndarray): First matrix of shape (370, 1000)
    arr2 (np.ndarray): Second matrix of shape (370, 1000)
    
    Returns:
    np.ndarray: 1D array of correlations for each column (size 1000)
    """
    # Ensure input arrays are numpy arrays
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    # Subtract the mean of each column (normalize)
    arr1_centered = arr1 - np.mean(arr1, axis=0)
    arr2_centered = arr2 - np.mean(arr2, axis=0)
    
    # Compute the numerator (covariance)
    numerator = np.sum(arr1_centered * arr2_centered, axis=0)
    
    # Compute the denominator (product of standard deviations)
    denominator = np.sqrt(np.sum(arr1_centered**2, axis=0) * np.sum(arr2_centered**2, axis=0))
    
    # Compute the Pearson correlation for each column
    correlation = numerator / denominator
    
    return correlation

    
def get_betas(ffa_batch,design_matrix):
    # Calculates betas for different regressors
    Y = ffa_batch
    do_norm=True # This is important, incase DeepCor scales the signal, we dont want the betas to be artificailly inflated
    if do_norm==True:
        Y = Y-Y.mean(axis=1)[:,np.newaxis]
        Y = Y/Y.std(axis=1)[:,np.newaxis]
    Y = Y.T
    X = design_matrix.values # Regressor information 
    beta = np.linalg.inv(X.T @ design_matrix) @ design_matrix.T @ Y
    beta = beta.T
    return beta
    
def get_contrast(ffa_fg,design_matrix):
    # Calculates face+body > house+scene+object+scramble
    beta = get_betas(ffa_fg,design_matrix).values
    contrast_vector = np.array([2,2,-1,-1,-1,-1,0,0,0,0])
    contrast_values = beta @ contrast_vector
    return contrast_values.mean()

def make_RDM(inVec,data_scale='ratio',metric='euclidean'):
    # Make RDM for RSA analyses
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    vec = inVec
    vec = (vec - min(vec.flatten())) / (max(vec.flatten())-min(vec.flatten()))
    
    if np.ndim(inVec)==1: # must be at least 2D
        vec = np.vstack((vec,np.zeros(vec.shape))).transpose()
                   
    mat = squareform(pdist(vec,metric=metric).transpose())
    if data_scale=='ordinal':
        mat[mat!=0]=1 # Make into zeros and ones
        
    return mat

def get_triu(inMat):
    assert np.ndim(inMat)==2, 'not 2 dim, wtf'
    assert inMat.shape[0]==inMat.shape[1], 'not a square'

    n = inMat.shape[0]
    triu_vec = inMat[np.triu_indices(n=n,k=1)]
    return triu_vec

class Scaler():
    def __init__(self,inputs):
        self.data = inputs
        self.mean = np.mean(inputs,axis=1)
        self.std = np.std(inputs, axis=1)
        self.vox, self.time = inputs.shape
    def transform(self,inputs):
        self.mean = np.reshape(self.mean,(self.vox,1))
        self.m_large = np.repeat(self.mean,self.time,axis=1)
        self.std = np.reshape(self.std,(self.vox,1))
        self.s_large = np.repeat(self.std,self.time,axis=1)
        return np.divide(inputs-self.m_large,self.s_large)
    def inverse_transform(self,outputs):
        return np.multiply(outputs,self.s_large)+self.m_large

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16,8]):
        super(Discriminator, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_corr_w_reg(ffa_list,batch_size,device,model,face_reg,place_reg):
    # Function that tracks correlations with task regressors during training
    ffa_batch = ffa_list[0:batch_size,:] # FFA BOLD (not denoised)
    ffa_batch = torch.tensor(ffa_batch[:,np.newaxis,:]).to(device)
    
    ffa_tg = model.forward_tg(ffa_batch) # Reconstruction
    ffa_tg = ffa_tg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_fg = model.forward_fg(ffa_batch) # Denoised
    ffa_fg = ffa_fg[0].detach().cpu().numpy()[:,0,:]
    
    ffa_bg = model.forward_bg(ffa_batch) # Noise features
    ffa_bg = ffa_bg[0].detach().cpu().numpy()[:,0,:]

    # Calculate correlations with regressors
    c_ffa_bold_facereg = np.corrcoef(ffa_list[:,:].mean(axis=0),face_reg)[0,1]
    c_ffa_bold_placereg = np.corrcoef(ffa_list[:,:].mean(axis=0),place_reg)[0,1]
    c_ffa_bold_diff = c_ffa_bold_facereg-c_ffa_bold_placereg
    
    c_tg_bold_facereg = np.corrcoef(ffa_tg[:,:].mean(axis=0),face_reg)[0,1]
    c_tg_bold_placereg = np.corrcoef(ffa_tg[:,:].mean(axis=0),place_reg)[0,1]
    c_tg_bold_diff = c_tg_bold_facereg-c_tg_bold_placereg
    
    c_fg_bold_facereg = np.corrcoef(ffa_fg[:,:].mean(axis=0),face_reg)[0,1]
    c_fg_bold_placereg = np.corrcoef(ffa_fg[:,:].mean(axis=0),place_reg)[0,1]
    c_fg_bold_diff = c_fg_bold_facereg-c_fg_bold_placereg
    
    c_bg_bold_facereg = np.corrcoef(ffa_bg[:,:].mean(axis=0),face_reg)[0,1]
    c_bg_bold_placereg = np.corrcoef(ffa_bg[:,:].mean(axis=0),place_reg)[0,1]
    c_bg_bold_diff = c_bg_bold_facereg-c_bg_bold_placereg
    
    return [c_ffa_bold_facereg,
    c_ffa_bold_placereg,
    c_ffa_bold_diff,
    c_tg_bold_facereg,
    c_tg_bold_placereg,
    c_tg_bold_diff,
    c_fg_bold_facereg,
    c_fg_bold_placereg,
    c_fg_bold_diff,
    c_bg_bold_facereg,
    c_bg_bold_placereg,
    c_bg_bold_diff]

### Functions for density estimation with normalizing flows

class FlowModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.flow_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

    def forward(self, x):
        log_prob = self.base_dist.log_prob(x)
        for layer in self.flow_layers:
            x = torch.tanh(layer(x))  # Example transformation
            log_prob += torch.log(torch.abs(1 - x**2)).sum(-1)
        return log_prob