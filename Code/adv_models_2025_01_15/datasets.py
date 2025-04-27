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


class TrainDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
    self.obs = X
    self.noi = Y

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])

  def __getitem__(self, index):
    observation = self.obs[index]
    noise = self.noi[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug
    #return observation, noise

class TrainDataset_coords(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y, gm_coords):
    self.obs = X
    self.noi = Y
    self.gm_coords = gm_coords

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])

  def __getitem__(self, index):
    observation = self.obs[index]
    noise = self.noi[index]
    coords = self.gm_coords[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug, coords
    #return observation, noise

class TrainDataset_coords2(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y, gm_coords):
    self.obs = np.concatenate((X,gm_coords),axis=1)
    self.noi = Y
    self.gm_coords = gm_coords

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])
      
  def __getitem__(self, index):
    observation = self.obs[index]
    noise = self.noi[index]
    coords = self.gm_coords[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug, coords
    #return observation, noise


class TrainDataset_coords2conv(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y, gm_coords):
    X_tensor = torch.from_numpy(X).unsqueeze(1)
    gm_tensor = torch.from_numpy(gm_coords).unsqueeze(2)
    gm_tensor = gm_tensor.repeat(1,1,X_tensor.shape[2])
    self.obs = torch.cat((X_tensor,gm_tensor),axis=1)
    self.noi = Y
    self.gm_coords = gm_coords

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])

  def __getitem__(self, index):
    observation = self.obs[index,:,:]
    min_val = torch.min(observation[0,:])
    max_val = torch.max(observation[0,:])
    observation[0,:] = (observation[0,:]-min_val)/(max_val-min_val)
    noise = self.noi[index]
    min_val = np.min(noise)
    max_val = np.max(noise)
    noise = (noise-min_val)/(max_val-min_val)
    coords = self.gm_coords[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug, coords
    #return observation, noise


class TrainDataset_coords_all(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y, gm_coords):
    print(type(X))
    if len(Y) < len(X):
        ratio = len(X)/len(Y)
        YY = np.repeat(Y, int(np.ceil(ratio)), axis=0)
        YY = YY[0:len(X)]
    self.obs = X
    self.noi = YY
    self.gm_coords = gm_coords

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])

  def __getitem__(self, index):
    observation = self.obs[index]
    noise = self.noi[index]
    coords = self.gm_coords[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug, coords
    #return observation, noise

class DenoiseDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X):
    self.obs = X
    
  def __len__(self):
    return self.obs.shape[0]

  def __getitem__(self, index):
    observation = self.obs[index]
    return observation