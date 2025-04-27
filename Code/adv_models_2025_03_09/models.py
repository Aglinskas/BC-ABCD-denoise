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
from torch.autograd import Function

import utils

# Auxiliary functions

def compute_in(x):
  return (x-3)/2+1

def compute_in_size(x):
  for i in range(4):
    x = compute_in(x)
  return x

def compute_out_size(x):
  return ((((x*2+1)*2+1)*2+1)*2+1)

def compute_padding(x):
  rounding = np.ceil(compute_in_size(x))-compute_in_size(x)
  y = ((((rounding*2)*2)*2)*2)
  pad = bin(int(y)).replace('0b', '')
  if len(pad) < 4:
      for i in range(4-len(pad)):
          pad = '0' + pad
  final_size = compute_in_size(x+y)
  pad_out = bin(int(compute_out_size(final_size)-x)).replace('0b','')
  if len(pad_out) < 4:
      for i in range(4-len(pad_out)):
          pad_out = '0' + pad_out
  return pad,final_size, pad_out

# Loss functions

def high_frequency_penalty_loss(signal):  # MODIFIED BY SA
    """
    Penalizes frequencies of 0.2 Hz or higher in a signal sampled at 0.5 Hz.

    Args:
        signal (torch.Tensor): The input signal, shape (batch_size, signal_length).
        base_loss (torch.Tensor, optional): The base loss to combine with the penalty.

    Returns:
        torch.Tensor: The combined loss.
    """
    # print(signal.shape)
    # Signal length and frequency resolution
    signal_length = signal.shape[2]
    freq_resolution = 0.5 / signal_length  # Sampling rate / signal length

    # Perform Fourier Transform
    fft_coeffs = torch.fft.rfft(signal,dim=2)
    fft_magnitudes = torch.abs(fft_coeffs)

    # Compute the frequency indices
    frequencies = torch.fft.rfftfreq(signal_length, d=1/0.5)  # d is 1/sampling_rate
    # print(frequencies)
    
    # Create a mask for frequencies >= 0.2 Hz
    high_freq_mask = (frequencies >= 0.2) & (frequencies <= 0.5)  # Include Nyquist
    # print(high_freq_mask)
    
    # Compute penalty for high frequencies
    penalty = torch.sum(fft_magnitudes[:,:,high_freq_mask].squeeze() ** 2)

    return penalty



def high_frequency_penalty_loss_fc(signal):  # MODIFIED BY SA
    """
    Penalizes frequencies of 0.2 Hz or higher in a signal sampled at 0.5 Hz.

    Args:
        signal (torch.Tensor): The input signal, shape (batch_size, signal_length).
        base_loss (torch.Tensor, optional): The base loss to combine with the penalty.

    Returns:
        torch.Tensor: The combined loss.
    """
    # print(signal.shape)
    # Signal length and frequency resolution
    signal_length = signal.shape[1]
    freq_resolution = 0.5 / signal_length  # Sampling rate / signal length

    # Perform Fourier Transform
    fft_coeffs = torch.fft.rfft(signal,dim=1)
    fft_magnitudes = torch.abs(fft_coeffs)

    # Compute the frequency indices
    frequencies = torch.fft.rfftfreq(signal_length, d=1/0.5)  # d is 1/sampling_rate
    # print(frequencies)
    
    # Create a mask for frequencies >= 0.2 Hz
    high_freq_mask = (frequencies >= 0.2) & (frequencies <= 0.5)  # Include Nyquist
    # print(high_freq_mask)
    
    # Compute penalty for high frequencies
    penalty = torch.sum(fft_magnitudes[:,high_freq_mask].squeeze() ** 2)

    return penalty
    
# cVAE models - Aidas version with the addition of frequency loss

class cVAE(nn.Module):

    def __init__(self,discriminator,batch_size,device,in_channels: int,in_dim: int, latent_dim: tuple,hidden_dims: List = None, beta : float = 1, gamma : float = 1, delta : float = 1, scale_MSE_GM : float = 1, scale_MSE_CF : float = 1, scale_MSE_FG : float = 1,do_disentangle = True, freq_exp : float = 1, freq_scale : float = 1) -> None:
        super(cVAE, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim_z = self.latent_dim[0]
        self.latent_dim_s = self.latent_dim[1]
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.scale_MSE_GM = scale_MSE_GM
        self.scale_MSE_CF = scale_MSE_CF
        self.do_disentangle = do_disentangle
        self.freq_exp = freq_exp
        self.freq_scale = freq_scale
        self.scale_MSE_FG = scale_MSE_FG
        self.device = device
        self.batch_size = batch_size
        self.discriminator = discriminator

        modules_z = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]
        
        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)

        # Build Encoder
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_z.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder_z = nn.Sequential(*modules_z)
        self.fc_mu_z = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_z)
        self.fc_var_z = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_z)

        modules_s = []
        in_channels = self.in_channels
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_s.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder_s = nn.Sequential(*modules_s)
        self.fc_mu_s = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_s)
        self.fc_var_s = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_s)


        # Build Decoder
        modules = []

        #self.decoder_input = nn.Linear(2*latent_dim, hidden_dims[-1] * int(self.final_size))
        self.decoder_input = nn.Linear(self.latent_dim_s+self.latent_dim_z, hidden_dims[-1] * int(self.final_size))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))
           #out_channels

    def encode_z(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_z(input)
  
        result = torch.flatten(result, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_z(result)
        log_var = self.fc_var_z(result)

        return [mu, log_var]

    def encode_s(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_s(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_s(result)
        log_var = self.fc_var_s(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1,256,int(self.final_size))
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward_tg(self, input: Tensor) -> List[Tensor]: # ORIGINAL
        tg_mu_z, tg_log_var_z = self.encode_z(input)
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_z = self.reparameterize(tg_mu_z, tg_log_var_z)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        output = self.decode(torch.cat((tg_z, tg_s),1))
        return  [output, input, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s]

    def forward_fg(self, input: Tensor) -> List[Tensor]:
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        zeros = torch.zeros(tg_s.shape[0],self.latent_dim_z)
        zeros = zeros.to(self.device)
        output = self.decode(torch.cat((zeros, tg_s),1))
        return  [output, input, tg_mu_s, tg_log_var_s]

    def forward_bg(self, input: Tensor) -> List[Tensor]:
        bg_mu_z, bg_log_var_z = self.encode_z(input)
        bg_z = self.reparameterize(bg_mu_z, bg_log_var_z)
        #zeros = torch.zeros_like(bg_z)
        zeros = torch.zeros(bg_z.shape[0],self.latent_dim_s)
        zeros = zeros.to(self.device)
        output = self.decode(torch.cat((bg_z, zeros),1))
        return  [output, input, bg_mu_z, bg_log_var_z]
        

    def loss_function(self,
                      *args,
                      ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        beta = self.beta
        gamma = self.gamma
        delta = self.delta
                          
        recons_tg = args[0]
        input_tg = args[1]
        tg_mu_z = args[2]
        tg_log_var_z = args[3]
        tg_mu_s = args[4]
        tg_log_var_s = args[5]
        tg_z = args[6]
        tg_s = args[7]
        recons_bg = args[8]
        input_bg = args[9]
        bg_mu_z = args[10]
        bg_log_var_z = args[11]
                          
        recons_loss_roi = F.mse_loss(recons_tg, input_tg) * self.scale_MSE_GM / self.batch_size # TG reconstrction loss
        recons_loss_roni = F.mse_loss(recons_bg, input_bg) * self.scale_MSE_CF / self.batch_size # BG reconstrction loss
        recons_loss_roni += F.mse_loss(self.forward_bg(recons_tg)[0], recons_bg)*self.scale_MSE_CF/self.batch_size # Reconstructed BG features of TG input should look like CF
        recons_loss = recons_loss_roi+recons_loss_roni


        recon_fg = self.forward_fg(input_tg)[0] # Denoised signal
        recons_loss_fg = F.mse_loss(recon_fg, input_tg)*self.scale_MSE_FG/self.batch_size # Regularization, denoised signal should look somewhat like the original signal
        recons_loss_fg = F.mse_loss(torch.zeros_like(input_bg), self.forward_fg(input_bg)[0])/self.batch_size # Denoised version of RONI, should be all zeros
        recons_loss_discourage = recons_loss_fg*delta

        frequency_loss = high_frequency_penalty_loss(recon_fg).to(self.device)
        fg_volatility_loss = torch.from_numpy(np.array(0)).to(self.device)


        
        do_disentangle=self.do_disentangle # Whether to do disentagling 
        disentangle_type = 1 # What type of disentangling to do
        
        if do_disentangle==True and disentangle_type==1: # TC based on Discriminator 

            qz_s = torch.cat([tg_z, tg_s], dim=1)
            # Shuffle to get factorized samples
            z_perm = tg_z[torch.randperm(tg_z.size(0))]
            s_perm = tg_s[torch.randperm(tg_s.size(0))]
            qz_s_perm = torch.cat([z_perm, s_perm], dim=1)

            # We only use D's outputs to compute TC term for the VAE
            joint_logits = self.discriminator(qz_s)           # Joint
            factorized_logits = self.discriminator(qz_s_perm) # Factorized
            # TC estimate: (E_j[joint] - E_f[factorized])
            # Minimizing w.r.t. VAE: encourage joint ~ factorized
            TC_loss = (joint_logits.mean() - factorized_logits.mean())
            total_contrastive_loss = gamma * TC_loss
            
            qz_s = torch.cat([tg_z, tg_s], dim=1)
            mu = torch.cat([tg_mu_z, tg_mu_s], dim=1)
            log_var = torch.cat([tg_log_var_z, tg_log_var_s], dim=1)

        elif do_disentangle==True and disentangle_type==2: # TC based on computing the joint and marginal log-likelihoods
            # Compute log q(z, s)
            qz_s = torch.cat([tg_z, tg_s], dim=1)
            log_qz_s = self._compute_log_density_gaussian(qz_s, mu, log_var)

            # Compute log q(z) + log q(s)
            # Shuffle z and s to break dependencies
            z_perm = tg_z[torch.randperm(tg_z.size(0))]
            s_perm = tg_s[torch.randperm(tg_s.size(0))]
            qz_s_perm = torch.cat([z_perm, s_perm], dim=1)
            log_qz_s_perm = self._compute_log_density_gaussian(qz_s_perm, mu, log_var)

            # Total Correlation Loss
            total_contrastive_loss = (log_qz_s - log_qz_s_perm).mean()
            
        elif do_disentangle==True and disentangle_type==3: # TC based on correlation between FG and BG
            arr1 = model.forward_fg(inputs_gm)[0][:,0,:]
            arr2 = model.forward_bg(inputs_gm)[0][:,0,:]
            
            arr1_centered = arr1 - arr1.mean(dim=1, keepdim=True)
            arr2_centered = arr2 - arr2.mean(dim=1, keepdim=True)
            
            # Compute the numerator: covariance
            numerator = (arr1_centered * arr2_centered).sum(dim=1)
            
            # Compute the denominator: product of standard deviations
            arr1_std = arr1_centered.pow(2).sum(dim=1).sqrt()
            arr2_std = arr2_centered.pow(2).sum(dim=1).sqrt()
            denominator = arr1_std * arr2_std
            
            # Compute Pearson correlation for each row
            row_correlations = numerator / denominator
            
            # Average the correlations
            average_correlation = row_correlations.mean()
            total_contrastive_loss = torch.abs(average_correlation)
        else:
            total_contrastive_loss = torch.from_numpy(np.array(0)).to(self.device)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + tg_log_var_z - tg_mu_z ** 2 - tg_log_var_z.exp(), dim = 1), dim = 0)
        kld_loss += torch.mean(-0.5 * torch.sum(1 + tg_log_var_s - tg_mu_s ** 2 - tg_log_var_s.exp(), dim = 1), dim = 0)
        kld_loss += torch.mean(-0.5 * torch.sum(1 + bg_log_var_z - bg_mu_z ** 2 - bg_log_var_z.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/3
        

        if do_disentangle==True:
            loss = torch.sum(recons_loss + beta*kld_loss + gamma*total_contrastive_loss + recons_loss_roi + recons_loss_roni + recons_loss_discourage + fg_volatility_loss + recons_loss_fg+frequency_loss)
            return {
                'loss': loss, 
                'Reconstruction_Loss':recons_loss.detach(), 
                'KLD': kld_loss.detach()*beta, 
                'total_contrastive_loss' : total_contrastive_loss*gamma, 
                'recons_loss_roni' : recons_loss_roni.detach(), 
                'recons_loss_roi' : recons_loss_roi.detach(), 
                'recons_loss_discourage' : recons_loss_discourage.detach(), 
                'fg_volatility_loss': fg_volatility_loss.detach(), 
                'recons_loss_fg' : recons_loss_fg.detach(),
                'frequency_loss': frequency_loss.detach()
                }
            
        else:
            loss = torch.sum(recons_loss + beta*kld_loss + gamma*total_contrastive_loss+frequency_loss)
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD': kld_loss.detach()*beta, 'total_contrastive_loss' : total_contrastive_loss.detach()*gamma, 'recons_loss_roni' : recons_loss_roni.detach(), 'recons_loss_roi' : recons_loss_roi.detach(), 'recons_loss_discourage' : recons_loss_discourage,'frequency_loss': frequency_loss.detach()}

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward_fg(x)[0]

    def _compute_log_density_gaussian(self, z, mu, log_var):
            """
            Computes the log density of a Gaussian for each sample in the batch.
            """
            normalization = -0.5 * (math.log(2 * math.pi) + log_var)
            log_prob = normalization - 0.5 * ((z - mu) ** 2 / log_var.exp())
            return log_prob.sum(dim=1)


### Adversarial network

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Pass the input as is

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_  # Reverse and scale gradients
        return grad_input, None  # Second element corresponds to lambda_, which has no gradient

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Adversarial(nn.Module):

    def __init__(self, in_dim, latent_dim = None, hidden_dims = None):

        super(Adversarial, self).__init__()
        self.in_dim = in_dim
        
        if hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 256]
        else:
            self.hidden_dims = hidden_dims

        if latent_dim is None:
            self.latent_dim = 64
        else:
            self.latent_dim = latent_dim

        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)    

        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Build Encoder
        in_channels = 1
        modules_encoder = []
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules_encoder)

        # Build transition to and from latent
        self.transition_in = nn.Linear(self.hidden_dims[-1]*int(self.final_size), self.latent_dim)
        self.transition_out = nn.Linear(self.latent_dim, self.hidden_dims[-1] * int(self.final_size))
        
        # Build Decoder Signal
        modules_decoder_s = []
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules_decoder_s.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],
                                    self.hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )
            )
        self.decoder_signal = nn.Sequential(*modules_decoder_s)

        self.final_layer_signal = nn.Sequential(
                            nn.ConvTranspose1d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(self.hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))

        # Build Decoder Noise
        modules_decoder_n = []
        modules_decoder_n.append(self.grl)
        for i in range(len(self.hidden_dims) - 1):
            modules_decoder_n.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],
                                    self.hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )
            )
        self.decoder_noise = nn.Sequential(*modules_decoder_n)

        self.final_layer_noise = nn.Sequential(
                            nn.ConvTranspose1d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(self.hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))

    def encode(self,input: Tensor):
            z = self.encoder(input)
            return z

    def decode_signal(self, z: Tensor):
            signal = self.decoder_signal(z)
            return signal

    def decode_noise(self, z: Tensor):
            noise = self.decoder_noise(z)
            return noise

    def forward(self,input):
            z = self.encode(input)
            z = torch.flatten(z, start_dim=1)
            z = self.transition_in(z)
            z = self.transition_out(z)
            z = z.view(-1,256,int(self.final_size))
            signal = self.decode_signal(z)
            signal = self.final_layer_signal(signal)
            noise = self.decode_noise(z)
            noise = self.final_layer_noise(noise)
            return signal, noise, z

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise


# Adversarial model that uses coordinates

class Adversarial_coords(nn.Module):

    def __init__(self, in_dim, latent_dim = None, hidden_dims = None, noise_weight = 1):

        super(Adversarial_coords, self).__init__()
        self.in_dim = in_dim
        self.noise_weight = noise_weight
        
        if hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 256]
        else:
            self.hidden_dims = hidden_dims

        if latent_dim is None:
            self.latent_dim = 64
        else:
            self.latent_dim = latent_dim

        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)    

        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Build Encoder
        in_channels = 1
        modules_encoder = []
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules_encoder)

        # Build transition to and from latent
        self.transition_in = nn.Linear(self.hidden_dims[-1]*int(self.final_size), self.latent_dim)
        self.transition_out_signal = nn.Linear(self.latent_dim+3, self.hidden_dims[-1] * int(self.final_size))
        self.transition_out_noise = nn.Linear(self.latent_dim, self.hidden_dims[-1] * int(self.final_size))
        
        # Build Decoder Signal
        modules_decoder_s = []
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules_decoder_s.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],
                                    self.hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )
            )
        self.decoder_signal = nn.Sequential(*modules_decoder_s)

        self.final_layer_signal = nn.Sequential(
                            nn.ConvTranspose1d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(self.hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))

        # Build Decoder Noise
        modules_decoder_n = []
        modules_decoder_n.append(self.grl)
        for i in range(len(self.hidden_dims) - 1):
            modules_decoder_n.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],
                                    self.hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )
            )
        self.decoder_noise = nn.Sequential(*modules_decoder_n)

        self.final_layer_noise = nn.Sequential(
                            nn.ConvTranspose1d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(self.hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))

    def encode(self,input: Tensor):
            z = self.encoder(input)
            return z

    def decode_signal(self, z: Tensor):
            signal = self.decoder_signal(z)
            return signal

    def decode_noise(self, z: Tensor):
            noise = self.decoder_noise(z)
            return noise

    def forward(self,input,coords):
            z = self.encode(input)
            z = torch.flatten(z, start_dim=1)
            z = self.transition_in(z)
            z_signal = self.transition_out_signal(torch.cat([z,coords],dim=-1))
            z_noise = self.transition_out_noise(z)
            z_signal = z_signal.view(-1,256,int(self.final_size))
            z_noise = z_noise.view(-1,256,int(self.final_size))
            signal = self.decode_signal(z_signal)
            signal = self.final_layer_signal(signal)
            noise = self.decode_noise(z_noise)
            noise = self.final_layer_noise(noise)
            return signal, noise, z

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + self.noise_weight*loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + self.noise_weight*loss_noise
        return loss, loss_signal, loss_noise

    def loss_ncc_gmm(self,inputs_gm,inputs_cf,signal,noise,pca,gmm,device):       
        neg_log_probs = -gmm.score_samples(pca.transform(inputs_gm.squeeze().cpu().numpy()).reshape(len(inputs_gm), -1))
        loss_signal = torch.mul(self.ncc(inputs_gm,signal),torch.tensor(neg_log_probs).to(device)).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + self.noise_weight*loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 
    
    def independence_loss(self, X):
        """
        Penalizes non-independence by minimizing off-diagonal entries of the covariance matrix.
    
        Args:
            X (torch.Tensor): Input tensor of shape (N, D).
        Returns:
            torch.Tensor: Independence loss.
        """
        N, D = X.size()
        
        # Center the data (mean removal)
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        # Covariance matrix (D x D)
        cov_matrix = (X_centered.T @ X_centered) / (N - 1)
        
        # Off-diagonal elements of the covariance matrix
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        
        # Penalize the squared off-diagonal terms
        loss = torch.sum(off_diag**2)
        return loss


# Autoencoder with no denoising


class Autoencoder(nn.Module):

    def __init__(self, in_dim, latent_dim = None, hidden_dims = None):

        super(Autoencoder, self).__init__()
        self.in_dim = in_dim
        
        if hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 256]
        else:
            self.hidden_dims = hidden_dims

        if latent_dim is None:
            self.latent_dim = 64
        else:
            self.latent_dim = latent_dim

        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)    

        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Build Encoder
        in_channels = 1
        modules_encoder = []
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules_encoder)

        # Build transition to and from latent
        self.transition_in = nn.Linear(self.hidden_dims[-1]*int(self.final_size), self.latent_dim)
        self.transition_out = nn.Linear(self.latent_dim, self.hidden_dims[-1] * int(self.final_size))
        
        # Build Decoder Signal
        modules_decoder_s = []
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules_decoder_s.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],
                                    self.hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                    )
            )
        self.decoder_signal = nn.Sequential(*modules_decoder_s)

        self.final_layer_signal = nn.Sequential(
                            nn.ConvTranspose1d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(self.hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))


    def encode(self,input: Tensor):
            z = self.encoder(input)
            return z

    def decode_signal(self, z: Tensor):
            signal = self.decoder_signal(z)
            return signal

    def forward(self,input):
            z = self.encode(input)
            z = torch.flatten(z, start_dim=1)
            z = self.transition_in(z)
            z = self.transition_out(z)
            z = z.view(-1,256,int(self.final_size))
            signal = self.decode_signal(z)
            signal = self.final_layer_signal(signal)
            return signal

    def loss_function(self,inputs_gm,signal):
            loss_signal = F.mse_loss(inputs_gm,signal)
            return loss_signal

### Fully connected model

class Adversarial_fc(nn.Module):

    def __init__(self, in_dim, latent_dim = None):


        self.noise_weight = 0.1
        super(Adversarial_fc, self).__init__()
        self.in_dim = in_dim
        
        self.hidden_dims = [64, 64, 32, 32]

        if latent_dim is None:
            self.latent_dim = 32
        else:
            self.latent_dim = latent_dim

        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Build Encoder
        modules_encoder = []
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_encoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules_encoder)

        # Build Decoder Signal
        modules_decoder_s = []
        self.hidden_dims.reverse()
        in_dim = self.hidden_dims[0]
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_decoder_s.append(
                nn.Sequential(
                    nn.Linear(in_dim,h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_dim = h_dim
        self.decoder_signal = nn.Sequential(*modules_decoder_s)

        self.final_layer_signal = nn.Linear(self.hidden_dims[-1],self.in_dim)

        # Build Decoder Noise
        in_dim = self.hidden_dims[0]
        modules_decoder_n = []
        modules_decoder_n.append(self.grl)
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_decoder_n.append(
                nn.Sequential(
                    nn.Linear(in_dim,h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_dim = h_dim
        self.decoder_noise = nn.Sequential(*modules_decoder_n)

        self.final_layer_noise = nn.Linear(self.hidden_dims[-1],self.in_dim)

    def encode(self,input: Tensor):
            z = self.encoder(input)
            return z

    def decode_signal(self, z: Tensor):
            signal = self.decoder_signal(z)
            return signal

    def decode_noise(self, z: Tensor):
            noise = self.decoder_noise(z)
            return noise

    def forward(self,input):
            z = self.encode(input)
            signal = self.decode_signal(z)
            signal = self.final_layer_signal(signal)
            noise = self.decode_noise(z)
            noise = self.final_layer_noise(noise)
            return signal, noise, z

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + self.noise_weight*loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 

    def independence_loss(self, X):
        """
        Penalizes non-independence by minimizing off-diagonal entries of the covariance matrix.
    
        Args:
            X (torch.Tensor): Input tensor of shape (N, D).
        Returns:
            torch.Tensor: Independence loss.
        """
        N, D = X.size()
        
        # Center the data (mean removal)
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        # Covariance matrix (D x D)
        cov_matrix = (X_centered.T @ X_centered) / (N - 1)
        
        # Off-diagonal elements of the covariance matrix
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        
        # Penalize the squared off-diagonal terms
        loss = torch.sum(off_diag**2)
        return loss

    def loss_ncc_gmm(self,inputs_gm,inputs_cf,signal,noise,pca,gmm,device):       
        neg_log_probs = -gmm.score_samples(pca.transform(inputs_gm.squeeze().cpu().numpy()).reshape(len(inputs_gm), -1))
        loss_signal = torch.mul(self.ncc(inputs_gm,signal),torch.tensor(neg_log_probs).to(device)).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + self.noise_weight*loss_noise
        return loss, loss_signal, loss_noise



class Adversarial_fc_coords(nn.Module):

    def __init__(self, in_dim, latent_dim = None):


        self.noise_weight = 0.1
        super(Adversarial_fc_coords, self).__init__()
        self.in_dim = in_dim
        
        self.hidden_dims = [64, 64, 32, 32]

        if latent_dim is None:
            self.latent_dim = 32
        else:
            self.latent_dim = latent_dim

        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Build Encoder
        in_dim = in_dim+3 # 3 extra for coordinates
        modules_encoder = []
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_encoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules_encoder)

        # Build Decoder Signal
        modules_decoder_s = []
        self.hidden_dims.reverse()
        in_dim = self.hidden_dims[0]
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_decoder_s.append(
                nn.Sequential(
                    nn.Linear(in_dim,h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_dim = h_dim
        self.decoder_signal = nn.Sequential(*modules_decoder_s)

        self.final_layer_signal = nn.Linear(self.hidden_dims[-1],self.in_dim)

        # Build Decoder Noise
        in_dim = self.hidden_dims[0]
        modules_decoder_n = []
        modules_decoder_n.append(self.grl)
        for i in range(len(self.hidden_dims)):
            h_dim = self.hidden_dims[i]
            modules_decoder_n.append(
                nn.Sequential(
                    nn.Linear(in_dim,h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_dim = h_dim
        self.decoder_noise = nn.Sequential(*modules_decoder_n)

        self.final_layer_noise = nn.Linear(self.hidden_dims[-1],self.in_dim)

    def encode(self,input: Tensor):
            z = self.encoder(input)
            return z

    def decode_signal(self, z: Tensor):
            signal = self.decoder_signal(z)
            return signal

    def decode_noise(self, z: Tensor):
            noise = self.decoder_noise(z)
            return noise

    def forward(self,input):
            z = self.encode(input)
            signal = self.decode_signal(z)
            signal = self.final_layer_signal(signal)
            noise = self.decode_noise(z)
            noise = self.final_layer_noise(noise)
            return signal, noise, z

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + self.noise_weight*loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 

    def independence_loss(self, X):
        """
        Penalizes non-independence by minimizing off-diagonal entries of the covariance matrix.
    
        Args:
            X (torch.Tensor): Input tensor of shape (N, D).
        Returns:
            torch.Tensor: Independence loss.
        """
        N, D = X.size()
        
        # Center the data (mean removal)
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        # Covariance matrix (D x D)
        cov_matrix = (X_centered.T @ X_centered) / (N - 1)
        
        # Off-diagonal elements of the covariance matrix
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        
        # Penalize the squared off-diagonal terms
        loss = torch.sum(off_diag**2)
        return loss

    def loss_ncc_gmm(self,inputs_gm,inputs_cf,signal,noise,pca,gmm,device):       
        neg_log_probs = -gmm.score_samples(pca.transform(inputs_gm.squeeze().cpu().numpy()).reshape(len(inputs_gm), -1))
        loss_signal = torch.mul(self.ncc(inputs_gm,signal),torch.tensor(neg_log_probs).to(device)).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + self.noise_weight*loss_noise
        return loss, loss_signal, loss_noise


### Newer convolutional model

class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DAutoencoder, self).__init__()

        # Encoder: Convolutional layers reduce the sequence length and extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/8)
            nn.ReLU()
        )

        # Decoder Signal: Deconvolutional layers restore the sequence length
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Noise
        self.decoder_noise = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_signal(latent)
        noise = self.decoder_noise(latent)
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1Ddenoise(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1Ddenoise, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder: Convolutional layers reduce the sequence length and extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/8)
            nn.ReLU()
        )

        # Decoder Signal: Deconvolutional layers restore the sequence length
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Noise
        self.decoder_noise = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_signal(latent)
        noise = self.decoder_noise(self.grl(latent))
        return reconstructed.squeeze(),noise.squeeze()

    def smoothness_loss(self,signal):
        smoothness_loss = torch.mean((signal[:, 1:] - signal[:, :-1])**2)
        return smoothness_loss

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1Ddenoise_motion(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1Ddenoise_motion, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder: Convolutional layers reduce the sequence length and extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/8)
            nn.ReLU()
        )

        # Decoder Signal: Deconvolutional layers restore the sequence length
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Noise
        self.decoder_noise = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 7, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_signal(latent)
        noise = self.decoder_noise(self.grl(latent))
        return reconstructed.squeeze(),noise.squeeze()

    def smoothness_loss(self,signal):
        smoothness_loss = torch.mean((signal[:, 1:] - signal[:, :-1])**2)
        return smoothness_loss

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = 0
        for i in range(noise.shape[1]):
            loss_noise += self.ncc(inputs_cf[:,i,:],noise[:,i,:]).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 



class Conv1Ddenoise_motion2(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1Ddenoise_motion2, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder: Convolutional layers reduce the sequence length and extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/8)
            nn.ReLU()
        )

        # Decoder Signal: Deconvolutional layers restore the sequence length
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Noise
        self.decoder_noise = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Confounds
        self.decoder_confounds = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 6, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_signal(latent)
        noise = self.decoder_noise(self.grl(latent))
        confounds_pred = self.decoder_confounds(self.grl(latent))
        return reconstructed.squeeze(),noise.squeeze(), confounds_pred

    def smoothness_loss(self,signal):
        smoothness_loss = torch.mean((signal[:, 1:] - signal[:, :-1])**2)
        return smoothness_loss

    def loss_function(self,inputs_gm,inputs_cf,signal,noise,confounds,confounds_pred):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss_confounds = F.mse_loss(confounds,confounds_pred)
            loss = loss_signal + loss_noise + loss_confounds*0.1
            return loss, loss_signal, loss_noise, loss_confounds

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise,confounds,confounds_pred):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss_confounds = 0
        for i in range(confounds.shape[1]):
            loss_confounds += self.ncc(confounds[:,i,:],confounds_pred[:,i,:]).mean()
        loss = loss_signal + loss_noise +loss_confounds*0.1
        return loss, loss_signal, loss_noise, loss_confounds

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1DdenoiseBN(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DdenoiseBN, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder: Convolutional layers reduce the sequence length and extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, L/2)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/8)
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        # Decoder Signal: Deconvolutional layers restore the sequence length
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Noise
        self.decoder_noise = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_signal(latent)
        noise = self.decoder_noise(self.grl(latent))
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1Ddenoise_large(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(Conv1Ddenoise_large, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder: Convolutional layers reduce the sequence length and extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),              # (B, 32, L/4)
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/8)
            nn.ReLU()
        )

        # Decoder Signal: Deconvolutional layers restore the sequence length
        self.decoder_signal = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder Noise
        self.decoder_noise = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, L/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),          # (B, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, L)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_signal(latent)
        noise = self.decoder_noise(self.grl(latent))
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1DUnet(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DUnet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1)
        # Decoder Signal
        self.decSig1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decSig2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)     
        self.decSig3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)     
        
        # Decoder Noise
        self.decNoi1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decNoi2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decNoi3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.enc1(x))
        x2 = F.leaky_relu(self.enc2(x1))
        x3 = F.leaky_relu(self.enc3(x2))
        s1 = F.leaky_relu(self.decSig1(x3))
        s2 = F.leaky_relu(self.decSig2(torch.cat((s1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        reconstructed = F.sigmoid(self.decSig3(torch.cat((s2,F.pad(x1,(0,2),"constant",0)),dim=1)))         
        n1 = F.leaky_relu(self.decNoi1(x3))
        n2 = F.leaky_relu(self.decNoi2(torch.cat((n1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        noise = F.sigmoid(self.decNoi3(torch.cat((n2,F.pad(x1,(0,2),"constant",0)),dim=1))) 
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1DUnet_denoise(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DUnet_denoise, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder
        self.enc1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1)
        # Decoder Signal
        self.decSig1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decSig2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)     
        self.decSig3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)     
        
        # Decoder Noise
        self.decNoi1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decNoi2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decNoi3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.enc1(x))
        x2 = F.leaky_relu(self.enc2(x1))
        x3 = F.leaky_relu(self.enc3(x2))
        s1 = F.leaky_relu(self.decSig1(x3))
        s2 = F.leaky_relu(self.decSig2(torch.cat((s1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        reconstructed = F.sigmoid(self.decSig3(torch.cat((s2,F.pad(x1,(0,2),"constant",0)),dim=1)))         
        n1 = F.leaky_relu(self.decNoi1(self.grl(x3)))
        n2 = F.leaky_relu(self.decNoi2(torch.cat((n1,F.pad(self.grl(x2),(0,1),"constant",0)),dim=1)))
        noise = F.sigmoid(self.decNoi3(torch.cat((n2,F.pad(self.grl(x1),(0,2),"constant",0)),dim=1))) 
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1DUnetBN(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DUnetBN, self).__init__()
        # Encoder
        self.enc1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.enc3 = nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        # Decoder Signal
        self.decSig1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decSig2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)     
        self.decSig3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)     
        
        # Decoder Noise
        self.decNoi1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decNoi2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decNoi3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.enc1(x))
        x2 = F.leaky_relu(self.enc2(x1))
        x3 = F.leaky_relu(self.enc3(x2))
        s1 = F.leaky_relu(self.decSig1(x3))
        s2 = F.leaky_relu(self.decSig2(torch.cat((s1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        reconstructed = F.sigmoid(self.decSig3(torch.cat((s2,F.pad(x1,(0,2),"constant",0)),dim=1)))         
        n1 = F.leaky_relu(self.decNoi1(x3))
        n2 = F.leaky_relu(self.decNoi2(torch.cat((n1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        noise = F.sigmoid(self.decNoi3(torch.cat((n2,F.pad(x1,(0,2),"constant",0)),dim=1))) 
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1DUnetBN2(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DUnetBN2, self).__init__()
        # Encoder
        self.enc1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.enc3 = nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        # Decoder Signal
        self.decSig1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnSig1 = nn.BatchNorm1d(32)
        self.decSig2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.bnSig2 = nn.BatchNorm1d(16)
        self.decSig3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Decoder Noise
        self.decNoi1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnNoi1 = nn.BatchNorm1d(32)
        self.decNoi2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnNoi2 = nn.BatchNorm1d(16)
        self.decNoi3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.enc1(x))
        x2 = F.leaky_relu(self.enc2(x1))
        x3 = F.leaky_relu(self.enc3(x2))
        s1 = F.leaky_relu(self.decSig1(x3))
        s2 = F.leaky_relu(self.decSig2(torch.cat((s1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        reconstructed = F.sigmoid(self.decSig3(torch.cat((s2,F.pad(x1,(0,2),"constant",0)),dim=1)))         
        n1 = F.leaky_relu(self.decNoi1(x3))
        n2 = F.leaky_relu(self.decNoi2(torch.cat((n1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        noise = F.sigmoid(self.decNoi3(torch.cat((n2,F.pad(x1,(0,2),"constant",0)),dim=1))) 
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 


class Conv1DUnetBN2_denoise(nn.Module):
    
    def __init__(self, input_channels=1, latent_dim=16):
        super(Conv1DUnetBN2_denoise, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        # Encoder
        self.enc1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.enc3 = nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        # Decoder Signal
        self.decSig1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnSig1 = nn.BatchNorm1d(32)
        self.decSig2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.bnSig2 = nn.BatchNorm1d(16)
        self.decSig3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Decoder Noise
        self.decNoi1 = nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnNoi1 = nn.BatchNorm1d(32)
        self.decNoi2 = nn.ConvTranspose1d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnNoi2 = nn.BatchNorm1d(16)
        self.decNoi3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.enc1(x))
        x2 = F.leaky_relu(self.enc2(x1))
        x3 = F.leaky_relu(self.enc3(x2))
        s1 = F.leaky_relu(self.decSig1(x3))
        s2 = F.leaky_relu(self.decSig2(torch.cat((s1,F.pad(x2,(0,1),"constant",0)),dim=1)))
        reconstructed = F.sigmoid(self.decSig3(torch.cat((s2,F.pad(x1,(0,2),"constant",0)),dim=1)))         
        n1 = F.leaky_relu(self.decNoi1(self.grl(x3)))
        n2 = F.leaky_relu(self.decNoi2(torch.cat((n1,F.pad(self.grl(x2),(0,1),"constant",0)),dim=1)))
        noise = F.sigmoid(self.decNoi3(torch.cat((n2,F.pad(self.grl(x1),(0,2),"constant",0)),dim=1))) 
        return reconstructed.squeeze(),noise.squeeze()

    def loss_function(self,inputs_gm,inputs_cf,signal,noise):
            loss_signal = F.mse_loss(inputs_gm,signal)
            loss_noise = F.mse_loss(inputs_cf,noise)
            loss = loss_signal + loss_noise
            return loss, loss_signal, loss_noise

    def loss_ncc(self,inputs_gm,inputs_cf,signal,noise):
        loss_signal = self.ncc(inputs_gm,signal).mean()
        loss_noise = self.ncc(inputs_cf,noise).mean()
        loss = loss_signal + loss_noise
        return loss, loss_signal, loss_noise

    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 