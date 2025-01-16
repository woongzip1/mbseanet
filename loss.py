import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.signal.windows import hann
# from soundstream.balancer import *

class LossCalculator:
    def __init__(self, config, discriminator):
        self.discriminator = discriminator
        self.lambda_adv_loss = config['loss']['lambda_adv_loss']
        self.lambda_fm_loss = config['loss']['lambda_fm_loss']
        self.lambda_mel_loss = config['loss']['lambda_mel_loss']
        self.lambda_commitment_loss = config['loss']['lambda_commitment_loss']
        self.lambda_codebook_loss = config['loss']['lambda_codebook_loss']
        self.ms_mel_loss_config = config['loss']['ms_mel_loss_config']
        
        self.lambda_subband_loss = config['loss']['lambda_subband_loss']
        self.num_taps = 481
        
    def _save_figures(self, hr, x_hat_full):
        ## for debugging
        hr = hr[0,...]
        x_hat_full = x_hat_full[0,...]
        from matplotlib import pyplot as plt
        # clip front samples
        plt.figure(figsize=(14, 4))
        plt.plot(hr.squeeze().cpu().detach().numpy(), label='gt')
        plt.plot(x_hat_full.squeeze().cpu().detach().numpy() + 0.1, label='recon')
        # start = x.shape[-1] - num_taps
        start = 0
        step = self.num_taps + 200
        plt.xlim(start, start + step)
        plt.legend()
        # Save the figure instead of showing it
        save_path = "output_plot.png"  # Define the output file path
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_generator_loss(self, hr, x_hat_full, commitment_loss, codebook_loss, hf_estimate=None, target_subbands=None):
        # self._save_figures(hr, x_hat_full)
        # import pdb
        # pdb.set_trace()
        
        hr = hr[...,self.num_taps//2:]
        x_hat_full = x_hat_full[...,self.num_taps//2:]
        assert hr.shape == x_hat_full.shape, f"size mismatch- {hr.shape}, {x_hat_full.shape}" # [B,T]

        # import pdb
        # pdb.set_trace()
        
        # adv loss
        g_loss_dict, g_loss_report = self.discriminator.g_loss(hr, x_hat_full, adv_loss_type='hinge')
        
        # mel loss
        # squeeze and careout for tensor shape
        ms_mel_loss_value = ms_mel_loss(hr, x_hat_full, **self.ms_mel_loss_config) # take care of tensor shape

        # subband loss
        if hf_estimate is not None:
            subband_loss_value = compute_subband_loss(hf_estimate, target_subbands) 
        else:
            subband_loss_value = 0
        
        # overall loss (for logging)
        loss_G = (
            self.lambda_adv_loss * g_loss_dict.get('adv_g', 0) +
            self.lambda_fm_loss * g_loss_dict.get('fm', 0) +
            self.lambda_mel_loss * ms_mel_loss_value +
            self.lambda_commitment_loss * commitment_loss +
            self.lambda_codebook_loss * codebook_loss +
            self.lambda_subband_loss * subband_loss_value
        )
        return loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report, subband_loss_value

    def compute_discriminator_loss(self, hr, x_hat_full):
        d_loss_dict, d_loss_report = self.discriminator.d_loss(hr, x_hat_full, adv_loss_type='hinge')
        loss_D = d_loss_dict.get('adv_d', 0)
        # import pdb
        # pdb.set_trace()
        # loss_D = sum(d_loss_dict.values())

        return loss_D, d_loss_dict, d_loss_report
    
def compute_subband_loss(hf_estimate: torch.Tensor, target_subbands: torch.Tensor):
    subband_loss, sc_loss, mag_loss = ms_stft_loss(x=target_subbands, x_hat=hf_estimate)
    return subband_loss

def ms_stft_loss(x, x_hat, n_fft_list=[16, 32, 64], hop_ratio=0.25, eps=1e-5):
    """
    reference: https://github.com/AppleHolic/multiband_melgan/
    
    Multi-Scale STFT Loss with Spectral Convergence and Magnitude Loss
    Args:
        x (torch.Tensor) [B, T]: ground truth waveform
        x_hat (torch.Tensor) [B, T]: generated waveform
        n_fft_list (List of int): list of n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
        sr (int): sampling rate
        eps (float): epsilon for numerical stability in log calculation
    Returns:
        total_loss: combined STFT loss (SC + Magnitude)
        sc_loss: Spectral Convergence loss
        mag_loss: Magnitude loss
    """
    assert len(n_fft_list) > 0, "n_fft_list must contain at least one value."
    assert x.shape == x_hat.shape, "input and target must have the same shape"
    
    total_loss, sc_loss, mag_loss = 0., 0., 0.
    
    if len(x.shape) == 3: # subband input
        assert x.shape[1] > 1, "Expected Subband Input with multiple channels, got shape: {}".format(x.shape)
        B,C,_ = x.shape
        x = x.reshape(B*C, -1) # [B,C,T] -> [BC,T]
        x_hat = x_hat.reshape(B*C, -1)
    else: # fullband input
        C = 1
        
    for n_fft in n_fft_list:
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), 
                                   power=1.0, normalized=False, center=True).to(x.device)

        # Compute spectrograms for x and x_hat
        spg_x = sig_to_spg(x)  # [B, F, T']
        spg_x_hat = sig_to_spg(x_hat)  # [B, F, T']
        # print(spg_x.shape)
        
        # SC, Mag loss
        sc_loss_ = ((spg_x - spg_x_hat).norm(p='fro', dim=(1, 2)) / spg_x.norm(p='fro', dim=(1, 2)) + eps).mean()
        mag_loss_ = torch.mean(torch.abs(torch.log(spg_x.clamp(min=eps)) - torch.log(spg_x_hat.clamp(min=eps))))
        
        # Accumulate losses
        sc_loss += sc_loss_
        mag_loss += mag_loss_
        total_loss += sc_loss_ + mag_loss_
    
    scale_factor = len(n_fft_list)
    return total_loss/scale_factor, sc_loss/scale_factor, mag_loss/scale_factor
    
def ms_mel_loss(x, x_hat, n_fft_list=[32, 64, 128, 256, 512, 1024, 2048], hop_ratio=0.25, 
                mel_bin_list=[5, 10, 20, 40, 80, 160, 320], fmin=0, fmax=None, sr=44100, mel_power=1.0,
                core_cutoff = 4500,
                eps=1e-5, reduction='mean', **kwargs):
    """
    Multi-scale spectral energy distance loss
    References:
        Kumar, Rithesh, et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS, 2023.
    Args:
        x (torch.Tensor) [B, ..., T]: ground truth waveform (Batched)
        x_hat (torch.Tensor) [B, ..., T]: generated waveform
        n_fft_list (List of int): list that contains n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
        mel_bin_list (List of int): list that contains the number of mel bins for each scale
        sr (int): sampling rate
        fmin (float): minimum frequency for mel-filterbank calculation
        fmax (float): maximum frequency for mel-filterbank calculation
        mel_power (float): power to raise magnitude to before taking log
    Returns:
    """
    assert len(n_fft_list) == len(mel_bin_list)
    assert x.shape == x_hat.shape, "input and target must have the same shape"
    
    loss = 0
    for n_fft, mel_bin in zip(n_fft_list, mel_bin_list):
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), 
                                    window_fn=hann, wkwargs={"sym": False},\
                                    power=1.0, normalized=False, center=True).to(x.device)
        spg_to_mel = T.MelScale(n_mels=mel_bin, sample_rate=sr, n_stft=n_fft//2+1, f_min=fmin, f_max=fmax, norm="slaney", mel_scale="slaney").to(x.device)  
        
        ## erase core bands considerations
        # x_spg = sig_to_spg(x) # [B,F,T]
        # f_start = int((core_cutoff / sr) * n_fft) + 1
        # x_spg[:,:f_start,:] = 0
        # x_mel = spg_to_mel(x_spg) # [B, C, mels, T]
        
        # x_hat_spg = sig_to_spg(x_hat)
        # x_hat_spg[:,:f_start,:] = 0
        # x_hat_mel = spg_to_mel(x_hat_spg)
        ################################
        x_mel = spg_to_mel(sig_to_spg(x))           # [B,F,T]
        x_hat_mel = spg_to_mel(sig_to_spg(x_hat))
        
        log_term = torch.sum(torch.abs(x_mel.clamp(min=eps).pow(mel_power).log10() - x_hat_mel.clamp(min=eps).pow(mel_power).log10()))
        if reduction == 'mean':
            log_term /= torch.numel(x_mel)
        elif reduction == 'sum':
            log_term /= x_mel.shape[0]
        
        loss += log_term
        
    return loss