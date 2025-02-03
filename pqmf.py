"""
Code from HS Lim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.optimize as optimize
import scipy.signal.windows as windows

PI = math.pi
class PQMF(nn.Module):
    def __init__(self, num_subbands=32, num_taps=512, cutoff_ratio=None, beta=9):
        super().__init__()

        # Attributes
        self.num_subbands = num_subbands
        self.num_taps = num_taps

        if cutoff_ratio is None:
            print("Find an optimizal cutoff ratio for the PQMF...")
            res = optimize.minimize(_objective,
                                    args=(num_subbands, num_taps, beta),
                                    x0=np.array([0.01]),
                                    bounds=optimize.Bounds(0.01, 0.99))
            cutoff_ratio = res.x[0]
            print(f"Optimal cutoff ratio: {cutoff_ratio:.8f}")

        ### Build filterbank ###
        # Prototype window
        cutoff_freq = cutoff_ratio * PI
        self.w_proto = self._get_prototype_window(num_subbands, num_taps, cutoff_freq, beta)

        # Analysis filter
        phase = torch.arange(num_taps) - (num_taps - 1) / 2   # [num_taps]
        phase = torch.tile(phase.view(1, -1), (num_subbands, 1)) * PI / num_subbands * (torch.arange(num_subbands).view(-1, 1) + 0.5)

        h = (2 * torch.tile(self.w_proto.view(1, -1), (num_subbands, 1))
               * torch.cos(phase - (-1) ** torch.arange(num_subbands).view(-1, 1) * PI / 4))   # [num_subbands, num_taps]
        self.register_buffer('h', h, False)

        # Syntheis filter
        g = torch.fliplr(h)
        self.register_buffer('g', g, False)

    def analysis(self,
            x, pad_input=None):
        """
        PQMF analysis

        Args:
            x (Tensor) [(B), T]: Input audio

        Returns:
            y (Tensor) [(B), num_subbands, T / num_subbands]: Sub-band signals

        """
        if pad_input is None:
            # pad_input = (self.num_taps - 1, self.num_taps - 1)
            pad_input = (0, self.num_taps - 1)

        # Check tensor shape
        if x.dim() == 1:
            x = x.unsqueeze(0)   # [T] ---> [1, T]
            need_squeezing = True
        elif x.dim() == 2:
            need_squeezing = False
        elif x.dim() == 3:
            x = x.squeeze(1)     # [B,1,T] ---> [B,T]
            need_squeezing = False             
        else: raise ValueError("Dimension of the input should be smaller than 3")

        # Pad zeros at the front of the waveform,
        # according to the tab size and the down-sampling ratio
        x = F.pad(x, pad_input, 'constant', 0.)
        # print("Pad Size", pad_input)

        # make multiple of stride
        pad_downsampling = ((self.num_subbands - (x.size(1) % self.num_subbands))
                            % self.num_subbands)
        
        x = F.pad(x, (0, pad_downsampling), 'constant', 0.)
        # print("Pad Size", pad_downsampling)
        # print("Pad Size", x.shape)

        # Filtering and down-sampling
        x = x.unsqueeze(1)   # [B, 1, T]
        y = F.conv1d(x, self.h.unsqueeze(1),
                bias=None,
                stride=self.num_subbands)   # [B, num_subbands, T // num_subbands]
        # print("Pad Size", y.shape)

        # Squeezing (if required)
        if need_squeezing:
            y = y.squeeze(0)   # [num_subbands, T // num_subbands]
        
        return y

    def synthesis(self,
            y, pad_input=None, length=None, delay=0):
        """
        PQMF synthesis

        Args:
            y (Tensor) [(B), num_subbands, T / num_subbands]: Sub-band signals
            length (int): (Optional) Desired output length

        Returns:
            x_hat (Tensor) [(B), T]: Output audio

        """
        if pad_input is None:
            pad_input = (self.num_taps - 1, self.num_taps - 1)

        if delay is None:
            delay = self.num_taps - 1

        # Check tensor shape
        if y.dim() == 2:
            y = y.unsqueeze(0)   # [1, num_subbands, T / num_subbands]
            need_squeezing = True
        elif y.dim() == 3:
            need_squeezing = False
        else: raise ValueError("Dimension of the input should be 2 or 3")

        # Up-sampling
        x_hat = torch.zeros(y.size(0), 
                            y.size(1),
                            y.size(2) * self.num_subbands).to(y.device)

        for n in range(y.size(2)):
            x_hat[:, :, n * self.num_subbands] = y[:, :, n]

        # Padding
        x_hat = F.pad(x_hat, pad_input, 'constant', 0.)

        # Convolution
        x_hat = F.conv1d(x_hat, self.g.unsqueeze(0), bias=None)   # [B, 1, T]
        x_hat = x_hat * self.num_subbands   
        x_hat = x_hat.squeeze(1)   # [B, T]

        # Remove delays
        x_hat = x_hat[:, delay:-(self.num_taps - 1)]

        # Length adjustment
        if length is not None:
            x_hat = x_hat[:, :length]

        # Batch squeezing (if required)
        if need_squeezing:
            x_hat = x_hat.squeeze(0)

        return x_hat

    def _get_prototype_window(self, num_subbands, num_taps, cutoff_freq, beta):
        if cutoff_freq is None:
            cutoff_freq = PI / (2 * num_subbands)

        w_proto = (torch.sin(cutoff_freq * (torch.arange(num_taps) - 0.5 * (num_taps - 1)))
                   / (PI * (torch.arange(num_taps) - 0.5 * (num_taps - 1))))
        w_proto[(num_taps - 1) // 2] = cutoff_freq / PI   # Specify the sinc function center

        w_proto *= torch.Tensor(windows.kaiser(num_taps, beta))

        return w_proto

def design_prototype_filter(taps=63, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 1, "The number of taps mush be an odd number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps) - 0.5 * (taps - 1))) / (
            np.pi * (np.arange(taps) - 0.5 * (taps - 1))
        )
    h_i[(taps - 1) // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = windows.kaiser(taps, beta)
    h = h_i * w

    return h

def _objective(cutoff_ratio, num_subbands, num_taps, beta):
    h_proto = design_prototype_filter(num_taps, cutoff_ratio, beta)
    # auto correlation 
    conv_h_proto = np.convolve(h_proto, h_proto[::-1], mode='full')
    length_conv_h = conv_h_proto.shape[0]
    half_length = length_conv_h // 2

    # attenuation
    check_steps = np.arange((half_length) // (2 * num_subbands)) * 2 * num_subbands
    _phi_new = conv_h_proto[half_length:][check_steps]
    phi_new = np.abs(_phi_new[1:]).max()
    
    # Since phi_new is not convex, This value should also be considered. 
    diff_zero_coef = np.abs(_phi_new[0] - 1 / (2 * num_subbands))
    
    return phi_new + diff_zero_coef


pqmf_cutoff_ratios = {3: 0.189, 4: 0.142, 8: 0.071, 16: 0.03552, 24: 0.0237, 32: 0.01777}
pqmf_taps = {3: 48, 4: 62, 8: 124, 16: 246, 24: 368, 32: 490}