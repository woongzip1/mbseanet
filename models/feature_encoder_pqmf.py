import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
import torch.nn.functional as F
import pickle
from einops import rearrange
import pickle
import pdb
from pqmf import PQMF

from models.modules import Conv1d,EncBlock,DecBlock
from models.feature_encoder import AutoEncoder
from models.quantize import ResidualVectorQuantize
# from vector_quantize_pytorch import ResidualVQ

class SubBandEncoder(nn.Module):
    def __init__(self, min_dim=32, strides=[2,2,4,4], 
                 c_in=27, 
                 visualize=False,
                 use_sfm=False, 
                 use_core=False,
                 **kwargs):
        super().__init__()
        
        self.visualize = visualize
        self.use_sfm = use_sfm
        self.use_core = use_core
        self.downsampling_factor = np.prod(strides)
        self._initialize_weights()

        self.pqmf_ = PQMF(num_subbands=32, num_taps=481, cutoff_ratio=None)
        self.n_core = 32 - c_in

        # feature encoder
        self.c_in = c_in if not use_core else 32
        print(f"USE CORE:{use_core}")
        
        # Input convolutional layers
        self.conv_in = Conv1d(
            in_channels=self.c_in,
            out_channels=min_dim,
            kernel_size=7,
            stride=1,
        )

        # Encoder blocks
        self.encoder_with_film = nn.ModuleList([
            nn.Sequential(
                EncBlock(min_dim * 2, strides[0]),
            ),
            nn.Sequential(
                EncBlock(min_dim * 4, strides[1]),
            ),
            nn.Sequential(
                EncBlock(min_dim * 8, strides[2]),
            ),
            nn.Sequential(
                EncBlock(min_dim * 16, strides[3]),
            )
        ])

    def _initialize_weights(self):
        # Iterate through all layers and apply Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _adjust_signal_len(self, x):
        B,C,sig_len = x.size() # [B,C,T]
        pad_len = sig_len % self.downsampling_factor
        if pad_len != 0:
            front_pad = pad_len // 2
            back_pad = pad_len - front_pad
            # x = x[:, :, :-pad_len]
            x = x[:, :, front_pad:-back_pad]
        else:
            front_pad = 0
            back_pad = 0
        return x, front_pad, back_pad

    def _pad_signal_len(self, x, front_pad, back_pad):
        padded_x = torch.cat((front_pad, x, back_pad), dim=-1)  # Concatenate along time axis
        return padded_x
    
    def forward(self, x, stft_cond=None, sfm=None, n_quantizers=None):
        """
        x: Original input signal [1,T]
        """
        ## Apply PQMF analysis to get subband representation
        # extract N_HF bands 
        x = self.pqmf_.analysis(x) # [B,32,T]
        if self.use_core is False:
            x = x[:, self.n_core:, :] # [B,32,T] -> [B,N_HF,T]
        
        # make sure input is multiple of downsampling
        # x, front_pad, back_pad = self._adjust_signal_len(x)

        # Skip connections
        skip = []
        x = self.conv_in(x)     # [C_in,T] -> [D, T]
        skip.append(x)

        # Encoder
        for block in self.encoder_with_film: # [D, T] -> [16D, T/8], T=t/32
            x = block[0](x)  # EncBlock
            skip.append(x)
            
        return x


if __name__ == "__main__":
    pass