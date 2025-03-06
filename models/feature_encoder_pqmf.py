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
                 c_in=10,
                 n_core=5,
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
        self.n_core = n_core # core bands

        # feature encoder
        self.c_in = c_in if not use_core else c_in+n_core # N_HF
        print(f"USE CORE:{use_core}")
        print(f"USED BANDS:{self.c_in}")
        
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
        if not self.use_core:
            ## only HF
            x = x[:, self.n_core:self.n_core+self.c_in, :] # [B,32,T] -> [B,Ncore:Nhf,T]
        else:
            ## use core
            x = x[:, :self.c_in, :]
        
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


import torch

def synthesize_and_draw_debug(
    x_subbands: torch.Tensor,      # [B, subbands, T] 지금 보고 싶은 subband들
    x_full: torch.Tensor,
    pqmf_fb,                       # PQMF 객체
    target_length: int,
    sr: int = 48000,               # 샘플링 레이트
    n_fft: int = 2048,
    win_len: int = 1024,
    hop_len: int = 256,
    save_path: str = "debug_spec",
    use_core = False,
    figsize = (8,6),
):
    """
    특정 subband만 0이 아닌 상태로 합성하고, 스펙트로그램을 그려 저장하는 디버그 함수.
    
    synthesize_and_draw_debug(
                x_subbands=x,
                x_full=temp,
                pqmf_fb=self.pqmf_,
                target_length=temp.shape[-1],  # 혹은 다른 길이
                sr=48000,                   # 쓰는 SR
                n_fft=2048,
                win_len=1024,
                hop_len=256,
                save_path="debug_hf_spec",
                use_core=self.use_core,
                )
    """
    from utils import draw_spec
    B, used_subbands, L = x_subbands.shape
    Ncore = 0 if use_core else 5 # 0 if use core
    
    # 1) 나머지 서브밴드 채널은 0으로 채운다
    #    여기서 PQMF 전체 채널 수가 32라는 전제
    
    freq_zeros = torch.zeros(B, Ncore, L, device=x_subbands.device)
    freq_zeros2 = torch.zeros(B, 32 - used_subbands - Ncore, L, device=x_subbands.device)
    
    # 2) concat -> full subband
    #    [사용 중인 subbands, 나머지 zeros]
    x_cat = torch.cat([freq_zeros, x_subbands, freq_zeros2], dim=1)  # shape [B, 32, T]

    # 3) full-band waveform 합성
    x_hat_full = pqmf_fb.synthesis(x_cat, delay=0, length=target_length)
    # x_hat_full.shape => [B, T']

    # (선택) 첫 번째 배치만 저장한다고 가정
    x_hat_np = x_hat_full[0].detach().cpu().numpy()

    # 4) draw_spec() 호출
    #    이미 정의하신 draw_spec 함수를 쓴다고 가정:
    draw_spec(
        x=x_hat_np,
        figsize=figsize,
        title='Extracted Bands',
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        sr=sr,
        # cmap='inferno',  # 원하면 지정
        vmin=-50,
        vmax=40,
        use_colorbar=False,
        ylim=None,
        return_fig=False,
        save_fig=True,            # 저장할 것이므로 True
        save_path=save_path       # 'debug_spec.png' 같은 경로
    )
    draw_spec(
        x=x_full.squeeze().detach().cpu().numpy(),
        figsize=figsize,
        title='Full Band',
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        sr=sr,
        # cmap='inferno',  # 원하면 지정
        vmin=-50,
        vmax=40,
        use_colorbar=False,
        ylim=None,
        return_fig=False,
        save_fig=True,            # 저장할 것이므로 True
        save_path=f"{save_path}_gt"       # 'debug_spec.png' 같은 경로
    )
    print(f"[DEBUG] spec saved to {save_path}")

def main():
    from models.feature_encoder_pqmf import SubBandEncoder
    from main import MODEL_MAP, load_config
    from models.prepare_models import prepare_generator

    config = load_config("configs/exp21.yaml")
    config['generator']['feature_encoder_config']['use_core'] = False

    generator = prepare_generator(config, MODEL_MAP)
    model = generator.feature_encoder

    subband_sig = torch.randn(7,1,45056)  # [B,N_hf,T] [7,32,1408]
    print(subband_sig.shape)
    print(config['generator']['c_in'])
    print(config['generator']['feature_encoder_config'])

    print(summary(model, input_data=subband_sig, depth=3, col_names=['input_size', 'output_size', 'num_params']))


if __name__ == "__main__":
    
    main()