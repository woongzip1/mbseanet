import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm
import torchaudio

from einops import rearrange
import typing as tp
##

class SSDiscriminatorBlock(nn.Module):
    """
    Combine all sub-discriminators into one object
        wd: Multi-scale discriminator (MSD) 
        sd: STFT-based discriminator (STFTD)
        pd: Multi-period discriminator (MPD) 
    """
    def __init__(self, 
                 wd_num=0, ds_factor_list=[],
                 sd_num=0, C=32, n_fft_list=[], hop_len_list=[], sd_mode='BS', band_split_ratio=[],
                 pd_num=0, period_list=[], C_period=64,
                 **kwargs):
        """
        Class that aggregates various discriminators used with SoundStream autoencoder.
        
        Args:
            wd_num (int): the number of MSD sub-discriminators.
            ds_factor_list (list): list of downsampling factor of MSD sub-discriminators.
            sd_num (int): the number of STFT-based discriminators from SoundStream.
            C (int): channel size of the STFT-discriminator.
            n_fft_list (list): list of n_fft of STFT-discriminators.
            hop_len_list (list): list of hop_len of STFT-discriminators.
            sd_mode (str): SS: SoundStream STFT-Disc, EC: EnCodec Multi-scale STFT-Disc, BS: DAC band-split STFT-Disc.
            pd_num (int): the number of MPD sub-discriminators.
            period_list (list): list of period factor of MPD sub-discriminators.
        """
        super().__init__()
        assert wd_num == len(ds_factor_list), "wd_num must be equal to the length of ds_factor_list!"
        assert sd_num == len(n_fft_list) and sd_num == len(hop_len_list), "sd_num must be equal to the length of n_fft_list and hop_len_list!"
        assert pd_num == len(period_list), "pd_num must be equal to the length of period_list!"
        
        self.wd_num = wd_num
        self.sd_num = sd_num
        self.pd_num = pd_num
        
        self.sd_mode = sd_mode
        
        self.wave_disc_list = nn.ModuleList()
        for i, ds_factor in enumerate(ds_factor_list):
            disc_name = f"wd{i + 1}"
            self.wave_disc_list.append(WaveDiscriminator(ds_factor, name=disc_name))
        
        self.stft_disc_list = nn.ModuleList()
        for i, (n_fft, hop_len) in enumerate(zip(n_fft_list, hop_len_list)):
            disc_name = f"sd{i + 1}"
            if self.sd_mode == 'SS':
                self.stft_disc_list.append(STFTDiscriminator(C, n_fft, hop_len, name=disc_name))
            elif self.sd_mode == 'EC':
                self.stft_disc_list.append(STFTDiscriminator_EnCodec(C=C, n_fft=n_fft, hop_length=hop_len, win_length=n_fft, name=disc_name))
            elif self.sd_mode == 'BS':
                self.stft_disc_list.append(MultiBandSTFTDiscriminator(C=C, n_fft=n_fft, hop_length=hop_len, bands=band_split_ratio, name=disc_name))
        
        self.period_disc_list = nn.ModuleList()
        for i, period in enumerate(period_list):
            disc_name = f"pd{i + 1}"
            self.period_disc_list.append(PeriodDiscriminator(period, name=disc_name, C=C_period))
    
   
    def d_loss(self, x, x_hat, adv_loss_type, **kwargs):
        """
        Collect losses from sub-discrimiantors in the SSDiscrimiantorBlock class.
        """
        loss_total = 0
        loss_dict = {}

        loss_report_total = {}
        
        # Multi-scale discriminators
        for wd in self.wave_disc_list:
            loss_d_ref, loss_d_syn, report = wd.d_loss(x, x_hat, adv_loss_type)
            
            loss_total += loss_d_ref + loss_d_syn
            loss_report_total = dict(loss_report_total, **report)       
            
        # STFT-based discriminators
        for sd in self.stft_disc_list:
            # each sub-d returns three outputs
            loss_d_ref, loss_d_syn, report = sd.d_loss(x, x_hat, adv_loss_type)
            
            loss_total += loss_d_ref + loss_d_syn
            loss_report_total = dict(loss_report_total, **report)  
            
        # Multi-period discriminators
        for pd in self.period_disc_list:
            loss_d_ref, loss_d_syn, report = pd.d_loss(x, x_hat, adv_loss_type)
            
            loss_total += loss_d_ref + loss_d_syn
            loss_report_total = dict(loss_report_total, **report)  
        
        loss_total /= (self.wd_num + self.sd_num + self.pd_num)
        
        loss_dict['adv_d'] = loss_total
        
        return loss_dict, loss_report_total   
    
    def g_loss(self, x, x_hat, adv_loss_type, **kwargs):
        """
        Collect losses from sub-discrimiantors in the SSDiscrimiantorBlock class.
        """
        loss_adv_g_total = 0
        loss_fm_total = 0
        loss_dict = {}

        loss_report_total = {}
        
        # Multi-scale discriminators
        for wd in self.wave_disc_list:
            loss_adv_g, loss_fm, report = wd.g_loss(x, x_hat, adv_loss_type)
            
            loss_adv_g_total += loss_adv_g
            loss_fm_total += loss_fm
            loss_report_total = dict(loss_report_total, **report)   
            
        # STFT-based discriminators
        for sd in self.stft_disc_list:
            loss_adv_g, loss_fm, report = sd.g_loss(x, x_hat, adv_loss_type)
            
            loss_adv_g_total += loss_adv_g
            loss_fm_total += loss_fm
            loss_report_total = dict(loss_report_total, **report)  
        
        # Multi-period discriminators
        for pd in self.period_disc_list:
            loss_adv_g, loss_fm, report = pd.g_loss(x, x_hat, adv_loss_type)
            
            loss_adv_g_total += loss_adv_g
            loss_fm_total += loss_fm
            loss_report_total = dict(loss_report_total, **report)   
        
        loss_adv_g_total /= (self.wd_num + self.sd_num + self.pd_num)
        loss_fm_total /= (self.wd_num + self.sd_num + self.pd_num)
        
        loss_dict['adv_g'] = loss_adv_g_total
        loss_dict['fm'] = loss_fm_total
        
        return loss_dict, loss_report_total
    
    def load_checkpoint(self, ckpt, strict=True):
        self.load_state_dict(ckpt['discriminator_state_dict'], strict=strict)
        
    def save_checkpoint(self, optimizer_d, epoch, iteration, save_path, tag):
        disc_dir = os.path.join(save_path, 'disc')
        disc_save_dict = {"epoch": epoch,
                    "iteration": iteration,
                    'discriminator_state_dict': self.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict()}
        torch.save(disc_save_dict, os.path.join(disc_dir, tag))
      

class DiscCore(nn.Module):
    """
    Core class for sub-discrimiantors.
    Every discriminator inherits DiscCore
    """
    def __init__(self):
        super().__init__()
        
    def forward(self):
        return
    
    def d_loss(self, x, x_hat, obj_func='hinge'):
        """
        Calculate adversarial loss for discrimiantor. Hinge loss and LSGAN loss are available.
        loss_ref:   real data must be 1
        loss_syn:   fake data must be 0
        report:     loss_ref, loss_syn, sum(nash_eq)
        """
        report = {}

        # Compute scores
        ref_scores, _ = self.forward(x, False)
        syn_scores, _ = self.forward(x_hat.detach(), False) ##

        if obj_func == 'hinge':
            # Adversarial loss for discriminators (hinge loss)
            loss_ref = torch.mean(torch.clamp(1 - ref_scores, min=0))
            report['d_adv_ref/' + self.name] = loss_ref.item()

            loss_syn = torch.mean(torch.clamp(1 + syn_scores, min=0))
            report['d_adv_syn/' + self.name] = loss_syn.item()
            
            report['nash_eq/' + self.name] = loss_ref.item() + loss_syn.item()  # Converge if == 2
            
        elif obj_func == 'ls':
            # Adversarial loss for discriminators (lsgan loss)
            loss_ref = torch.mean((1 - ref_scores) ** 2)
            report['d_adv_ref/' + self.name] = loss_ref.item()

            loss_syn = torch.mean((syn_scores) ** 2)
            report['d_adv_syn/' + self.name] = loss_syn.item()
            
            report['nash_eq/' + self.name] = loss_ref.item() + loss_syn.item()  # Converge if ?
        
        return loss_ref, loss_syn, report 
    
    def g_loss(self, x, x_hat, obj_func='hinge'):
        """
        Calculate adversarial loss for generator and feature matching loss. Hinge loss and LSGAN loss are available.
        """
        report = {}

        # Compute scores and get features maps
        _, ref_features = self.forward(x, True)
        syn_scores, syn_features = self.forward(x_hat, True)
        # ref_features = [f.detach() for f in ref_features]

        if obj_func == 'hinge':
            # Adversarial loss for generators (hinge loss)
            loss_adv = torch.mean(-syn_scores)
            report['g_adv/' + self.name] = loss_adv.item()
            
        elif obj_func == 'ls':
            # Adversarial loss for generators (lsgan loss)
            loss_adv = torch.mean((1 - syn_scores) ** 2)
            report['g_adv/' + self.name] = loss_adv.item()

        # Feature matching loss
        loss_fm = 0
        num_features = 0
        for ref_feature, syn_feature in zip(ref_features, syn_features):
            loss_fm += torch.mean(torch.abs(ref_feature - syn_feature)) / torch.mean(torch.abs(ref_feature))
            num_features += 1
        loss_fm /= num_features
        report['fm/' + self.name] = loss_fm.item()
        
        return loss_adv, loss_fm, report
        

#### Multi-Scale Discriminator ####
def WNConv1d(*args, **kwargs):
    """
    1d-convolution with wieght normalization
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConv2d(*args, **kwargs):
    """
    2d-convolution with wieght normalization
    """
    return weight_norm(nn.Conv2d(*args, **kwargs))

class WaveDiscriminator(DiscCore):
    def __init__(self, downsample_factor, name):
        super().__init__()
        self.name = name
        self.downsample_factor = downsample_factor
        self.downsampler = nn.AvgPool1d(kernel_size=2 * downsample_factor, stride=downsample_factor, padding=downsample_factor,
                                        count_include_pad=False)

        self.layers = nn.ModuleList([
            nn.Sequential(
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])

    def forward(self, x, return_features):
        x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        if self.downsample_factor != 1:
            x = self.downsampler(x)
        
        if return_features:
            feature_map = []
        else:
            feature_map = None
  
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_features and i < len(self.layers) - 1:
                feature_map.append(x)
                
        return x, feature_map         


#### STFT-based Discriminator ####
class ResidualUnit2d(nn.Module):
    """
    Resudial unit for STFT-discriminator
    """
    def __init__(self, in_channels, out_channels, stride_t, stride_f):
        super().__init__()
        
        self.stride_t = stride_t
        self.stride_f = stride_f

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding="same"
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(stride_f+2, stride_t+2),
                stride=(stride_f, stride_t)
            )
        )
        
        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1), stride=(stride_f, stride_t)
        )

    def forward(self, x):
        return self.layers(F.pad(x, [self.stride_t + 1, 0, self.stride_f + 1, 0])) + self.skip_connection(x)


class STFTDiscriminator(DiscCore):
    def __init__(self, C, n_fft, hop_len, name):
        super().__init__()
        
        self.name = name
        
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.f_bins = int(self.n_fft / 2)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=C, kernel_size=(7, 7), padding='same'),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=C,  out_channels=2*C, stride_t=1, stride_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2*C, out_channels=4*C, stride_t=2, stride_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, out_channels=4*C, stride_t=1, stride_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, out_channels=8*C, stride_t=2, stride_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C, out_channels=8*C, stride_t=1, stride_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C,  out_channels=16*C, stride_t=2, stride_f=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16*C, out_channels=1, kernel_size=(self.f_bins//2**6, 1))
        ])

    def forward(self, x, return_features):        
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_len, return_complex=True)   # [B, freq_bins, #frames]
        x = x.unsqueeze(1)
        x = torch.cat([x.real, x.imag], dim=1)  # Concatenate real and imaginary parts in channel dimension: [B, 2, freq_bins, #frames]

        if return_features:
            feature_map = []
        else:
            feature_map = None
        
        for i, layer in enumerate(self.layers):
            x = layer(x)    
            if return_features and i < len(self.layers) - 1:
                feature_map.append(x)
                
        return x, feature_map


'''
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_group_norm'])

def apply_parametrization_norm(module: nn.Module, norm: str = 'none'):
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs):
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class STFTDiscriminator_EnCodec(DiscCore):
    """STFT sub-discriminator from Encodec.

    Args:
        C (int): Output channel size of initial convolutional layer.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        C_growth (int): Growth factor for the internal channel size.
        C_max (int): maximum number of internal channel size.
        n_fft (int): Size of FFT for each scale.
        hop_length (int): Length of hop between STFT windows for each scale.
        win_length (int): Window size for each scale.
        normalized (bool): Whether to normalize by magnitude after stft.
        kernel_size (tuple of int): Inner Conv2d kernel sizes.
        stride (tuple of int): Inner Conv2d strides.
        dilations (list of int): Inner Conv2d dilation on the time dimension.
        norm (str): Normalization method.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.     
        name (str): name of discrimiantor
    """
    def __init__(self, C: int, in_channels: int = 1, out_channels: int = 1, C_growth: int = 1, C_max: int = 1024,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, normalized: bool = True,
                 kernel_size: tp.Tuple[int, int] = (3, 9), stride: tp.Tuple[int, int] = (1, 2), dilations: tp.List = [1, 2, 4],
                 norm: str = 'weight_norm', activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2},
                 name: str = "", **kwargs):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        
        self.name = name
        
        self.C = C
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = getattr(torch.nn, activation)(**activation_params)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=self.normalized, center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        
        self.layers = nn.ModuleList()
        
        self.layers.append(
            NormConv2d(spec_channels, self.C, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        
        in_chs = min(C_growth * self.C, C_max)
        for i, dilation in enumerate(dilations):
            out_chs = min((C_growth ** (i + 1)) * self.C, C_max)
            self.layers.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                         dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm=norm))
            in_chs = out_chs
        out_chs = min((C_growth ** (len(dilations) + 1)) * self.C, C_max)
        
        self.layers.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm=norm))
        
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)

    def forward(self, x: torch.Tensor, return_features):
        if return_features:
            feature_map = []
        else:
            feature_map = None

        x = x.unsqueeze(1)                      # [B, T] -> [B, 1, T]
        z = self.spec_transform(x)              # [B, C, F, T', 2]
        z = torch.cat([z.real, z.imag], dim=1)  # [B, 2C, F, T']
        z = rearrange(z, 'b c w t -> b c t w')  # [B, 2C, T', F]
        for i, layer in enumerate(self.layers):
            z = layer(z)
            z = self.activation(z)
            if return_features:
                feature_map.append(z)
            
        z = self.conv_post(z)
        
        return z, feature_map


'''
MIT License

Copyright (c) 2023-present, Descript

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
class MultiBandSTFTDiscriminator(DiscCore):
    def __init__(
        self,
        C: int,
        n_fft: int,
        hop_length: int,
        bands: list = BANDS,
        name: str = "", 
        **kwargs
    ):
        """Complex multi-band spectrogram discriminator, from DAC.
        Parameters
        ----------
        C : int
            channel size of the discriminator.
        n_fft : int
            FFT size of STFT.
        hop_length : int
            Hop length of the STFT.
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()
        
        self.name = name
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        n_bins = n_fft // 2 + 1
        bands = [(int(b[0] * n_bins), int(b[1] * n_bins)) for b in bands] # band index extraction (start,end)
        self.bands = bands

        ch = C
        layers = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)), # downsampling along F axis
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.layers_per_band = nn.ModuleList([layers() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1))
        self.activation = nn.LeakyReLU(0.1)
    
    def stft_band_split(self, x):
        # STFT and rearrange
        x_stft = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)   # [B, T] -> [B, F, T]
        x_stft = torch.view_as_real(x_stft)                 # [B, F, T, 2]
        x_stft = rearrange(x_stft, "b f t c -> b c t f")    # [B, 2, T, F]
        
        # Split into bands
        x_bands = [x_stft[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x, return_features):
        """
        feature_map: list of intermediate convolution feature maps [B,C,T,F]
        """
        x_bands = self.stft_band_split(x)   # [B, T] -> List[B, 2, T, f]
        
        feature_map = []

        outputs_per_band = []
        for band, layers in zip(x_bands, self.layers_per_band):
            for layer in layers:
                band = layer(band)
                band = self.activation(band)
                if return_features:
                    feature_map.append(band)
                
            # print(f"band shape: {band.shape}")
            outputs_per_band.append(band)

        z = torch.cat(outputs_per_band, dim=-1) # [B, C, T, f] cat in f dim
        z = self.conv_post(z) # [B, 1, T, F]

        return z, feature_map


#### Multi-Period Discriminator ####
class PeriodDiscriminator(DiscCore):
    def __init__(self, period, C=64, name=""):
        super().__init__()
        self.name = name
        self.period = period
        self.layers = nn.ModuleList([
            nn.Sequential(
                WNConv2d(in_channels=1, out_channels=C, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            nn.Sequential(
                WNConv2d(in_channels=C, out_channels=2*C, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            nn.Sequential(
                WNConv2d(in_channels=2*C, out_channels=4*C, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            nn.Sequential(
                WNConv2d(in_channels=4*C, out_channels=8*C, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            nn.Sequential(
                WNConv2d(in_channels=8*C, out_channels=16*C, kernel_size=(5, 1), padding=(2, 0)),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            WNConv2d(in_channels=16*C, out_channels=1, kernel_size=(3, 1), padding=(1, 0))
        ])

    def forward(self, x, return_features):
        """
        for each period
        x: Tensor [B,1,T_down,p] 
        fm: List of [B,C,T_down,p]
        """
        x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        x = F.pad(x, (0, self.period - x.size(-1) % self.period), 'constant')   # Padding before squeezing
        x = x.view(x.size(0), 1, -1, self.period)   # [B, 1, T_padded / P, P]
        
        if return_features:
            feature_map = []
        else:
            feature_map = None
  
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_features and i < len(self.layers) - 1:
                feature_map.append(x)
                
        return x, feature_map         
