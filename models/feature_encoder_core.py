import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm

import math
from torchinfo import summary
                    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, isfinal=False):
        super(BasicBlock, self).__init__()
        self.conv1 = weight_norm(CausalConv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =  weight_norm(CausalConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False))
        self.downsample = downsample  # Used for downsampling (channel modificiation)
        self.isfinal = isfinal

    def forward(self, x):
        # print("input shape", x.shape)
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        # Skip ReLU for Final output
        if not self.isfinal:
            out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=64, 
                 use_condition=False, condition_channels=1, 
                 use_sfm=False, sfm_channels=0,
                 visualize=False):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.bottleneckdim = in_channels
        self.use_condition = use_condition
        self.use_sfm = use_sfm
        
        self.conv1 = weight_norm(CausalConv2d(1, self.bottleneckdim, kernel_size=(7,7), stride=(2,1), bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,1), padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, self.bottleneckdim, layers[0])
        self.layer2 = self._make_layer(block, self.bottleneckdim*2, layers[1], stride=(2,1))
        self.layer3 = self._make_layer(block, self.bottleneckdim*4, layers[2], stride=(2,1))
        self.layer4 = self._make_layer(block, self.bottleneckdim*8, layers[3], stride=(2,1), isfinal=True)
        
        if self.use_condition:
            self.feature_pool = nn.Conv1d(in_channels=condition_channels, out_channels=condition_channels, 
                                          kernel_size=8, stride=8, bias=False, groups=1)
            if self.use_sfm:
                condition_channels = condition_channels + sfm_channels
            self.film1 = FiLMLayer2D(n_channels=self.bottleneckdim, condition_channel=condition_channels, visualize=visualize)
            self.film2 = FiLMLayer2D(n_channels=self.bottleneckdim, condition_channel=condition_channels, visualize=visualize)
            self.film3 = FiLMLayer2D(n_channels=self.bottleneckdim*2, condition_channel=condition_channels, visualize=visualize)
            self.film4 = FiLMLayer2D(n_channels=self.bottleneckdim*4, condition_channel=condition_channels, visualize=visualize)
        
    def _make_layer(self, block, out_channels, blocks, stride=1, isfinal=False):
        downsample = None
        if stride != 1 : # Downsampling layer needs channel modification
            downsample = CausalConv2d(self.in_channels, out_channels,
                             kernel_size=1, stride=stride, bias=False)
                        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, isfinal=isfinal))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, condition=None, sfm_feature=None):
        """
        x: (B, 1, F, T) - Input spectrogram
        condition: (B, C_condition, T) - Spectral Flatness Measure, only used if use_condition=True
        """
        if self.use_condition and condition is None:
            raise ValueError("use_condition=True, but condition is None. Please provide SFM input")
        
        if self.use_condition:
            core_condition = self.feature_pool(condition) # [B,C_core,T] -> [B,C_core,T/8]
            assert core_condition.shape[-1] == x.shape[-1], "Condition must be aligned!"
            
        if self.use_sfm:
            print(core_condition.shape, sfm_feature.shape)
            core_condition = torch.cat((core_condition, sfm_feature), dim=1)
            
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet Layers with optional FiLM
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        films = [self.film1, self.film2, self.film3, self.film4] if self.use_condition else [None] * 4

        for layer, film in zip(layers, films):
            if film is not None and core_condition is not None:
                print(core_condition.shape)
                x = film(x, core_condition)  # Apply FiLM if enabled
            x = layer(x)  # Apply ResNet Layer
        return x
    
def ResNet18(in_channels=64, condition_channels=1, use_condition=False, use_sfm=False, sfm_channels=0, visualize=False, ):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, condition_channels=condition_channels, use_condition=use_condition, 
                  use_sfm=use_sfm, sfm_channels=sfm_channels,
                  visualize=visualize)

from einops import rearrange
class FiLMLayer2D(nn.Module):
    def __init__(self, n_channels, condition_channel, visualize=False):
        """
        n_channels: CNN Feature map's channel count (e.g., 64, 128, ...)
        condition_channel: The number of channels in condition information (C_sfm)
        """
        super().__init__()
        self.n_channels = n_channels
        self.condition_channel = condition_channel
        self.film_gen = nn.Linear(self.condition_channel, 2*self.n_channels) # (C_s) → (2*n_channels)
        self.visualize = visualize

    def forward(self, x, condition,):
        """
        x: (B, C, F, T) - Feature Encoder's 2D CNN Feature Map
        condition: (B, C_condition, T) - condition information without a frequency axis
        """
        if self.visualize: print('\t', x.shape, "Feature Map Shape")

        # Generate FiLM modulation parameters (gamma, beta)
        condition = rearrange(condition, 'b c t -> b t c')  # (B,C_condition,T) -> (B,T,C_condition)
        film_params = self.film_gen(condition)        # (B,T,C_condition) -> (B,T,2C)
        film_params = rearrange(film_params, 'b t c-> b c t')  # (B,2C,T)
        
        # Separate gamma and beta
        gamma = film_params[:, :self.n_channels,:,].unsqueeze(2)     # (B,2C,T) -> (B,C,1,T)
        beta = film_params[:, :self.n_channels, :,].unsqueeze(2)      # (B,2C,T) -> (B,C,1,T)
        
        # Apply FiLM
        x = gamma * x + beta
        return x
    
## Conv-ReLU-Conv with Residual Connection
class ResBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = weight_norm(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1))

    def forward(self, x, final=False):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += identity
        if final:
            out = x
        else:
            out = self.relu(x)
        return out

""" Causal Conv """
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + 1 - self.stride[0]
 
    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Calculate padding for temporal dimension (T)
        self.temporal_padding = self.dilation[1] * (self.kernel_size[1] - 1) - (self.stride[1] - 1)
        
        # Calculate total padding for frequency dimension (F)
        total_f_padding = self.dilation[0] * (self.kernel_size[0] - 1) - (self.stride[0] - 1)
        
        # Split total padding into top and bottom (asymmetrical padding if needed)
        self.frequency_padding_top = math.ceil(total_f_padding / 2)
        self.frequency_padding_bottom = math.floor(total_f_padding / 2)
        
    def forward(self, x):
        # Apply padding: F (top and bottom), T (only to the left)
        # print(f"Temporal Padding (T): {self.temporal_padding}")
        # print(f"Frequency Padding (F): top={self.frequency_padding_top}, bottom={self.frequency_padding_bottom}")
        x = F.pad(x, [self.temporal_padding, 0, self.frequency_padding_top, self.frequency_padding_bottom])
        return self._conv_forward(x, self.weight, self.bias)

def main():
    ## usage
    film_layer = FiLMLayer2D(n_channels=32, condition_channel=10)
    feature_map = torch.rand(1,32,512,22)
    sfm = torch.rand(1,10,22)
    out = film_layer(feature_map,sfm)
    print(out.shape)


