import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.utils.weight_norm as weight_norm
from torch.nn.utils.parametrizations import weight_norm


import math
from torchinfo import summary

class AutoEncoder(nn.Module):
    """ New AutoEncoder for 26 subband feature extraction """
    def __init__(self, in_channels=16):
        super(AutoEncoder, self).__init__()
        
        self.encoder = ResNet18(in_channels=in_channels)
        self.decoder = Decoder_new(bottleneck_shape=in_channels * 8)

        self.initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def initialize_weights(self):
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
                    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("**** CHECKPOINT LOADED for Feature Encoder! **** ")
                    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, isfinal=False):
        super(BasicBlock, self).__init__()
        self.conv1 = weight_norm(CausalConv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False))
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =  weight_norm(CausalConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False))
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
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
    def __init__(self, block, layers, in_channels=64, sfm_channels=1, use_sfm=False, visualize=False):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.bottleneckdim = in_channels
        self.use_sfm = use_sfm
        
        self.conv1 = weight_norm(CausalConv2d(1, self.bottleneckdim, kernel_size=(7,7), stride=(2,1), bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,1), padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, self.bottleneckdim, layers[0])
        self.layer2 = self._make_layer(block, self.bottleneckdim*2, layers[1], stride=(2,1))
        self.layer3 = self._make_layer(block, self.bottleneckdim*4, layers[2], stride=(2,1))
        self.layer4 = self._make_layer(block, self.bottleneckdim*8, layers[3], stride=(2,1), isfinal=True)
        
        if self.use_sfm:
            self.film1 = FiLMLayer2D(n_channels=self.bottleneckdim, sfm_dim=sfm_channels, visualize=visualize)
            self.film2 = FiLMLayer2D(n_channels=self.bottleneckdim, sfm_dim=sfm_channels, visualize=visualize)
            self.film3 = FiLMLayer2D(n_channels=self.bottleneckdim*2, sfm_dim=sfm_channels, visualize=visualize)
            self.film4 = FiLMLayer2D(n_channels=self.bottleneckdim*4, sfm_dim=sfm_channels, visualize=visualize)
        
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

    def forward(self, x, sfm=None):
        """
        x: (B, 1, F, T) - Input spectrogram
        sfm: (B, C_sfm, T) - Spectral Flatness Measure, only used if use_sfm=True
        """
        if self.use_sfm and sfm is None:
            raise ValueError("use_sfm=True, but sfm is None. Please provide SFM input")
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet Layers with optional FiLM
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        films = [self.film1, self.film2, self.film3, self.film4] if self.use_sfm else [None] * 4

        for layer, film in zip(layers, films):
            if film is not None and sfm is not None:
                x = film(x, sfm)  # Apply FiLM if enabled
            x = layer(x)  # Apply ResNet Layer
        return x
    
    
def ResNet18(in_channels=64, sfm_channels=1, use_sfm=False, visualize=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, sfm_channels=sfm_channels, use_sfm=use_sfm, visualize=visualize)

from einops import rearrange
class FiLMLayer2D(nn.Module):
    def __init__(self, n_channels, sfm_dim, visualize=False):
        """
        n_channels: CNN Feature map's channel count (e.g., 64, 128, ...)
        sfm_dim: The number of channels in SFM information (C_sfm)
        """
        super().__init__()
        self.n_channels = n_channels
        self.sfm_dim = sfm_dim
        self.film_gen = nn.Linear(self.sfm_dim, 2*self.n_channels) # (C_s) → (2*n_channels)
        self.visualize = visualize

    def forward(self, x, sfm, ):
        """
        x: (B, C, F, T) - Feature Encoder's 2D CNN Feature Map
        sfm: (B, C_sfm, T) - SFM information without a frequency axis
        """
        if self.visualize: print('\t', x.shape, "Feature Map Shape")

        # Generate FiLM modulation parameters (gamma, beta)
        sfm = rearrange(sfm, 'b c t -> b t c')  # (B,C_sfm,T) -> (B,T,C_sfm)
        film_params = self.film_gen(sfm)        # (B,T,C_sfm) -> (B,T,2C)
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
    
class Decoder_new(nn.Module):
    def __init__(self, bottleneck_shape=512):
        super(Decoder_new, self).__init__()

        self.bottleneck_shape = bottleneck_shape
        
        self.c1 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape, self.bottleneck_shape//2, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv1 = ResBlock(self.bottleneck_shape//2)

        self.c2 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//2, self.bottleneck_shape//4, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv2 = ResBlock(self.bottleneck_shape//4)
        
        self.c3 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//4, self.bottleneck_shape//8, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv3 = ResBlock(self.bottleneck_shape//8)
        
        self.c4 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//8, self.bottleneck_shape//16, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv4 = ResBlock(self.bottleneck_shape//16)
        
        self.c5 = weight_norm(nn.ConvTranspose2d(self.bottleneck_shape//16, 1, kernel_size=(4,3), stride=(2,1), padding=(1,1)))
        self.conv5 = ResBlock(1)

    def forward(self, x):
        x = self.c1(x)
        x = self.conv1(x)
        x = self.c2(x)
        x = self.conv2(x)
        x = self.c3(x)
        x = self.conv3(x)
        x = self.c4(x)
        x = self.conv4(x)
        x = self.c5(x)
        x = self.conv5(x, final=True)
        return x

""" Causal Conv """
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + 1 - self.stride[0]
 
    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

# class CausalConv1d(nn.Conv1d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + 1 - self.stride[0]
#         self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

#     def forward(self, x):
#         x = F.pad(x, (self.causal_padding, 0))
#         return super().forward(x)
#         # return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)
        
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
    film_layer = FiLMLayer2D(n_channels=32, sfm_dim=10)
    feature_map = torch.rand(1,32,512,22)
    sfm = torch.rand(1,10,22)
    out = film_layer(feature_map,sfm)
    print(out.shape)


