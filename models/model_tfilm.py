import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
import torch.nn.functional as F
import pickle
from einops import rearrange
import pickle

from models.modules import Conv1d,EncBlock,DecBlock
from models.feature_encoder import AutoEncoder
from models.quantize import ResidualVectorQuantize
# from vector_quantize_pytorch import ResidualVQ

"""
****** Temporal FiLMing ******

n_channels: Number of Conv feature map channels

do not input N x L x C

x: Conv Feature Map (N x C x L)
cond: SSL Condition (N x L/320 x 1024)
        --> N x #FramePatches x F

output:  modulated feature map (N x C x L)

"""
class FiLMLayer(nn.Module):
    def __init__(self, n_channels=10, subband_num=3, visualize=False):
        """ What does melpatch_num do ? """
        super().__init__()
        self.n_channels = n_channels # channels of conv layer
        # 320: subband dim 
        self.subband_num = subband_num
        # 32: output of FeatureReduction layer
        self.film_gen = nn.Linear(32*self.subband_num, 2*n_channels)
        self.visualize = visualize

    def forward(self, x, condition):

        ## X -> N C L
        ## Condi -> N L C
        subblock_num = condition.size(1)
        original_length = x.size(-1) # length of x
        
        if self.visualize:
            print("Original Feature Map Shape:", x.size())

        # Padding 
        padding_size = (subblock_num - original_length % subblock_num) % subblock_num
        if padding_size > 0:
            x = F.pad(x, (0, padding_size), mode='constant', value=0)
            if self.visualize:
                print("Padded Feature Map Shape:", x.size())
                print("Condition Shape", condition.size())

        film_params = self.film_gen(condition)
        # if self.visualize: print(film_params.shape, "FiLM Generated Shape")
        film_params = rearrange(film_params, 'n l c -> n c l')
        # Extract (ch x subblock_num) gamma and beta 
        gamma = film_params[:,:self.n_channels,:].unsqueeze(-1)
        beta = film_params[:,self.n_channels:,:].unsqueeze(-1)

        ## Reshape Feature Map
        if self.visualize: 
            print("Subblock_num (Frame num):", subblock_num)
            print("\t",x.shape,"-->", end=' ')

        x = rearrange(x, "n c (b t) -> n c b t", b=subblock_num)

        if self.visualize: 
            print("\t",x.shape)
            print("\t",beta.shape, "BETA Shape")
            print("\t",x.shape, "Feature Map Shape")

        # Linear Modulation
        x = gamma * x + beta
        x = rearrange(x, 'n c b t -> n c (b t)')
        if padding_size > 0:    
            x = x[:, :, :original_length]
            if self.visualize:
                print("Cropped Feature Map Shape:", x.shape,"\n")

        return x

"""
# x as a conv feature map shape (B x C x L)
x = torch.rand(3, 4, 16000)
cond = torch.rand(3, #frames, #features)
model = FiLMLayer(n_channels=4, visualize=True)
y = model(x,cond)
print(y.shape)
"""

"""
Input Shape: B x Patch x 6144
Output Shape: B x Patch x 256 
"""
class FeatureReduction(nn.Module):
    def __init__(self, subband_num=10, D=512):
        super(FeatureReduction, self).__init__()

        self.subband_num = subband_num
        self.patch_len = D #  8D for feature encoder
        self.layers = nn.ModuleList([nn.Linear(self.patch_len,32) for _ in range(self.subband_num)])
        # 8 Feature Encoder 512x10 -> 32x10 dim
    
    def forward(self, embeddings):
        # embedding: B x Patch x 6144 (5120)
        ## input must be: B x T x (D F)

        outs = []
        for idx, layer in enumerate(self.layers):
            patch_embeddings = embeddings[:, :, idx*self.patch_len:(idx+1)*self.patch_len] # extract per subband
            # print(idx, patch_embeddings.shape)
            out = layer(patch_embeddings)
            outs.append(out)
            # print(out.shape)
        final_output = torch.cat(outs, dim=-1)

        return final_output

class MBSEANet_film(nn.Module):
    def __init__(self, min_dim=8, strides=[1,2,2,2], 
                 in_channels=16, subband_num=27, 
                 c_in=5, c_out=32, out_bias=True, visualize=False,
                 **kwargs):
        super().__init__()
        
        self.visualize = visualize
        self.downsampling_factor = np.prod(strides)
        self._initialize_weights()

        ## Load SSL model
        from models.feature_encoder import ResNet18
        self.feature_encoder = ResNet18(in_channels=in_channels)

        self.rvq = ResidualVectorQuantize(
            input_dim=subband_num*32,        #
            n_codebooks=10,         # 
            codebook_size=1024,     # 
            codebook_dim=8,       # 
            quantizer_dropout=0.5  # 
        )

        # Feature Extracted SSL Layers
        self.subband_num = subband_num
        self.EmbeddingReduction = FeatureReduction(self.subband_num, D=in_channels*8)

        # Encoder blocks and FiLM layers combined
        self.encoder_with_film = nn.ModuleList([
            nn.Sequential(
                EncBlock(min_dim * 2, strides[0]),
                FiLMLayer(n_channels=min_dim * 2, subband_num=subband_num, visualize=visualize)
            ),
            nn.Sequential(
                EncBlock(min_dim * 4, strides[1]),
                FiLMLayer(n_channels=min_dim * 4, subband_num=subband_num, visualize=visualize)
            ),
            nn.Sequential(
                EncBlock(min_dim * 8, strides[2]),
                FiLMLayer(n_channels=min_dim * 8, subband_num=subband_num, visualize=visualize)
            ),
            nn.Sequential(
                EncBlock(min_dim * 16, strides[3]),
                FiLMLayer(n_channels=min_dim * 16, subband_num=subband_num, visualize=visualize)
            )
        ])

        # Bottleneck layers
        self.conv_bottle1 = Conv1d(
            in_channels=min_dim * 16,
            out_channels=min_dim * 16 // 4,
            kernel_size=7,
            stride=1,
        )
        self.film_b1 = FiLMLayer(n_channels=min_dim * 16 // 4, subband_num=subband_num, visualize=visualize)

        self.conv_bottle2 = Conv1d(
            in_channels=min_dim * 16 // 4,
            out_channels=min_dim * 16,
            kernel_size=7,
            stride=1,
        )
        self.film_b2 = FiLMLayer(n_channels=min_dim * 16, subband_num=subband_num, visualize=visualize)

        # Decoder blocks and FiLM layers combined
        self.decoder_with_film = nn.ModuleList([
            nn.Sequential(
                DecBlock(min_dim * 8, strides[3]),
                FiLMLayer(n_channels=min_dim * 8, subband_num=subband_num)
            ),
            nn.Sequential(
                DecBlock(min_dim * 4, strides[2]),
                FiLMLayer(n_channels=min_dim * 4, subband_num=subband_num)
            ),
            nn.Sequential(
                DecBlock(min_dim * 2, strides[1]),
                FiLMLayer(n_channels=min_dim * 2, subband_num=subband_num)
            ),
            nn.Sequential(
                DecBlock(min_dim, strides[0]),
                FiLMLayer(n_channels=min_dim, subband_num=subband_num)
            )
        ])

        # Input and output convolutional layers
        self.conv_in = Conv1d(
            in_channels=c_in,
            out_channels=min_dim,
            kernel_size=7,
            stride=1,
        )
        self.conv_out = Conv1d(
            in_channels=min_dim,
            out_channels=c_out,
            kernel_size=7,
            stride=1,
            bias=out_bias,
        )

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
            x = x[:, :, :-pad_len]
        print(pad_len)
        return x, pad_len

    def forward(self, x, condition):
        x, pad_len = self._adjust_signal_len(x)

        # Feature extraction and reduction
        embedding = self.feature_encoder(condition)
        embedding = rearrange(embedding, 'b d f t -> b t (d f)')
        embedding = self.EmbeddingReduction(embedding)
        embedding = rearrange(embedding, 'b t f -> b f t')

        # Residual vector quantization (RVQ)
        embedding, codes, latents, commitment_loss, codebook_loss = self.rvq(embedding)
        embedding = rearrange(embedding, 'b f t -> b t f')

        # Skip connections
        skip = []
        x = self.conv_in(x)
        skip.append(x)

        if self.visualize:
            print("Input Signal Length:", x.shape[2], "Fragment:", pad_len)

        # Encoder with FiLM layers
        for block in self.encoder_with_film:
            x = block[0](x)  # EncBlock
            x = block[1](x, embedding)  # FiLMLayer
            skip.append(x)

        # Bottleneck
        x = self.conv_bottle1(x)
        x = self.film_b1(x, embedding)
        x = self.conv_bottle2(x)
        x = self.film_b2(x, embedding)

        # Decoder with FiLM layers
        skip = skip[::-1]
        for block, skip_connection in zip(self.decoder_with_film, skip):
            x = x + skip_connection
            x = block[0](x)  # DecBlock
            x = block[1](x, embedding)  # FiLMLayer

        x = self.conv_out(x)

        padval = torch.zeros([x.size(0), x.size(1), pad_len]).to(x.device) # [B,C_out,T]
        x = torch.cat((x, padval), dim=-1)

        return x, commitment_loss, codebook_loss
