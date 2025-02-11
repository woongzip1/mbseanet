import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from modules import Conv1d,EncBlock,DecBlock
from models.modules import Conv1d,EncBlock,DecBlock


class MBSEANet(nn.Module):
    def __init__(self, min_dim=8, strides=[1,2,2,2], 
                 c_in=32, c_out=32, out_bias=True,
                 **kwargs):
        super().__init__()
        
        self.min_dim = min_dim # first conv output channels
        self.downsampling_factor = np.prod(strides)
        self._initialize_weights()
        
        self.conv_in = Conv1d(
            in_channels=c_in,
            out_channels=min_dim,
            kernel_size=7,
            stride=1
        )
        
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, strides[0]),
                                    EncBlock(min_dim*4, strides[1]),
                                    EncBlock(min_dim*8, strides[2]),
                                    EncBlock(min_dim*16, strides[3])                                        
                                    ])
        
        self.conv_bottle = nn.Sequential(
                                        Conv1d(
                                            in_channels=min_dim*16,
                                            out_channels = min_dim*16//4,
                                            kernel_size = 7, 
                                            stride = 1,
                                            ),
                                        
                                        Conv1d(
                                            in_channels=min_dim*16//4,
                                            out_channels = min_dim*16,
                                            kernel_size = 7,
                                            stride = 1,
                                            ),
                                        )
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, strides[3]),
                                    DecBlock(min_dim*4, strides[2]),
                                    DecBlock(min_dim*2, strides[1]),
                                    DecBlock(min_dim, strides[0]),
                                    ])
        
        self.conv_out = Conv1d(
            in_channels=min_dim,
            out_channels=c_out,
            kernel_size=7,
            stride=1,
            bias=out_bias,
        )
        
    def _adjust_signal_len(self, x):
        B,C,sig_len = x.size() # [B,C,T]
        pad_len = sig_len % self.downsampling_factor
        if pad_len != 0:
            x = x[:, :, :-pad_len]
            # print(pad_len) ### for debug
        return x, pad_len
    
    def forward(self, x, HR=None,):
        x, pad_len = self._adjust_signal_len(x) # adjust length
        
        # conv in
        skip = []
        x = self.conv_in(x)
        skip.append(x)
        # enc
        for encoder in self.encoder:
            x = encoder(x)
            skip.append(x)
        # bottleneck
        x = self.conv_bottle(x)
        # dec
        skip = skip[::-1]
        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)
        x = x + skip[4]
        # conv out
        x = self.conv_out(x)
        
        # pad
        padval = torch.zeros([x.size(0), x.size(1), pad_len]).to(x.device) # [B,C_out,T]
        x = torch.cat((x, padval), dim=-1)
        
        return x
    
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


if __name__ == "__main__":
    """ Multi band SEANet """
    from torchinfo import summary
    model = MBSEANet(c_out=32, c_in=5, min_dim=32, out_bias=False)
    wav = torch.rand(4,5,55400)
    summary(
        model, input_data = wav,
        col_names=['input_size','output_size'],
        depth=2
    )