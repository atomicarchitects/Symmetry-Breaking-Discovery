import torch
import torch.nn.functional as F
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NonEquivResBlock(torch.nn.Module):
    """Non-Equiv Upscaling ResNet Block"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation = True,
                 upscale = False
                 ):
        super(NonEquivResBlock, self).__init__()

        self.upscale = upscale
        if self.upscale:
            self.block = nn.Sequential(
                torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(out_channels, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)
            )
        else:
            self.block = nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                torch.nn.ReLU(),
                torch.nn.Conv3d(out_channels, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)
            )
        
        if self.upscale:
            # residual connection with simple trilinear interpolation
            self.residual = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        
        self.activation = activation


    def forward(self, x):
        # input shape: [bz, #in, d, h, w]
        # output shape: [bz, #out, d*2, h*2, w*2]
        if self.upscale:
            out = self.block(x) + self.residual(x)
        else:
            out = self.block(x) + x
        
        if self.activation:
            return F.relu(out)
        return out
    

class NonEquivResNet(torch.nn.Module): 
    """Non-Equiv SuperResolution ResNet"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_dim, 
                 kernel_size,
                 ):
        super(NonEquivResNet, self).__init__()
        self.embedding = nn.Sequential(
            torch.nn.Conv3d(in_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            torch.nn.ReLU()
        )
        self.model = [NonEquivResBlock(hidden_dim, hidden_dim, kernel_size, activation = True, upscale = True), 
                      NonEquivResBlock(hidden_dim, hidden_dim, kernel_size, activation = True, upscale = False),
                      NonEquivResBlock(hidden_dim, hidden_dim, kernel_size, activation = True ,upscale = True),
                      NonEquivResBlock(hidden_dim, hidden_dim, kernel_size, activation = True, upscale = False)]
        self.model = torch.nn.Sequential(*self.model)
        self.final_layer = torch.nn.Conv3d(hidden_dim, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)
        
    def forward(self, x):
        out = self.embedding(x)
        out = self.model(out)
        return self.final_layer(out)


