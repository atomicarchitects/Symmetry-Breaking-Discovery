import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Relaxed_Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 h_size, # Image size
                 w_size,
                 num_filter_banks, # L
                 activation
                ):

        super(Relaxed_Conv2d, self).__init__()

        # Initialize multiple filters.
        self.convs = nn.Sequential(*[nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size,
                                               padding=(kernel_size-1)//2)
                                     for i in range(num_filter_banks)])
        self.num_filter_banks = num_filter_banks

        # For every location we initialize a set of equal coefficients for linear combinations.
        self.relaxed_weights = nn.Parameter(torch.ones(h_size, w_size, num_filter_banks).float())
        self.activation = activation

    def forward(self, x):
        # Compute Convolution for each filter
        outs = torch.stack([self.convs[i](x) for i in range(self.num_filter_banks)], dim  = 0)
        
        # Compute the linear combination of kernels for each spatial location
        out = torch.einsum("ijr, rboij -> boij", self.relaxed_weights, outs)
        if self.activation:
            return F.relu(out)
        else:
            return out
    
    
class TranSymDisNet(torch.nn.Module): 
    """Relaxed Translation Group ConvNet"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 h_size, 
                 w_size,
                 hidden_dim, 
                 num_filter_basis
                 ):
        super(TranSymDisNet, self).__init__()
        
        self.model = torch.nn.Sequential(
            Relaxed_Conv2d(in_channels, hidden_dim, kernel_size, h_size, w_size, num_filter_basis, activation = True),
            Relaxed_Conv2d(hidden_dim, hidden_dim, kernel_size, h_size, w_size, num_filter_basis, activation = True),
            Relaxed_Conv2d(hidden_dim, out_channels, kernel_size, h_size, w_size, num_filter_basis, activation = False)
        )
        
    def forward(self, inp):
        return self.model(inp)
 


