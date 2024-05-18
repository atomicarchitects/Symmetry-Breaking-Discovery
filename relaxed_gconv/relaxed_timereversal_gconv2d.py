import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F

class RelaxedTimeRev_LiftConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_filter_banks, # L
                 activation
                ):

        super(RelaxedTimeRev_LiftConv2d, self).__init__()
        
        self.num_filter_banks = num_filter_banks
        self.activation = activation
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Initialize multiple filters.
        # We assume the input is a stack of consectutive 1D frames, 
        # And the first dimension is time dimension. 
        self.kernel = torch.nn.Parameter(torch.zeros(self.num_filter_banks,
                                                     self.out_channels,
                                                     self.in_channels,
                                                     self.kernel_size,
                                                     self.kernel_size))
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))
        
        # We only consider the reflection over time dimension.
        self.relaxed_weights = nn.Parameter(torch.ones(self.num_filter_banks, 2).float())
        
        
    def generate_filter_bank(self):
        # we assume the first dim of a input 3d tensor is the time 
        filter_bank = torch.stack([self.kernel, torch.flip(self.kernel, dims = [3])], dim = 2)
        return filter_bank

    def forward(self, x):
        # The input shape: [bz, #in, t, w, d]
        filter_bank = self.generate_filter_bank()
        relaxed_conv_weights = torch.einsum("ng, nog... -> og...", self.relaxed_weights, filter_bank)
        
        
        out = F.conv2d(
            input=x,
            weight=relaxed_conv_weights.reshape(
                self.out_channels * 2,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ),
            padding = (self.kernel_size-1)//2
        )
        
        # ==============================
        # reshape output signal to shape [bz, #out, group order, 2, h, w, d]
        # ==============================
        out = out.view(
            out.shape[0],
            self.out_channels,
            2,
            out.shape[-2],
            out.shape[-1]
        )
        # ==============================
        if self.activation:
            return F.leaky_relu(out)
        else:
            return out
        
        

class RelaxedTimeRev_GroupConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_filter_banks, # L
                 activation
                ):

        super(RelaxedTimeRev_GroupConv2d, self).__init__()
        
        self.num_filter_banks = num_filter_banks
        self.activation = activation
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Initialize multiple filters.
        # We assume the input is a stack of consectutive 2D frames, 
        # And the third dimension is time dimension. 
        self.kernel = torch.nn.Parameter(torch.zeros(self.num_filter_banks,
                                                     self.out_channels,
                                                     self.in_channels,
                                                     2, 
                                                     self.kernel_size,
                                                     self.kernel_size))
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))
        
        # We only consider the reflection over time dimension.
        self.relaxed_weights = nn.Parameter(torch.ones(self.num_filter_banks, 2).float())
        
        
    def generate_filter_bank(self):
        filter_bank = torch.stack([self.kernel, torch.flip(self.kernel, dims = [3, 4])], dim = 2)
        return filter_bank

    def forward(self, x):
        # The input shape: [bz, #in, 2, h, w, d]
        filter_bank = self.generate_filter_bank()
        # print(self.relaxed_weights.shape, filter_bank.shape)
        relaxed_conv_weights = torch.einsum("ng, nogi... -> ogi...", self.relaxed_weights, filter_bank)
        relaxed_conv_weights.reshape(self.out_channels * 2,
                                     self.in_channels,
                                     2,
                                     self.kernel_size,
                                     self.kernel_size)
        x = x.reshape(x.shape[0], 
                      self.in_channels*2,  
                      x.shape[-2], 
                      x.shape[-1])
        
        out = F.conv2d(
            input=x,
            weight=relaxed_conv_weights.reshape(
                self.out_channels * 2,
                self.in_channels * 2,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )
        
        # ==============================
        # reshape output signal to shape [bz, #out, group order, 2, h, w, d]
        # ==============================
        out = out.view(
            out.shape[0],
            self.out_channels,
            2,
            out.shape[-2],
            out.shape[-1]
        )
        # ==============================
        if self.activation:
            return F.leaky_relu(out)
        else:
            return out
        
    
class TimeRevNet(torch.nn.Module): 
    """Relaxed Time Reversal Group ConvNet"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 hidden_dim, 
                 num_filter_basis, 
                 ):
        super(TimeRevNet, self).__init__()
        
        self.model = torch.nn.Sequential(
            RelaxedTimeRev_LiftConv2d(in_channels, hidden_dim, kernel_size, num_filter_basis, activation = True),
            RelaxedTimeRev_GroupConv2d(hidden_dim, hidden_dim, kernel_size, num_filter_basis, activation = True),
            RelaxedTimeRev_GroupConv2d(hidden_dim, out_channels, kernel_size, num_filter_basis, activation = False)
        )
        
    def forward(self, x):
        out = self.model(x).mean(2)
        return out