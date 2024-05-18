import torch
import numpy as np
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
##### 2D Relaxed Rotation Lifting Convolution Layer #####
class RelaxedRotLiftConv2d(torch.nn.Module):
    """Relaxed lifting convolution Layer for 2D finite rotation group"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order, # the order of 2d finite rotation group
                 num_filter_banks,
                 activation = True # whether to apply relu in the end
                 ):
        super(RelaxedRotLiftConv2d, self).__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation
  
        # The relaxed weights are initialized as equal
        # they do not need to be equal across different filter bank
        self.relaxed_weights = torch.nn.Parameter(torch.ones(num_filter_banks, group_order).float())

        # Initialize an unconstrained kernel.
        self.kernel = torch.nn.Parameter(torch.zeros(self.num_filter_banks, # Additional dimension
                                                     self.out_channels,
                                                     self.in_channels,
                                                     self.kernel_size,
                                                     self.kernel_size))
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))
        
    def generate_filter_bank(self):
        """ Obtain a stack of rotated filters"""
        weights = self.kernel.reshape(self.num_filter_banks*self.out_channels,
                                      self.in_channels,
                                      self.kernel_size,
                                      self.kernel_size)
        filter_bank = torch.stack([rot_img(weights, -np.pi*2/self.group_order*i)
                                   for i in range(self.group_order)])
        filter_bank = filter_bank.transpose(0,1).reshape(self.num_filter_banks, # Additional dimension
                                                         self.out_channels,
                                                         self.group_order,
                                                         self.in_channels,
                                                         self.kernel_size,
                                                         self.kernel_size)
        return filter_bank


    def forward(self, x):
        # input shape: [bz, #in, h, w]
        # output shape: [bz, #out, group order, h, w]

        # generate filter bank given input group order
        filter_bank = self.generate_filter_bank()

        # for each rotation, we have a linear combination of multiple filters with different coefficients.
        relaxed_conv_weights = torch.einsum("na, noa... -> oa...", self.relaxed_weights, filter_bank)

        # concatenate the first two dims before convolution.
        # ==============================
        x = F.conv2d(
            input=x,
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )
        # ==============================

        # reshape output signal to shape [bz, #out, group order, h, w].
        # ==============================
        x = x.view(
            x.shape[0],
            self.out_channels,
            self.group_order,
            x.shape[-1],
            x.shape[-2]
        )
        # ==============================

        if self.activation:
            return F.relu(x)
        return x

##### 2D Relaxed Rotation Group Convolution Layer #####
class RelaxedRotGroupConv2d(torch.nn.Module):
    """Relaxed group convolution Layer for 2D finite rotation group"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order, # the order of 2d finite rotation group
                 num_filter_banks,
                 activation = True # whether to apply relu in the end
                ):

        super(RelaxedRotGroupConv2d, self).__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation


        # Initialize weights
        # If relaxed_weights are equal values, then the model is still equivariant
        # Relaxed weights do not need to be equal across different filter bank
        self.relaxed_weights = torch.nn.Parameter(torch.ones(group_order, num_filter_banks).float())
        self.kernel = torch.nn.Parameter(torch.randn(self.num_filter_banks, # additional dimension
                                                     self.out_channels,
                                                     self.in_channels,
                                                     self.group_order,
                                                     self.kernel_size,
                                                     self.kernel_size))

        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))

        
    def generate_filter_bank(self):
        """ Obtain a stack of rotated and cyclic shifted filters"""
        filter_bank = []
        weights = self.kernel.reshape(self.num_filter_banks*self.out_channels*self.in_channels,
                                      self.group_order,
                                      self.kernel_size,
                                      self.kernel_size)

        for i in range(self.group_order):
            # planar rotation
            rotated_filter = rot_img(weights, -np.pi*2/self.group_order*i)

            # cyclic shift
            shifted_indices = torch.roll(torch.arange(0, self.group_order, 1), shifts = i)
            shifted_rotated_filter = rotated_filter[:,shifted_indices]


            filter_bank.append(shifted_rotated_filter.reshape(self.num_filter_banks,
                                                              self.out_channels,
                                                              self.in_channels,
                                                              self.group_order,
                                                              self.kernel_size,
                                                              self.kernel_size))
        # stack
        filter_bank = torch.stack(filter_bank).permute(1,2,0,3,4,5,6)
        return filter_bank
    
    def forward(self, x):

        filter_bank = self.generate_filter_bank()

        relaxed_conv_weights = torch.einsum("na, aon... -> on...", self.relaxed_weights, filter_bank)

        x = torch.nn.functional.conv2d(
            input=x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                x.shape[4]
                ),
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels * self.group_order,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )

        # Reshape signal back [bz, #out * g_order, h, w] -> [bz, out, g_order, h, w]
        x = x.view(x.shape[0], self.out_channels, self.group_order, x.shape[-2], x.shape[-1])
        # ========================
        if self.activation:
            return F.relu(x)
        return x

    
class RelaxedRotCNN2d(torch.nn.Module):
    """A small relaxed rotation 2d CNN model"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 hidden_dim, 
                 group_order, # the order of 2d finite rotation group
                 num_gconvs, # number of group conv layers
                 num_filter_banks
                ):
        super(RelaxedRotCNN2d, self).__init__()
        
        self.group_order = group_order
        
        self.gconvs = [RelaxedRotLiftConv2d(in_channels = in_channels,
                                            out_channels = hidden_dim,
                                            kernel_size = kernel_size,
                                            group_order = group_order,
                                            num_filter_banks = num_filter_banks,
                                            activation = True)]

        for i in range(num_gconvs-2):
            self.gconvs.append(RelaxedRotGroupConv2d(in_channels = hidden_dim,
                                                     out_channels = hidden_dim,
                                                     kernel_size = kernel_size,
                                                     group_order = group_order,
                                                     num_filter_banks = num_filter_banks,
                                                     activation = True))
            
        self.gconvs.append(RelaxedRotGroupConv2d(in_channels = hidden_dim,
                                                 out_channels = out_channels,
                                                 kernel_size = kernel_size,
                                                 group_order = group_order,
                                                 num_filter_banks = num_filter_banks,
                                                 activation = False))

        self.gconvs = torch.nn.Sequential(*self.gconvs)

    def forward(self, x, mean = True): 
        # average over h axis or not         
        out = self.gconvs(x)
        if mean:
            out = out.mean(2)
        return out

            


class IsoSymDisNet2D(torch.nn.Module): 
    """2D relaxed rotation  group ConvNet for discovering isotropy breaking"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_dim, 
                 num_filter_basis,
                 group_order,
                 kernel_size,
                 freq_list = None,
                 ):
        super(IsoSymDisNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        
        # The inputs are velocity fields, we need to first lift rho_1 rep to regular rep
        theta = torch.tensor(2*np.pi/self.group_order).float()
        self.lift_weights = torch.from_numpy(np.array([[np.cos(theta*i), np.sin(theta*i)] 
                                                       for i in range(self.group_order)])).float().to(device)
      
        # perform the scale separation given the freq_list
        if freq_list is None:
            self.freq_list = torch.fft.rfftfreq(128)
        else:
            self.freq_list = freq_list
            
        # define a distinct layer for each scale
        self.model = torch.nn.ModuleList([
            torch.nn.Sequential(
                RelaxedRotGroupConv2d(in_channels = in_channels,
                                      out_channels = out_channels,
                                      kernel_size = kernel_size,
                                      group_order = group_order,
                                      num_filter_banks = num_filter_basis,
                                      activation = False)
            ).to(device)
            for _ in range(len(self.freq_list)-1)])
        
    def lowpass_torch(self, inp, lower, upper):
        pass1 = (torch.abs(torch.fft.fftfreq(inp.shape[-1])) >= lower) & (torch.abs(torch.fft.fftfreq(inp.shape[-1])) <= upper)
        pass2 = (torch.abs(torch.fft.fftfreq(inp.shape[-2])) >= lower) & (torch.abs(torch.fft.fftfreq(inp.shape[-2])) <= upper)
        kernel = torch.outer(pass2, pass1).to(inp.device)
        fft_input = torch.fft.fft2(inp)
        return torch.fft.ifft2(fft_input * kernel, s=inp.shape[-2:]).real
    
    def forward(self, inp):
        # inp shape: [bz, 2, h, w]
        bz = inp.shape[0]
        separate_scales = []
        for i in range(len(self.freq_list)-1):  
            separate_scales.append(self.lowpass_torch(inp, self.freq_list[i], self.freq_list[i+1]))
        smoothed_inps = torch.stack([separate_scales[i] for i in range(len(separate_scales))], dim = 0)
        lift_smoothed_inps = torch.einsum("cbvhw, gv -> cbghw", smoothed_inps, self.lift_weights) 

        group_out = torch.stack([self.model[i](lift_smoothed_inps[i].unsqueeze(1)) for i in range(lift_smoothed_inps.shape[0])])
        
        # [bz, #out, 48or24, h, w, d] x [48or24, 3] -> [bz, #out, 3, h, w, d]
        out = torch.einsum("cbighw, gv -> bivhw", group_out, self.lift_weights)        
        # output shape: [bz, 2, h, w]
        return out[:,0]
        
    
def rot_img(x, theta):
    """ Rotate 2D images
    Args:
        x : input images with shape [N, C, H, W]
        theta: angle
    Returns:
        rotated images
    """
    # Rotation Matrix (2 x 3)
    rot_mat = torch.FloatTensor([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0]]).to(x.device)

    # The affine transformation matrices should have the shape of N x 2 x 3
    rot_mat = rot_mat.repeat(x.shape[0],1,1)

    # Obtain transformed grid
    # grid is the coordinates of pixels for rotated image
    # F.affine_grid assumes the origin is in the middle
    # and it rotates the positions of the coordinates
    # r(f(x)) = f(r^-1 x)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False).float().to(x.device)
    x = F.grid_sample(x, grid)
    return x

def rot_vector(x, theta):
    #x has the shape [c x 2 x h x w]
    rho = torch.FloatTensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]]).to(x.device)
    out = torch.einsum("ab, bc... -> ac...",(rho, x.transpose(0,1))).transpose(0,1)
    return out
