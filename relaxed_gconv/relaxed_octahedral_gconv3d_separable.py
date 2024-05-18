import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import octahedral_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Relaxed_Octahedral_LiftConv3d(nn.Module):
    """Relaxed Lifting Convolution Layer for Octahedral Group"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 num_filter_basis, 
                 reflection,
                 activation = True
                 ):
        super(Relaxed_Octahedral_LiftConv3d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_filter_basis = num_filter_basis
        self.reflection = reflection
        
        # the Octahedral group contain 24/48 elements
        if self.reflection:
            self.group_order = 48
        else:
            self.group_order = 24
        
        self.octahedral_transform = octahedral_transform(self.reflection)
        self.activation = activation
        
        
        self.relaxed_weights = nn.Parameter(torch.ones(self.num_filter_basis, self.group_order).float())
        # Initialize an unconstrained kernel. 
        self.kernel = torch.nn.Parameter(torch.zeros(self.num_filter_basis,
                                                     self.out_channels,
                                                     self.in_channels,
                                                     self.kernel_size,
                                                     self.kernel_size,
                                                     self.kernel_size))
        
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))
        
    def generate_filter_bank(self):
        """ Obtain a stack of rotated filters"""
        filter_bank = []
        weights = self.kernel.reshape(self.num_filter_basis*self.out_channels,
                                      self.in_channels,
                                      self.kernel_size,
                                      self.kernel_size,
                                      self.kernel_size)
        
        for i in range(self.group_order):
            filter_bank.append(self.octahedral_transform(weights, i))
            
        filter_bank = torch.stack(filter_bank).transpose(0,1)
        filter_bank = filter_bank.reshape(self.num_filter_basis,
                                          self.out_channels,
                                          self.group_order,
                                          self.in_channels,
                                          self.kernel_size,
                                          self.kernel_size,
                                          self.kernel_size)
        
        return filter_bank
    
    
    def forward(self, x):
        # input shape: [bz, #in, h, w, d]
        # output shape: [bz, #out, group_order, 3, h, w, d]
        
        # generate filter bank given input group order
        filter_bank = self.generate_filter_bank()
        
        relaxed_conv_weights = torch.einsum("na, noa... -> oa...", self.relaxed_weights, filter_bank)


        # concatenate the first two dims before convolution. 
        # ==============================
        x = F.conv3d(
            input=x,
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )
        
        # ==============================

        # reshape output signal to shape [bz, #out, group order, 3, h, w, d]
        # ==============================
        x = x.view(
            x.shape[0],
            self.out_channels,
            self.group_order,
            x.shape[-3],
            x.shape[-2],
            x.shape[-1]
        )
        # ==============================
        
        if self.activation:
                return F.relu(x)
        return x  
    
    
    
    
    
class Relaxed_Octahedral_GroupConv3d(nn.Module):
    """Relaxed Group Convolution Layer for finite rotation group"""
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 num_filter_basis,
                 reflection,
                 stride = 1,
                 padding = 1,
                 upscale = False, 
                 activation = False
                 ):
        super(Relaxed_Octahedral_GroupConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_filter_basis = num_filter_basis
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.upscale = upscale
        self.reflection = reflection
        
        # A regular octahedron has 24 rotational symmetries 
        # or 48 symmetries if include reflections. 
        if self.reflection:
            self.group_order = 48
        else:
            self.group_order = 24
        
        # initialize a octahedral_transform class 
        self.octahedral_transform = octahedral_transform(self.reflection)

        # relaxed weights
        self.relaxed_weights = nn.Parameter(torch.ones(self.num_filter_basis, self.group_order).float())
                                                
        # Since we use separable convolutions
        # we use two separate kernels for O and translation group.
        self.g_weight = torch.nn.Parameter(torch.zeros(self.num_filter_basis, self.group_order, self.out_channels, self.in_channels))
        
        self.t_weight = torch.nn.Parameter(torch.zeros(self.out_channels,
                                                       1,
                                                       self.kernel_size,
                                                       self.kernel_size,
                                                       self.kernel_size))
        
        self.g_bias = torch.nn.Parameter(torch.zeros(1, self.out_channels, 1, 1, 1, 1))
        self.t_bias = torch.nn.Parameter(torch.zeros(1))
        
        # stdv = np.sqrt(1/(self.in_channels*self.kernel_size))
        # self.t_weight.data.uniform_(-stdv, stdv)
        # self.g_weight.data.uniform_(-stdv, stdv)
        g_stdv = np.sqrt(2/self.in_channels/self.group_order)
        t_stdv = np.sqrt(2/self.out_channels/self.group_order)
        self.g_weight.data.normal_(0, g_stdv)
        self.t_weight.data.normal_(0, t_stdv)
        
        
    def generate_filter_bank_G(self):
        """ Permute the indices of group elements """
        filter_bank_g = []
        for g in range(self.group_order):
            shifted_indices = self.octahedral_transform.shifted_indices[g]
            filter_bank_g.append(self.g_weight[:, shifted_indices])
        filter_bank_g = torch.stack(filter_bank_g, dim = 1)
        return filter_bank_g
    
    def generate_filter_bank_T(self):
        """ Obtain a stack of rotated filters """
        filter_bank_t = []
        for i in range(self.group_order):
            filter_bank_t.append(self.octahedral_transform(self.t_weight, i))
        
        return torch.stack(filter_bank_t, dim = 1)

    def forward(self, x):
        # input shape: [bz, in, group order, 3, h, w, d]
        # output shape: [bz, out, group order, 3, h, w, d]
        
        g_filter_bank = self.generate_filter_bank_G()
        t_filter_bank = self.generate_filter_bank_T()
        
        # [#basis, #g] x [#basis, #g, #g, #out, #in]
        # -> [#g, #g, #out, #in]
        relaxed_conv_weights = torch.einsum("ng, nhgoi -> ghoi", self.relaxed_weights, g_filter_bank)

        
        # ==============================
        # [bz, #in, #g_in, h, w, d] x [#g, #g, #out, #in]
        # -> [bz, #out, #g, h, w, d] 
        x = torch.einsum("bighwd,zgoi->bozhwd", x, relaxed_conv_weights) + self.g_bias
        if self.upscale:
            x = x.reshape(x.shape[0], self.out_channels*self.group_order, x.shape[-3], x.shape[-2], x.shape[-1])
            out = F.conv_transpose3d(
                input=x,
                weight=t_filter_bank.reshape(
                    self.out_channels*self.group_order,
                    1,
                    self.kernel_size,
                    self.kernel_size,
                    self.kernel_size
                ),
                padding = self.padding,
                stride = self.stride,
                groups=(self.out_channels * self.group_order),
                bias = self.t_bias.repeat(self.out_channels * self.group_order)
            )
            
        else:
            x = x.reshape(x.shape[0], self.out_channels*self.group_order, x.shape[-3], x.shape[-2], x.shape[-1])
            out = F.conv3d(
                input=x,
                weight=t_filter_bank.reshape(
                    self.out_channels*self.group_order,
                    1,
                    self.kernel_size,
                    self.kernel_size,
                    self.kernel_size
                ),
                padding = self.padding,
                stride = self.stride,
                groups=(self.out_channels * self.group_order),
                bias = self.t_bias.repeat(self.out_channels * self.group_order)
            )
        
        out = out.reshape(x.shape[0], 
                          self.out_channels, 
                          self.group_order,
                          out.shape[2], 
                          out.shape[3],
                          out.shape[4])
    
        if self.activation:
            return F.relu(out)
        return out
    
class Relaxed_EquivResBlock(torch.nn.Module):
    """Relaxed Equiv Upscaling ResNet Block"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_filter_basis,
                 reflection, 
                 activation = True,
                 upscale = False
                 ):
        super(Relaxed_EquivResBlock, self).__init__()

        self.upscale = upscale
        if self.upscale:
            self.layer1 = Relaxed_Octahedral_GroupConv3d(in_channels = in_channels, 
                                                         out_channels = out_channels, 
                                                         num_filter_basis = num_filter_basis,
                                                         reflection = reflection,
                                                         kernel_size = 4, 
                                                         upscale=True, 
                                                         stride = 2, 
                                                         padding = 1, 
                                                         activation = True)
        else:
            self.layer1 = Relaxed_Octahedral_GroupConv3d(in_channels = in_channels, 
                                                         out_channels = out_channels, 
                                                         num_filter_basis = num_filter_basis,
                                                         reflection = reflection,
                                                         kernel_size = 3, 
                                                         upscale=False, 
                                                         stride = 1, 
                                                         padding = 1, 
                                                         activation = True)
        
        self.layer2 = Relaxed_Octahedral_GroupConv3d(in_channels = out_channels, 
                                                     out_channels = out_channels,
                                                     num_filter_basis = num_filter_basis,
                                                     reflection = reflection,
                                                     kernel_size = 3,
                                                     upscale=False, 
                                                     stride = 1,
                                                     padding = 1, 
                                                     activation = False)
        
        if self.upscale:
            # residual connection with simple trilinear interpolation
            self.residual = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        
        self.activation = activation


    def forward(self, x):
        # input shape: [bz, #in, d, h, w]
        # output shape: [bz, #out, d*2, h*2, w*2]
        if self.upscale:
            residual = self.residual(x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4], x.shape[5]))
            residual = residual.reshape(x.shape[0], x.shape[1], x.shape[2], residual.shape[-3], residual.shape[-2], residual.shape[-1])
        
            out = self.layer2(self.layer1(x)) + residual
        else:
            out = self.layer2(self.layer1(x)) + x
        
        if self.activation:
            return F.relu(out)
        return out
    

class RelaxedOctahedralConvNet(torch.nn.Module):
    """A small relaxed Oh group convolution model"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 hidden_dim,
                 num_filter_basis,
                 reflection
                 ):
        super(RelaxedOctahedralConvNet, self).__init__()

        self.model = torch.nn.Sequential(*[Relaxed_Octahedral_LiftConv3d(in_channels, hidden_dim, kernel_size, num_filter_basis, reflection, 
                                                                         activation = True),
                                           Relaxed_Octahedral_GroupConv3d(hidden_dim, hidden_dim, kernel_size, num_filter_basis, reflection,
                                                                          padding = (kernel_size-1)//2, activation = True),
                                           Relaxed_Octahedral_GroupConv3d(hidden_dim, out_channels, kernel_size, num_filter_basis, reflection,
                                                                          padding = (kernel_size-1)//2, activation = False)]).to(device)


    def forward(self, x):
        return self.model(x)
    
# class Relaxed_Octahedral_EquivResNet(torch.nn.Module): 
#     """Relaxed Equiv SuperResolution ResNet"""
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  hidden_dim, 
#                  num_filter_basis,
#                  reflection,
#                  kernel_size,
#                  vec_inp = False
#                  ):
#         super(Relaxed_Octahedral_EquivResNet, self).__init__()
#         self.in_channels = in_channels
#         self.vec_inp = vec_inp
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
        
#         if self.vec_inp:
#             self.lift_weights = torch.nn.Parameter(torch.zeros(1,self.in_channels,1,1,1,1))
#             self.back_weights = torch.nn.Parameter(torch.zeros(1,self.out_channels,1,1,1,1))
#             stdv = np.sqrt(1/(self.in_channels))
#             self.lift_weights.data.uniform_(-stdv, stdv)
#             self.back_weights.data.uniform_(-stdv, stdv)
#             self.embedding = Relaxed_Octahedral_GroupConv3d(in_channels = in_channels, 
#                                                   out_channels = hidden_dim,
#                                                   num_filter_basis = num_filter_basis, 
#                                                   kernel_size = kernel_size,
#                                                   reflection = reflection,
#                                                   activation = True)
#         else:
#             self.embedding = Relaxed_Octahedral_LiftConv3d(in_channels = in_channels, 
#                                                            out_channels = hidden_dim, 
#                                                            num_filter_basis = num_filter_basis,
#                                                            kernel_size = kernel_size,
#                                                            reflection = reflection,
#                                                            activation = True)
                          
#         self.model = [Relaxed_EquivResBlock(hidden_dim, hidden_dim, kernel_size, num_filter_basis, reflection, activation = True, upscale = True), 
#                       Relaxed_EquivResBlock(hidden_dim, hidden_dim, kernel_size, num_filter_basis, reflection, activation = True, upscale = False), 
#                       Relaxed_EquivResBlock(hidden_dim, hidden_dim, kernel_size, num_filter_basis, reflection, activation = True, upscale = True), 
#                       Relaxed_EquivResBlock(hidden_dim, hidden_dim, kernel_size, num_filter_basis, reflection, activation = True, upscale = False)]
#         self.model = torch.nn.Sequential(*self.model)
        
#         # self.final_layer = Relaxed_Octahedral_GroupConv3d(in_channels = hidden_dim, 
#         #                                                   out_channels = out_channels,
#         #                                                   num_filter_basis = num_filter_basis, 
#         #                                                   kernel_size = kernel_size,
#         #                                                   reflection = reflection,
#         #                                                   activation = False).to(device)
#         self.final_layer = torch.nn.Conv3d(hidden_dim, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)
        
#         self.o_transform = octahedral_transform(reflection)
#         self.SHs = self.o_transform.l1_spherical_harmonics_real()
        
#     def forward(self, x):
#         if self.vec_inp:
#             lift_inp = torch.einsum("bishwd, gs->bighwd", x*self.lift_weights, self.SHs.to(x.device))
#             out = self.model(self.embedding(lift_inp))#)self.final_layer(
#             #[bz, #out, 48or24, h, w, d] x [48or24, 3] -> [bz, #out, 3, h, w, d]
#             back_out = torch.einsum("boghwd, gs->boshwd", out*self.back_weights, self.SHs.to(out.device))
#             # return back_out
#             back_out = back_out.transpose(1,2)
#             out = back_out.reshape(back_out.shape[0]*back_out.shape[1],
#                                   back_out.shape[2], 
#                                   back_out.shape[3],
#                                   back_out.shape[4],
#                                   back_out.shape[5])
#             out = self.final_layer(out).reshape(back_out.shape[0],
#                                                 back_out.shape[1], 
#                                                 self.out_channels,
#                                                 back_out.shape[3],
#                                                 back_out.shape[4],
#                                                 back_out.shape[5]).transpose(1,2)
#             return out
#         else:
#             out = self.embedding(x)
#             out = self.model(out)
#             return self.final_layer(out)[:,:,0]#.mean(2)
      