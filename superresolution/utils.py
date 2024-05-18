import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def Rodrigues(vec, theta):
    vec = vec / np.linalg.norm(vec)

    K = np.array([[0, -vec[2], vec[1]],
                 [vec[2], 0, -vec[0]],
                 [-vec[1], vec[0], 0]])

    R = np.identity(3) + np.sin(theta)*K + (1-np.cos(theta))*np.matmul(K, K)
    return np.round(R, 8)

def rot_vector(inp,index):
    out = torch.einsum("bkvhwd, uv->bkuhwd", inp, torch.from_numpy(refl_octahedron_rots[index]).float().to(inp.device))
    return out

def compute_l1_spherical_harmonics_real(R):
    # Define original direction vector (pointing along z-axis)
    d = np.array([1, 0, 0])
    
    # Rotate the direction vector using the rotation matrix
    d_rotated = np.dot(R, d)
    
    # Compute the angles theta and phi from the rotated direction vector
    theta = np.arccos(d_rotated[1])
    phi = np.arctan2(d_rotated[0], d_rotated[2])
    
    # Compute the l=1 real spherical harmonics
    Y_1_minus1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    Y_1_0 = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
    Y_1_1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)
    
    return Y_1_minus1, Y_1_0, Y_1_1

def compute_all_l1_spherical_harmonics_real(Rs):
    SHs = []
    for i in range(len(Rs)):
        Y_10, Y_11, Y_1m1 = compute_l1_spherical_harmonics_real(Rs[i])
        SHs.append([Y_10, Y_11, Y_1m1])
    return torch.from_numpy(np.array(SHs)).float()


class octahedral_transform(nn.Module):
    def __init__(self, reflection = False):
        super(octahedral_transform, self).__init__()
        if reflection:
            self.octahedron_rots = refl_octahedron_rots
        else:
            self.octahedron_rots = octahedron_rots
        shifted_indices = []
        
        for g in range(len(self.octahedron_rots)):
            out = np.einsum("xy, nyz-> nxz", np.linalg.inv(self.octahedron_rots[g]), self.octahedron_rots)
            indices = []
            for i in range(len(self.octahedron_rots)):
                indices.append(np.where(np.all(out[i] == self.octahedron_rots, axis = (1,2)) == True)[0][0])
            shifted_indices.append(indices)
        shifted_indices = torch.from_numpy(np.array(shifted_indices))#.to(device)

        self.shifted_indices = shifted_indices
        
    def l1_spherical_harmonics_real(self):
        return compute_all_l1_spherical_harmonics_real(self.octahedron_rots)#.to(device)

        
    def forward(self, input_tensor, element):
    
        rotation_matrix = torch.FloatTensor(np.linalg.inv(self.octahedron_rots[element]))
        padding = torch.zeros(3, 1)

        rotation_matrix = torch.cat([rotation_matrix, padding], dim = 1).unsqueeze(0).repeat(input_tensor.shape[0], 1, 1).to(input_tensor.device)
        # print(rotation_matrix)


        # Get the size of the input tensor
        size = input_tensor.size()

        # Generate the grid
        grid = F.affine_grid(rotation_matrix, size, align_corners=False).to(input_tensor.dtype)
    

        # Resample the tensor
        rotated_tensor = F.grid_sample(input_tensor, grid)
        return rotated_tensor
    
def rotate_3d(input_tensor, theta, axis):
    """
    Rotates a 3D tensor along the specified axis.
    
    Parameters:
    - input_tensor (Tensor): the tensor to rotate
    - angle (float): angle to rotate in degrees
    - axis (str): 'x', 'y' or 'z'
    
    Returns:
    - Tensor: the rotated tensor
    """
    
    # Define the rotation matrix for each axis
    if axis == 'x':
        rotation_matrix = [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0]
        ]
    elif axis == 'y':
        rotation_matrix = [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0]
        ]
    elif axis == 'z':
        rotation_matrix = [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0]
        ]
    else:
        raise ValueError('Axis should be x, y, or z.')

    # Convert rotation matrix to tensor
    rotation_matrix = torch.FloatTensor(rotation_matrix).repeat(input_tensor.shape[0],1,1).to(input_tensor.device)

    # Get the size of the input tensor
    size = input_tensor.size()
    
    # Generate the grid
    grid = F.affine_grid(rotation_matrix, size, align_corners=False).to(input_tensor.dtype)
    
    # Resample the tensor
    rotated_tensor = F.grid_sample(input_tensor, grid)
    
    return rotated_tensor

### The octahedron are the coordinate axes
### The vertices be at the points (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
# identity

octahedron_rots = np.array([[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]],
# 3 rotations along x axis
[[1, 0, 0],
 [0, 0, -1],
 [0, 1, 0]],
[[1, 0, 0],
 [0, -1, 0], 
 [0, 0, -1]],
[[1, 0, 0],
 [0, 0, 1],
 [0, -1, 0]],
# 3 rotations along y axis
[[0, 0, 1],
 [0, 1, 0],
 [-1, 0, 0]],
[[-1, 0, 0],
 [0, 1, 0], 
 [0, 0, -1]],
[[0, 0, -1],
 [0, 1, 0],
 [1, 0, 0]],
# 3 rotations along z axis
[[0, -1, 0],
 [1, 0, 0],
 [0, 0, 1]],
[[-1, 0, 0],
 [0, -1, 0], 
 [0, 0, 1]],
[[0, 1, 0],
 [-1, 0, 0],
 [0, 0, 1]],
# 120 and 240 Rotations about an Axis through the Midpoints of Opposite Edges
# For the axis through (1,1,0) and (-1,-1,0)
[[0, 1, 0],
 [1, 0, 0],
 [0, 0, -1]],
# For the axis through (1,-1,0) and (-1,1,0)
[[0, -1, 0], 
 [-1, 0, 0],
 [0, 0, -1]], 
# For the axis through (1,0,1) and (-1,0,1)
[[0, 0, 1],
 [0, -1, 0],
 [1, 0, 0]],
# For the axis through (1,0,-1) and (-1,0,1)
[[0, 0, -1],
 [0, -1, 0], 
 [-1, 0, 0]], 
# For the axis through (0,1,1) and (0,-1,-1)
[[-1, 0, 0], 
 [0, 0, 1],
 [0, 1, 0]], 
# For the axis through (0,1,-1) and (0,-1,1)
[[-1, 0, 0], 
 [0, 0, -1],
 [0, -1, 0]], 
# Rotations about an Axis through the Midpoints of Opposite Faces
# For the axis through (1,1,1) and (-1,-1,-1)
# 120 degrees
[[0., 0., 1.],
[ 1., 0., 0.],
[ 0., 1., 0.]],
# 240 degrees
[[0., 1., 0.],
[ 0., 0., 1.],
[ 1., 0., 0.]],
# For the axis through (1,-1,1) and (-1,1,-1)
# 120 degrees
[[0., -1.,  0.],
[ 0., -0., -1.],
[ 1.,  0., 0.]],
# 240 degrees
[[0., 0.,  1.],
[-1., 0., 0.],
[ 0., -1., 0.]],
# For the axis through (-1,-1,1) and (1,1,-1)
[[0.,  0., -1.],
[ 1., 0.,  0.],
[ 0., -1., 0.]],
[[0.,  1., 0.],
[ 0., 0., -1.],
[-1., 0., 0.]],
# For the axis through (-1,1,1) and (1,-1,-1)
[[0., -1.,  0.],
[ 0., 0.,  1.],
[-1.,  0., 0.]],
[[0., 0., -1.],
[-1., 0.,  0.],
[0.,  1., 0.]]])



### The octahedron are the coordinate axes
### The vertices be at the points (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
# identity
refl_octahedron_rots = np.array([[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]],
# 3 rotations along x axis
[[1, 0, 0],
 [0, 0, -1],
 [0, 1, 0]],
[[1, 0, 0],
 [0, -1, 0], 
 [0, 0, -1]],
[[1, 0, 0],
 [0, 0, 1],
 [0, -1, 0]],
# 3 rotations along y axis
[[0, 0, 1],
 [0, 1, 0],
 [-1, 0, 0]],
[[-1, 0, 0],
 [0, 1, 0], 
 [0, 0, -1]],
[[0, 0, -1],
 [0, 1, 0],
 [1, 0, 0]],
# 3 rotations along z axis
[[0, -1, 0],
 [1, 0, 0],
 [0, 0, 1]],
[[-1, 0, 0],
 [0, -1, 0], 
 [0, 0, 1]],
[[0, 1, 0],
 [-1, 0, 0],
 [0, 0, 1]],
# 120 and 240 Rotations about an Axis through the Midpoints of Opposite Edges
# For the axis through (1,1,0) and (-1,-1,0)
[[0, 1, 0],
 [1, 0, 0],
 [0, 0, -1]],
# For the axis through (1,-1,0) and (-1,1,0)
[[0, -1, 0], 
 [-1, 0, 0],
 [0, 0, -1]], 
# For the axis through (1,0,1) and (-1,0,1)
[[0, 0, 1],
 [0, -1, 0],
 [1, 0, 0]],
# For the axis through (1,0,-1) and (-1,0,1)
[[0, 0, -1],
 [0, -1, 0], 
 [-1, 0, 0]], 
# For the axis through (0,1,1) and (0,-1,-1)
[[-1, 0, 0], 
 [0, 0, 1],
 [0, 1, 0]], 
# For the axis through (0,1,-1) and (0,-1,1)
[[-1, 0, 0], 
 [0, 0, -1],
 [0, -1, 0]], 
# Rotations about an Axis through the Midpoints of Opposite Faces
# For the axis through (1,1,1) and (-1,-1,-1)
# 120 degrees
[[0., 0., 1.],
[ 1., 0., 0.],
[ 0., 1., 0.]],
# 240 degrees
[[0., 1., 0.],
[ 0., 0., 1.],
[ 1., 0., 0.]],
# For the axis through (1,-1,1) and (-1,1,-1)
# 120 degrees
[[0., -1.,  0.],
[ 0., -0., -1.],
[ 1.,  0., 0.]],
# 240 degrees
[[0., 0.,  1.],
[-1., 0., 0.],
[ 0., -1., 0.]],
# For the axis through (-1,-1,1) and (1,1,-1)
[[0.,  0., -1.],
[ 1., 0.,  0.],
[ 0., -1., 0.]],
[[0.,  1., 0.],
[ 0., 0., -1.],
[-1., 0., 0.]],
# For the axis through (-1,1,1) and (1,-1,-1)
[[0., -1.,  0.],
[ 0., 0.,  1.],
[-1.,  0., 0.]],
[[0., 0., -1.],
[-1., 0.,  0.],
[0.,  1., 0.]],
# Reflection through the XY plane
[[1., 0., 0.],
[0., 1.,  0.],
[0.,  0., -1.]],
# Reflection through the XZ plane    
[[1., 0., 0.],
[0., -1.,  0.],
[0.,  0., 1.]],
# Reflection through the YZ plane
[[-1., 0., 0.],
[0., 1.,  0.],
[0.,  0., 1.]],    
# Other Reflections
[[1.0, 0.0, 0.0], 
[0.0, 0.0, -1.0], 
[0.0, -1.0, 0.0]],
[[1.0, 0.0, 0.0], 
[0.0, 0.0, 1.0], 
[0.0, 1.0, 0.0]],
[[-1.0, 0.0, 0.0],
[0.0, 0.0, 1.0], 
[0.0, -1.0, 0.0]],
[[-1.0, 0.0, 0.0],
 [0.0, -1.0, 0.0], 
 [0.0, 0.0, -1.0]],
[[-1.0, 0.0, 0.0],
 [0.0, 0.0, -1.0], 
 [0.0, 1.0, 0.0]],
[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
[[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
[[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
[[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
[[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
[[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
 [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
[[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
[[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])



def spectrum(result):
    import time as tt
    skip=1
    dim=int(round(result.shape[-1]/skip))
    # print(dim)
    L=2*np.pi

    start = tt.time()
    uu_fft=np.fft.fftn(result[0,::skip,::skip,::skip])
    vv_fft=np.fft.fftn(result[1,::skip,::skip,::skip])
    ww_fft=np.fft.fftn(result[2,::skip,::skip,::skip])
    # print(tt.time() - start)

    start = tt.time()
    uu_fft=(np.abs(uu_fft)/dim**3)**2
    vv_fft=(np.abs(vv_fft)/dim**3)**2
    ww_fft=(np.abs(ww_fft)/dim**3)**2
    # print(tt.time() - start)

    k_end=int(dim/2)
    rx=np.array(range(dim))-dim/2+1
    rx=np.roll(rx,int(dim/2)+1)

    #start = tt.time()
    #r,Y,Z = np.meshgrid(rx, rx, rx)
    #r=np.sqrt(r**2+Y**2+Z**2)
    #print(tt.time() - start)
    start = tt.time()
    r=np.zeros((rx.shape[0],rx.shape[0],rx.shape[0]))
    for i in range(rx.shape[0]):
        for j in range(rx.shape[0]):
                r[i,j,:]=rx[i]**2+rx[j]**2+rx[:]**2
    r=np.sqrt(r)
    # print(tt.time() - start)

    dx=2*np.pi/L
    k=(np.array(range(k_end))+1)*dx

    start = tt.time()
    bins=np.zeros((k.shape[0]+1))
    for N in range(k_end):
        if N==0:
            bins[N]=0
        else:
            bins[N]=(k[N]+k[N-1])/2    
    bins[-1]=k[-1]

    inds = np.digitize(r*dx, bins, right=True)
    spectrum=np.zeros((k.shape[0]))
    bin_counter=np.zeros((k.shape[0]))

    for N in range(k_end):
        spectrum[N]=np.sum(uu_fft[inds==N+1])+np.sum(vv_fft[inds==N+1])+np.sum(ww_fft[inds==N+1])
        bin_counter[N]=np.count_nonzero(inds==N+1)

    spectrum=spectrum*2*np.pi*(k**2)/(bin_counter*dx**3)
    # print(tt.time() - start)
    return spectrum

# import numpy as np
# from numpy.fft import fftn
# from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

# #------------------------------------------------------------------------------

# def movingaverage(interval, window_size):
#     window= ones(int(window_size))/float(window_size)
#     return convolve(interval, window, 'same')

# #------------------------------------------------------------------------------

# def compute_tke_spectrum_1d(u,lx,ly,lz,smooth):
#   """
#   Given a velocity field u this function computes the kinetic energy
#   spectrum of that velocity field in spectral space. This procedure consists of the 
#   following steps:
#   1. Compute the spectral representation of u using a fast Fourier transform.
#   This returns uf (the f stands for Fourier)
#   2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf)* conjugate(uf)
#   3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
#   Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
#   the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
#   E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

#   Parameters:
#   -----------  
#   u: 3D array
#     The x-velocity component.
#   v: 3D array
#     The y-velocity component.
#   w: 3D array
#     The z-velocity component.    
#   lx: float
#     The domain size in the x-direction.
#   ly: float
#     The domain size in the y-direction.
#   lz: float
#     The domain size in the z-direction.
#   smooth: boolean
#     A boolean to smooth the computed spectrum for nice visualization.
#   """
#   nx = len(u[:,0,0])
#   ny = len(u[0,:,0])
#   nz = len(u[0,0,:])
  
#   nt= nx*ny*nz
#   n = max(nx,ny,nz) #int(np.round(np.power(nt,1.0/3.0)))
  
#   uh = fftn(u)/nt
  
#   tkeh = zeros((nx,ny,nz))
#   tkeh = 0.5*(uh*conj(uh)).real
  

#   l = max(lx,ly,lz)  
  
#   knorm = 2.0*pi/l
  
#   kxmax = nx/2
#   kymax = ny/2
#   kzmax = nz/2
  
#   wave_numbers = knorm*arange(0,n)
  
#   tke_spectrum = zeros(len(wave_numbers))
  
#   for kx in xrange(nx):
#     rkx = kx
#     if (kx > kxmax):
#       rkx = rkx - (nx)
#     for ky in xrange(ny):
#       rky = ky
#       if (ky>kymax):
#         rky=rky - (ny)
#       for kz in xrange(nz):        
#         rkz = kz
#         if (kz>kzmax):
#           rkz = rkz - (nz)
#         rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
#         k = int(np.round(rk))
#         tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]

#   tke_spectrum = tke_spectrum/knorm
#   if smooth:
#     tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
#     tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
#     tke_spectrum = tkespecsmooth

#   knyquist = knorm*min(nx,ny,nz)/2 

#   return knyquist, wave_numbers, tke_spectrum

# #------------------------------------------------------------------------------

# def compute_tke_spectrum(u,v,w,lx,ly,lz,smooth):
#   """
#   Given a velocity field u, v, w, this function computes the kinetic energy
#   spectrum of that velocity field in spectral space. This procedure consists of the 
#   following steps:
#   1. Compute the spectral representation of u, v, and w using a fast Fourier transform.
#   This returns uf, vf, and wf (the f stands for Fourier)
#   2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf, vf, wf)* conjugate(uf, vf, wf)
#   3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
#   Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
#   the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
#   E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

#   Parameters:
#   -----------  
#   u: 3D array
#     The x-velocity component.
#   v: 3D array
#     The y-velocity component.
#   w: 3D array
#     The z-velocity component.    
#   lx: float
#     The domain size in the x-direction.
#   ly: float
#     The domain size in the y-direction.
#   lz: float
#     The domain size in the z-direction.
#   smooth: boolean
#     A boolean to smooth the computed spectrum for nice visualization.
#   """
#   nx = len(u[:,0,0])
#   ny = len(v[0,:,0])
#   nz = len(w[0,0,:])
  
#   nt= nx*ny*nz
#   n = nx #int(np.round(np.power(nt,1.0/3.0)))
  
#   uh = fftn(u)/nt
#   vh = fftn(v)/nt
#   wh = fftn(w)/nt
  
#   tkeh = zeros((nx,ny,nz))
#   tkeh = 0.5*(uh*conj(uh) + vh*conj(vh) + wh*conj(wh)).real
  
#   k0x = 2.0*pi/lx
#   k0y = 2.0*pi/ly
#   k0z = 2.0*pi/lz
  
#   knorm = (k0x + k0y + k0z)/3.0
  
#   kxmax = nx/2
#   kymax = ny/2
#   kzmax = nz/2
  
#   wave_numbers = knorm*arange(0,n)
  
#   tke_spectrum = zeros(len(wave_numbers))
  
#   for kx in range(nx):
#     rkx = kx
#     if (kx > kxmax):
#       rkx = rkx - (nx)
#     for ky in range(ny):
#       rky = ky
#       if (ky>kymax):
#         rky=rky - (ny)
#       for kz in range(nz):        
#         rkz = kz
#         if (kz>kzmax):
#           rkz = rkz - (nz)
#         rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
#         k = int(np.round(rk))
#         tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]

#   tke_spectrum = tke_spectrum/knorm
# #  tke_spectrum = tke_spectrum[1:]
# #  wave_numbers = wave_numbers[1:]
#   if smooth:
#     tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
#     tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
#     tke_spectrum = tkespecsmooth

#   knyquist = knorm*min(nx,ny,nz)/2 

#   return knyquist, wave_numbers, tke_spectrum
