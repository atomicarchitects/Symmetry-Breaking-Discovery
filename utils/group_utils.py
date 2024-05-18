import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # compute the permutation matrix for the group convolution
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


        # Get the size of the input tensor
        size = input_tensor.size()

        # Generate the grid
        grid = F.affine_grid(rotation_matrix, size, align_corners=False).to(input_tensor.dtype)
    

        # Resample the tensor
        rotated_tensor = F.grid_sample(input_tensor, grid)
        return rotated_tensor
    
### O and Oh rotation matrices ###
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


refl_octahedron_rots  = np.array([[[1, 0, 0],
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
[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
[[-1.0, 0.0, 0.0],[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],[0.0, 0.0, -1.0]],
[[-1.0, 0.0, 0.0],[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
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

