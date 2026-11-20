# -*- coding: utf-8 -*-
# tps_3d.py
"""
Author:   Estelle Trieu
Created on Oct. 4th, 2023
Last modified on Oct. 17, 2023

Description
-----------
3D thin plate splines (TPS) deformation algorithm for 
quantitative multi-parametric MRI correlation. 

If you would like to understand the mathematics better
of this algorithm, feel free to read my OverLeaf doc:
https://www.overleaf.com/read/gcfrcktsrdnw.
"""
################################################################################
import torch
import numpy as np
from Common.Interp import *
from Common.StructuredGridClass import *
from copy import deepcopy
import matplotlib.pyplot as plt
import array
# from Common.Nifti_IO import *
# from Common.Landmarks_IO import *
################################################################################

def l_matrix(numpoints: int, source_landmarks: torch.Tensor) -> torch.Tensor:
    """
    Build the 'big' matrix L necessary to find the coefficients
    for the thin plate splines interpolant function. L holds 
    matrices K, P, P.T, and O.
    
    Parameters
    ----------
        numpoints (int): the number of landmarks chosen. The number
                         of source landmarks is equal to the 
                         number of target landmarks.

        source_landmarks (tensor): 2D tensor of source landmarks with shape
                                   [[x1, y1, z1],
                                    [x2, y2, x2],
                                          .
                                          .
                                          .
                                    [xn, yn, zn]]

    Returns
    -------
        L (tensor): 2D square (i.e n x n) tensor necessary to find the 
                    coefficients of the thin plate splines linear sytem.
    """
    # Build K - the matrix of distances between source points and kernels
    x_idxs = torch.tile(source_landmarks, (1, numpoints))
    y_idxs = torch.tile(source_landmarks, (numpoints, 1)).reshape(numpoints, 3*numpoints)
    # distance between two source points
    diff = (y_idxs - x_idxs)**2
    diff[:,0::3] = diff[:,0::3] + diff[:,1::3] + diff[:,2::3]
    K = torch.sqrt(diff[:,0::3])

    # Build P - the matrix of the form [xi, yi, zi, 1]
    P = torch.ones((numpoints, 4))
    P[:, 0:3] = source_landmarks        # Fill in columns xi, yi, and zi

    # Note: O is the null matrix, so there's no need to fill
    # it into L. O is in the lower right quadrant of L. 
    L = torch.zeros((numpoints + 4, numpoints + 4))

    # Fill in K in the upper left quadrant of matrix L
    L[:numpoints, :numpoints] = K

    # Fill in P in the upper right quadrant of matrix L
    L[:numpoints, -4:] = P

    # Fill in P.T in the lower left quadrant of matrix L
    L[-4:, :numpoints] = P.T
    return L


def solve_coefficients(L: torch.Tensor, source_landmarks: torch.Tensor,
                       target_landmarks: torch.Tensor, device) -> torch.Tensor:
    """
    Solve for the 3D TPS interpolation coefficients.
    
    Parameters
    ----------
        L (tensor): 2D square (i.e n x n) tensor necessary to find the 
                    coefficients of the thin plate splines linear sytem.

        source_landmarks (tensor): 2D tensor of source landmarks with shape
                                   [[x1, y1, z1],
                                    [x2, y2, x2],
                                          .
                                          .
                                          .
                                    [xn, yn, zn]]

        target_landmarks (tensor): 2D tensor of target landmarks with shape
                                   [[x1, y1, z1],
                                    [x2, y2, x2],
                                          .
                                          .
                                          .
                                    [xn, yn, zn]]

    Returns
    -------
        coefficients (tensor): 2D tensor of solved coefficients with shape
                                [[w_x1, w_y1, w_z1],
                                 [w_x2, w_y2, w_x2],
                                         .
                                         . 
                                 [w_x(n-4), w_y(n-4), w_z(n-4)],
                                 [a_x1, a_x2, a_x3],
                                 [a_y1, a_y2, a_y3],
                                 [a_z1, a_z2, a_z3],
                                 [a_0x, a_0y, a0_z]]
    """
    # Building (v|0)
    diff = torch.sub(target_landmarks, source_landmarks)
    rhs = torch.vstack((diff, torch.zeros((4,3)).to(device)))

    # Solve for coefficients 
    # torch.mm only takes float32 as input, so cast
    coefficients = torch.mm(torch.linalg.inv(L).to(device), rhs.float())  # MMKline 02/11/2026 added device
    return coefficients


def apply_tps(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
              numpoints: int, source_landmarks: torch.Tensor,
              coefficients: torch.Tensor, device) -> tuple:
    """
    Applies the TPS transformation function to identity trans-
    formation grids x, y, and z.

    Parameters
    ----------
        x (tensor): 3D identity transformation grid in the x dimension -
                    has shape (img_height, img_width, img_depth).

        y (tensor): 3D identity transformation grid in the y dimension -
                    has shape (img_height, img_width, img_depth).

        z (tensor): 3D identity transformation grid in the z dimension -
                    has shape (img_height, img_width, img_depth).

        numpoints (int): the number of landmarks chosen. The number
                         of source landmarks is equal to the 
                         number of target landmarks.

        source_landmarks (tensor): 2D tensor of source landmarks.
        coefficients (tensor): 2D tensor of solved coefficients.
                        
    Returns
    -------
        x_new, y_new, z_new (tuple): tuple containing the 3D tensors 
                                     of deformed identity transformation
                                     grids.
    """
    x_new, y_new, z_new = (torch.zeros(np.shape(x)).to(device),
                           torch.zeros(np.shape(y)).to(device),
                           torch.zeros(np.shape(z)).to(device))

    # Sum of kernel weights times coefficients w_i 
    # give the local deformations within the image
    for i in range(numpoints):
        x_comp = torch.sub(x, source_landmarks[i][0])
        y_comp = torch.sub(y, source_landmarks[i][1])
        z_comp = torch.sub(z, source_landmarks[i][2])
        r = torch.sqrt(x_comp**2 + y_comp**2 + z_comp**2)
        # We simply multiply by r since kernel is U(r) = r
        x_new += coefficients[i, 0] * r
        y_new += coefficients[i, 1] * r
        z_new += coefficients[i, 2] * r

    # Add linear part to account for affine motion
    x_new += coefficients[numpoints, 0] * x + coefficients[numpoints + 1, 0] * y + \
             coefficients[numpoints + 2, 0] * z + coefficients[numpoints + 3, 0]
    y_new += coefficients[numpoints, 1] * x + coefficients[numpoints + 1, 1] * y + \
             coefficients[numpoints + 2, 1] * z + coefficients[numpoints + 3, 1]
    z_new += coefficients[numpoints, 2] * x + coefficients[numpoints + 1, 2] * y + \
             coefficients[numpoints + 2, 2] * z + coefficients[numpoints + 3, 2]
    return x_new, y_new, z_new


def create_deformed_img(source_volume: torch.Tensor, numpoints: int,
                        source_landmarks: torch.Tensor,
                        coeff: torch.Tensor, device) -> torch.Tensor:
    """
    Returns the image deformed by 3D TPS.

    Parameters
    ----------
        source_volume (tensor): the 3D source image.
        numpoints (int): the number of landmarks chosen. The number
                         of source landmarks is equal to the
                         number of target landmarks.
        source_landmarks (tensor): 2D tensor of source landmarks.
        coeff (tensor): 2D tensor of solved coefficients.

    Returns
    -------
        deformation (tensor): the 3D image deformed by 3D TPS.
    """
    img_shape = np.shape(source_volume)

    # Create deformation grid
    ex, ey, ez = np.meshgrid(np.arange(img_shape[0]),
                             np.arange(img_shape[1]),
                             np.arange(img_shape[2]), indexing='ij')
    # MMKline - added device 02/11/2026
    ex, ey, ez = torch.tensor(ex).to(device), torch.tensor(ey).to(device), torch.tensor(ez).to(device)

    # Apply TPS on the identity transformation 
    ux, uy, uz = apply_tps(ex, ey, ez, numpoints, source_landmarks, coeff, device)

    # Add the identity transformation to the displacements
    hx = ex + ux
    hy = ey + uy
    hz = ez + uz

    # Amanpreet's interp function requires you to
    # stack the transformations in the order z, y, x
    # and not x, y, z - he should really change this
    diffeo = torch.stack((hz, hy, hx))
    deformation = interp(deepcopy(torch.tensor(source_volume)).squeeze(), diffeo)
    return deformation, diffeo


def tps_3d(source_landmarks: array.array, target_landmarks: array.array,
           source_volume: array.array, device) -> tuple:
    """
    Main function for 3D TPS algorithm.

    Parameters
    ----------
        source_landmarks (array): 2D numpy array of source landmarks.
        target_landmarks (array): 2D numpy array of target landmarks.
        source_volume (array): the 3D source image.
        device: can be cpu or gpu.
    
    Returns
    -------
        deformed_img (array): the 3D image deformed by 3D TPS.
        transformed_points (array): 2D array of deformed source landmarks.
    """
    # # Convert numpy data arrays to PyTorch tensors
    # source_landmarks = torch.tensor(source_landmarks)
    # target_landmarks = torch.tensor(target_landmarks)

    # MMKline - added device to list of arguments 02/11/2026


    num_pts = source_landmarks.size()[0]

    # Set up and solve linear system
    # [K   P] [w] = [dst]
    # [P^T 0] [a]   [ 0 ]
    L = l_matrix(num_pts, source_landmarks)
    coefficients = solve_coefficients(L, source_landmarks, target_landmarks, device)  # MMKline 02/11/2026 added device
    deformed_img, diffeo = create_deformed_img(source_volume, num_pts,
                                       source_landmarks, coefficients, device)  # MMKline 02/11/2026 added device

    # Convert back to numpy array for Michelle's GUI
    transformed_points = 1      # TODO: add deformed points
    return diffeo
   
