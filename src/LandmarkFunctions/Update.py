import torch
from LandmarkFunctions.GaussianKernel import *


def update_def(B, tar_pt, deformation, k_sigma, df, device='cpu'):
    dims = deformation.shape[0]

    all_pts = deformation.reshape(dims, -1).to(device)
    g = GaussianKernel(all_pts, tar_pt, k_sigma, df, field=True, device=device)
    B = B.repeat(all_pts.shape[1], 1, 1).to(device)
    ret = torch.matmul(B, g.float()).squeeze().t()
    ret = ret.reshape(deformation.shape)

    return ret