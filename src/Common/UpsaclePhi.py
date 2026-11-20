import torch
from skimage.transform import rescale
from Common.composeHfields import *
from Common.eyeHField import *


def UpscalePhi(phi0, device='cpu'):

    phi0 -= eyeHField(list(phi0.shape[1:]), device)
    phi_up = []


    for d in range(phi0.shape[0]):
        phi_up.append(2 * torch.from_numpy(rescale(phi0[d].data.cpu().numpy().squeeze(), 2, anti_aliasing=False)).to(
            device))

    phi_up = torch.stack(phi_up)

    phi_up += eyeHField(list(phi_up.shape[1:]), device)

	# J. Bonaventura
    # Clamp back into the valid coordinate range [1, s] -- rescale()'s interpolation
    # can overshoot by a fraction of a voxel right at the boundary, and even a
    # sub-voxel overshoot makes composeHfields' downstream interp() hard-zero the
    # whole point (via checkBounds) instead of just clamping it. Same [1, s] clamp
    # sdiff.py already applies to Vp, applied here too so an overshoot never reaches
    # that unclamped compose step in the first place.
    s = list(phi_up.shape[1:])
    dims = phi_up.shape[0]
    if dims == 2:
        phi_up[0][phi_up[0] > s[0]] = s[0]
        phi_up[1][phi_up[1] > s[1]] = s[1]
    elif dims == 3:
        phi_up[0][phi_up[0] > s[2]] = s[2]
        phi_up[1][phi_up[1] > s[1]] = s[1]
        phi_up[2][phi_up[2] > s[0]] = s[0]
    phi_up[phi_up < 1] = 1

    return phi_up


def ScaleList(phiList, device):

    if len(phiList) == 1:
        return UpscalePhi(phiList[0], device)

    else:
        temp = composeHfields(phiList.pop(-1), ScaleList(phiList, device), None, None)

        return UpscalePhi(temp, device)