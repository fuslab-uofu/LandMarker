import torch
from Common.transformPoints import *
from Common.eyeHField import *
from Common.Interp import *

# I0 is moving I1 is fixed - these are structured grid objects
def oR(I0, I1, device='cpu'):
    s = list(I0.data.squeeze().shape)
    space = I0.spacing

    a = I0.origin
    b = I0.origin + (space * (torch.tensor(s).to(device) - 1))  # because spacing is negative sometimes
    o1 = torch.min(torch.vstack([a, b]), axis=0)[0]

    s = list(I1.data.squeeze().shape)
    space = I1.spacing

    a1 = I1.origin
    b1 = I1.origin + (space * (torch.tensor(s).to(device) - 1))
    o2 = torch.min(torch.vstack([a1, b1]), axis=0)[0]

    sh1 = torch.tensor(list(I0.data.squeeze().shape)).to(device)
    sh2 = torch.tensor(list(I1.data.squeeze().shape)).to(device)

    e1 = o1 + torch.abs(I0.spacing) * (sh1 - 1)
    e2 = o2 + torch.abs(I1.spacing) * (sh2 - 1)

    startIdx = coords2idx(I0, o2, sh1).cpu().numpy() + 1
    endIdx = coords2idx(I0, e2, sh1).cpu().numpy() + 1

    phi = Hfield(startIdx, endIdx, sh2, device)

    ret = interp(I0.data.squeeze(), phi)  # structured grid

    return ret

