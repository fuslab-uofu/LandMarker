import torch
import numpy as np
from Common.imageGradient import *


def jacobianHField(h):
    dim = h.shape[0]
    J = torch.zeros([dim] + list(h.shape)).to(h.device)

    J[0] = imageGradient(h[0].squeeze())
    J[1] = imageGradient(h[1].squeeze())

    if dim == 3:
        J[2] = imageGradient(h[2].squeeze())

    return J
