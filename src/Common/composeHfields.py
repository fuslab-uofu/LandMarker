import torch
import torch.nn.functional as F
from Common.deformImage import *


def composeHfields(f, g, s, dim):
    # `deformImage()` -> `interp()` expects `im` and `grid` to live on the same device.
    # When composing deformation fields, it's common to have `f` on CUDA (computed)
    # and `g` on CPU (imported). Align devices to avoid "Expected all tensors to be
    # on the same device" errors inside interp().
    if isinstance(f, torch.Tensor) and isinstance(g, torch.Tensor):
        if f.device != g.device:
            g = g.to(f.device)
    ret = torch.zeros_like(f)

    for x in range(f.shape[0]):
        ret[x] = deformImage(f[x].float(), g, s, dim)

    return ret


