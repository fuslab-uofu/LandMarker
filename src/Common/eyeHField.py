import torch
import numpy as np
torch.set_default_dtype(torch.float32)


def eyeHField(dim, device='cpu'):
    dims = len(dim)
    n = np.zeros((dims))
    c = np.zeros((dims))

    xx = []
    for x in range(dims):
        n[x] = dim[x]
        c[x] = n[x] / 2
        xx.append(np.linspace(1, int(n[x]), int(n[x])))

    if dims == 2:
        ey, ex = np.meshgrid(xx[1], xx[0])
        return torch.from_numpy(np.vstack([np.expand_dims(ex, 0), np.expand_dims(ey, 0)])).to(device).float()
    elif dims == 3:
        ey, ex, ez = np.meshgrid(xx[1], xx[0], xx[2])
        return torch.from_numpy(np.vstack([np.expand_dims(ez, 0),
                                           np.expand_dims(ey, 0), np.expand_dims(ex, 0)])).to(device).float()
    else:
        raise NotImplementedError


def Hfield(start, stop, dim, device='cpu'):
    dims = len(dim)
    n = np.zeros((dims))

    xx = []
    for x in range(dims):
        n[x] = dim[x]
        xx.append(np.linspace(start[x], stop[x], int(n[x])))

    if dims == 2:
        raise NotImplementedError
    elif dims == 3:
        ey, ex, ez = np.meshgrid(xx[1], xx[0], xx[2])
        return torch.from_numpy(np.vstack([np.expand_dims(ez, 0),
                                           np.expand_dims(ey, 0), np.expand_dims(ex, 0)])).to(device).float()
    else:
        raise NotImplementedError
