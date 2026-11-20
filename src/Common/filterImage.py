import torch
import math
import numpy as np
import torch.nn.functional as F

# Gaussian Blur Filter

def _gaussian_1d_kernel(sigma, device, dtype):
    # build a 1D Gaussian kernel; size = max(3, int(6*sigma)+1) odd
    radius = max(1, int(math.ceil(3 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(-radius, radius+1, dtype=dtype, device=device)
    kernel = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size)   # shape for conv1d

def _pad_same_1d(x, kernel_size, dim):
    pad = (kernel_size - 1) // 2
    # pad format for F.pad: (left, right) pairs for last dims
    if dim == -1:   # width
        return F.pad(x, (pad, pad))
    elif dim == -2: # height
        return F.pad(x, (0,0, pad, pad))
    elif dim == -3: # depth
        return F.pad(x, (0,0,0,0, pad, pad))
    else:
        return x

def filterImage(im, dim, sigma):
    """
    Separable Gaussian blur applied in float32, using conv1d/conv2d reshapes.
    im: torch tensor with shape (Z,Y,X) for dim==3 or (Y,X) for dim==2
    sigma: scalar float
    """
    # ensure float32 and device alignment
    device = im.device
    dtype = torch.float32
    im = im.to(dtype=dtype)

    if dim == 2:
        # im: (H, W) -> reshape to (N=1, C=1, W) after conv along height, etc.
        H, W = im.shape
        # create 1D kernels for y and x
        kx = _gaussian_1d_kernel(sigma, device, dtype).view(1, 1, -1)  # (1,1,Kx)
        ky = _gaussian_1d_kernel(sigma, device, dtype).view(1, 1, -1)

        # first convolve along rows (width) using conv1d: need shape (N=H, C=1, W)
        x = im.unsqueeze(1)           # (H,1,W)? careful: im is (H,W) so:
        x = im.unsqueeze(1)           # (H,1,W) via (H, W) -> (H,1,W)
        # conv1d expects (N, C, L) so treat rows as batch dimension
        x = F.pad(x, ((kx.shape[2]-1)//2,)*2)    # pad width
        x = F.conv1d(x, kx)            # (H,1,W)
        # now convolve along columns: treat columns as batch for conv1d
        x = x.permute(2, 1, 0)         # (W,1,H)
        x = F.pad(x, ((ky.shape[2]-1)//2,)*2)
        x = F.conv1d(x, ky)            # (W,1,H)
        x = x.permute(2, 1, 0)         # back (H,1,W)
        ret = x[ :, 0, : ]             # (H, W)
        return ret

    elif dim == 3:
        # im: (Z, Y, X)
        Z, Y, X = im.shape
        kz = _gaussian_1d_kernel(sigma, device, dtype).view(1, 1, -1)  # along depth
        ky = _gaussian_1d_kernel(sigma, device, dtype).view(1, 1, -1)  # along height
        kx = _gaussian_1d_kernel(sigma, device, dtype).view(1, 1, -1)  # along width

        # 1) convolve along X (width): treat each (Z*Y) row as batch
        x = im.view(Z*Y, 1, X)   # (N=Z*Y, C=1, L=X)
        padx = (kx.shape[2]-1)//2
        x = F.pad(x, (padx, padx))
        x = F.conv1d(x, kx)      # (Z*Y,1,X)
        x = x.view(Z, Y, X)

        # 2) convolve along Y (height): treat each (Z) slice rows as batch
        x = x.permute(0, 2, 1).contiguous()   # (Z, X, Y)
        x = x.view(Z*X, 1, Y)                 # (N=Z*X,1,Y)
        pady = (ky.shape[2]-1)//2
        x = F.pad(x, (pady, pady))
        x = F.conv1d(x, ky)                    # (Z*X,1,Y)
        x = x.view(Z, X, Y).permute(0, 2, 1)   # back to (Z, Y, X)

        # 3) convolve along Z (depth): treat each (Y,X) column as batch
        x = x.permute(1, 2, 0).contiguous()    # (Y, X, Z)
        x = x.view(Y*X, 1, Z)                 # (N=Y*X,1,Z)
        padz = (kz.shape[2]-1)//2
        x = F.pad(x, (padz, padz))
        x = F.conv1d(x, kz)                    # (Y*X,1,Z)
        x = x.view(Y, X, Z).permute(2, 0, 1)   # (Z, Y, X)

        return x
    else:
        raise ValueError("dim must be 2 or 3")
