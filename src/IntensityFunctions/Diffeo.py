import torch
from Common.eyeHField import *
from Common.imageGradient import *
from Common.Interp import *
import math
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


def diffeomorphic(i2, i1, sigma, l, step_size, n_iters, cancel_check=None, cancel_exception=None):

    device = i1.device

    s = list(i1.shape)
    dims = len(s)

    id = eyeHField(s, device)
    def_field = torch.zeros_like(id)

    # --- Build L vectorized (replaces triple nested Python loop) ---
    u = torch.arange(s[0], device=device).float()
    v = torch.arange(s[1], device=device).float()
    w = torch.arange(s[2], device=device).float()
    cu = -2.0 * torch.cos(2 * math.pi * u / s[0])
    cv = -2.0 * torch.cos(2 * math.pi * v / s[1])
    cw = -2.0 * torch.cos(2 * math.pi * w / s[2])
    L = cu[:, None, None] + cv[None, :, None] + cw[None, None, :] + 6.0

    # Precompute: stable inverse of L — reused every iteration
    Linv = 1.0 / (L + 0.0001)

    # Precompute scalar step factor — constant across all iterations
    scale = step_size / (sigma ** 2 * l)

    def _smooth(der):
        """Apply inverse-Laplacian smoothing in Fourier space, return smoothed field."""
        fft = torch.view_as_real(torch.fft.fftn(der))
        fft[..., 0] *= Linv
        fft[..., 1] *= Linv
        return torch.fft.ifftn(torch.view_as_complex(fft), norm="backward").real

    for iter in range(n_iters):
        if cancel_check and cancel_exception and cancel_check():
            raise cancel_exception()

        phi = id + def_field

        i2_u = interp(i2, phi)
        diff = (i2_u - i1)
        print(f"Iteration n:{iter}, difference:{(diff**2).sum():.2g}")

        grad = imageGradient(i2_u)

        # Smooth each gradient component
        vx = scale * _smooth(diff * grad[0])
        vy = scale * _smooth(diff * grad[1])
        vz = scale * _smooth(diff * grad[2])

        # Diffeomorphic update: compose phi with (id - v) instead of additive update
        vf = torch.stack([vx, vy, vz])
        temp = id - vf

        def_field = torch.stack([
            interp(phi[0], temp),
            interp(phi[1], temp),
            interp(phi[2], temp),
        ]) - id

    phi = id + def_field

    return phi, None
