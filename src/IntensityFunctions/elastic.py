from Common.eyeHField import *
from Common.imageGradient import *
from Common.Interp import *
import math
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


def elastic(i2, i1, sigma, l, step_size, n_iters, cancel_check=None, cancel_exception=None):

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

    # Precompute: stable inverse of L — used every iteration
    Linv = 1.0 / (L + 0.0001)   # shape (X, Y, Z), on device

    # Precompute scalar step factor — constant across all iterations
    scale = step_size / (sigma ** 2 * l)

    def _smooth_and_update(component_idx, der):
        """Apply inverse-Laplacian smoothing in Fourier space and update def_field in-place."""
        fft = torch.fft.fftn(der)
        smoothed = torch.view_as_real(fft)
        smoothed_r = smoothed[..., 0] * Linv
        smoothed_c = smoothed[..., 1] * Linv
        fft_smooth = torch.view_as_complex(torch.stack((smoothed_r, smoothed_c), -1))
        def_field[component_idx] -= scale * torch.fft.ifftn(fft_smooth, norm="backward").real

    for iter in range(n_iters):
        if cancel_check and cancel_exception and cancel_check():
            raise cancel_exception()

        phi = id + def_field

        i2_u = interp(i2, phi)
        diff = (i2_u - i1)
        print(f"Iteration n:{iter}, difference:{(diff**2).sum():.2g}")

        grad = imageGradient(i2_u)

        _smooth_and_update(0, diff * grad[0])
        _smooth_and_update(1, diff * grad[1])
        _smooth_and_update(2, diff * grad[2])

    phi = id + def_field

    return phi, None
