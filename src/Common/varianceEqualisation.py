import torch
from typing import Union
from Common.filterImage import filterImage

def varEq(Im: torch.Tensor, sigma: float = 2.0, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Variance equalisation / local contrast normalisation.

    Parameters
    ----------
    Im : torch.Tensor
        Input image tensor. Expected shape: (1, Z, Y, X) or (1, Y, X) or similar where .data.squeeze()
        yields (Z, Y, X) for 3D or (Y, X) for 2D. This function will operate on Im.data.squeeze().
    sigma : float
        Gaussian sigma used for local averaging.
    device : str or torch.device
        Device to move tensors to (e.g., 'cpu' or 'cuda:0').

    Returns
    -------
    torch.Tensor
        The same `Im` object with its .data replaced by the variance-equalised result (keeps original container).
    """
    # Defensive: ensure device and dtype
    device = torch.device(device)
    eps = 1e-3  # small epsilon to avoid division by zero

    # Put Im on requested device and use float32 to reduce memory footprint
    Im = Im.to(device)
    # Work on a squeezed view of the data (consistent with prior code paths)
    im_data = Im.data.squeeze().to(dtype=torch.float32, device=device)

    # Replace any pre-existing NaN/Inf values in the input to avoid propagation
    if not torch.isfinite(im_data).all():
        im_data = torch.nan_to_num(im_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Detect dimensionality (2 or 3)
    dims = len(im_data.shape)
    if dims not in (2, 3):
        raise ValueError(f"varEq: expected 2D or 3D image after squeeze, got shape {im_data.shape}")

    # Compute local mean (g_im) using the separable filter implementation
    g_im = filterImage(im_data, dims, float(sigma))
    # Ensure finite values and correct dtype
    g_im = g_im.to(dtype=torch.float32, device=device)
    g_im = torch.nan_to_num(g_im, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute local energy (mean of squares) and guard it
    energy = filterImage(im_data * im_data, dims, float(sigma))
    energy = energy.to(dtype=torch.float32, device=device)
    energy = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

    # sqrt of clamped energy to avoid sqrt(negative) producing NaN
    den = torch.sqrt(torch.clamp(energy, min=0.0))

    # Compute normalized (zero-mean, unit-local-variance) image
    temp = im_data - g_im
    temp = temp / (den + eps)

    # Final guard: remove any remaining NaN/Inf (be conservative and replace with 0)
    temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

    # Restore to original container shape convention (e.g., (1, Z, Y, X) or (1, Y, X))
    # If original Im.data had a singleton leading dimension, keep that.
    # We set Im.data to temp with a leading singleton dimension.
    Im.data = temp.unsqueeze(0)

    return Im

