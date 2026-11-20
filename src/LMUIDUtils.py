"""
LMUIDUtils.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

import hashlib
import numpy as np
import nibabel as nib

UID_HEADER = b"LandMarkerUID:v1\n"

def _round_affine(a: np.ndarray, decimals: int = 6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.round(a, decimals=decimals)

def compute_image_uid_from_nifti(nib_img: nib.Nifti1Image, *, display_convention="RAS") -> dict:
    """
    Compute a stable UID from canonical (RAS) image geometry + raw voxel bytes.
    Returns a small dict with the UID and key metadata needed for a sidecar.
    """
    # Canonicalize to RAS
    img_ras = nib.as_closest_canonical(nib_img)
    affine = img_ras.affine
    affine_r = _round_affine(affine, decimals=6)

    # Raw voxel array (preserve original dtype; avoid get_fdata() casting)
    arr = np.asanyarray(img_ras.dataobj)  # loads on demand
    dtype_str = arr.dtype.str

    # Shape & spacing (pad to 3D for consistency)
    shape = tuple(int(s) for s in arr.shape)
    if len(shape) == 2:
        shape3 = (1, shape[0], shape[1])          # Z,Y,X
    elif len(shape) == 3:
        shape3 = shape
    else:
        # Fallback: flatten higher dims into Z
        z = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
        shape3 = (z, shape[-2], shape[-1])

    zooms = img_ras.header.get_zooms()
    dz = float(zooms[0]) if len(zooms) > 0 else 1.0
    dy = float(zooms[1]) if len(zooms) > 1 else 1.0
    dx = float(zooms[2]) if len(zooms) > 2 else 1.0
    spacing3 = (dz, dy, dx)

    # Build payload (order matters)
    payload_parts = [
        UID_HEADER,
        dtype_str.encode("utf-8"),
        np.asarray(shape3, dtype=np.int64).tobytes(order="C"),
        np.asarray(spacing3, dtype=np.float64).tobytes(order="C"),
        np.asarray(affine_r, dtype=np.float64).ravel(order="C").tobytes(),
        arr.tobytes(order="C"),
    ]
    payload = b"".join(payload_parts)

    # Hashes
    uid_full = hashlib.sha256(payload).hexdigest()
    data_hash_full = hashlib.sha256(arr.tobytes(order="C")).hexdigest()

    return {
        "image_uid": uid_full[:16],              # short, friendly
        "uid_algo": "sha256",
        "space": display_convention,
        "dtype": dtype_str,
        "shape": list(shape3),                   # [Z, Y, X]
        "spacing": [dz, dy, dx],
        "affine_rounded_1e-6": _round_affine(affine, 6).ravel(order="C").tolist(),
        "data_sha256_short": data_hash_full[:16],
    }
