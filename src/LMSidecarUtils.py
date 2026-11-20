"""
LMSidecarUtils.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

import json, os, time, tempfile
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox

def _old_sidecar_path_for(image_path: str) -> Path:
    """
    Returns the old sidecar path format for backward compatibility.
    Old format: <image_stem>.landmarker.json next to the image.
    """
    p = Path(image_path)
    name = p.name.lower()
    if name.endswith(".nii.gz"):
        stem = p.name[:-7]        # drop ".nii.gz"
    elif name.endswith(".nii"):
        stem = p.name[:-4]        # drop ".nii"
    else:
        stem = p.stem
    return p.with_name(f"{stem}.landmarker.json")

def sidecar_path_for(image_path: str) -> Path:
    """
    Returns the path to the sidecar file in .landmarker/ subdirectory.
    New format: .landmarker/<image_name>.lm.json
    """
    p = Path(image_path)
    name = p.name.lower()
    if name.endswith(".nii.gz"):
        stem = p.name[:-7]        # drop ".nii.gz"
    elif name.endswith(".nii"):
        stem = p.name[:-4]        # drop ".nii"
    else:
        stem = p.stem
    # Create path in .landmarker subdirectory with .lm.json extension
    landmarker_dir = p.parent / ".landmarker"
    return landmarker_dir / f"{stem}.lm.json"

def write_sidecar(image_path: str, uid_meta: dict, *, landmarker_version: str = "0.1.0", parent=None) -> Path | None:
    """
    Try to write .landmarker/<image_stem>.lm.json next to image_path.
    On any failure, warn the user and return None.
    """
    img = Path(image_path)
    sc_path = sidecar_path_for(image_path)

    # Optional file stats
    try:
        st = img.stat()
        file_size = int(st.st_size)
        file_mtime_epoch = float(st.st_mtime)
    except Exception:
        file_size = None
        file_mtime_epoch = None

    payload = {
        "schema_version": "1",
        "landmarker_version": landmarker_version,
        "image_uid": uid_meta.get("image_uid"),
        "uid_algo": uid_meta.get("uid_algo", "sha256"),
        "space": uid_meta.get("space", "RAS"),
        "dtype": uid_meta.get("dtype"),
        "shape": uid_meta.get("shape"),
        "spacing": uid_meta.get("spacing"),
        "affine_rounded_1e-6": uid_meta.get("affine_rounded_1e-6"),
        "data_sha256_short": uid_meta.get("data_sha256_short"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_full_path": str(img.resolve()),
        "source_basename": img.name,
    }
    if file_size is not None:
        payload["file_size"] = file_size
    if file_mtime_epoch is not None:
        payload["file_mtime_epoch"] = file_mtime_epoch

    # Try atomic write in .landmarker directory
    try:
        sc_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(sc_path.parent), encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(sc_path)  # atomic on same filesystem
        return sc_path
    except Exception as e:
        error_msg = f"Unable to save sidecar in .landmarker directory:\n{sc_path}\n\nError:\n{e}"
        if parent is not None and hasattr(parent, 'post_status_message'):
            parent.print_log_message(error_msg)
        QMessageBox.warning(
            parent,
            "Sidecar not written",
            error_msg,
            QMessageBox.Ok,
        )
        return None

def read_sidecar_if_exists(image_path: str) -> dict | None:
    """
    Try to read sidecar file. Checks new location first (.landmarker/<image_name>.lm.json),
    then falls back to old location (<image_stem>.landmarker.json) for backward compatibility.
    """
    # Try new location first
    sc = sidecar_path_for(image_path)
    if sc.exists():
        try:
            return json.loads(sc.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    # Fall back to old location for backward compatibility
    sc_old = _old_sidecar_path_for(image_path)
    if sc_old.exists():
        try:
            return json.loads(sc_old.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    return None
