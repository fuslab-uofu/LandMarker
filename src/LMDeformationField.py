"""
LMDeformationField.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

import hashlib
import datetime
import os
import numpy as np
import torch
import uuid

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import h5py
except ImportError:
    h5py = None


def _landmark_set_fingerprint(landmarks):
    """Build a deterministic string fingerprint of the landmark set for use in unique_id_hash.

    Uses source (moving) and target (fixed) voxel coordinates in order so that the same
    landmark set produces the same fingerprint and different sets (e.g. after adding a
    landmark) produce different fingerprints.
    """
    if not landmarks:
        return ""
    parts = []
    for lm in landmarks:
        src = []
        tgt = []
        try:
            if lm.source_voxel_coords is not None:
                arr = np.asarray(lm.source_voxel_coords, dtype=np.float64)
                if arr.size >= 3:
                    src = arr.flatten()[:3].tolist()
            if lm.target_voxel_coords is not None:
                arr = np.asarray(lm.target_voxel_coords, dtype=np.float64)
                if arr.size >= 3:
                    tgt = arr.flatten()[:3].tolist()
        except (TypeError, ValueError):
            pass
        parts.append(repr((tuple(src), tuple(tgt))))
    return str(len(landmarks)) + "_" + "|".join(parts)


def _normalize_deformation_field_to_tensor(arr):
    """Convert array to torch tensor of shape (3, D, H, W) in float32.

    Accepts (3, D, H, W) or (D, H, W, 3). Raises ValueError if shape is invalid.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(
            f"Deformation field must be 4D, got shape {arr.shape}. "
            "Expected (3, D, H, W) or (D, H, W, 3)."
        )
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        # (3, D, H, W)
        pass
    elif arr.shape[-1] == 3 and arr.shape[0] != 3:
        # (D, H, W, 3) -> (3, D, H, W)
        arr = np.transpose(arr, (3, 0, 1, 2))
    else:
        raise ValueError(
            f"Deformation field shape must be (3, D, H, W) or (D, H, W, 3), got {arr.shape}."
        )
    return torch.from_numpy(arr.astype(np.float32))


def _squeeze_deformation_array(arr: np.ndarray) -> np.ndarray:
    """
    Attempt to reduce common singleton dimensions from deformation exports.

    Examples:
      - (D, H, W, 1, 3) -> (D, H, W, 3)
      - (1, 3, D, H, W) -> (3, D, H, W)
    """
    arr = np.asarray(arr)

    # Common NIfTI-like convention: (D, H, W, 1, 3)
    if arr.ndim == 5 and arr.shape[-2:] == (1, 3):
        return arr.squeeze(-2)

    # Handle a couple additional singleton layouts seen in the wild
    if arr.ndim == 5 and arr.shape[:2] == (1, 3):
        return arr.squeeze(0)
    if arr.ndim == 5 and arr.shape[-1] == 1 and arr.shape[0] == 3:
        return arr.squeeze(-1)

    # Generic: squeeze singleton dims while preserving the vector channel.
    channel_first = arr.ndim > 0 and arr.shape[0] == 3
    channel_last = arr.ndim > 0 and arr.shape[-1] == 3
    while arr.ndim > 4:
        squeezed_any = False
        for axis in reversed(range(arr.ndim)):
            if arr.shape[axis] != 1:
                continue
            # Avoid squeezing the assumed channel dimension.
            if channel_first and axis == 0:
                continue
            if channel_last and axis == (arr.ndim - 1):
                continue
            arr = np.squeeze(arr, axis=axis)
            squeezed_any = True
            channel_first = arr.ndim > 0 and arr.shape[0] == 3
            channel_last = arr.ndim > 0 and arr.shape[-1] == 3
            break
        if not squeezed_any:
            break

    return arr


# TEMPORARILY DISABLED DURING REFACTORING
# def _load_deformation_field_from_h5(file_path: str) -> torch.Tensor:
#     """
#     Load a deformation/displacement field from an HDF5 (.h5) file.
#
#     The loader scans datasets to find a 3-component vector field compatible with
#     LandMarker's expected shapes:
#       - (3, D, H, W) or (D, H, W, 3)
#     """
#     if h5py is None:
#         raise ValueError("HDF5 deformation field import requires h5py. Install with: pip install h5py")
#
#     file_path = os.path.abspath(file_path)
#     best_candidate = None  # (score, dataset_name)
#
#     def _score_candidate(name: str, shape: tuple[int, ...]) -> int:
#         lname = name.lower()
#         score = 0
#
#         if "deformation" in lname:
#             score += 5
#         if "displacement" in lname:
#             score += 4
#         if "warp" in lname:
#             score += 3
#         if "vector" in lname:
#             score += 2
#
#         if len(shape) == 4 and (shape[0] == 3 or shape[-1] == 3):
#             score += 8
#         if len(shape) == 5 and shape[-2:] == (1, 3):
#             score += 6
#
#         # Strong preference for channel-last layouts.
#         if len(shape) >= 1 and shape[-1] == 3:
#             score += 2
#         if len(shape) >= 1 and shape[0] == 3:
#             score += 2
#
#         if len(shape) not in (4, 5):
#             score -= 100
#         return score
#
#     with h5py.File(file_path, "r") as f:
#         def visit(name, obj):
#             nonlocal best_candidate
#             try:
#                 if not isinstance(obj, h5py.Dataset):
#                     return
#             except Exception:
#                 return
#
#             shape = getattr(obj, "shape", None)
#             if shape is None:
#                 return
#
#             shape = tuple(int(s) for s in shape)
#             if len(shape) not in (4, 5):
#                 return
#
#             # Quick compatibility check before reading any data
#             compatible = (
#                 (len(shape) == 4 and (shape[0] == 3 or shape[-1] == 3))
#                 or (len(shape) == 5 and shape[-2:] == (1, 3))
#                 or (len(shape) == 5 and shape[:2] == (1, 3))
#             )
#             if not compatible:
#                 return
#
#             score = _score_candidate(name, shape)
#             if best_candidate is None or score > best_candidate[0]:
#                 best_candidate = (score, name)
#
#         f.visititems(visit)
#
#         if best_candidate is None:
#             raise ValueError(
#                 "No compatible 3-component deformation/displacement dataset was found in this .h5 file. "
#                 "3D Slicer exports may store the field under a different dataset name or layout."
#             )
#
#         _, dataset_name = best_candidate
#         arr = np.asarray(f[dataset_name][...])
#
#     arr = _squeeze_deformation_array(arr)
#     return _normalize_deformation_field_to_tensor(arr)


# TEMPORARILY DISABLED DURING REFACTORING
# def load_from_external_file(parent, file_path, idx):
#     """Load a deformation field from an external file (.npy, .pt data-only, NIfTI, or HDF5).
#
#     Creates an LMDeformationField with is_imported=True and generated name/description/unique_id_hash.
#     Supported formats:
#     - .npy: NumPy array, shape (3, D, H, W) or (D, H, W, 3)
#     - .pt: PyTorch tensor or dict with "data" key; shape (3, D, H, W) or (D, H, W, 3)
#     - .nii, .nii.gz: NIfTI image with 3 components (e.g. last dimension 3), shape (3, D, H, W) or (D, H, W, 3)
#     - .h5: HDF5 file containing a 3-component vector field dataset exported by 3D Slicer
#
#     :param parent: LandMarker parent (e.g. main window)
#     :param file_path: path to the file
#     :param idx: index for the new deformation field in the list
#     :return: LMDeformationField instance with is_imported=True
#     :raises: ValueError or other exception on unsupported format or invalid data
#     """
#     file_path = os.path.abspath(file_path)
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     if base_name.endswith(".nii"):
#         base_name = os.path.splitext(base_name)[0]
#     ext = os.path.splitext(file_path)[1].lower()
#     if file_path.endswith(".nii.gz"):
#         ext = ".nii.gz"
#
#     data_tensor = None
#     rigid_matrix = None
#
#     if ext == ".npy":
#         arr = np.load(file_path)
#         data_tensor = _normalize_deformation_field_to_tensor(arr)
#     elif ext == ".pt":
#         try:
#             loaded = torch.load(file_path, weights_only=False)
#         except TypeError:
#             loaded = torch.load(file_path)
#         if isinstance(loaded, dict) and "data" in loaded:
#             arr = loaded["data"]
#             if isinstance(arr, torch.Tensor):
#                 arr = arr.cpu().numpy()
#             else:
#                 arr = np.asarray(arr)
#             data_tensor = _normalize_deformation_field_to_tensor(arr)
#             rigid_matrix = loaded.get("rigid_matrix")
#         elif isinstance(loaded, torch.Tensor):
#             arr = loaded.cpu().numpy()
#             data_tensor = _normalize_deformation_field_to_tensor(arr)
#         else:
#             raise ValueError("External .pt file must contain a tensor or a dict with 'data' key.")
#     elif ext in (".nii", ".nii.gz"):
#         if nib is None:
#             raise ValueError("NIfTI deformation field import requires nibabel. Install with: pip install nibabel")
#         img = nib.load(file_path)
#         arr = np.asarray(img.dataobj)
#         if arr.ndim == 5 and arr.shape[-2:] == (1, 3):
#             # (D, H, W, 1, 3) -> (D, H, W, 3)
#             arr = arr.squeeze(-2)
#         data_tensor = _normalize_deformation_field_to_tensor(arr)
#     elif ext in (".h5", ".hdf5"):
#         data_tensor = _load_deformation_field_from_h5(file_path)
#     else:
#         raise ValueError(
#             f"Unsupported deformation field file extension: {ext}. Use .npy, .pt, .nii, .nii.gz, or .h5."
#         )
#
#     imported_name = "Imported_" + base_name
#     return LMDeformationField(
#         parent, idx, data_tensor, False,
#         _moving_vol=None, _fixed_vol=None, _algorithm=None,
#         _landmarks=None, _source_dfields=None, _rigid_matrix=rigid_matrix,
#         _is_imported=True, _imported_name=imported_name
#     )


def compute_unique_id_hash(moving_vol, fixed_vol, algorithm, landmarks):
    """Compute the unique_id_hash that a non-composed LMDeformationField would have.

    Used to detect an existing deformation field with the same inputs (moving/fixed volumes,
    algorithm, and landmark set) so it can be reused instead of creating a duplicate.
    """
    mv = (moving_vol.full_file_name if moving_vol is not None else "")
    fv = (fixed_vol.full_file_name if fixed_vol is not None else "")
    algo = (algorithm.algorithm_description if algorithm is not None else "")
    concat_vals = mv + fv + algo
    fp = _landmark_set_fingerprint(landmarks)
    if fp:
        concat_vals += "_" + fp
    return hashlib.md5(concat_vals.encode()).hexdigest()


class LMDeformationField:
    """
    This file contains the LMDeformationField class, which is a class for managing a LandMarker deformation field.
    It simply stores the deformation field data and some metadata about the field, including a unique id hash formed
    from the moving volume, fixed volume, algorithm, and (when present) a fingerprint of the landmark set used
    to generate the field.
    """

    def __init__(self, _parent, _idx, _dfield, _is_composed, _moving_vol=None, _fixed_vol=None, _algorithm=None,
                 _landmarks=None, _source_dfields=None, _rigid_matrix=None, _is_imported=False, _imported_name=None):
        self.parent = _parent

        self.data = _dfield  # tensor

        # is this deformation field a result of composing other deformation fields?
        self.is_composed = _is_composed
        # is this deformation field imported from an external file (no LandMarker provenance)?
        self.is_imported = _is_imported

        # rigid/affine transformation matrix (None if not applicable)
        self.rigid_matrix = _rigid_matrix
        # if not self.is_composed:
        #     if _moving_vol is None or _fixed_vol is None or _algorithm is None:
        #         raise ValueError("LMDeformationField: _moving_vol, _fixed_vol, and _algorithm must be provided for "
        #                          "non-composed deformation fields.")

        # the moving volume, fixed volume, and algorithm used to generate the field (None if composed)
        if _moving_vol is None:
            self.moving_vol = ""
            mv_base_name = ""
        else:
            self.moving_vol = _moving_vol.full_file_name
            mv_base_name = _moving_vol.file_base_name
        if _fixed_vol is None:
            self.fixed_vol = ""
            fv_base_name = ""
        else:
            self.fixed_vol = _fixed_vol.full_file_name
            fv_base_name = _fixed_vol.file_base_name
        if _algorithm is None:
            self.algorithm = ""
        else:
            self.algorithm = _algorithm.algorithm_description
        self.landmarks = _landmarks  # landmarks used in registration are optional
        self.idx = _idx  # optional index of the deformation field into the parent list of dfields (integer)

        # the source deformation fields used to generate this deformation field (None if not composed)
        # ok to be None, even if composed
        self.source_dfields = _source_dfields

        if self.is_composed:
            self.unique_id_hash = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
            # self.unique_id_hash = "to be implemented"
            # self.name = (mv_base_name + "_" + fv_base_name + "_" + self.algorithm + "_composed")
            self.name = f"{mv_base_name}_{fv_base_name}_{self.algorithm}_composed_{self.unique_id_hash}"
            self.description = "composed deformation field"
        elif self.is_imported:
            # imported external deformation field: no LandMarker provenance
            self.unique_id_hash = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self.name = _imported_name if _imported_name else ("Imported_" + self.unique_id_hash)
            self.description = "Imported deformation field"
        else:
            # this deformation field is a result of a single registration
            # generate an id using volume names, algorithm, and landmark set fingerprint
            self.name = (mv_base_name + "_" + fv_base_name + "_" + self.algorithm)
            concat_vals = (self.moving_vol + self.fixed_vol + self.algorithm)
            landmark_fp = _landmark_set_fingerprint(_landmarks)
            if landmark_fp:
                concat_vals += "_" + landmark_fp
            self.unique_id_hash = hashlib.md5(concat_vals.encode()).hexdigest()
            self.description = (mv_base_name + " to " + fv_base_name + " using " +
                                self.algorithm)

    def set_dfield_source_list(self, source_dfields):
        self.source_dfields = source_dfields

    # TEMPORARILY DISABLED DURING REFACTORING
    # def save_data_only(self, file_path, format='numpy'):
    #     """Save only the deformation field data (no metadata).
    #
    #     :param file_path: path for the output file
    #     :param format: 'numpy' or 'tensor' — output format ('numpy' → .npy, 'tensor' → PyTorch .pt)
    #     :return: None
    #     """
    #     if format == 'numpy':
    #         arr = self.data.detach().cpu().numpy()
    #         np.save(file_path, arr)
    #     elif format == 'tensor':
    #         torch.save(self.data, file_path)
    #     else:
    #         raise ValueError(f"format must be 'numpy' or 'tensor', got '{format}'")

    def _deformation_as_channel_first_numpy(self) -> np.ndarray:
        """Return deformation data as numpy array with shape (3, D, H, W)."""
        if self.data is None:
            raise ValueError("No deformation field data available to export.")

        if isinstance(self.data, torch.Tensor):
            arr = self.data.detach().cpu().numpy()
        else:
            arr = np.asarray(self.data)

        # Expect either channel-first (3, D, H, W) or channel-last (D, H, W, 3)
        if arr.ndim != 4:
            raise ValueError(f"Deformation field must be 4D, got shape {arr.shape}.")

        if arr.shape[0] == 3:
            return arr
        if arr.shape[-1] == 3:
            return np.transpose(arr, (3, 0, 1, 2))

        raise ValueError(
            f"Deformation field must have a 3-component channel either first or last; got shape {arr.shape}."
        )

    def _deformation_as_channel_last_numpy(self) -> np.ndarray:
        """Return deformation data as numpy array with shape (D, H, W, 3)."""
        arr_cf = self._deformation_as_channel_first_numpy()
        return np.transpose(arr_cf, (1, 2, 3, 0))

    # TEMPORARILY DISABLED DURING REFACTORING
    # def save_nifti(self, file_name: str, *, affine=None):
    #     """
    #     Save deformation field as a NIfTI image.
    #
    #     Output layout: (D, H, W, 3) where the last dimension stores the 3 displacement components.
    #     """
    #     if nib is None:
    #         raise ValueError("NIfTI deformation field export requires nibabel. Install with: pip install nibabel")
    #
    #     data_cl = self._deformation_as_channel_last_numpy().astype(np.float32, copy=False)
    #
    #     if affine is None:
    #         affine = np.eye(4, dtype=np.float64)
    #     affine = np.asarray(affine, dtype=np.float64)
    #     if affine.shape != (4, 4):
    #         raise ValueError(f"affine must be 4x4, got shape {affine.shape}.")
    #
    #     # Nibabel supports vector fields stored as 4D (X, Y, Z, C) where C=3.
    #     img = nib.Nifti1Image(data_cl, affine=affine)
    #     nib.save(img, file_name)

    # TEMPORARILY DISABLED DURING REFACTORING
    # def save_h5(self, file_name: str, *, dataset_name: str = "deformationField"):
    #     """Save deformation field as an HDF5 dataset with a name searchable by the importer."""
    #     if h5py is None:
    #         raise ValueError("HDF5 deformation field export requires h5py. Install with: pip install h5py")
    #
    #     data_cl = self._deformation_as_channel_last_numpy().astype(np.float32, copy=False)
    #
    #     file_name = os.path.abspath(file_name)
    #     with h5py.File(file_name, "w") as f:
    #         # The importer scans dataset names for "deformation"/"displacement"/"warp"/"vector".
    #         f.create_dataset(dataset_name, data=data_cl, dtype=np.float32)

    def save_to_file(self, file_name):
        meta_dict = {"data": self.data,
                     "is_composed": self.is_composed,
                     "is_imported": getattr(self, "is_imported", False),
                     "moving_vol": self.moving_vol,
                     "fixed_vol": self.fixed_vol,
                     "algorithm": self.algorithm,
                     "landmarks": [],  # TODO
                     "name": self.name,
                     "description": self.description,
                     "unique_id_hash": self.unique_id_hash,
                     "source_dfields": None,  # TODO
                     "rigid_matrix": self.rigid_matrix}
        torch.save(meta_dict, file_name)

    def load_from_file(self, file_name):
        try:
            meta_dict = torch.load(file_name, weights_only=False)
        except TypeError:
            meta_dict = torch.load(file_name)
        self.data = meta_dict["data"]
        self.is_composed = meta_dict["is_composed"]
        self.is_imported = meta_dict.get("is_imported", False)
        self.moving_vol = meta_dict["moving_vol"]
        self.fixed_vol = meta_dict["fixed_vol"]
        self.algorithm = meta_dict["algorithm"]
        self.landmarks = meta_dict["landmarks"]
        self.name = meta_dict["name"]
        self.description = meta_dict["description"]
        self.unique_id_hash = meta_dict["unique_id_hash"]
        self.source_dfields = meta_dict["source_dfields"]
        # Handle rigid_matrix - may not exist in older saved files
        self.rigid_matrix = meta_dict.get("rigid_matrix", None)