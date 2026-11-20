"""
LMVolume.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Dict
import nibabel as nib

from LMEnumerations import ViewDir

try:
    import pyqtgraph as pg
except Exception:
    pg = None
import numpy as np


class LMVolume:
    """
    A simple 3D image container with orientation-aware slice extraction and
    voxel/world coordinate helpers.

    Public attributes:
        parent, file_type, full_file_name, file_path, file_name, file_base_name,
        data, data_type, canonical_nifti, header, original_nifti_header, original_nifti_affine,
        dx, dy, dz, num_rows, num_cols, num_slices, scan_direction,
        data_min, data_max, transform, x_dir, y_dir, z_dir,
        origin, resolution, shape, visible

    Public methods:
        populate_with_dicom(...)
        populate_with_nifti(nifti_image, full_path_name, base_name=None, settings_dict=None)
        get_slice(view, slice_num)
        _get_x_slice(slice_num)
        _get_y_slice(slice_num)
        _get_z_slice(slice_num)
        screenxy_to_voxelijk(slice_orientation, plot_col, plot_row, slice_index)
        voxelijk_to_screenxy(slice_orientation, voxel_x, voxel_y, voxel_z)

    Private helpers (API-neutral):
        _slice_2d(view, index)
        voxel_to_world(ijk)
        world_to_voxel(xyz)

    Key points:
      - No data/name are passed in __init__; populate_with_nifti() handles core fields.
      - All *display* state lives here: is_discrete, LUT, opacity, window (display_range),
        and colormap selection (name/index).
      - Discrete labels may be non-contiguous; we keep value<->row mappings.
      - LUTs are always uint8 (N x 4), ready for pyqtgraph ImageItem.setLookupTable(lut).
    """

    # ------------------------- init -------------------------

    def __init__(
        self,
        _parent,
        *,
        default_discrete_palette: Optional[np.ndarray] = None,
        continuous_palette_names: Optional[Sequence[str]] = None,
        _is_discrete: bool = False,
    ):

        # identification
        self.parent = _parent
        self.file_type = ''  # 'dicom' or 'nifti'
        self.full_file_name = ''  # full path and name of the dataset
        self.file_path = ''  # path to the dataset
        self.file_name = ''  # file name, including extension(s)
        self.file_base_name = ''  # file name without extension(s)

        # voxel data & data type
        self.data: np.ndarray | None = None
        self.data_type = None

        # keep reference to canonical NiBabel image
        self.canonical_nifti: nib.Nifti1Image | None = None

        # geometry & stats
        self.header = None
        self.original_nifti_header = None  # header before canonicalization
        self.original_nifti_affine = None  # affine matrix before canonicalization
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.num_rows = 1
        self.num_cols = 1
        self.num_slices = 1
        self.scan_direction = None
        self.data_min = None
        self.data_max = None

        # orientation
        self.transform = np.eye(4)  # affine ijk→world
        self.x_dir = None  # 'R' or 'L'
        self.y_dir = None  # 'A' or 'P'
        self.z_dir = None  # 'S' or 'I'

        self.origin = None
        self.resolution = None
        self.shape = None

        self.visible = True

        # Classification
        self.is_discrete: bool = bool(_is_discrete)

        self.uid = None

        # Discrete bookkeeping (filled during populate_with_nifti / registered)
        self.labels: Optional[np.ndarray] = None                     # sorted unique label values (int64)
        self.label_to_index: Dict[int, int] = {}                     # label value -> LUT row
        self.index_to_label: Optional[np.ndarray] = None             # LUT row -> label value
        self.null_value: Optional[int] = 0                           # conventional background

        # Display state
        self.lut: Optional[np.ndarray] = None                        # uint8 (N x 4)
        self.opacity: float = 1.0
        self.blend_opacity: float = 1.0  # Separate opacity for blending in registered viewports (default 1.0)
        self._display_range: Optional[Tuple[float, float]] = None    # continuous window (lo, hi)
        self.colormap_name: Optional[str] = None                     # continuous selection name
        self.colormap_index: Optional[int] = None                    # index into continuous palette
        self.interpolation: str = "nearest" if self.is_discrete else "smooth"

        # Palettes
        self._default_discrete_palette: Optional[np.ndarray] = None
        if default_discrete_palette is not None:
            pal = np.asarray(default_discrete_palette)
            if pal.ndim != 2 or pal.shape[1] != 4 or pal.dtype != np.uint8:
                raise ValueError("default_discrete_palette must be (N x 4) uint8.")
            self._default_discrete_palette = pal

        self._continuous_palette_names: Tuple[str, ...] = tuple(continuous_palette_names or ())

        # Rebuild hints (optional)
        self._lut_dirty: bool = True
        self._palette_dirty: bool = False

        # Registration references
        self.dfield = None
        self.fixed_vol_reference = None
        self.moving_vol_reference = None

    # ------------------------- properties -------------------------

    @property
    def display_range(self) -> Optional[Tuple[float, float]]:
        return self._display_range

    @display_range.setter
    def display_range(self, rng: Optional[Tuple[float, float]]):
        if rng is None:
            self._display_range = None
            return
        lo, hi = float(rng[0]), float(rng[1])
        if hi < lo:
            lo, hi = hi, lo
        self._display_range = (lo, hi)

    @property
    def display_min(self) -> Optional[float]:
        return None if self._display_range is None else self._display_range[0]

    @display_min.setter
    def display_min(self, v: float):
        if self._display_range is None:
            self._display_range = (float(v), float(v))
        else:
            self._display_range = (float(v), float(self._display_range[1]))

    @property
    def display_max(self) -> Optional[float]:
        return None if self._display_range is None else self._display_range[1]

    @display_max.setter
    def display_max(self, v: float):
        if self._display_range is None:
            self._display_range = (float(v), float(v))
        else:
            self._display_range = (float(self._display_range[0]), float(v))

    # ------------------------- public display helpers -------------------------

    def set_opacity(self, value: float) -> None:
        self.opacity = max(0.0, min(1.0, float(value)))

    def set_display_range(self, lo: float, hi: float) -> None:
        self.display_range = (float(lo), float(hi))

    # ------------------------- populate methods -------------------------

    def populate_with_nifti(self, nifti_image, full_path_name, base_name=None, uid_meta=None, convention='RAS'):
        """
        Populate from a NIfTI image using NiBabel.

        Parameters
        ----------
        nifti_image : nib.Nifti1Image
        full_path_name : str
        base_name : Optional[str]
        """
        # For neuro/medical images (especially NIfTI), the standard most tools use is RAS+:
        # - Right is the positive x direction
        # - Anterior is the positive y direction
        # - Superior is the positive z direction
        # as_closest_canonical() will flip and/or permute axes to match this convention.
        # The affine transform will be adjusted accordingly.
        # There is no resampling or data type conversion here; the original data type is preserved.

        # Store original header and affine before canonicalization
        self.original_nifti_header = nifti_image.header.copy()
        self.original_nifti_affine = nifti_image.affine.copy()

        # TODO: support other conventions if needed
        # if convention == 'RAS':
        self.canonical_nifti = nib.as_closest_canonical(nifti_image)
        # else:
        #     pass

        # Load voxel data eagerly for interactive use; preserve on-disk dtype
        self.data = np.asanyarray(self.canonical_nifti.dataobj).astype(nifti_image.header.get_data_dtype())
        self.header = self.canonical_nifti.header
        self.data_type = str(self.data.dtype)

        # Filenames / types
        self.full_file_name = full_path_name
        self.file_path = os.path.dirname(full_path_name)
        self.file_name = os.path.basename(full_path_name)
        if base_name is not None:
            self.file_base_name = base_name
        else:
            base_name_only = os.path.splitext(self.file_name)[0]
            self.file_base_name = os.path.splitext(base_name_only)[0]  # handles .nii.gz
        self.file_type = 'nifti'

        # Spacing (NiBabel header pixdim is float32; cast to float for safety)
        self.dx = float(self.canonical_nifti.header['pixdim'][1:4][0])  # ROW height
        self.dy = float(self.canonical_nifti.header['pixdim'][1:4][1])  # COL width
        self.dz = float(self.canonical_nifti.header['pixdim'][1:4][2])

        # Dimensions: data shape is (cols=x, rows=y, slices=z)
        self.num_rows = int(self.data.shape[1])
        self.num_cols = int(self.data.shape[0])
        self.num_slices = int(self.data.shape[2])

        # Affine & axis codes
        self.transform = self.canonical_nifti.affine
        ax_codes = nib.orientations.aff2axcodes(self.transform)
        self.x_dir, self.y_dir, self.z_dir = ax_codes[0], ax_codes[1], ax_codes[2]

        # Min/max & geometry summaries
        self.data_min = float(np.min(self.data))
        self.data_max = float(np.max(self.data))
        self.origin = list(self.transform[:3, 3])
        self.resolution = [self.dx, self.dy, self.dz]
        self.shape = [int(s) for s in self.data.shape]

        # Extract just the image_uid from the metadata dict
        if uid_meta and isinstance(uid_meta, dict):
            self.uid = uid_meta.get("image_uid")
        else:
            self.uid = uid_meta

        if self.is_discrete:
            # Unique sorted labels (as int64) and mappings
            labels = np.asarray(sorted(np.unique(self.data.astype(np.int32))), dtype=np.int32)
            self.labels = labels
            self.index_to_label = labels.copy()
            self.label_to_index = {int(v): i for i, v in enumerate(labels)}
            self.null_value = 0

            # Build a K-row LUT from default palette (repeat/trim)
            base = self._default_discrete_palette
            self.set_discrete_colormap(base, labels=labels)

            # Window not used for label rendering—fill for UI
            self.display_range = (float(labels.min()), float(labels.max()))
            self.interpolation = "nearest"
        else:
            # Continuous: choose an initial colormap and window
            self.labels = None
            self.label_to_index = {}
            self.index_to_label = None
            self.null_value = None

            initial_name = self._continuous_palette_names[0] if len(self._continuous_palette_names) else "viridis"
            self.set_continuous_colormap(initial_name)

            dataf = np.asarray(self.data, dtype=float)
            finite = np.isfinite(dataf)
            if finite.any():
                lo = np.percentile(dataf[finite], 2.0)
                hi = np.percentile(dataf[finite], 98.0)
            else:
                lo, hi = float(np.nanmin(dataf)), float(np.nanmax(dataf))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.nanmin(dataf)), float(np.nanmax(dataf))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = 0.0, 1.0
            self.display_range = (float(lo), float(hi))
            self.interpolation = "antialiased"

        # Ensure a LUT exists for the viewer
        if self.lut is None:
            if self.is_discrete:
                self.build_lut_discrete()
            else:
                self.build_lut_continuous()

    def populate_with_dicom(self, datasets, dataset_name, is_enhanced: bool = False):
        """Populate from a DICOM series (not yet implemented in this version)."""
        # Placeholder to preserve API; implement as needed
        # Record provenance source metadata even though data loading is TBD.
        # self.provenance["source"] = {
        #     "type": "dicom",
        #     "series_name": dataset_name,
        #     "enhanced": bool(is_enhanced),
        # }
        pass

    def populate_with_registered_data(self, reg_data, fixed_vol, dfield, moving_vol):
        """
        Initialize this LMVolume with the result of registering `moving_vol` to `fixed_vol`.
        Geometry/header copied from fixed; display intent preserved from moving_vol.
        """
        # Identity
        self.file_type = "registered"
        self.file_base_name = "registered"

        self.header = fixed_vol.header
        self.fixed_vol_reference = fixed_vol
        self.moving_vol_reference = moving_vol
        self.dfield = dfield

        if getattr(self.dfield, "is_composed", False):
            tag = getattr(dfield, "unique_id_hash", "composed")
        else:
            tag = getattr(dfield, "algorithm", "registered")
        self.file_name = f"{moving_vol.file_name} --> {fixed_vol.file_name}: {tag}"
        self.full_file_name = os.path.join(getattr(moving_vol, "file_path", ""), self.file_name)

        # Data
        self.data = np.array(reg_data, copy=True)
        self.data_min = float(np.nanmin(self.data))
        self.data_max = float(np.nanmax(self.data))
        self.num_cols = int(self.data.shape[0])
        self.num_rows = int(self.data.shape[1])
        self.num_slices = int(self.data.shape[2]) if self.data.ndim >= 3 else 1

        # Geometry from fixed (shallow copy semantics preserved)
        self.transform = fixed_vol.transform
        self.x_dir = fixed_vol.x_dir
        self.y_dir = fixed_vol.y_dir
        self.z_dir = fixed_vol.z_dir
        self.dx = fixed_vol.dx
        self.dy = fixed_vol.dy
        self.dz = fixed_vol.dz

        # Display intent from moving volume
        self.is_discrete = bool(moving_vol.is_discrete)
        # Get opacity from moving volume (should be 0.0-1.0), default to 1.0 if not present
        self.opacity = float(getattr(moving_vol, "opacity", 1.0))
        # Ensure opacity is in valid range [0.0, 1.0]
        self.opacity = max(0.0, min(1.0, self.opacity))
        self.interpolation = "nearest" if self.is_discrete else "smooth"

        if self.is_discrete:
            # Recompute labels from new data; try to preserve per-label colors if label sets match.
            labels = np.asarray(sorted(np.unique(self.data.astype(np.int32))), dtype=np.int32)
            self.labels = labels
            self.index_to_label = labels.copy()
            self.label_to_index = {int(v): i for i, v in enumerate(labels)}
            self.null_value = 0

            preserved = False
            if getattr(moving_vol, "labels", None) is not None and getattr(moving_vol, "lut", None) is not None:
                mv_labels = np.asarray(moving_vol.labels)
                mv_lut = np.asarray(moving_vol.lut)
                if mv_lut.ndim == 2 and mv_lut.shape[1] == 4 and set(map(int, mv_labels)) == set(map(int, labels)):
                    lut = np.empty((labels.size, 4), dtype=np.uint8)
                    mv_map = {int(v): i for i, v in enumerate(mv_labels)}
                    for i, lab in enumerate(labels):
                        lut[i] = mv_lut[mv_map[int(lab)]]
                    self.lut = lut
                    preserved = True

            if not preserved:
                base = self._default_discrete_palette
                self.set_discrete_colormap(base, labels=labels)

            self.display_range = (float(labels.min()), float(labels.max()))
        else:
            # Continuous: keep colormap choice if available; otherwise pick a default.
            initial_name = getattr(moving_vol, "colormap_name", None)
            if not initial_name:
                palette_names = tuple(getattr(moving_vol, "_continuous_palette_names", ())) or self._continuous_palette_names
                initial_name = palette_names[0] if len(palette_names) else "viridis"
            self.set_continuous_colormap(initial_name)

            # Carry over window if set, else recompute on new data
            if getattr(moving_vol, "display_range", None):
                lo, hi = moving_vol.display_range
            else:
                dataf = self.data.astype(float, copy=False)
                finite = np.isfinite(dataf)
                if finite.any():
                    lo = np.percentile(dataf[finite], 2.0)
                    hi = np.percentile(dataf[finite], 98.0)
                else:
                    lo, hi = float(np.nanmin(dataf)), float(np.nanmax(dataf))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = float(np.nanmin(dataf)), float(np.nanmax(dataf))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
            self.display_range = (float(lo), float(hi))

        # Ensure LUT
        if self.lut is None:
            if self.is_discrete:
                self.build_lut_discrete()
            else:
                self.build_lut_continuous()

    # ------------------------- LUT builders & setters -------------------------

    def build_lut_discrete(self) -> np.ndarray:
        """
        Build a LUT of length len(labels) using the default discrete palette (repeat/trim).
        """
        if not self.is_discrete:
            raise RuntimeError("build_lut_discrete called on a continuous volume.")
        if self.labels is None or self.labels.size == 0:
            self.lut = None
            return np.zeros((0, 4), dtype=np.uint8)

        base = self._default_discrete_palette
        self.set_discrete_colormap(base, labels=self.labels)
        return self.lut

    def set_discrete_colormap(self, palette: Optional[np.ndarray], *, labels: np.ndarray):
        """
        Accepts an (N,4) uint8 base palette and expands/tiles to exactly len(labels) rows.
        Does not assume label value == LUT row.
        """
        if labels is None or labels.size == 0:
            self.lut = None
            return

        K = int(labels.size)
        if palette is None:
            # Safe fallback palette (first row transparent)
            palette = np.array(
                [
                    [0, 0, 0, 0],
                    [228, 25, 27, 255],
                    [54, 126, 184, 255],
                    [76, 175, 74, 255],
                    [151, 77, 163, 255],
                    [255, 127, 0, 255],
                    [255, 255, 51, 255],
                    [165, 85, 40, 255],
                    [246, 128, 191, 255],
                ],
                dtype=np.uint8,
            )
        base = np.asarray(palette, dtype=np.uint8)
        if base.ndim != 2 or base.shape[1] != 4:
            raise ValueError("Discrete palette must be (N x 4) uint8.")

        reps = (K + len(base) - 1) // len(base)
        lut = np.vstack([base] * reps)[:K].astype(np.uint8, copy=False)

        # Optional: make the null label transparent if present
        if self.null_value is not None and int(self.null_value) in self.label_to_index:
            lut[self.label_to_index[int(self.null_value)], 3] = np.uint8(0)

        self.lut = lut

    def build_lut_continuous(self, name: Optional[str] = None, *, n: int = 256) -> np.ndarray:
        """
        Build a continuous LUT from a named map (or current selection if name is None).
        """
        if name is not None:
            self.colormap_name = str(name)
        if not self.colormap_name:
            self.colormap_name = self._continuous_palette_names[0] if len(self._continuous_palette_names) else "viridis"

        # Index bookkeeping (not mandatory if name isn't in palette list)
        try:
            self.colormap_index = list(self._continuous_palette_names).index(self.colormap_name)
        except ValueError:
            self.colormap_index = None

        self.lut = _resolve_named_colormap_to_lut(self.colormap_name, n=n)
        return self.lut

    def set_continuous_colormap(self, name: str, *, n: int = 256):
        """
        Select and build a continuous colormap by name.
        """
        self.build_lut_continuous(name=name, n=n)

    def set_label_color(self, label_id: int, rgba_uint8) -> None:
        """
        Update a single label color in-place (discrete only).
        """
        if not self.is_discrete or self.lut is None or self.labels is None or self.index_to_label is None:
            return
        if label_id not in self.label_to_index:
            return
        row = self.label_to_index[int(label_id)]
        rgba = np.asarray(rgba_uint8, dtype=np.uint8)
        if rgba.shape != (4,):
            raise ValueError("rgba_uint8 must be length-4.")
        self.lut[row] = rgba

    # ---------------------------------------------------------------------
    # Slice extraction (public API preserved)
    # ---------------------------------------------------------------------
    def get_slice(self, view, slice_num):
        """
        Return a 2D slice for the requested view direction and slice index,
        respecting RAS display convention and stored axis codes.
        """
        if view == ViewDir.AX.dir:
            if 0 <= slice_num < self.num_slices:
                return self._get_z_slice(slice_num)
            return None
        elif view == ViewDir.SAG.dir:
            if 0 <= slice_num < self.num_cols:
                return self._get_x_slice(slice_num)
            return None
        elif view == ViewDir.COR.dir:
            if 0 <= slice_num < self.num_rows:
                return self._get_y_slice(slice_num)
            return None
        else:
            return None

    def _get_x_slice(self, slice_num):
        """SAGITTAL: slice along x index (y–z plane), with orientation flips to match RAS display."""
        # Data is (x, y, z) == (cols, rows, slices)
        if self.parent.display_convention == 'RAS':
            if self.y_dir == 'A':
                x_slice = np.flip(self.data[slice_num, :, :], axis=0)
            else:  # 'P'
                x_slice = self.data[slice_num, :, :]
        else:
            # Other conventions can be added as needed
            x_slice = self.data[slice_num, :, :]
        return x_slice

    def _get_y_slice(self, slice_num):
        """CORONAL: slice along y index (x–z plane), with orientation flips to match RAS display."""
        if self.parent.display_convention == 'RAS':
            if self.x_dir == 'R':
                y_slice = np.flipud(self.data[:, slice_num, :])
            else:  # 'L'
                y_slice = self.data[:, slice_num, :]
        else:
            y_slice = self.data[:, slice_num, :]
        return y_slice

    def _get_z_slice(self, slice_num):
        """AXIAL: slice along z index (x–y plane), with orientation flips to match RAS display."""
        if self.parent.display_convention == 'RAS':
            if self.x_dir == 'R':
                if self.y_dir == 'A':
                    z_slice = np.flipud(self.data[:, :, slice_num])
                else:  # 'P'
                    z_slice = np.flipud(np.fliplr(self.data[:, :, slice_num]))
            else:  # 'L'
                if self.y_dir == 'A':
                    z_slice = self.data[:, :, slice_num]
                else:  # 'P'
                    z_slice = np.fliplr(self.data[:, :, slice_num])
        else:
            z_slice = self.data[:, :, slice_num]
        return z_slice

    # Convenience (internal)—not used by callers, but helpful for maintenance
    def _slice_2d(self, view, index):
        if view == ViewDir.AX.dir:
            return self._get_z_slice(self._clamp_index(self.num_slices, index))
        if view == ViewDir.SAG.dir:
            return self._get_x_slice(self._clamp_index(self.num_cols, index))
        if view == ViewDir.COR.dir:
            return self._get_y_slice(self._clamp_index(self.num_rows, index))
        return None

    @staticmethod
    def _clamp_index(axis_len, idx):
        return int(np.clip(idx, 0, axis_len - 1))

    def _clamp_voxel(self, r, c, p):
        r = int(np.clip(r, 0, self.data.shape[1] - 1))
        c = int(np.clip(c, 0, self.data.shape[0] - 1))
        p = int(np.clip(p, 0, self.data.shape[2] - 1))
        return r, c, p

    # ---------------------------------------------------------------------
    # Coordinate transforms
    # ---------------------------------------------------------------------
    def voxel_to_world(self, ijk):
        """Map voxel (i,j,k) → world (x,y,z) using affine."""
        return nib.affines.apply_affine(self.transform, ijk)

    def world_to_voxel(self, xyz):
        """Map world (x,y,z) → voxel (i,j,k) using inverse affine."""
        inv_aff = np.linalg.inv(self.transform)
        return nib.affines.apply_affine(inv_aff, xyz)

    # ---------------------------------------------------------------------
    # Screen↔voxel conversions (public API preserved)
    # ---------------------------------------------------------------------
    def screenxy_to_voxelijk(self, slice_orientation, plot_col, plot_row, slice_index):
        """Given screen (x=col, y=row) in the current orientation, return voxel ijk coords into the canonical data. """
        if self.parent.display_convention == "RAS":
            if slice_orientation == ViewDir.AX.dir:
                # right on left, posterior at bottom
                if self.x_dir == 'L':
                    voxel_x = plot_col
                else:  # 'R'
                    voxel_x = self.num_cols - 1 - plot_col
                if self.y_dir == 'A':
                    voxel_y = plot_row
                else:  # 'P'
                    voxel_y = self.num_rows - 1 - plot_row
                voxel_z = slice_index
                ijk = np.array([voxel_x, voxel_y, voxel_z])
            elif slice_orientation == ViewDir.SAG.dir:
                # anterior on left, inferior at bottom
                voxel_x = slice_index
                if self.y_dir == 'A':
                    voxel_y = self.num_rows - 1 - plot_col
                else:  # 'P'
                    voxel_y = plot_col
                voxel_z = plot_row
                ijk = np.array([voxel_x, voxel_y, voxel_z])
            elif slice_orientation == ViewDir.COR.dir:
                # right on left, inferior at bottom
                if self.x_dir == 'L':
                    voxel_x = plot_col
                else:  # 'R'
                    voxel_x = self.num_cols - 1 - plot_col
                voxel_y = slice_index
                voxel_z = plot_row
                ijk = np.array([voxel_x, voxel_y, voxel_z])
            else:
                ijk = None
        else:
            # Other conventions could be implemented here
            ijk = None
        return ijk

    def voxelijk_to_screenxy(self, slice_orientation, voxel_x, voxel_y, voxel_z):
        """Inverse of screenxy_to_voxelijk for RAS: return (plot_col, plot_row, slice_index)."""
        if self.parent.display_convention == "RAS":
            if slice_orientation == ViewDir.AX.dir:
                if self.x_dir == 'L':
                    plot_col = voxel_x
                else:
                    plot_col = self.num_cols - 1 - voxel_x
                if self.y_dir == 'A':
                    plot_row = voxel_y
                else:
                    plot_row = self.num_rows - 1 - voxel_y
                slice_index = voxel_z
                xyz = np.array([plot_col, plot_row, slice_index])
            elif slice_orientation == ViewDir.SAG.dir:
                slice_index = voxel_x
                if self.y_dir == 'A':
                    plot_col = self.num_rows - 1 - voxel_y
                else:
                    plot_col = voxel_y
                plot_row = voxel_z
                xyz = np.array([plot_col, plot_row, slice_index])
            elif slice_orientation == ViewDir.COR.dir:
                if self.x_dir == 'L':
                    plot_col = voxel_x
                else:
                    plot_col = self.num_cols - 1 - voxel_x
                slice_index = voxel_y
                plot_row = voxel_z
                xyz = np.array([plot_col, plot_row, slice_index])
            else:
                xyz = None
        else:
            xyz = None
        return xyz


# ------------------------- Colormap resolution helpers -------------------------

def _resolve_named_colormap_to_lut(name: str, *, n: int = 256) -> np.ndarray:
    """
    Resolve a named colormap to a uint8 (n x 4) LUT.
    Tries, in order:
      1) pyqtgraph.colormap.get(name)
      2) matplotlib.cm.get_cmap(name)
      3) cmcrameri (e.g. 'crameri:navia' -> cmcrameri.cm.navia)
      4) colorcet
      5) cmap.Colormap(name).to_pyqtgraph()  <-- NEW fallback per your note
    """
    # ---- 1) pyqtgraph ----
    try:
        import pyqtgraph as pg
        try:
            cm = pg.colormap.get(name)  # modern API
            lut = cm.getLookupTable(nPts=n, alpha=True)
            return _ensure_u8_rgba(lut)
        except Exception:
            pass
    except Exception:
        pass

    # ---- 2) matplotlib ----
    try:
        from matplotlib import cm as _cm
        colors = _cm.get_cmap(name)(np.linspace(0.0, 1.0, n))
        return _ensure_u8_rgba(colors)
    except Exception:
        pass

    # ---- 3) cmcrameri ----
    try:
        import cmcrameri.cm as _crm
        key = name.split(":", 1)[1] if ":" in name else name
        if hasattr(_crm, key):
            colors = getattr(_crm, key)(np.linspace(0.0, 1.0, n))
            return _ensure_u8_rgba(colors)
    except Exception:
        pass

    # ---- 4) colorcet ----
    try:
        import colorcet as _cc
        candidates = [name, name.replace("-", "_"), name.replace(":", "_")]
        seq = None
        for cand in candidates:
            if hasattr(_cc, cand):
                seq = getattr(_cc, cand)
                break
            if hasattr(_cc, "palette") and cand in getattr(_cc, "palette"):
                seq = _cc.palette[cand]
                break
        if seq is not None:
            arr = np.asarray(seq, dtype=float)
            if arr.ndim == 2 and arr.shape[1] in (3, 4):
                if arr.shape[0] != n:
                    arr = _resample_colors(arr, n)
                return _ensure_u8_rgba(arr)
    except Exception:
        pass

    # ---- 5) cmap.Colormap(...).to_pyqtgraph() ----
    # Your project-specific colormap constructor (e.g., superqt-based or custom).
    try:
        import cmap  # your module providing Colormap
        try:
            cm_obj = cmap.Colormap(name)                  # construct from string
            pg_cm = cm_obj.to_pyqtgraph()                 # -> pyqtgraph.ColorMap
            lut = pg_cm.getLookupTable(nPts=n, alpha=True)
            return _ensure_u8_rgba(lut)
        except Exception:
            pass
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve colormap '{name}' via pyqtgraph/matplotlib/cmcrameri/colorcet/cmap."
    )

def _ensure_u8_rgba(colors: np.ndarray) -> np.ndarray:
    """
    Normalize arbitrary color arrays to uint8 RGBA, shape (N, 4).
    Accepts float 0..1 or 0..255, or uint8; Nx3 or Nx4.
    """
    arr = np.asarray(colors)
    if arr.ndim != 2:
        raise ValueError("Colors must be a 2D array.")

    arr = arr.astype(float, copy=False)
    vmax = np.nanmax(arr) if arr.size else 1.0
    if vmax > 1.5:  # looks like 0..255 floats
        arr = np.clip(arr / 255.0, 0.0, 1.0)

    if arr.shape[1] == 3:
        alpha = np.ones((arr.shape[0], 1), dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=1)
    elif arr.shape[1] != 4:
        raise ValueError("Colors must have 3 or 4 channels.")

    return np.clip(np.round(arr * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)

def _resample_colors(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Linear resample an NxC color table (C=3 or 4) to length n.
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError("Expected (N, 3) or (N, 4) color table.")
    if arr.shape[0] == n:
        return arr

    x_old = np.linspace(0.0, 1.0, arr.shape[0])
    x_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, arr.shape[1]), dtype=float)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(x_new, x_old, arr[:, c])
    return out
