"""
LMLandmarkManager.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import numpy as np


# -------------------------------------------------------------------------
# Helpers and types
# -------------------------------------------------------------------------

PairKey = Tuple[str, str]  # (moving_uid, fixed_uid)


def _uid_of(vol) -> str:
    """Return a normalized string UID for a volume.
    
    Args:
        vol: Volume object with a uid attribute
        
    Returns:
        Normalized string UID
        
    Raises:
        ValueError: If volume is missing a UID
    """
    u = getattr(vol, "uid", None)
    if isinstance(u, dict):
        # Handle old metadata structures
        u = u.get("image_uid") or u.get("uid") or u.get("id") or u.get("value")
    if u is None:
        raise ValueError("Volume missing UID")
    return str(u)


def pair_key_from_vols(moving_vol, fixed_vol) -> PairKey:
    """Generate a normalized (moving_uid, fixed_uid) tuple.
    
    Args:
        moving_vol: Moving volume object
        fixed_vol: Fixed volume object
        
    Returns:
        Tuple of (moving_uid, fixed_uid) as strings
    """
    return (_uid_of(moving_vol), _uid_of(fixed_vol))

# -------------------------------------------------------------------------
# LMLandmark class
# -------------------------------------------------------------------------

class LMLandmark:
    """Landmark class for the manager (non-Qt version).
    
    This is a simplified version used by LMLandmarkManager that doesn't
    inherit from QWidget. For the full Qt-enabled version, see LMLandmark.py.
    """
    
    def __init__(self, parent,
                 source_voxel_coords,
                 source_patient_coords,
                 source_marker_id,
                 target_voxel_coords,
                 target_patient_coords,
                 target_marker_id,
                 lm_id: int = 0):
        # Identifying info
        self.parent = parent  # main window reference
        self.id = int(lm_id)
        self.selected = False

        # Coordinates are NumPy arrays [col, row, slice, 1]
        # Source (moving) volume coordinates
        self.source_voxel_coords = source_voxel_coords      # int array (shape 4 or 3)
        self.source_patient_coords = source_patient_coords  # float array (shape 4 or 3)
        
        # Target (fixed) volume coordinates
        self.target_voxel_coords = target_voxel_coords
        self.target_patient_coords = target_patient_coords
        
        # Registered volume coordinates (set after registration)
        self.registered_voxel_coords = None
        self.registered_patient_coords = None

        # Custom scatter plot marker IDs created by the viewport
        self.source_marker_id = source_marker_id
        self.target_marker_id = target_marker_id

    def __repr__(self):
        """Return a string representation of the landmark for debugging."""
        def _format_coords(coords, precision=1):
            """Format coordinates for display."""
            if coords is None:
                return "None"
            if isinstance(coords, np.ndarray):
                if len(coords) >= 3:
                    return f"({coords[0]:.{precision}f}, {coords[1]:.{precision}f}, {coords[2]:.{precision}f})"
            return str(coords)
        
        selected_str = "selected" if self.selected else "unselected"
        source_vox = _format_coords(self.source_voxel_coords, precision=1)
        source_pat = _format_coords(self.source_patient_coords, precision=2)
        target_vox = _format_coords(self.target_voxel_coords, precision=1)
        target_pat = _format_coords(self.target_patient_coords, precision=2)
        reg_vox = _format_coords(self.registered_voxel_coords, precision=1) if self.registered_voxel_coords is not None else "None"
        reg_pat = _format_coords(self.registered_patient_coords, precision=2) if self.registered_patient_coords is not None else "None"
        
        return (f"LMLandmark(id={self.id}, {selected_str}, "
                f"source_vox={source_vox}, source_pat={source_pat}, "
                f"target_vox={target_vox}, target_pat={target_pat}, "
                f"registered_vox={reg_vox}, registered_pat={reg_pat}, "
                f"marker_ids=({self.source_marker_id}, {self.target_marker_id}))")

    # --- Utility methods ---
    
    def _ensure_homog4(self, arr):
        """Ensure array is 4D homogeneous coordinates [x, y, z, 1].
        
        Args:
            arr: Input array (3D or 4D, or None)
            
        Returns:
            4D homogeneous coordinate array, or None if input is None
        """
        if arr is None:
            return None
        a = np.asarray(arr).reshape(-1)
        if a.shape[0] == 3:
            a = np.append(a, 1)
        return a

    # --- Coordinate access ---
    
    def get_coords(self, vol, space):
        """Get coordinates for the specified volume and coordinate space.
        
        Args:
            vol: The volume object (moving, fixed, or registered)
            space: Coordinate space, either "VOX" (voxel) or "PAT" (patient)
            
        Returns:
            The requested coordinate array, or None if not available
        """
        if vol == self.parent.moving_volume:
            return self.source_voxel_coords if space == "VOX" else self.source_patient_coords
        elif vol == self.parent.fixed_volume:
            return self.target_voxel_coords if space == "VOX" else self.target_patient_coords
        else:  # registered volume
            return self.registered_voxel_coords if space == "VOX" else self.registered_patient_coords

    # --- Update voxel coordinates based on transform ---
    
    def update_moving_voxel_coords(self, vol):
        """Update source (moving) voxel coordinates based on patient coordinates and volume transform.
        
        Args:
            vol: The moving volume object with transform matrix, or None to clear coordinates
        """
        if vol is None:
            self.source_voxel_coords = None
            return
        self.source_patient_coords = self._ensure_homog4(self.source_patient_coords)
        coords = np.dot(np.linalg.inv(vol.transform), self.source_patient_coords)
        self.source_voxel_coords = coords.astype(int)

    def update_fixed_voxel_coords(self, vol):
        """Update target (fixed) voxel coordinates based on patient coordinates and volume transform.
        
        Args:
            vol: The fixed volume object with transform matrix, or None to clear coordinates
        """
        if vol is None:
            self.target_voxel_coords = None
            return
        self.target_patient_coords = self._ensure_homog4(self.target_patient_coords)
        coords = np.dot(np.linalg.inv(vol.transform), self.target_patient_coords)
        self.target_voxel_coords = coords.astype(int)

# -------------------------------------------------------------------------
# Landmark ID Manager
# -------------------------------------------------------------------------

class LandmarkIdManager:
    """Manages integer landmark IDs unique per (moving_uid, fixed_uid) pair."""
    
    def __init__(self, start_at: int = 1):
        """Initialize the ID manager.
        
        Args:
            start_at: Starting ID number for new landmarks (default: 1)
        """
        self._next_idx: Dict[PairKey, int] = {}
        self._in_use: Dict[PairKey, Set[int]] = {}
        self._start = int(start_at)

    def register_existing(self, pair: PairKey, lm_id: int) -> None:
        """Register an existing landmark ID to prevent reuse.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            lm_id: The landmark ID to register
        """
        d = self._in_use.setdefault(pair, set())
        lm_id = int(lm_id)
        d.add(lm_id)
        # Update next index to be at least one more than the registered ID
        self._next_idx[pair] = max(self._next_idx.get(pair, self._start), lm_id + 1)

    def next_id(self, pair: PairKey) -> int:
        """Get the next available landmark ID for the given pair.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            
        Returns:
            The next available unique landmark ID
        """
        n = self._next_idx.get(pair, self._start)
        # Skip any IDs that are already in use
        while n in self._in_use.setdefault(pair, set()):
            n += 1
        self._in_use[pair].add(n)
        self._next_idx[pair] = n + 1
        return n

    def release(self, pair: PairKey, lm_id: int) -> None:
        """Release a landmark ID so it can be reused.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            lm_id: The landmark ID to release
        """
        if pair in self._in_use:
            self._in_use[pair].discard(int(lm_id))

# -------------------------------------------------------------------------
# Main Manager
# -------------------------------------------------------------------------

class LMLandmarkManager:
    """Tracks all LMLandmark objects per (moving_uid, fixed_uid) pair."""
    
    def __init__(self, *, id_start_at: int = 1):
        """Initialize the landmark manager.
        
        Args:
            id_start_at: Starting ID number for new landmarks (default: 1)
        """
        self._by_pair: Dict[PairKey, List[LMLandmark]] = {}
        self.id_mgr = LandmarkIdManager(start_at=id_start_at)

    # --- Internal normalizer ---
    
    @staticmethod
    def _as_pairkey(pair: Any) -> PairKey:
        """Ensure pair is always a (str, str) tuple.
        
        Args:
            pair: Input that should be a 2-tuple of UIDs
            
        Returns:
            Normalized (str, str) tuple
            
        Raises:
            TypeError: If pair is not a 2-tuple
        """
        if isinstance(pair, tuple) and len(pair) == 2:
            return (str(pair[0]), str(pair[1]))
        raise TypeError("pair must be a 2-tuple (moving_uid, fixed_uid)")

    # --- Pair management ---
    
    def ensure_pair(self, pair: Any) -> None:
        """Ensure a pair exists in the manager (creates empty list if needed).
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
        """
        pair = self._as_pairkey(pair)
        self._by_pair.setdefault(pair, [])

    def pairs(self) -> Iterable[PairKey]:
        """Get all pair keys that have landmarks.
        
        Returns:
            Iterable of (moving_uid, fixed_uid) pair keys
        """
        return self._by_pair.keys()

    def has_pair(self, pair: Any) -> bool:
        """Check if a pair exists in the manager.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            
        Returns:
            True if the pair exists, False otherwise
        """
        pair = self._as_pairkey(pair)
        return pair in self._by_pair

    # --- CRUD operations ---
    
    def list(self, pair: Any) -> List[LMLandmark]:
        """Get all landmarks for a given pair.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            
        Returns:
            List of LMLandmark objects for the pair
        """
        pair = self._as_pairkey(pair)
        return list(self._by_pair.get(pair, []))

    def get(self, pair: Any, lm_id: int) -> Optional[LMLandmark]:
        """Get a specific landmark by ID.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            lm_id: The landmark ID to retrieve
            
        Returns:
            The LMLandmark object if found, None otherwise
        """
        pair = self._as_pairkey(pair)
        lm_id = int(lm_id)
        for lm in self._by_pair.get(pair, []):
            if lm.id == lm_id:
                return lm
        return None

    def add(self, pair: Any,
            parent,
            *,
            source_voxel_coords,
            source_patient_coords,
            source_marker_id,
            target_voxel_coords,
            target_patient_coords,
            target_marker_id,
            lm_id: Optional[int] = None) -> LMLandmark:
        """Add a new landmark to the manager.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            parent: Parent window reference
            source_voxel_coords: Source voxel coordinates
            source_patient_coords: Source patient coordinates
            source_marker_id: Source marker ID
            target_voxel_coords: Target voxel coordinates
            target_patient_coords: Target patient coordinates
            target_marker_id: Target marker ID
            lm_id: Optional landmark ID (auto-generated if None)
            
        Returns:
            The newly created LMLandmark object
            
        Raises:
            ValueError: If lm_id is provided and already exists
        """
        pair = self._as_pairkey(pair)
        self.ensure_pair(pair)

        if lm_id is None:
            lm_id = self.id_mgr.next_id(pair)
        else:
            lm_id = int(lm_id)
            if self.get(pair, lm_id) is not None:
                raise ValueError(f"Duplicate landmark id in pair {pair}: {lm_id}")
            self.id_mgr.register_existing(pair, lm_id)

        lm = LMLandmark(
            parent=parent,
            source_voxel_coords=source_voxel_coords,
            source_patient_coords=source_patient_coords,
            source_marker_id=source_marker_id,
            target_voxel_coords=target_voxel_coords,
            target_patient_coords=target_patient_coords,
            target_marker_id=target_marker_id,
            lm_id=lm_id,
        )
        self._by_pair[pair].append(lm)
        return lm

    def remove(self, pair: Any, lm_id: int) -> bool:
        """Remove a landmark from the manager.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            lm_id: The landmark ID to remove
            
        Returns:
            True if landmark was found and removed, False otherwise
        """
        pair = self._as_pairkey(pair)
        lst = self._by_pair.get(pair)
        if not lst:
            return False
        lm_id = int(lm_id)
        for i, lm in enumerate(lst):
            if lm.id == lm_id:
                del lst[i]
                self.id_mgr.release(pair, lm_id)
                return True
        return False

    def rename(self, pair: Any, old_id: int, new_id: int) -> None:
        """Rename a landmark (change its ID).
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            old_id: Current landmark ID
            new_id: New landmark ID
            
        Raises:
            ValueError: If new_id already exists
            KeyError: If old_id doesn't exist
        """
        pair = self._as_pairkey(pair)
        old_id = int(old_id)
        new_id = int(new_id)
        if self.get(pair, new_id) is not None:
            raise ValueError(f"Target id already exists in pair {pair}: {new_id}")
        lm = self.get(pair, old_id)
        if lm is None:
            raise KeyError(old_id)
        lm.id = new_id
        self.id_mgr.register_existing(pair, new_id)
        self.id_mgr.release(pair, old_id)

    def clear_pair(self, pair: Any) -> None:
        """Remove all landmarks for a given pair.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
        """
        pair = self._as_pairkey(pair)
        self._by_pair.pop(pair, None)

    def replace_all(self, pair: Any, landmarks: Iterable[LMLandmark]) -> None:
        """Replace all landmarks for a pair with a new set.
        
        Args:
            pair: The (moving_uid, fixed_uid) pair key
            landmarks: Iterable of LMLandmark objects to replace existing ones
        """
        pair = self._as_pairkey(pair)
        self._by_pair[pair] = []
        for lm in landmarks:
            self.add(
                pair,
                lm.parent,
                source_voxel_coords=lm.source_voxel_coords,
                source_patient_coords=lm.source_patient_coords,
                source_marker_id=lm.source_marker_id,
                target_voxel_coords=lm.target_voxel_coords,
                target_patient_coords=lm.target_patient_coords,
                target_marker_id=lm.target_marker_id,
                lm_id=lm.id,
            )

    def landmarks_by_pair(self) -> Dict[PairKey, List[LMLandmark]]:
        """Return the full mapping of pairs to landmarks.
        
        Returns:
            Dictionary mapping (moving_uid, fixed_uid) pairs to lists of landmarks
        """
        return self._by_pair
