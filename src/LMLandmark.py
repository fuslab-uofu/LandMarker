"""
LMLandmark.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from PyQt5 import QtWidgets
import numpy as np


class LMLandmark(QtWidgets.QWidget):
    """Class for managing landmarks used for registration of 3D medical images.
    
    An LMLandmark consists of six sets of coordinates:
    - two (voxel ijk and patient) for the source volume,
    - two (voxel ijk and patient) for the target volume, and
    - two (voxel ijk and patient) for the registered volume.
    
    Voxel coordinates are 1x4 np arrays of int, patient coordinates are 1x4 np arrays of float.
    The fourth array entry is a 1.

    Each LMLandmark object also stores the custom marker IDs created by the viewport (source and target).
    """
    
    def __init__(self, parent, 
                 source_voxel_coords, source_patient_coords, source_marker_id,
                 target_voxel_coords, target_patient_coords, target_marker_id, 
                 registered_voxel_coords=None, registered_patient_coords=None, lm_id=0):
        super(LMLandmark, self).__init__(parent)
        
        # Identifying info
        self.parent = parent  # main window
        self.id = lm_id
        self.selected = False

        # Coordinates are NumPy arrays [col, row, slice, 1]
        # Source (moving) volume coordinates
        self.source_voxel_coords = source_voxel_coords      # int array
        self.source_patient_coords = source_patient_coords  # float array
        
        # Target (fixed) volume coordinates
        self.target_voxel_coords = target_voxel_coords
        self.target_patient_coords = target_patient_coords
        
        # Registered volume coordinates (set after registration)
        self.registered_voxel_coords = registered_voxel_coords
        self.registered_patient_coords = registered_patient_coords

        # Custom scatter plot marker IDs created by the viewport
        self.source_marker_id = source_marker_id
        self.target_marker_id = target_marker_id


    def get_coords(self, vol, space):
        """Get coordinates for the specified volume and coordinate space.
        
        Args:
            vol: The volume object (moving, fixed, or registered)
            space: Coordinate space, either "VOX" (voxel) or "PAT" (patient)
            
        Returns:
            The requested coordinate array, or None if not available.
        """
        if vol == self.parent.moving_volume:
            return self.source_voxel_coords if space == "VOX" else self.source_patient_coords
        elif vol == self.parent.fixed_volume:
            return self.target_voxel_coords if space == "VOX" else self.target_patient_coords
        else:  # vol == self.parent.registered_volume
            return self.registered_voxel_coords if space == "VOX" else self.registered_patient_coords

    def update_moving_voxel_coords(self, vol):
        """Update source (moving) voxel coordinates based on patient coordinates and volume transform.
        
        Args:
            vol: The moving volume object with transform matrix
            
        Note:
            TODO: Ensure coords is a 1x4 array with last element = 1 (or remove the 1 everywhere).
        """
        if vol is None:
            self.source_voxel_coords = None
        else:
            # Ensure patient coords are 4D homogeneous coordinates
            if len(self.source_patient_coords) == 3:
                self.source_patient_coords = np.append(self.source_patient_coords, 1)
            
            # Transform from patient space to voxel space
            coords = np.dot(np.linalg.inv(vol.transform), self.source_patient_coords)
            self.source_voxel_coords = coords.astype(int)

    def update_fixed_voxel_coords(self, vol):
        """Update target (fixed) voxel coordinates based on patient coordinates and volume transform.
        
        Args:
            vol: The fixed volume object with transform matrix
            
        Note:
            TODO: Ensure coords is a 1x4 array with last element = 1 (or remove the 1 everywhere).
        """
        if vol is None:
            self.target_voxel_coords = None
        else:
            # Ensure patient coords are 4D homogeneous coordinates
            if len(self.target_patient_coords) == 3:
                self.target_patient_coords = np.append(self.target_patient_coords, 1)
            
            # Transform from patient space to voxel space
            coords = np.dot(np.linalg.inv(vol.transform), self.target_patient_coords)
            self.target_voxel_coords = coords.astype(int)