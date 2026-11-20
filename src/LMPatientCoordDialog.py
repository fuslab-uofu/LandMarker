"""
LMPatientCoordDialog.py
Dialog for entering patient coordinates to create a landmark.

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QGroupBox


class LMPatientCoordDialog(QDialog):
    """Dialog for entering patient coordinates (R, A, S) for both moving and fixed images to create a landmark."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Landmark by Patient Coordinates")
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Enter patient coordinates (R, A, S) in millimeters for both images:")
        layout.addWidget(instructions)
        
        # Moving image coordinates group
        moving_group = QGroupBox("Moving Image Patient Coordinates")
        moving_layout = QHBoxLayout()
        
        self.moving_x_edit = QLineEdit()
        self.moving_x_edit.setPlaceholderText("R")
        moving_layout.addWidget(QLabel("R:"))
        moving_layout.addWidget(self.moving_x_edit)
        
        self.moving_y_edit = QLineEdit()
        self.moving_y_edit.setPlaceholderText("A")
        moving_layout.addWidget(QLabel("A:"))
        moving_layout.addWidget(self.moving_y_edit)
        
        self.moving_z_edit = QLineEdit()
        self.moving_z_edit.setPlaceholderText("S")
        moving_layout.addWidget(QLabel("S:"))
        moving_layout.addWidget(self.moving_z_edit)
        
        moving_group.setLayout(moving_layout)
        layout.addWidget(moving_group)
        
        # Fixed image coordinates group
        fixed_group = QGroupBox("Fixed Image Patient Coordinates")
        fixed_layout = QHBoxLayout()
        
        self.fixed_x_edit = QLineEdit()
        self.fixed_x_edit.setPlaceholderText("R")
        fixed_layout.addWidget(QLabel("R:"))
        fixed_layout.addWidget(self.fixed_x_edit)
        
        self.fixed_y_edit = QLineEdit()
        self.fixed_y_edit.setPlaceholderText("A")
        fixed_layout.addWidget(QLabel("A:"))
        fixed_layout.addWidget(self.fixed_y_edit)
        
        self.fixed_z_edit = QLineEdit()
        self.fixed_z_edit.setPlaceholderText("S")
        fixed_layout.addWidget(QLabel("S:"))
        fixed_layout.addWidget(self.fixed_z_edit)
        
        fixed_group.setLayout(fixed_layout)
        layout.addWidget(fixed_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_coordinates(self):
        """Get the entered coordinates as a tuple (moving_coords, fixed_coords) or None if invalid.
        
        Returns:
            tuple: ((moving_x, moving_y, moving_z), (fixed_x, fixed_y, fixed_z)) or None if invalid
        """
        try:
            moving_x = float(self.moving_x_edit.text())
            moving_y = float(self.moving_y_edit.text())
            moving_z = float(self.moving_z_edit.text())
            
            fixed_x = float(self.fixed_x_edit.text())
            fixed_y = float(self.fixed_y_edit.text())
            fixed_z = float(self.fixed_z_edit.text())
            
            return ([moving_x, moving_y, moving_z], [fixed_x, fixed_y, fixed_z])
        except ValueError:
            return None

