""" LandMarker.py
    LandMarker, 2023-2026
    Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
    Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

    Copyright (c) 2026 University of Utah
    Licensed under the MIT License. See LICENSE file for details.

    LandMarker is an application for registering 3D medical images to one another using a choice of neural network-based 
    algorithms. Some of the algorithms are intensity-based, and the others use landmarks for the registration. Therefore 
    this application allows the user to interactively place landmarks on the images.



    TODO: DICOM file support
    TODO: LPS display convention
    
"""
import os
from dotenv import load_dotenv
import sys
import datetime
import io
import contextlib
import cmap
from pathlib import Path
import re

# Add parent directory to path so we can import from ui directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import pydicom
from pydicom.dataset import FileDataset
import nibabel as nib
import nilearn.image as nlimg
import json
import webbrowser
from pathlib import Path
import argparse
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QElapsedTimer, QTimer, QThread
from PyQt5.QtWidgets import (QTableWidgetItem, QMessageBox, QActionGroup, QApplication, QMainWindow, QLabel,
                             QVBoxLayout, QStackedWidget, QInputDialog, QDialog, QFontDialog, QCheckBox)
from PyQt5.QtGui import QIcon

from LMMainWindow import Ui_MainWindow
from LMVolume import LMVolume
import LMLandmarkManager
from LMLandmarkManager import pair_key_from_vols
from LMTable import LMTable
from LMEnumerations import Algorithm
from LMRegistrationParams import LMRegistrationParams
from LMPatientCoordDialog import LMPatientCoordDialog
from LMDeformationField import LMDeformationField, compute_unique_id_hash
from LMdFieldManager import LMdFieldManager
from LMUIDUtils import compute_image_uid_from_nifti
from LMPlatformUtils import get_platform_regular_slider_stylesheet
import LMSidecarUtils
# from LMVolumeSelector import LMVolumeSelector
from IntensityBasedRegistration import IntensityBasedRegistration, RegistrationCancelled
from LandmarkBasedRegistration import *

from LMEnumerations import ViewDir

from interaction_method import InteractionMethod
from colormap_combo import ColormapCombo
from viewport import Viewport
from display_settings import DisplaySettings
from active_combo import ActiveComboController


def set_hidpi_safely():
    # trying to deal with teeny-tiny text and widgets on high-res display. For example, 189 dpi
    # only set before app exists; for example, in tests, it already exists.
    if QtWidgets.QApplication.instance() is None:
        if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class RegisteredViewportsWindow(QMainWindow):
    """ A separate window (QMainWindow) that displays the registered image viewports.
        Displaying this window is optional for the user. Simply transfers the frame containing the registered
        viewports from the main window to this window.
    """

    def __init__(self, frame, main_window):
        super().__init__()
        self.setCentralWidget(frame)
        self.frame = frame
        self.main_window = main_window

    def closeEvent(self, event):
        # Check if main window is closing - if so, don't re-dock, just close
        if hasattr(self.main_window, '_closing') and self.main_window._closing:
            event.accept()
            return
        
        # Normal close: re-dock the frame back to main window
        self.frame.setParent(self.main_window.ui.centralwidget)  # FIXME: is this correct?
        self.main_window.ui.expand_window_button.setIcon(QIcon(str(icons_dir / "expandIcon.svg")))
        # self.main_window.ui.centralwidget.layout().addWidget(self.frame)
        self.main_window.ui.main_splitter.addWidget(self.frame)
        self.main_window.reg_vol_window = None
        event.accept()


class VerticalLabel(QLabel):
    """
    This class is simply a QLabel that displays text vertically.
    """
    def __init__(self, *args):
        QLabel.__init__(self, *args)
        self.setAlignment(QtCore.Qt.AlignCenter)  # type: ignore
        self.setAlignment(QtCore.Qt.AlignBottom)  # type: ignore

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        # calculate the size of the font
        fm = QtGui.QFontMetrics(painter.font())
        xoffset = int(fm.boundingRect(self.text()).width() / 2)
        yoffset = int(fm.boundingRect(self.text()).height() / 2)
        # x = int(self.width() / 2) + yoffset # MMKline - this is the original line
        x = int(self.width() / 2)   # MMKline - this is the updated line. Trying to get text to center left-right
        y = int(self.height() / 2) - xoffset
        # because we rotated the label, x affects the vertical placement, and y affects the horizontal
        painter.drawText(y, x, self.text())
        painter.end()

    def minimumSizeHint(self):
        size = QLabel.minimumSizeHint(self)
        return QtCore.QSize(size.height(), size.width())

    def sizeHint(self):
        size = QLabel.sizeHint(self)
        return QtCore.QSize(size.height(), size.width())


class WaitDialog(QDialog):
    """Modal wait dialog for registration or loading: shows output, optional Cancel, then OK on completion."""
    def __init__(self, parent=None, message="Registering images. Please wait...", show_cancel_button=False, title=None):
        super().__init__(parent)
        self.setWindowTitle(title if title is not None else "Please wait...")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        layout = QVBoxLayout(self)

        # Initial message label
        self.initial_label = QLabel(message, self)
        self.initial_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.initial_label)

        # Text area for displaying messages
        from PyQt5.QtWidgets import QTextEdit, QPushButton
        self.message_text = QTextEdit(self)
        self.message_text.setReadOnly(True)
        self.message_text.setMaximumHeight(150)
        self.message_text.setFont(QtGui.QFont("Courier", 9))
        self.message_text.setVisible(False)  # Initially hidden
        layout.addWidget(self.message_text)

        # Optional Cancel button (e.g. for intensity-based registration)
        self._cancel_btn = None
        self._cancel_callback = None
        if show_cancel_button:
            self._cancel_btn = QPushButton("Cancel", self)
            self._cancel_btn.clicked.connect(self._on_cancel_clicked)
            layout.addWidget(self._cancel_btn)

        self.resize(500, 200)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._ok_button = None  # added in show_success() / show_cancelled() / show_error()
        self._allow_close = False

    def set_cancel_callback(self, callback):
        self._cancel_callback = callback

    def _on_cancel_clicked(self):
        if self._cancel_callback:
            self._cancel_callback()
        if self._cancel_btn:
            self._cancel_btn.setEnabled(False)
        self.initial_label.setText("Cancelling...")

    def closeEvent(self, event):
        if not self._allow_close:
            event.ignore()
            return
        super().closeEvent(event)

    def allow_close(self):
        """Allow the dialog to be closed (by X or programmatic accept/reject)."""
        self._allow_close = True
        if self._cancel_btn:
            self._cancel_btn.setEnabled(False)

    def _show_done_state(self, title, summary_text):
        """Common final state: show text area, summary, hide Cancel, show OK."""
        self._allow_close = True
        self.setWindowTitle(title)
        self.initial_label.setVisible(False)
        self.message_text.setVisible(True)
        if summary_text:
            self.message_text.append(summary_text)
        if self._cancel_btn is not None:
            self._cancel_btn.setVisible(False)
        if self._ok_button is None:
            from PyQt5.QtWidgets import QPushButton
            self._ok_button = QPushButton("OK", self)
            self._ok_button.clicked.connect(self.accept)
            layout = self.layout()
            if layout is not None:
                layout.addWidget(self._ok_button)
        self._ok_button.setVisible(True)
        QApplication.processEvents()

    def show_success(self, moving_name, fixed_name, algorithm_name, device):
        """Switch to success state: summary and OK button."""
        self._allow_close = True
        self.setWindowTitle("Registration successful")
        self.initial_label.setVisible(False)
        self.message_text.setVisible(True)
        self.message_text.append(
            "\nRegistration completed successfully.\n\n"
            "Moving image:  {}\n"
            "Fixed image:   {}\n"
            "Algorithm:    {}\n"
            "Device:       {}"
            .format(moving_name, fixed_name, algorithm_name, device)
        )
        if self._cancel_btn is not None:
            self._cancel_btn.setVisible(False)
        if self._ok_button is None:
            from PyQt5.QtWidgets import QPushButton
            self._ok_button = QPushButton("OK", self)
            self._ok_button.clicked.connect(self.accept)
            layout = self.layout()
            if layout is not None:
                layout.addWidget(self._ok_button)
        self._ok_button.setVisible(True)
        QApplication.processEvents()

    def show_cancelled(self):
        """Switch to cancelled state: message and OK button."""
        self._show_done_state("Registration cancelled", "\nRegistration cancelled by user.")

    def show_error(self, err_msg):
        """Switch to error state: append error and OK button."""
        self._show_done_state("Registration error", "\nError: " + (err_msg or "Unknown error."))

    def append_message(self, message):
        """Append a message to the dialog and show the text area."""
        if not self.message_text.isVisible():
            self.message_text.setVisible(True)
            self.initial_label.setVisible(False)
        self.message_text.append(message)
        self.message_text.verticalScrollBar().setValue(
            self.message_text.verticalScrollBar().maximum()
        )
        QApplication.processEvents()


class RegistrationWorker(QThread):
    """Runs calc_deformation in a background thread and supports cancellation."""
    finished = pyqtSignal(object, object, bool, object)  # dfield, dmatrix, cancelled, error_msg
    output_line = pyqtSignal(str)  # lines from stdout during registration (for dialog)

    def __init__(self, main_window, landmarks, moving_vol, fixed_vol, algorithm, params, device, cancel_token):
        super().__init__(main_window)
        self.main_window = main_window
        self.landmarks = landmarks
        self.moving_vol = moving_vol
        self.fixed_vol = fixed_vol
        self.algorithm = algorithm
        self.params = params
        self.device = device
        self.cancel_token = cancel_token

    def run(self):
        err_msg = None
        cancelled = False
        dfield = None
        dmatrix = None
        # Redirect stdout so print() from registration is emitted to the wait dialog
        old_stdout = sys.stdout

        class LineEmitter:
            def __init__(self, worker):
                self.worker = worker
                self.buf = ""

            def write(self, s):
                self.buf += s
                while "\n" in self.buf or "\r" in self.buf:
                    for sep in ("\n", "\r"):
                        if sep in self.buf:
                            line, self.buf = self.buf.split(sep, 1)
                            t = line.strip()
                            if t:
                                self.worker.output_line.emit(t)
                            break

            def flush(self):
                pass

        emitter = LineEmitter(self)
        sys.stdout = emitter
        try:
            dfield, dmatrix = self.main_window.calc_deformation(
                self.landmarks, self.moving_vol, self.fixed_vol,
                self.algorithm, self.params, self.device,
                cancel_check=lambda: getattr(self.cancel_token, "cancelled", False))
        except RegistrationCancelled:
            cancelled = True
        except Exception as e:
            err_msg = str(e)
        finally:
            sys.stdout = old_stdout
            if emitter.buf.strip():
                self.output_line.emit(emitter.buf.strip())
        self.finished.emit(dfield, dmatrix, cancelled, err_msg)


class OutputCapture:
    """Context manager to capture stdout/stderr and route to multiple destinations."""
    def __init__(self, wait_dialog=None, status_callback=None, log_callback=None):
        self.wait_dialog = wait_dialog
        self.status_callback = status_callback
        self.log_callback = log_callback
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.captured_output = io.StringIO()
    
    def write(self, text):
        """Write to all destinations."""
        if text.strip():  # Only process non-empty lines
            # Write to captured output
            self.captured_output.write(text)
            
            # Update wait dialog
            if self.wait_dialog:
                self.wait_dialog.append_message(text.strip())
            
            # Update status label and log file
            if self.status_callback:
                self.status_callback(text.strip())
            
            # Log to file
            if self.log_callback:
                self.log_callback(text.strip())
    
    def flush(self):
        """Flush all outputs."""
        self.captured_output.flush()
        if self.original_stdout:
            self.original_stdout.flush()
        if self.original_stderr:
            self.original_stderr.flush()
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return False
    
    def get_output(self):
        """Get all captured output as a string."""
        return self.captured_output.getvalue()


class VolumeHeaderInfoDialog(QDialog):
    """
    Custom dialog to display comprehensive Image Header Information.
    Shows file name, path, type, data type, dimensions, resolution, direction cosines, and origin.
    """
    def __init__(self, volume, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Header Information")
        self.setModal(False)
        self.resize(600, 700)
        
        layout = QVBoxLayout(self)
        
        # Create scrollable text area
        from PyQt5.QtWidgets import QTextEdit, QScrollArea
        text_edit = QTextEdit(self)
        text_edit.setReadOnly(True)
        text_edit.setFont(QtGui.QFont("Courier", 9))
        
        # Build the information string
        info_text = self._build_info_text(volume)
        text_edit.setPlainText(info_text)
        
        layout.addWidget(text_edit)
        
        # Add close button
        from PyQt5.QtWidgets import QPushButton, QHBoxLayout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
    
    def _build_info_text(self, volume):
        """Build a formatted string with all image information."""
        lines = []
        
        # File Information
        lines.append("=" * 70)
        lines.append("FILE INFORMATION")
        lines.append("=" * 70)
        
        file_name = getattr(volume, 'file_name', 'N/A')
        lines.append(f"File Name:        {file_name}")
        
        full_file_name = getattr(volume, 'full_file_name', None)
        if full_file_name:
            lines.append(f"Full Path:        {full_file_name}")
        else:
            file_path = getattr(volume, 'file_path', None)
            if file_path:
                lines.append(f"File Path:        {file_path}")
        
        file_type = getattr(volume, 'file_type', None)
        if file_type:
            lines.append(f"File Type:        {file_type}")
        
        file_base_name = getattr(volume, 'file_base_name', None)
        if file_base_name:
            lines.append(f"Base Name:        {file_base_name}")
        
        # LandMarker UID
        uid = getattr(volume, 'uid', None)
        if uid:
            lines.append(f"LandMarker UID:   {uid}")
        else:
            lines.append("LandMarker UID:   N/A (not generated)")
        
        lines.append("")
        
        # Data Information
        lines.append("=" * 70)
        lines.append("DATA INFORMATION")
        lines.append("=" * 70)
        
        if hasattr(volume, 'data') and volume.data is not None:
            data_type = str(volume.data.dtype)
            lines.append(f"Data Type:        {data_type}")
            
            data_min = getattr(volume, 'data_min', None)
            data_max = getattr(volume, 'data_max', None)
            if data_min is not None and data_max is not None:
                lines.append(f"Data Range:       [{data_min:.6f}, {data_max:.6f}]")
        else:
            lines.append("Data Type:        N/A")
        
        lines.append("")
        
        # Image Dimensions
        lines.append("=" * 70)
        lines.append("IMAGE DIMENSIONS")
        lines.append("=" * 70)
        
        if hasattr(volume, 'data') and volume.data is not None:
            num_cols = getattr(volume, 'num_cols', None)
            num_rows = getattr(volume, 'num_rows', None)
            num_slices = getattr(volume, 'num_slices', None)
            
            if num_cols is not None and num_rows is not None and num_slices is not None:
                lines.append(f"Dimensions:       {num_cols} x {num_rows} x {num_slices} (cols x rows x slices)")
            else:
                shape = volume.data.shape
                if len(shape) >= 3:
                    lines.append(f"Dimensions:       {shape[0]} x {shape[1]} x {shape[2]} (cols x rows x slices)")
                else:
                    lines.append("Dimensions:       N/A")
        else:
            lines.append("Dimensions:       N/A")
        
        lines.append("")
        
        # Image Spacing
        lines.append("=" * 70)
        lines.append("IMAGE SPACING")
        lines.append("=" * 70)
        
        dx = getattr(volume, 'dx', None)
        dy = getattr(volume, 'dy', None)
        dz = getattr(volume, 'dz', None)
        
        if dx is not None and dy is not None and dz is not None:
            lines.append(f"Voxel Spacing:    {dx:.6f} x {dy:.6f} x {dz:.6f} mm")
        else:
            lines.append("Voxel Spacing:    N/A")
        
        lines.append("")
        
        # Direction Cosines
        lines.append("=" * 70)
        lines.append("COORDINATE SYSTEM CONVENTION")
        lines.append("=" * 70)
        
        x_dir = getattr(volume, 'x_dir', None)
        y_dir = getattr(volume, 'y_dir', None)
        z_dir = getattr(volume, 'z_dir', None)
        
        if x_dir is not None:
            if isinstance(x_dir, np.ndarray):
                lines.append(f"X Direction:      [{x_dir[0]:.6f}, {x_dir[1]:.6f}, {x_dir[2]:.6f}]")
            else:
                lines.append(f"X Direction:      {x_dir}")
        else:
            lines.append("X Direction:      N/A")
        
        if y_dir is not None:
            if isinstance(y_dir, np.ndarray):
                lines.append(f"Y Direction:      [{y_dir[0]:.6f}, {y_dir[1]:.6f}, {y_dir[2]:.6f}]")
            else:
                lines.append(f"Y Direction:      {y_dir}")
        else:
            lines.append("Y Direction:      N/A")
        
        if z_dir is not None:
            if isinstance(z_dir, np.ndarray):
                lines.append(f"Z Direction:      [{z_dir[0]:.6f}, {z_dir[1]:.6f}, {z_dir[2]:.6f}]")
            else:
                lines.append(f"Z Direction:      {z_dir}")
        else:
            lines.append("Z Direction:      N/A")
        
        lines.append("")
        
        # Image Origin (from original NIfTI affine, before canonicalization)
        lines.append("=" * 70)
        lines.append("IMAGE ORIGIN")
        lines.append("=" * 70)
        
        # Use original affine if available (before canonicalization), otherwise fall back to transform
        original_affine = getattr(volume, 'original_nifti_affine', None)
        if original_affine is not None and isinstance(original_affine, np.ndarray) and original_affine.shape == (4, 4):
            # Origin is the translation part of the original affine matrix (last column, first 3 elements)
            original_origin = original_affine[:3, 3]
            lines.append(f"Image Origin:     [{original_origin[0]:.6f}, {original_origin[1]:.6f}, {original_origin[2]:.6f}] mm")
        else:
            lines.append("Image Origin:     N/A (original affine not available)")
        
        lines.append("")
        
        # Direction Cosines (computed from original NIfTI affine, before canonicalization)
        lines.append("=" * 70)
        lines.append("DIRECTION COSINES (from original NIfTI affine)")
        lines.append("=" * 70)
        
        # Use original affine if available (before canonicalization), otherwise fall back to transform
        original_affine = getattr(volume, 'original_nifti_affine', None)
        if original_affine is not None and isinstance(original_affine, np.ndarray) and original_affine.shape == (4, 4):
            transform = original_affine
        else:
            transform = getattr(volume, 'transform', None)
        
        if transform is not None and isinstance(transform, np.ndarray) and transform.shape == (4, 4):
            # Extract the 3x3 rotation/scale block from the affine
            M = transform[:3, :3]
            
            # Extract column vectors (each column corresponds to i, j, k axes)
            dx = M[:, 0]  # i-axis direction
            dy = M[:, 1]  # j-axis direction
            dz = M[:, 2]  # k-axis direction
            
            # Normalize to get direction cosines
            try:
                norm_dx = np.linalg.norm(dx)
                norm_dy = np.linalg.norm(dy)
                norm_dz = np.linalg.norm(dz)
                
                if norm_dx > 0 and norm_dy > 0 and norm_dz > 0:
                    dir_x = dx / norm_dx
                    dir_y = dy / norm_dy
                    dir_z = dz / norm_dz
                    
                    # Assemble as a matrix (column vectors)
                    direction_cosines = np.column_stack((dir_x, dir_y, dir_z))
                    
                    lines.append("Direction Cosines Matrix (3x3):")
                    lines.append("  Column 0 → i-axis, Column 1 → j-axis, Column 2 → k-axis")
                    lines.append("")
                    for i in range(3):
                        lines.append(f"  [{direction_cosines[i,0]:10.6f} {direction_cosines[i,1]:10.6f} {direction_cosines[i,2]:10.6f}]")
                else:
                    lines.append("Direction Cosines: N/A (zero-length column vectors in affine)")
            except Exception as e:
                lines.append(f"Direction Cosines: Error computing ({str(e)})")
        else:
            lines.append("Direction Cosines: N/A (affine matrix not available)")
        
        lines.append("")
        
        # Affine Matrix (canonical, after transformation)
        lines.append("=" * 70)
        lines.append("AFFINE (canonical, after transformation)")
        lines.append("=" * 70)
        
        canonical_transform = getattr(volume, 'transform', None)
        if canonical_transform is not None and isinstance(canonical_transform, np.ndarray) and canonical_transform.shape == (4, 4):
            lines.append("Affine Matrix (4x4):")
            for i in range(4):
                lines.append(f"  [{canonical_transform[i,0]:12.6f} {canonical_transform[i,1]:12.6f} {canonical_transform[i,2]:12.6f} {canonical_transform[i,3]:12.6f}]")
        else:
            lines.append("Affine Matrix: N/A")
        
        lines.append("")
        
        # Origin
        lines.append("=" * 70)
        lines.append("ORIGIN")
        lines.append("=" * 70)
        
        if canonical_transform is not None and isinstance(canonical_transform, np.ndarray) and canonical_transform.shape == (4, 4):
            # Origin is the translation part of the transform matrix (last column, first 3 elements)
            origin = canonical_transform[:3, 3]
            lines.append(f"Origin:           [{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}] mm")
        else:
            lines.append("Origin:           N/A")
        
        lines.append("")
        
        # Additional Header Information
        header = getattr(volume, 'header', None)
        if header is not None and isinstance(header, dict):
            lines.append("=" * 70)
            lines.append("ADDITIONAL HEADER INFORMATION")
            lines.append("=" * 70)
            
            # Display header dictionary items
            for key, value in sorted(header.items()):
                if isinstance(value, (list, tuple, np.ndarray)):
                    if isinstance(value, np.ndarray) and value.size > 10:
                        lines.append(f"{key:20s}: {str(value.dtype)} array, shape {value.shape}")
                    else:
                        lines.append(f"{key:20s}: {value}")
                elif isinstance(value, dict):
                    lines.append(f"{key:20s}: (dictionary with {len(value)} items)")
                else:
                    lines.append(f"{key:20s}: {value}")
        
        return "\n".join(lines)

# ----------------------------- small local helpers -----------------------------

def cmap_from_pg(pg_color_map, n: int = 256) -> cmap.Colormap:
    """
    Convert a pyqtgraph.ColorMap -> superqt cmap.Colormap.
    Handles alpha and normalization to [0,1].
    """
    # pyqtgraph import only when needed
    lut = pg_color_map.getLookupTable(nPts=n, alpha=True)  # uint8 (n x 4) or (n x 3)
    if lut.ndim != 2 or lut.shape[1] not in (3, 4):
        raise ValueError(f"Unexpected LUT shape: {lut.shape}")
    if lut.shape[1] == 3:
        lut = np.hstack([lut, np.full((lut.shape[0], 1), 255, dtype=lut.dtype)])
    lut = lut.astype(np.float32) / 255.0
    return cmap.Colormap(lut)


class LandMarker(QtWidgets.QMainWindow):
    """
    This is the main LandMarker class. It is the entry point for the application and serves as the controller for the
    interface, data, and backend registration algorithms.
    The LandMarker instance is a QMainWindow made up of the following main widgets:
         1. Nine LMViewport objects
            (displaying slices in axial, sagittal, and coronal orientation for each image),
         2. Three LMTable objects (displaying moving landmarks, fixed landmarks,and registered landmarks),
         3.

    1. Three types of LMVolume objects (moving image, fixed image, registered image),
         2. A list of LMLandmark objects,
    No arguments are needed to run LandMarker, but it can accept four optional arguments:
         1. Theme
         2. Debugging mode
         3. Source file (moving image)
         4. Target file (fixed image)
    """

    # NOTE about terminology: 'source' and 'moving' are used interchangeably, as are 'target' and 'fixed'.

    # signal emitted when a list of deformation fields is updated in the deformation field manager window
    dfields_updated_signal = pyqtSignal(list)

    def __init__(self, source_file=None, target_file=None, lm_dir=None, theme="dark", debug_mode=False, *args, **kwargs):
        super().__init__()

        # TODO: device count for GPU
        # torch.cuda.is_available()
        # torch.cuda.device_count()  (dev count - 1)
        # name device 'cuda:[dev_number]'

        self.version = '0.1.0'

        self.source_file = source_file  # moving image
        self.target_file = target_file  # fixed image
        self.landmarks_dir = lm_dir
        self.theme = theme  # 'dark' or 'light'  # TODO
        # self.debug_mode = debug_mode  # debugging mode
        self.debug_mode = False

        # load environment variables from .env file
        load_dotenv()
        self.last_directory = os.getenv('LAST_ACTIVE_DIR', None)
        
        # Initialize rolling logger for status messages
        self._setup_logger()

        # the three types of images: a moving image, a fixed image, and the resulting registered image
        self.moving_volume = None
        self.moving_volumes = []  # list of loaded moving images
        self.current_moving_volume_index = -1  # index of currently displayed moving image
        self.moving_overlay = None
        self.moving_overlay_volumes = []
        self.current_moving_overlay_volume_index = -1
        self.fixed_volume = None
        self.fixed_volumes = []
        self.current_fixed_volume_index = -1
        self.fixed_overlay = None
        self.fixed_overlay_volumes = []
        self.current_fixed_overlay_volume_index = -1
        self.registered_volume = None
        self.registered_volumes = []
        self.current_registered_volume_index = 0
        self.registered_save_filename = None
        # structure for saving difference in geometry between all moving and all fixed images
        self.match_matrix = []  # list of dictionaries: {moving_image_name: {fixed_image_name: [difference]}}

        # the deformation field created by the registration process
        self.current_deformation_field = None
        self.deformation_fields = []  # list of deformation fields

        # the registration parameters popup # TODO
        # self.params = LMRegistrationParams(self)
        self.params = {}  # Dictionary to store registration parameters

        # landmarks for the session
        self.lm_mgr = LMLandmarkManager.LMLandmarkManager()
        self.current_landmarks = []  # list of LMLandmark objects for the current moving-fixed pair

        # indicates status of landmark creation
        # 2: creating landmark, but neither image has a point placed yet for a (two-point) landmark
        # 1: creating landmark, a point has been placed in one image for the (two-point) landmark
        # 0: no landmark being created in this image
        # -1 no landmark being created
        self.lm_status = -1
        # Track if we're in edit mode (separate from lm_status)
        self.lm_edit_mode = False

        # reusable, temporary, point coordinate arrays for creating landmarks
        self.temp_source_point_voxel = None
        self.temp_source_point_patient = None
        self.temp_source_marker_id = None
        self.temp_target_point_voxel = None
        self.temp_target_point_patient = None
        self.temp_target_marker_id = None

        # user guide
        # self.path_to_user_guide = os.path.join('..', 'docs', 'user-guide.html')
        self.path_to_user_guide = (
                Path(__file__).resolve().parent.parent / "docs" / "user-guide.html"
        )

        # ...other display conventions?...
        self.display_convention = 'RAS'

        # cpu is the default
        self.device = 'cpu'

        # set the default
        self.algorithm = Algorithm.LMR

        self.is_linked_zoom = True

        # Track open volume info dialogs for cleanup on quit
        self.open_volume_info_dialogs = []

        # Shutdown state
        self._closing = False
        self.prompt_on_exit = True

        # set up the ui
        # ==============================================================================================================
        # menu, main layouts, etc. created in Qt Designer
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Store the original font for reset functionality
        # The original font is Segoe UI 8pt as set in the UI file for most widgets
        # Explicitly create it as Segoe UI 8pt to ensure consistency
        self.original_font = QtGui.QFont("Segoe UI", 8)

        # menu actions
        self.ui.action_load_moving_nifti.triggered.connect(lambda: self.action_load_moving_triggered('nifti'))
        self.ui.action_load_moving_overlay.triggered.connect(lambda: self.action_load_moving_overlay_triggered('nifti'))
        self.ui.action_save_moving.triggered.connect(self.action_save_moving_triggered)
        self.ui.action_load_fixed_nifti.triggered.connect(lambda: self.action_load_fixed_triggered('nifti'))
        self.ui.action_load_fixed_overlay.triggered.connect(lambda: self.action_load_fixed_overlay_triggered('nifti'))
        self.ui.action_save_registered_nifti.triggered.connect(lambda: self.action_save_registered_triggered('nifti'))
        self.ui.action_save_fixed.triggered.connect(self.action_save_fixed_triggered)
        self.ui.action_meta_moving.triggered.connect(lambda: self.action_view_meta_triggered(0))
        self.ui.action_meta_fixed.triggered.connect(lambda: self.action_view_meta_triggered(1))
        self.ui.action_meta_registered.triggered.connect(lambda: self.action_view_meta_triggered(2))
        #   landmarks
        self.ui.action_save_lm.triggered.connect(self.action_save_lm_triggered)
        self.ui.action_load_lm.triggered.connect(self.action_load_lm_triggered)
        
        # Add menu action for creating landmark by patient coordinates
        self.action_add_lm_by_coords = QtWidgets.QAction(self)
        self.action_add_lm_by_coords.setText("Add by Patient Coordinates...")
        self.action_add_lm_by_coords.setObjectName("action_add_lm_by_coords")
        self.ui.menu_landmarks.addAction(self.action_add_lm_by_coords)
        self.action_add_lm_by_coords.triggered.connect(self.action_add_lm_by_coords_triggered)
        
        # Add menu action for importing landmarks from individual json files
        # (instead of LandMarker source.mrk.json and target.mrk.json pair)
        self.action_import_json_files = QtWidgets.QAction(self)
        self.action_import_json_files.setText("Import Individual json Files")
        self.action_import_json_files.setObjectName("action_import_json_files")
        self.ui.menu_landmarks.addAction(self.action_import_json_files)
        self.action_import_json_files.triggered.connect(self.action_import_json_triggered)
        
        #   file menu
        self.ui.action_quit.triggered.connect(self.action_quit_triggered)
        #   registration > device
        self.ui.action_CPU.triggered.connect(lambda: self.action_set_device('cpu'))
        self.ui.action_GPU.triggered.connect(lambda: self.action_set_device('cuda'))

        #   registration > parameters
        # self.ui.actionParameters.triggered.connect(self.open_params_dialog)  # TODO
        #   registration > landmarks-based algorithms
        self.ui.action_lm_rigid.triggered.connect(lambda: self.action_set_algorithm(Algorithm.LMR))
        self.ui.action_lm_affine.triggered.connect(lambda: self.action_set_algorithm(Algorithm.LMA))
        self.ui.action_lm_tps.triggered.connect(lambda: self.action_set_algorithm(Algorithm.LMT))
        self.ui.action_lm_ksf.triggered.connect(lambda: self.action_set_algorithm(Algorithm.LMK))
        self.ui.action_lm_fft.triggered.connect(lambda: self.action_set_algorithm(Algorithm.LMF))
        self.ui.actionRigid_then_FFT.triggered.connect(lambda: self.action_set_algorithm(Algorithm.RTS))
        #   registration > intensity-based algorithms
        self.ui.action_intens_rigid.triggered.connect(lambda: self.action_set_algorithm(Algorithm.INR))
        self.ui.action_intens_affine.triggered.connect(lambda: self.action_set_algorithm(Algorithm.INA))
        self.ui.action_intens_elastic.triggered.connect(lambda: self.action_set_algorithm(Algorithm.INE))
        self.ui.action_intens_sd.triggered.connect(lambda: self.action_set_algorithm(Algorithm.INS))
        self.ui.action_intens_ldmm.triggered.connect(lambda: self.action_set_algorithm(Algorithm.INL))
        self.ui.action_intens_fft.triggered.connect(lambda: self.action_set_algorithm(Algorithm.INF))
        self.ui.action_intens_sdiff.triggered.connect(lambda: self.action_set_algorithm(Algorithm.IND))
        #   help menu
        self.ui.action_help.setEnabled(True)
        self.ui.action_help.triggered.connect(self.action_open_user_guide)
        #   preferences menu
        self.ui.action_font.triggered.connect(self.action_font_triggered)
        self.ui.action_reset_font.triggered.connect(self.action_reset_font_triggered)

        # registration buttons
        self.ui.register_button.clicked.connect(self.register_button_clicked)
        self.dfield_manager_window = LMdFieldManager(self)
        self.ui.compose_dfields_button.clicked.connect(self.open_dfield_manager_window)
        self.ui.compose_dfields_button.setStyleSheet("QPushButton#compose_dfields_button {background-color: #447CF9; border: 0.04em solid #447CF9;}")  # theme blue
        # self.volume_selector_window = LMVolumeSelector(self)
        # self.ui.apply_to_button.clicked.connect(lambda: self.volume_selector_window.show())

        # tool panel buttons
        # https://phosphoricons.com/
        # https://icones.js.org/
        from PyQt5.QtGui import QIcon

        self.ui.moving_resamp_button.setIcon(QIcon(str(icons_dir / 'black_resamp.svg')))
        self.ui.moving_resamp_button.setStyleSheet("QPushButton#moving_resamp_button {background-color: #FFC90E; border: 0.04em solid #FFC90E;}") # gold yellow

        self.ui.moving_resamp_button.clicked.connect(self.moving_resample_button_clicked)
        self.ui.moving_home_button.setIcon(QIcon(str(icons_dir / 'home_icon.svg')))
        self.ui.moving_home_button.clicked.connect(lambda: self.home_button_clicked(0))
        self.ui.moving_link_button.setIcon(QIcon(str(icons_dir / 'link_icon_white.svg')))
        self.ui.moving_link_button.clicked.connect(lambda: self.link_button_click(0))  # 0 = moving image
        # Store icon paths for toggle functionality
        self.moving_disp_caret_down_icon = QIcon(str(icons_dir / 'caret-down.svg'))
        self.moving_disp_caret_right_icon = QIcon(str(icons_dir / 'caret-right.svg'))
        self.ui.moving_disp_show_button.setIcon(self.moving_disp_caret_down_icon)
        self.ui.moving_disp_show_button.setCheckable(True)
        # Set stylesheet for disp show button
        self.ui.moving_disp_show_button.setStyleSheet(
            "#moving_disp_show_button:hover { background-color: transparent; }"
            "#moving_disp_show_button:checked:hover { background-color: #447CF9; }"
        )
        self.ui.moving_active.setIcon(QIcon(str(icons_dir / 'colorWheelIcon.svg')))
        self.ui.moving_active.setCheckable(True)
        # Prevent hover background color change for active button
        self.ui.moving_active.setStyleSheet(
            "#moving_active:hover { background-color: transparent; }"
            "#moving_active:checked:hover { background-color: #447CF9; }"
        )
        self.ui.moving_overlay_active.setIcon(QIcon(str(icons_dir / 'colorWheelIcon.svg')))
        self.ui.moving_overlay_active.setCheckable(True)
        # Prevent hover background color change for overlay active button
        self.ui.moving_overlay_active.setStyleSheet(
            "#moving_overlay_active:hover { background-color: transparent; }"
            "#moving_overlay_active:checked:hover { background-color: #447CF9; }"
        )
        self.ui.moving_disp_show_button.clicked.connect(lambda: self.settings_button_clicked(0))  # 0 = moving image
        self.ui.moving_info_button.clicked.connect(lambda: self.action_view_meta_triggered(0))  # 0 = moving image
        self.ui.active_moving_group.buttonClicked.connect(self.moving_active_button_clicked)

        self.ui.fixed_home_button.setIcon(QIcon(str(icons_dir / 'home_icon.svg')))
        self.ui.fixed_home_button.clicked.connect(lambda: self.home_button_clicked(1))
        self.ui.fixed_link_button.setIcon(QIcon(str(icons_dir / 'link_icon_white.svg')))
        self.ui.fixed_link_button.clicked.connect(lambda: self.link_button_click(1))
        # Store icon paths for toggle functionality
        self.fixed_disp_caret_down_icon = QIcon(str(icons_dir / 'caret-down.svg'))
        self.fixed_disp_caret_right_icon = QIcon(str(icons_dir / 'caret-right.svg'))
        self.ui.fixed_disp_show_button.setIcon(self.fixed_disp_caret_down_icon)
        self.ui.fixed_disp_show_button.setCheckable(True)
        # Set stylesheet for disp show button
        self.ui.fixed_disp_show_button.setStyleSheet(
            "#fixed_disp_show_button:hover { background-color: transparent; }"
            "#fixed_disp_show_button:checked:hover { background-color: #447CF9; }"
        )
        self.ui.fixed_active.setIcon(QIcon(str(icons_dir / 'colorWheelIcon.svg')))
        self.ui.fixed_active.setCheckable(True)
        # Prevent hover background color change for active button
        self.ui.fixed_active.setStyleSheet(
            "#fixed_active:hover { background-color: transparent; }"
            "#fixed_active:checked:hover { background-color: #447CF9; }"
        )
        self.ui.fixed_overlay_active.setIcon(QIcon(str(icons_dir / 'colorWheelIcon.svg')))
        self.ui.fixed_overlay_active.setCheckable(True)
        # Prevent hover background color change for overlay active button
        self.ui.fixed_overlay_active.setStyleSheet(
            "#fixed_overlay_active:hover { background-color: transparent; }"
            "#fixed_overlay_active:checked:hover { background-color: #447CF9; }"
        )
        self.ui.fixed_disp_show_button.clicked.connect(lambda: self.settings_button_clicked(1))  # 1 = fixed image
        self.ui.fixed_info_button.clicked.connect(lambda: self.action_view_meta_triggered(1))  # 1 = fixed image
        self.ui.active_fixed_group.buttonClicked.connect(self.fixed_active_button_clicked)

        self.ui.registered_home_button.setIcon(QIcon(str(icons_dir / 'home_icon.svg')))
        self.ui.registered_home_button.clicked.connect(lambda: self.home_button_clicked(2))
        self.ui.registered_link_button.setIcon(QIcon(str(icons_dir / 'link_icon_white.svg')))
        self.ui.registered_link_button.clicked.connect(lambda: self.link_button_click(2))

        # TODO - not implemented yet
        self.ui.registered_disp_show_button.setVisible(False)

        # slice guide line visibility button
        crosshair_icon = QIcon(str(icons_dir / 'plus.svg'))
        self.ui.moving_slice_guides_button.setIcon(crosshair_icon)
        self.ui.moving_slice_guides_button.clicked.connect(lambda: self.toggle_slice_guides('moving'))
        self.ui.fixed_slice_guides_button.setIcon(crosshair_icon)
        self.ui.fixed_slice_guides_button.clicked.connect(lambda: self.toggle_slice_guides('fixed'))
        self.ui.reg_slice_guides_button.setIcon(crosshair_icon)
        self.ui.reg_slice_guides_button.clicked.connect(lambda: self.toggle_slice_guides('reg'))
        # buttons for setting visibility, loading or closing the image
        self.ui.show_hide_moving_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        self.ui.show_hide_moving_button.toggled.connect(lambda state: self.toggle_show_hide_button(state, 'moving', 0))
        self.ui.close_moving_button.setIcon(QIcon(str(icons_dir / 'close_icon.svg')))
        self.ui.close_moving_button.clicked.connect(self.moving_close_button_clicked)
        self.ui.load_moving_button.setIcon(QIcon(str(icons_dir / 'open_icon.svg')))
        self.ui.load_moving_button.clicked.connect(lambda: self.action_load_moving_triggered('nifti'))
        # self.ui.show_hide_moving_lm_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        # self.ui.show_hide_moving_lm_button.toggled.connect(
        #     lambda state: self.toggle_show_hide_button(state, 'moving', 2))  # use layer 2 for landmark table
        self.ui.show_hide_mov_overlay_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        self.ui.show_hide_mov_overlay_button.toggled.connect(lambda state: self.toggle_show_hide_button(state,'moving', 1))
        self.ui.close_moving_overlay_button.setIcon(QIcon(str(icons_dir /'close_icon.svg')))
        self.ui.close_moving_overlay_button.clicked.connect(self.moving_overlay_close_button_clicked)
        self.ui.load_moving_overlay_button.setIcon(QIcon(str(icons_dir / 'open_icon.svg')))
        self.ui.load_moving_overlay_button.clicked.connect(lambda: self.action_load_moving_overlay_triggered('nifti'))
        self.ui.show_hide_fixed_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        self.ui.show_hide_fixed_button.toggled.connect(lambda state: self.toggle_show_hide_button(state, 'fixed', 0))
        self.ui.close_fixed_button.setIcon(QIcon(str(icons_dir / 'close_icon.svg')))
        self.ui.close_fixed_button.clicked.connect(self.fixed_close_button_clicked)
        self.ui.load_fixed_button.setIcon(QIcon(str(icons_dir / 'open_icon.svg')))
        self.ui.load_fixed_button.clicked.connect(lambda: self.action_load_fixed_triggered('nifti'))
        # self.ui.show_hide_fixed_lm_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        # self.ui.show_hide_fixed_lm_button.toggled.connect(lambda state: self.toggle_show_hide_button(state, 'fixed', 2))
        self.ui.show_hide_fix_overlay_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        self.ui.show_hide_fix_overlay_button.toggled.connect(
            lambda state: self.toggle_show_hide_button(state, 'fixed', 1))
        self.ui.close_fixed_overlay_button.setIcon(QIcon(str(icons_dir / 'close_icon.svg')))
        self.ui.close_fixed_overlay_button.clicked.connect(self.fixed_overlay_close_button_clicked)
        self.ui.load_fixed_overlay_button.setIcon(QIcon(str(icons_dir / 'open_icon.svg')))
        self.ui.load_fixed_overlay_button.clicked.connect(lambda: self.action_load_fixed_overlay_triggered('nifti'))
        self.ui.show_hide_reg_button.toggled.connect(lambda state: self.toggle_show_hide_button(state, 'reg', 0))
        self.ui.show_hide_reg_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
        self.ui.close_registered_button.setIcon(QIcon(str(icons_dir / 'close_icon.svg')))
        self.ui.close_registered_button.clicked.connect(self.registered_close_button_clicked)

        # combo boxes
        self.focused_combo_box_style = """background-color: #1A4C66;
                                          border: 0.04em solid #447CF9;"""
        self.unfocused_combo_box_style = """background-color: #31363B;
                                            border: 0.04em solid #EFF0F1;"""

        self.current_moving_settings_layer = None  # 'background' or 'overlay'

        self.ui.current_moving_combo.addItem('Select moving image...')
        self.ui.current_moving_combo.setObjectName('current_moving_combo')
        self.ui.current_moving_combo.currentIndexChanged.connect(self.moving_combo_changed)

        self.ui.moving_overlay_combo.addItem('Select moving image overlay...')
        self.ui.moving_overlay_combo.setObjectName('moving_overlay_combo')
        self.ui.moving_overlay_combo.currentIndexChanged.connect(self.moving_overlay_combo_changed)

        self.ui.current_fixed_combo.addItem('Select fixed image...')
        self.ui.current_fixed_combo.setObjectName('current_fixed_combo')
        self.ui.current_fixed_combo.currentIndexChanged.connect(self.fixed_combo_changed)

        self.ui.fixed_overlay_combo.addItem('Select fixed image overlay...')
        self.ui.fixed_overlay_combo.setObjectName('fixed_overlay_combo')
        self.ui.fixed_overlay_combo.currentIndexChanged.connect(self.fixed_overlay_combo_changed)

        self.ui.current_registered_combo.addItem('Select registered image...')
        self.ui.current_registered_combo.currentIndexChanged.connect(self.registered_combo_changed)

        # stand-alone window for optionally displaying the registered image viewports
        self.reg_vol_window = None
        self.ui.expand_window_button.setIcon(QIcon(str(icons_dir / 'expandIcon.svg')))
        self.ui.expand_window_button.clicked.connect(self.registered_expand_button_clicked)

        # add custom widgets like Viewport and LMTable to the main window
        # self.ui only holds the widgets and layouts created in Qt Designer
        # ==============================================================================================================

        # set the window icon
        icon_path = icons_dir / "landmarker_icon.ico"
        if not icon_path.exists():
            # Try alternative formats if .ico doesn't exist
            icon_path = icons_dir / "landmarker_icon.png"
        if not icon_path.exists():
            icon_path = icons_dir / "landmarker_icon.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # vertical labels for the moving, fixed, and registered image frames
        lbl1 = VerticalLabel('Moving Images')
        lbl1.setFont(QtGui.QFont('Segoe UI', 9))
        self.ui.moving_label_frame.layout().addWidget(lbl1)
        lbl1.setStyleSheet('QLabel { font-style:italic; }')
        lbl2 = VerticalLabel('Fixed Images')
        lbl2.setFont(QtGui.QFont('Segoe UI', 9))
        # lbl2.setMinimumHeight(30)
        self.ui.fixed_label_frame.layout().addWidget(lbl2)
        lbl2.setStyleSheet('QLabel { font-style:italic; }')
        lbl3 = VerticalLabel('Registered Images')
        lbl3.setFont(QtGui.QFont('Segoe UI', 9))
        lbl3.setStyleSheet('QLabel { font-style:italic; }')
        self.ui.registered_label_frame.layout().addWidget(lbl3)

        # make the main frames look like they are sunken
        self.ui.session_frame.setStyleSheet("""
                                    QFrame#session_frame {
                                        border-style: inset;
                                        border-width: 2px;
                                        border-color: #1D2023;
                                    }
                                    """)
        self.ui.moving_disp_settings_frame.setStyleSheet("""
                                            QFrame#moving_disp_settings_frame {
                                                border-style: inset;
                                                border-width: 2px;
                                                border-color: #1D2023;
                                            }
                                            """)
        self.ui.moving_vol_main_frame.setStyleSheet("""
                            QFrame#moving_vol_main_frame {
                                border-style: inset;
                                border-width: 2px;
                                border-color: #1D2023;
                            }
                            """)
        self.ui.fixed_vol_main_frame.setStyleSheet("""
                            QFrame#fixed_vol_main_frame {
                                border-style: inset;
                                border-width: 2px;
                                border-color: #1D2023;
                            }
                            """)
        self.ui.fixed_disp_settings_frame.setStyleSheet("""
                                                    QFrame#fixed_disp_settings_frame {
                                                        border-style: inset;
                                                        border-width: 2px;
                                                        border-color: #1D2023;
                                                    }
                                                    """)
        self.ui.registered_vol_main_frame.setStyleSheet("""
                            QFrame#registered_vol_main_frame {
                                border-style: inset;
                                border-width: 2px;
                                border-color: #1D2023;
                            }
                            """)
        # self.ui.current_dfield_label.setStyleSheet("""
        #                     QLabel {background-color: #626568;
        #                     border-color: #626568;
        #                     }
        #                     """)
        self.ui.device_label.setStyleSheet("""
                            QLabel {background-color: #626568;
                            border-color: #626568;
                            }
                            """)
        self.ui.algorithm_label.setStyleSheet("""
                            QLabel {background-color: #626568;
                            border-color: #626568;
                            }
                            """)

        self.range_slider_style = """
            QSlider {
                background-color: none;
            }
            QSlider::add-page:horizontal {
                background: none;
                border: none;
            }
            QSlider::sub-page:horizontal {
                background: none;
                border: none;
            }
            QSlider::groove:horizontal {
                background: black;
                border: none;
            }
            QRangeSlider {
                qproperty-barColor: #005FB8;
            }"""

        # moving display settings --------------------------------------------------------------------------------------
        # NOTE about PyQtGraph lookup tables and colormaps:
        # PyQtGraph’s ImageItem and ImageView expect color mapping information as a lookup table (LUT) — a numpy.ndarray
        # of shape (N, 4) and dtype uint8.  Each row corresponds to an RGBA color for an intensity index from 0 to N-1.
        #
        # For example:
        # lut = np.array([
        #     [0, 0, 0, 255],       # black
        #     [255, 0, 0, 255],     # red
        #     [255, 255, 0, 255],   # yellow
        #     [255, 255, 255, 255]  # white
        # ], dtype=np.uint8)
        # imageItem.setLookupTable(lut)
        #
        # A colormap is an abstract definition of how to convert numerical values to colors.
        # the cmap package (https://pypi.org/project/cmap/) provides a way to convert colormaps to LUTs compatible
        # with PyQtGraph.

        # Maximum number of discrete values supported
        self.max_discrete_values = 20
        
        # discrete color map for images with label, segmentation, or mask data
        self.default_discrete_palette = np.array(
            [[0, 0, 0, 0],  # transparent
             [228, 25, 27, 255],  # red
             [54, 126, 184, 255],  # blue
             [76, 175, 74, 255],  # green
             [151, 77, 163, 255],  # purple
             [255, 127, 0, 255],  # orange
             [255, 255, 51, 255],  # yellow
             [165, 85, 40, 255],  # brown
             [246, 128, 191, 255],  # pink
             [166, 206, 227, 255],  # light blue
             [31, 120, 180, 255],  # dark blue
             [178, 223, 138, 255],  # light green
             [51, 160, 44, 255],  # dark green
             [251, 154, 153, 255],  # light red
             [253, 191, 111, 255],  # light orange
             [202, 178, 214, 255],  # light purple
             [106, 61, 154, 255],  # dark purple
             [255, 255, 153, 255],  # light yellow
             [177, 89, 40, 255],  # dark brown
             [255, 192, 203, 255]],  # light pink
            dtype='uint8')
        # colormaps for images with continuous data such as MRI or CT
        self.continous_palette = [
            'CET-L1',  # bLack -> white (Linear)
            'crameri:navia',  # recommended for T2/T2* images
            'crameri:lipari',  # recommended for T1 images
            'CET-L3',  # bLack -> red -> yeLLow -> white (Linear)
            'CET-L4',  # bLack -> red -> yeLLow (Linear)
            'cividis',
            'inferno',
            'magma',
            'plasma',
            'viridis',
            'CET-L13',  # reds (Linear)
            'CET-L14',  # greens (Linear)
            'CET-L15',  # bLues (Linear)
            'CET-CBC1',  # coLorbLind-friendLy (cycLic)
            'CET-CBL1',  # coLorbLind-friendLy (Linear)
            'CET-D13']
        self._CMAP_ALIASES = {
                "lipari (for T1)": "crameri:lipari",
                "navia (for T2/T2*)": "crameri:navia"
                # add others if your combo shows friendly labels
            }

        # currently active display settings image - either moving image or moving overlay,
        # depending on which combo has focus or was last changed
        self.active_moving_settings_image = None

        # custom widget for display settings
        self.moving_display_settings = DisplaySettings(parent=self, colormaps=self.continous_palette,
                                                       discrete_palette=self.default_discrete_palette,
                                                       tag='moving_display_settings')
        # add it to the place-holder frame in the UI
        self.ui.moving_disp_settings_frame.layout().addWidget(self.moving_display_settings)
        # capture its signals
        self.moving_display_settings.colormap_changed.connect(lambda name, idx: self.disp_handle_colormap_changed_signal('moving', name, idx))
        self.moving_display_settings.opacity_changed.connect( lambda _val: self.disp_handle_opacity_changed_signal('moving', _val))
        # per-label updates:
        self.moving_display_settings.labelColorChanged.connect(lambda idx, qcolor: print("color", idx, qcolor))
        self.moving_display_settings.labelAlphaChanged.connect(lambda _active_im, idx, a: print("alpha", a))
        # whole palette when anything changes:
        self.moving_display_settings.discreteLUTChanged.connect(lambda cmap: self.disp_handle_discrete_LUT_changed_signal('moving', cmap))
        self.moving_display_settings.set_active_image(self.active_moving_settings_image)
        self.moving_display_settings.histogramChanged.connect(lambda new_range: self.disp_handle_histogram_changed_signal('moving', new_range))
        self.ui.moving_disp_settings_frame.setVisible(False)
        self._hide_display_settings_widgets(self.moving_display_settings)

        # fixed display settings =--------------------------------------------------------------------------------------
        self.active_fixed_settings_image = None

        # custom widget for display settings
        self.fixed_display_settings = DisplaySettings(parent=self, colormaps=self.continous_palette,
                                                       discrete_palette=self.default_discrete_palette,
                                                       tag='fixed_display_settings')
        # add it to the place-holder frame in the UI
        self.ui.fixed_disp_settings_frame.layout().addWidget(self.fixed_display_settings)
        # capture its signals
        self.fixed_display_settings.colormap_changed.connect(
            lambda name, idx: self.disp_handle_colormap_changed_signal('fixed', name, idx))
        self.fixed_display_settings.opacity_changed.connect(
            lambda _val: self.disp_handle_opacity_changed_signal('fixed', _val))
        # per-label updates:
        self.fixed_display_settings.labelColorChanged.connect(lambda idx, qcolor: print("color", idx, qcolor))
        self.fixed_display_settings.labelAlphaChanged.connect(lambda _active_im, idx, a: print("alpha", a))
        # whole palette when anything changes:
        self.fixed_display_settings.discreteLUTChanged.connect(lambda cmap: self.disp_handle_discrete_LUT_changed_signal('fixed', cmap))
        self.fixed_display_settings.set_active_image(self.active_fixed_settings_image)
        self.fixed_display_settings.histogramChanged.connect(
            lambda new_range: self.disp_handle_histogram_changed_signal('fixed', new_range))
        self.ui.fixed_disp_settings_frame.setVisible(False)
        self._hide_display_settings_widgets(self.fixed_display_settings)

        # landmark tables ------------------------------------------------------------------------------------------------
        self.moving_lm_table = LMTable(self, 0)
        self.ui.mv_tbl_frame.layout().addWidget(self.moving_lm_table, 1) # 1 = stretch
        self.ui.moving_lm_new_cancel_button.clicked.connect(self.lm_new_cancel_button_clicked)
        self.ui.moving_lm_delete_done_button.clicked.connect(self.lm_done_delete_button_clicked)
        self.ui.moving_lm_edit_cancel_button.clicked.connect(self.lm_edit_done_button_clicked)
        self.ui.moving_lm_delete_all_button.clicked.connect(self.lm_delete_all_button_clicked)

        self.fixed_lm_table = LMTable(self, 1)
        self.ui.fx_tbl_frame.layout().addWidget(self.fixed_lm_table)
        self.ui.fixed_lm_new_cancel_button.clicked.connect(self.lm_new_cancel_button_clicked)
        self.ui.fixed_lm_delete_done_button.clicked.connect(self.lm_done_delete_button_clicked)
        self.ui.fixed_lm_edit_cancel_button.clicked.connect(self.lm_edit_done_button_clicked)
        self.ui.fixed_lm_delete_all_button.clicked.connect(self.lm_delete_all_button_clicked)

        # self.registered_lm_table = LMTable(self, 2)
        # self.ui.reg_tbl_frame.layout().addWidget(self.registered_lm_table)


        # create the nine viewports for displaying slices of the three images
        #
        #                   axial  sagittal  coronal
        #                ---------------------------
        #     moving vol |  0   |    1    |    2   |
        #                ---------------------------
        #      fixed vol |  3   |    4    |    5   |
        #                ---------------------------
        # registered vol |  6   |    7    |    8   |
        #                ---------------------------
        self.viewports = []
        self.viewports.append(Viewport(self, 0, view_dir=ViewDir.AX, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 1, ViewDir.SAG, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 2, ViewDir.COR, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 3, ViewDir.AX, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 4, ViewDir.SAG, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 5, ViewDir.COR, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 6, ViewDir.AX, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 7, ViewDir.SAG, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))
        self.viewports.append(Viewport(self, 8, ViewDir.COR, num_vols=2,
                                       mark_method=InteractionMethod(Qt.LeftButton, Qt.ControlModifier)))

        # add the viewports to the main window ui
        self.ui.moving_vp_frame_0.layout().addWidget(self.viewports[0])
        self.ui.moving_vp_frame_1.layout().addWidget(self.viewports[1])
        self.ui.moving_vp_frame_2.layout().addWidget(self.viewports[2])

        self.ui.fixed_vp_frame_0.layout().addWidget(self.viewports[3])
        self.ui.fixed_vp_frame_1.layout().addWidget(self.viewports[4])
        self.ui.fixed_vp_frame_2.layout().addWidget(self.viewports[5])

        self.alpha_slider_value_pct = 0.5
        self.ui.reg_vp_frame_0.layout().addWidget(self.viewports[6])
        self.ui.reg_alpha_slider.valueChanged.connect(lambda value: self.registered_alpha_slider_change(value, 0))
        self.ui.reg_vp_frame_1.layout().addWidget(self.viewports[7])
        self.ui.reg_vp_frame_2.layout().addWidget(self.viewports[8])

        # Apply platform-specific styling to alpha sliders
        alpha_slider_stylesheet = get_platform_regular_slider_stylesheet()
        self.ui.reg_alpha_slider.setStyleSheet(alpha_slider_stylesheet)
        # # Scope to this widget so it overrides the app stylesheet
        # alpha_slider_stylesheet = alpha_slider_stylesheet.replace("QSlider {", "QSlider#reg_alpha_slider {")
        # alpha_slider_stylesheet = alpha_slider_stylesheet.replace("QSlider::", "QSlider#reg_alpha_slider::")
        # self.ui.reg_alpha_slider.setStyleSheet(alpha_slider_stylesheet)

        # connect the signal emitted from the vp when new marker is created
        for vp in self.viewports:
            vp.marker_added_signal.connect(self.handle_marker_added_signal, type=Qt.QueuedConnection)  # trying to speed it up
            vp.marker_selected_signal.connect(self.handle_marker_selected_signal)
            vp.markers_cleared_signal.connect(self.handle_markers_cleared_signal)
            vp.marker_moved_signal.connect(self.handle_marker_moved_signal)
            vp.marker_drag_ended_signal.connect(self.handle_marker_drag_ended_signal)
            vp.slice_changed_signal.connect(self.handle_slice_changed_signal)

        # trying to speed up
        self._refresh_pending = False

        self.moving_lm_table.row_clicked_signal.connect(self.handle_table_row_clicked_signal)
        self.fixed_lm_table.row_clicked_signal.connect(self.handle_table_row_clicked_signal)
        # self.registered_lm_table.row_clicked_signal.connect(self.handle_table_row_clicked_signal)

        # action groups for the options menu (makes actions mutually exclusively selectable)
        # coordinate system basis - currently not implemented, using RAS+ basis
        # self.coords_action_group = QActionGroup(self)
        # self.coords_action_group.addAction(self.ui.actionLPS)
        # self.coords_action_group.addAction(self.ui.actionRAS)

        # here, we are grouping some menu actions together, to make them mutually selective
        # device (processor) to use for registration
        self.device_action_group = QActionGroup(self)
        self.device_action_group.addAction(self.ui.action_CPU)
        self.device_action_group.addAction(self.ui.action_GPU)

        # algorithm to use for registration
        self.algorithm_action_group = QActionGroup(self)
        self.algorithm_action_group.addAction(self.ui.action_lm_rigid)
        self.algorithm_action_group.addAction(self.ui.action_lm_affine)
        self.algorithm_action_group.addAction(self.ui.action_lm_tps)
        self.algorithm_action_group.addAction(self.ui.action_lm_ksf)
        self.algorithm_action_group.addAction(self.ui.action_lm_fft)
        self.algorithm_action_group.addAction(self.ui.actionRigid_then_FFT)
        self.algorithm_action_group.addAction(self.ui.action_intens_rigid)
        self.algorithm_action_group.addAction(self.ui.action_intens_affine)
        self.algorithm_action_group.addAction(self.ui.action_intens_elastic)
        self.algorithm_action_group.addAction(self.ui.action_intens_sd)
        self.algorithm_action_group.addAction(self.ui.action_intens_ldmm)
        self.algorithm_action_group.addAction(self.ui.action_intens_fft)
        self.algorithm_action_group.addAction(self.ui.action_intens_sdiff)

        # # displays info at the top of the registered image row
        # self.info_label = QLabel()
        # # self.info_label.setStyleSheet('QLabel { text-align: middle;}')
        # self.info_label.setFont(self.base_font)
        # self.info_label.setAlignment(QtCore.Qt.AlignCenter)  # PyQt issue - can ignore warning
        # self.ui.main_grid_layout.addWidget(self.info_label, 4, 3, 1, 1)

        # signals and slots  ========================================================
        self.dfield_manager_window.dfield_selected_signal.connect(self.handle_dfield_selected_signal)
        self.dfield_manager_window.save_dfield_signal.connect(self.handle_save_dfield_signal)
        self.dfield_manager_window.rename_dfield_signal.connect(self.handle_rename_dfield_signal)
        self.dfield_manager_window.load_dfield_signal.connect(self.handle_load_dfield_signal)
        self.dfield_manager_window.apply_dfield_signal.connect(self.handle_apply_dfield_signal)

        # set the initial size of the main window to 80% of the user's screen size
        # self.set_initial_window_size(0.9)

        # optional input arguments
        if source_file is not None:
            if os.path.exists(source_file):
                dataset_type = 'nifti'
                dataset, data_class, uid_meta = self.read_data(dataset_type, source_file)
                if not dataset:
                    # TODO: handle this
                    # QMessageBox.warning(self, "Error", "Problem reading moving image data.", QMessageBox.Ok)
                    return
                if data_class is None:
                    # QMessageBox.warning(self, "Error", "Problem reading moving image data.", QMessageBox.Ok)
                    return

                # ask user for verification if the image is classified as discrete
                if data_class == "Discrete":
                    if len(np.unique(dataset.get_fdata())) > self.max_discrete_values:
                        error_msg = f"The image is classified as a segmentation, but has more than the supported number ({self.max_discrete_values}) unique values (labels)."
                        self.print_log_message(error_msg)
                        QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                        return

                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=True)
                else:
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=False)

                new_vol.populate_with_nifti(dataset, source_file, uid_meta=uid_meta)

                if new_vol.data is None:
                    error_msg = "Problem populating moving image with data."
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return
                LMSidecarUtils.write_sidecar(source_file, uid_meta, landmarker_version=self.version, parent=self)

                # append the new image to list
                self.moving_volumes.append(new_vol)

                # populate the moving combo boxes
                self.moving_update_combo()

                # for convenience, set the current image to the newly loaded image
                self.ui.current_moving_combo.setCurrentIndex(len(self.moving_volumes))  # triggers slot
                # # this calls refresh() and updates the current landmarks list

                # update the display settings image list with the new image - this calls disp_refresh()

                self.print_log_message("Loaded moving image " + source_file)
                if new_vol:
                    self.print_log_message(f"Image shape: {new_vol.shape}")
                    self.print_log_message(f"Image resolution: {new_vol.resolution}")
                    self.print_log_message(f"Image origin: {new_vol.origin}")

        if target_file is not None:
            if os.path.exists(target_file):
                dataset, data_class, uid_meta = self.read_data('nifti', target_file)
                if not dataset:
                    error_msg = f"Problem reading fixed image data from file: {target_file}"
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return
                if data_class is None:
                    error_msg = f"Problem reading fixed image data from file: {target_file}"
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return

                # Ask user for verification if the image is classified as discrete
                if data_class == "Discrete":
                    if len(np.unique(dataset.get_fdata())) > self.max_discrete_values:
                        error_msg = f"The image is classified as discrete, but has more than currently supported number ({self.max_discrete_values}) unique values."
                        self.print_log_message(error_msg)
                        QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                        return
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=True)
                else:
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=False)

                new_vol.populate_with_nifti(dataset, target_file, uid_meta=uid_meta)
                if new_vol.data is None:
                    error_msg = "Problem populating image with data."
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return

                LMSidecarUtils.write_sidecar(target_file, uid_meta, landmarker_version=self.version, parent=self)

                # append the new image to list
                self.fixed_volumes.append(new_vol)

                # populate the moving combo boxes
                self.fixed_update_combo()

                # for convenience, set the current image to the newly loaded image
                # triggers slot fixed_combo_changed()
                self.ui.current_fixed_combo.setCurrentIndex(len(self.fixed_volumes))

                # update the display settings image list with the new image - this calls disp_refresh()
                # self.disp_update_images("fixed")

                self.print_log_message("Loaded fixed image " + target_file)
                self.print_log_message(f"Image shape: {new_vol.shape}")
                self.print_log_message(f"Image resolution: {new_vol.resolution}")
                self.print_log_message(f"Image origin: {new_vol.origin}")
            else:
                error_msg = f"Fixed image file not found: {target_file}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return

        # Load landmarks from directory if provided
        if lm_dir is not None and os.path.isdir(lm_dir):
            # Wait until after volumes are loaded, so landmarks will be loaded after volumes
            # We'll call this at the end of __init__ via a QTimer to ensure UI is ready
            QTimer.singleShot(100, lambda: self._load_landmarks_from_dir(lm_dir))

        self.refresh(False)

    def update_last_active_dir(self, new_path):
        # Load existing environment variables
        load_dotenv()

        # Path to the .env file
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')

        # Read current lines
        if os.path.exists(env_path):
            with open(env_path, 'r') as file:
                lines = file.readlines()
        else:
            lines = []

        # Update or add LAST_ACTIVE_DIR
        updated = False
        with open(env_path, 'w') as file:
            for line in lines:
                if line.startswith("LAST_ACTIVE_DIR="):
                    file.write(f'LAST_ACTIVE_DIR={new_path}\n')
                    updated = True
                else:
                    file.write(line)
            if not updated:
                file.write(f'LAST_ACTIVE_DIR={new_path}\n')

        # Update the instance variable
        self.last_directory = new_path

    def _setup_logger(self):
        """
        Set up a rotating file handler for logging status messages.
        Logs are saved to a 'logs' directory with automatic rotation.
        """
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Set up logger with a rotating file handler
        logger = logging.getLogger('LandMarker')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # Create rotating file handler
        # Max file size: 10MB, keep 5 backup files
        log_file = log_dir / 'landmarker_status.log'
        handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Set format: timestamp, level, message
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Store logger instance
        self.logger = logger
        
        # Log application start
        self.logger.info("=" * 60)
        self.logger.info(f"LandMarker application started - Version {self.version}")
        self.logger.info("=" * 60)

    def closeEvent(self, event):
        """ Save last directory on app close and cleanup resources """
        if not self._closing:
            if self.prompt_on_exit:
                reply = QMessageBox.question(
                    self,
                    "Quit LandMarker",
                    "Are you sure you want to quit LandMarker?\nUnsaved work will be lost.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
            self._closing = True

        # Log application shutdown
        if hasattr(self, 'logger'):
            self.logger.info("=" * 60)
            self.logger.info("LandMarker application closing")
            self.logger.info("=" * 60)
        
        # Close registered window if it's open (undocked)
        if hasattr(self, 'reg_vol_window') and self.reg_vol_window is not None:
            # Close the registered window
            self.reg_vol_window.close()
            self.reg_vol_window = None
        
        # Close deformation fields manager window if it's open
        if hasattr(self, 'dfield_manager_window') and self.dfield_manager_window is not None:
            if self.dfield_manager_window.isVisible():
                self.dfield_manager_window.close()
        
        # Close all open volume info dialogs
        if hasattr(self, 'open_volume_info_dialogs'):
            for dialog in self.open_volume_info_dialogs[:]:  # Use slice copy to avoid modification during iteration
                try:
                    dialog.close()
                except Exception:
                    pass  # Ignore errors if dialog is already closed
            self.open_volume_info_dialogs.clear()
        
        self.update_last_active_dir(self.last_directory)
        
        # Cleanup viewports - clear references before widgets are destroyed
        # This prevents eventFilter from trying to process events during shutdown
        try:
            if hasattr(self, 'viewports') and self.viewports:
                # Clear the list to allow garbage collection
                # The widgets will be cleaned up by Qt when parent is destroyed
                self.viewports.clear()
        except Exception as e:
            self.print_log_message(f"Error during viewport cleanup: {e}")
            pass
        
        # Cleanup PyTorch/CUDA resources if available
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize to ensure cleanup completes
                torch.cuda.synchronize()
            # Force garbage collection to clean up any remaining references
            gc.collect()
        except ImportError:
            pass
        except Exception as e:
            # Log cleanup errors during shutdown (non-critical)
            self.print_log_message(f"Warning during cleanup: {e}")
            pass

        event.accept()

    def set_initial_window_size(self, percent_of_screen):
        # get the geometry of the user's screen
        screen_geometry = QApplication.desktop().screenGeometry()

        # calculate the desired size for the main window (e.g., 80% of the screen size)
        window_width = int(screen_geometry.width() * percent_of_screen)
        window_height = int(screen_geometry.height() * percent_of_screen)

        # set the size of the main window
        self.setGeometry(
            (screen_geometry.width() - window_width) // 2,  # Center horizontally
            (screen_geometry.height() - window_height) // 2,  # Center vertically
            window_width,
            window_height
        )

    # handle external signals ==========================================================================================
    def handle_rename_dfield_signal(self, idx, new_name):
        """ Update the description attribute of the selected deformation field.
        :param idx: current index of the deformation field to rename
        :param new_name:
        :return: success (bool)
        """
        success = False

        try:
            self.deformation_fields[idx].name = new_name
        except Exception as e:
            error_msg = f"An error occurred while renaming this deformation field: {e}"
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return success

        self.refresh()
        success = True
        return success

    def handle_dfield_selected_signal(self, dfield_idx):
        if dfield_idx < len(self.deformation_fields):
            self.current_deformation_field = self.deformation_fields[dfield_idx]
            self.refresh()

    def handle_apply_dfield_signal(self, vols):
        """ Called when LMdFieldManager Apply button is clicked; applies current dfield to selected volumes. """
        try:
            self.apply_dfield(vols)
        except Exception as e:
            error_msg = f"Apply failed: {e}"
            if hasattr(self, 'post_status_message'):
                self.post_status_message(error_msg)
            QMessageBox.critical(self, "Apply failed", error_msg)

    def handle_save_dfield_signal(self, idx):
        """ Called by deformation field manager window LMdFieldManager::save_selected_dfield()
                    Calls save_to_file method of the selected deformation field.
                :return: success (bool)
                """
        success = False

        selected_file = self.select_write_file_or_dir("dfield")
        if selected_file is None:
            # user canceled - do nothing
            return

        try:
            dfield = self.deformation_fields[idx]
            selected_lower = selected_file.lower()

            # TEMPORARILY DISABLED DURING REFACTORING: Only .pt format is supported
            # Export format is selected via filename extension.
            # - .pt: includes LandMarker metadata (save_to_file)
            if selected_lower.endswith(".pt"):
                dfield.save_to_file(selected_file)
            # TEMPORARILY DISABLED: External formats (.npy/.nii/.h5) during refactoring
            # elif selected_lower.endswith(".npy"):
            #     dfield.save_data_only(selected_file, format="numpy")
            # elif selected_lower.endswith(".nii.gz") or selected_lower.endswith(".nii"):
            #     affine = None
            #     dfield_fixed = getattr(dfield, "fixed_vol", "")
            #
            #     if dfield_fixed:
            #         for vol in getattr(self, "fixed_volumes", []) or []:
            #             if getattr(vol, "full_file_name", None) == dfield_fixed:
            #                 affine = getattr(vol, "transform", None)
            #                 break
            #
            #     if affine is None:
            #         affine = getattr(getattr(self, "fixed_volume", None), "transform", None)
            #
            #     dfield.save_nifti(selected_file, affine=affine)
            # elif selected_lower.endswith(".h5") or selected_lower.endswith(".hdf5"):
            #     dfield.save_h5(selected_file, dataset_name="deformationField")
            else:
                raise ValueError(f"Only .pt format is supported during refactoring. Got: {selected_file}")
        except Exception as e:
            error_msg = f"An error occurred while saving this deformation field: {e}"
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return success

        success = True
        return success

    def handle_load_dfield_signal(self):
        """ Called by deformation field manager window LMdFieldManager::import_dfield()
            Loads a deformation field from file and appends it to the list. Supports full LandMarker .pt
            and external formats (.npy, .pt data-only, .nii, .nii.gz, .h5). Sets the current deformation field
            to the newly loaded dfield.
        :return: success (bool)
        """
        success = False

        file_name = self.select_read_file_or_dir("dfield", "Select Deformation Field File")
        if not file_name:
            return
        path = file_name[0]
        idx = len(self.deformation_fields) + 1
        ext = os.path.splitext(path)[1].lower()

        new_dfield = None
        # TEMPORARILY DISABLED DURING REFACTORING: Only .pt format with full LandMarker metadata
        if ext == ".pt":
            try:
                new_dfield = LMDeformationField(self, idx, None, False)
                new_dfield.load_from_file(path)
            except Exception as e:
                error_msg = f"An error occurred while importing this deformation field: {e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return success
        else:
            # TEMPORARILY DISABLED: External formats (.npy, .nii, .h5) during refactoring
            error_msg = "Only .pt format with full LandMarker metadata is supported during refactoring."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return success

        if new_dfield is not None:
            self.deformation_fields.append(new_dfield)
            self.current_deformation_field = new_dfield
            self.refresh()
            success = True
        return success

    def handle_table_row_clicked_signal(self, table_id, table_item):
        # if table_item is not None and table_item.text() != '':
        #     item_text_as_list = table_item.text().strip()
        #     # select the landmark in this row
        #     for lm in self.current_landmarks:
        #         if lm.id == item_text_as_list:
        #             # set this landmark to selected
        #             self.lm_select(lm)
        #             self.lm_marker_selects(lm)
        #             break
        # else:
        #     self.lm_clear_selected()
        #     for vp in self.viewports:
        #         vp.marker_clear_selected(False)
        # Empty click → clear selection
        if table_item is None or not table_item.text().strip():
            self.lm_clear_selected()
            for vp in self.viewports:
                vp.marker_clear_selected(False)
            return

        # Parse landmark ID from cell text
        txt = table_item.text().strip().strip('"').strip("'")
        try:
            lm_id = int(txt)
        except Exception as e:
            self.print_log_message(f"Error parsing landmark ID from table cell: {e}")
            return  # cell does not contain a valid integer ID

        # Resolve pair and landmark
        if self.moving_volume is None or self.fixed_volume is None:
            return
        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        lm = self.lm_mgr.get(pair, lm_id)
        if lm is None:
            self.lm_clear_selected()
            for vp in self.viewports:
                vp.marker_clear_selected(False)
            return

        # If in edit mode and user clicked a different row, ask to save or cancel before switching
        if self.lm_edit_mode:
            selected_lm = self.lm_get_selected_landmark()
            if selected_lm is not None and selected_lm.id != lm_id:
                msg = QMessageBox(self)
                msg.setWindowTitle("Landmark in edit")
                msg.setText("A landmark is being edited. Save edits (Save), discard edits (Discard), or keep editing this landmark (Continue)?")
                done_btn = msg.addButton("Save", QMessageBox.AcceptRole)
                discard_btn = msg.addButton("Discard", QMessageBox.NoRole)
                stay_btn = msg.addButton("Continue", QMessageBox.RejectRole)

                msg.setDefaultButton(stay_btn)
                msg.exec()
                if msg.clickedButton() == stay_btn:
                    # Stay in edit mode on current landmark; restore table selection to its row
                    row_num = next((i for i, landmark in enumerate(self.current_landmarks) if landmark.id == selected_lm.id), None)
                    if row_num is not None:
                        self.moving_lm_table.table.selectRow(row_num)
                        self.fixed_lm_table.table.selectRow(row_num)
                    return
                if msg.clickedButton() == discard_btn:
                    # Discard changes and exit edit mode (restore marker position)
                    for vp in self.viewports:
                        vp.marker_set_edit_mode(False, restore_position=True)
                        vp.marker_clear_selected()
                        vp.refresh_preserve_extent()
                    self.lm_edit_mode = False
                else:
                    # Save changes and exit edit mode (same as Done button)
                    if self.current_landmarks:
                        for landmark in self.current_landmarks:
                            landmark.registered_voxel_coords = None
                            landmark.registered_patient_coords = None
                            landmark.registered_marker_id = None
                    if self.registered_volume is not None:
                        for vp_idx in (6, 7, 8):
                            self.viewports[vp_idx].marker_delete_all()
                    for vp in self.viewports:
                        vp.marker_set_edit_mode(False)
                        vp.marker_clear_selected()
                        vp.refresh_preserve_extent()
                    self.lm_edit_mode = False
                self.refresh(False)
                # Now select the clicked landmark
                self.lm_select(lm)
                self.lm_marker_selects(lm)
                return
            # Same row clicked while editing, or no selected landmark – allow selection as usual
            if selected_lm is not None and selected_lm.id == lm_id:
                # Same landmark already selected and in edit mode; no change
                return

        # Select the clicked landmark (lm_select clears previous selection and sets this one)
        self.lm_select(lm)
        self.lm_marker_selects(lm)

    def handle_marker_added_signal(self, new_marker, vp_id):

        if self.debug_mode:  # print debug messages
            # print(f"handle_marker_added_signal.() with marker {new_marker} viewport {vp_id}")
            print("LandMarker::handle_marker_added_signal")

        if self.lm_status <= 0:
            return

        all_src_vps = [0, 1, 2]
        all_trg_vps = [3, 4, 5]

        if vp_id not in all_src_vps + all_trg_vps:
            # we are in one of the registered image viewports (bottom row) - do nothing
            return

        if vp_id in all_src_vps:
            # FIXME: during testing and dev
            # t_all = QElapsedTimer()
            # t_all.start()  # start global timer
            # t = QElapsedTimer()
            # t.start()

            # we added a point to one of the moving image viewports (top row) (note: "moving" == "source")
            voxel_coords = np.array([new_marker['image_col'], new_marker['image_row'], new_marker['image_slice'], 1])
            # convert to patient coords
            patient_coords = np.dot(self.moving_volume.transform, voxel_coords)
            # set coordinates of temporary landmark
            self.temp_source_point_voxel = voxel_coords[:-1]
            self.temp_source_point_patient = patient_coords[:-1]
            self.temp_source_marker_id = new_marker['id']
            # add point to corresponding viewports and unset add point mode for all source viewports
            # self.viewports[vp_id].marker_set_add_mode(False)

            # FIXME: during testing and dev
            # took_setup = t.elapsed()
            # t.restart()

            # clone the point to other source viewports
            other_src_vps = [id_ for id_ in all_src_vps if id_ != vp_id]
            for vp_id in other_src_vps:
                # plot_data_crs = self.viewports[vp_id].imagecrs_to_plotdatacrs(voxel_coords[0], voxel_coords[1], voxel_coords[2])
                # plot_xy = self.viewports[vp_id].plotdatacr_to_plotxy(plot_data_crs[0], plot_data_crs[1])
                new_mkr = self.viewports[vp_id].add_marker(voxel_coords[0], voxel_coords[1], voxel_coords[2], 0, new_marker['id']) #TODO: set image index
                self.viewports[vp_id].marker_select(new_mkr, notify=False) # calls vp.refresh_preserve_extent()
                # Keep marker_mode as 'adding' so user can click on the selected marker to drag it
                # Lock selection to prevent selecting other markers while in add mode
                self.viewports[vp_id].marker_mode = 'idle'

            # Enable edit mode for source viewports so the marker can be moved immediately
            for vp_idx in all_src_vps:
                self.viewports[vp_idx].marker_set_edit_mode(True)

            # FIXME: during testing and dev
            # took_clone = t.elapsed()
            # t.restart()

            # FIXME: during testing and dev
            # took_refresh = t.elapsed()
            # total = t_all.elapsed()
            # print(f"[marker_added] setup={took_setup} ms, clone={took_clone} ms, "
            #       f"refresh={took_refresh} ms, total={total} ms")

        elif vp_id in all_trg_vps:
            # we are in one of the target image viewports (second row)  ("target" == "fixed")
            # we are creating a landmark in the moving image
            voxel_coords = np.array([new_marker['image_col'], new_marker['image_row'], new_marker['image_slice'], 1])
            # convert to patient coords
            patient_coords = np.dot(self.fixed_volume.transform, voxel_coords)
            self.temp_target_point_voxel = voxel_coords[:-1]  # [col, row, slice] of target image
            self.temp_target_point_patient = patient_coords[:-1]
            self.temp_target_marker_id = new_marker['id']
            # add point to corresponding viewports and unset add point mode for all target viewports
            # self.viewports[vp_id].marker_set_add_mode(False)
            # clone the point to other target viewports
            other_trg_vps = [id_ for id_ in all_trg_vps if id_ != vp_id]
            for vp_id in other_trg_vps:
                # plot_data_crs = self.viewports[vp_id].imagecrs_to_plotdatacrs(voxel_coords[0], voxel_coords[1],
                #                                                                    voxel_coords[2])
                # plot_xy = self.viewports[vp_id].plotdatacr_to_plotxy(plot_data_crs[0], plot_data_crs[1])
                new_pt = self.viewports[vp_id].add_marker(voxel_coords[0],
                                                          voxel_coords[1], voxel_coords[2], 0, new_marker['id'])  # TODO: set image index
                # self.viewports[temp_vp_id].current_slice_index = plot_crs[2]  # refresh will go to slice of new pt.
                # self.viewports[vp_id].marker_set_add_mode(False)
                self.viewports[vp_id].marker_select(new_pt, notify=False)  # calls vp.refresh_preserve_extent()
                # Keep marker_mode as 'adding' so user can click on the selected marker to drag it
                # Lock selection to prevent selecting other markers while in add mode
                self.viewports[vp_id].marker_mode = 'idle'
            
            # Enable edit mode for target viewports so the marker can be moved immediately
            for vp_idx in all_trg_vps:
                self.viewports[vp_idx].marker_set_edit_mode(True)

        # 2: waiting for target AND source points, 1: waiting for target OR source point, 0: both points in place
        self.lm_status = self.lm_status - 1
        # note: this calls refresh_plot() for the other viewports in this image
        # self.sync_viewport_slices(vp_id, voxel_coords)
        # self.refresh()
        self.refresh_throttled()  # trying to speed it up

    def handle_marker_selected_signal(self, marker, vp_id):
        """
        This method is called when a point is clicked in a viewport. It calls lm_select and then calls
        lm_marker_selects() to select the corresponding points in all viewports.

        :param marker:
        :param vp_id:
        :return:
        """
        # FIXME: during testing and dev
        # t = QElapsedTimer()
        # t.start()

        if self.debug_mode:  # print debug messages
            # print(f"handle_marker_selected_signal() with marker {marker} viewport {vp_id}")
            print("LandMarker::handle_marker_selected_signal")

        # find the landmark that corresponds to this point
        lm = self.lm_find_by_marker_id(marker['id'])
        if lm is not None:
            self.lm_select(lm)
            self.lm_marker_selects(lm)

        # print(f"handle_marker_selected_signal -> {t.elapsed()} ms")

    def handle_markers_cleared_signal(self, vp_id):
        """
        Clears (deselects) the point in all viewports when points are cleared in one viewport.
        Clears selected landmarks.

        :param point:
        :param vp_id:
        :param view_dir:
        :return:
        """

        if self.debug_mode:  # print debug messages
            # print(f"handle_markers_cleared_signal() viewport {vp_id}")
            print("LandMarker::handle_markers_cleared_signal")

        for vp in self.viewports:
            vp.marker_clear_selected(False)

        # deselect landmarks in the landmark tables
        self.lm_clear_selected()
        # self.refresh()
        self.refresh_throttled()  # trying to speed it up




    def ensure_transform_cache(self):
        """Cache moving/fixed transforms as (3x3 linear, 3x1 offset) float tuples."""

        def split_affine(a44):
            # a44 is expected to be a 4x4-like object (NumPy or nested lists)
            # Convert once to floats/tuples for super-fast scalar math.
            m00, m01, m02 = float(a44[0][0]), float(a44[0][1]), float(a44[0][2])
            m10, m11, m12 = float(a44[1][0]), float(a44[1][1]), float(a44[1][2])
            m20, m21, m22 = float(a44[2][0]), float(a44[2][1]), float(a44[2][2])
            t0, t1, t2 = float(a44[0][3]), float(a44[1][3]), float(a44[2][3])
            linear = ((m00, m01, m02),
                      (m10, m11, m12),
                      (m20, m21, m22))
            offset = (t0, t1, t2)
            return linear, offset

        if not hasattr(self, "_xf_cache"):
            self._xf_cache = {}

        mv_key = id(self.moving_volume.transform) if self.moving_volume is not None else None
        fx_key = id(self.fixed_volume.transform) if self.fixed_volume is not None else None

        if mv_key and mv_key not in self._xf_cache:
            self._xf_cache[mv_key] = split_affine(self.moving_volume.transform)
        if fx_key and fx_key not in self._xf_cache:
            self._xf_cache[fx_key] = split_affine(self.fixed_volume.transform)

        self._moving_lin, self._moving_off = self._xf_cache.get(mv_key, (None, None))
        self._fixed_lin, self._fixed_off = self._xf_cache.get(fx_key, (None, None))

    @staticmethod
    def voxel_to_patient_fast(c, r, s, linear, offset):
        """allocation-free voxel->patient using cached floats"""
        # linear: ((m00,m01,m02),(m10,m11,m12),(m20,m21,m22))
        # offset: (t0,t1,t2)
        x = linear[0][0] * c + linear[0][1] * r + linear[0][2] * s + offset[0]
        y = linear[1][0] * c + linear[1][1] * r + linear[1][2] * s + offset[1]
        z = linear[2][0] * c + linear[2][1] * r + linear[2][2] * s + offset[2]
        return (x, y, z)

    def init_drag_refresh_throttle(self):
        """Coalesce expensive refresh calls to ~30Hz during drags."""
        if getattr(self, "_drag_refresh_timer", None) is None:
            self._drag_refresh_timer = QtCore.QTimer(self)
            self._drag_refresh_timer.setSingleShot(True)
            self._drag_refresh_timer.setInterval(33)  # ~30 FPS
            self._drag_refresh_timer.timeout.connect(self.do_drag_refresh)
            self._pending_drag_refresh = False

    def schedule_light_refresh(self):
        """Prefer a markers-only refresh if available; coalesces bursts."""
        self.init_drag_refresh_throttle()
        self._pending_drag_refresh = True
        # restart timer (coalesce multiple calls)
        self._drag_refresh_timer.start()

    def do_drag_refresh(self):
        if not self._pending_drag_refresh:
            return
        self._pending_drag_refresh = False
        # If you have a lightweight overlay refresh, call it here:
        if hasattr(self, "refresh_markers_only"):
            self.refresh_markers_only()
        else:
            # Fall back to your existing throttle
            self.refresh_throttled()

    def handle_marker_moved_signal(self, marker, vp_id, vp_view_dir):
        # FAST PATH EXIT
        if marker is None:
            return

        # Ensure we have cached transforms
        self.ensure_transform_cache()

        c = int(marker['image_col'])
        r = int(marker['image_row'])
        s = int(marker['image_slice'])
        marker_id = marker['id']

        # Sync marker position to the other viewports in the same row (so all 3 views update)
        if vp_id in (0, 1, 2):
            for other_vp_id in (0, 1, 2):
                if other_vp_id != vp_id:
                    self.viewports[other_vp_id].marker_update_position(marker_id, c, r, s, 0)
        elif vp_id in (3, 4, 5):
            for other_vp_id in (3, 4, 5):
                if other_vp_id != vp_id:
                    self.viewports[other_vp_id].marker_update_position(marker_id, c, r, s, 0)

        lm = self.lm_find_by_marker_id(marker_id)

        # TEMP landmark (no lm yet)
        if lm is None:
            if vp_id in (0, 1, 2):  # moving image side
                if self.temp_source_point_voxel is not None and self._moving_lin is not None:
                    # Update voxel coords (no NumPy array)
                    self.temp_source_point_voxel = (c, r, s)
                    # Fast voxel->patient
                    self.temp_source_point_patient = self.voxel_to_patient_fast(
                        c, r, s, self._moving_lin, self._moving_off
                    )
                    self.temp_source_marker_id = marker_id

            elif vp_id in (3, 4, 5):  # fixed image side
                if self.temp_target_point_voxel is not None and self._fixed_lin is not None:
                    self.temp_target_point_voxel = (c, r, s)
                    self.temp_target_point_patient = self.voxel_to_patient_fast(
                        c, r, s, self._fixed_lin, self._fixed_off
                    )
                    self.temp_target_marker_id = marker_id

            # Coalesced, lightweight refresh
            self.schedule_light_refresh()
            return

        # EXISTING landmark (lm is not None)
        if vp_id in (0, 1, 2):  # moved a source image marker
            lm.source_voxel_coords = (c, r, s)
            if self._moving_lin is not None:
                lm.source_patient_coords = self.voxel_to_patient_fast(
                    c, r, s, self._moving_lin, self._moving_off
                )
        elif vp_id in (3, 4, 5):  # moved a fixed image marker
            lm.target_voxel_coords = (c, r, s)
            if self._fixed_lin is not None:
                lm.target_patient_coords = self.voxel_to_patient_fast(
                    c, r, s, self._fixed_lin, self._fixed_off
                )

        # Coalesced, lightweight refresh
        self.schedule_light_refresh()

    def handle_marker_drag_ended_signal(self, marker, vp_id, vp_view_dir):
        """
        When the user releases the mouse after dragging a marker, make the other two viewports
        in the same row jump to the slice that contains the marker.
        """
        if marker is None:
            return
        c = int(marker['image_col'])
        r = int(marker['image_row'])
        s = int(marker['image_slice'])
        if vp_id in (0, 1, 2):
            for other_vp_id in (0, 1, 2):
                if other_vp_id != vp_id:
                    plot_crs = self.viewports[other_vp_id].imagecrs_to_plotdatacrs(c, r, s)
                    if plot_crs is not None:
                        self.viewports[other_vp_id].goto_slice(int(plot_crs[2]))
        elif vp_id in (3, 4, 5):
            for other_vp_id in (3, 4, 5):
                if other_vp_id != vp_id:
                    plot_crs = self.viewports[other_vp_id].imagecrs_to_plotdatacrs(c, r, s)
                    if plot_crs is not None:
                        self.viewports[other_vp_id].goto_slice(int(plot_crs[2]))

    def handle_slice_changed_signal(self, vp_id, slice_index):
        """
        This method is called when the user changes the slice index in a viewport. It updates the slice intersection
        lines in all moving or fixed viewports.

        :param vp_id:
        :param slice_index:
        :return:
        """
        # decided to put this in the parent class, since it involves all viewports and viewports are not
        # aware of each other
        if vp_id == 0:
            self.viewports[1].horizontal_line_idx = slice_index
            # self.viewports[1].refresh_preserve_extent()
            self.viewports[1].update_crosshairs()
            self.viewports[2].horizontal_line_idx = slice_index
            # self.viewports[2].refresh_preserve_extent()
            self.viewports[2].update_crosshairs()
        elif vp_id == 1:
            self.viewports[0].vertical_line_idx = slice_index
            # self.viewports[0].refresh_preserve_extent()
            self.viewports[0].update_crosshairs()
            self.viewports[2].vertical_line_idx = slice_index
            # self.viewports[2].refresh_preserve_extent()
            self.viewports[2].update_crosshairs()
        elif vp_id == 2:
            self.viewports[0].horizontal_line_idx = slice_index
            # self.viewports[0].refresh_preserve_extent()
            self.viewports[0].update_crosshairs()
            self.viewports[1].vertical_line_idx = slice_index
            # self.viewports[1].refresh_preserve_extent()
            self.viewports[1].update_crosshairs()
        elif vp_id == 3:
            self.viewports[4].horizontal_line_idx = slice_index
            # self.viewports[4].refresh_preserve_extent()
            self.viewports[4].update_crosshairs()
            self.viewports[5].horizontal_line_idx = slice_index
            # self.viewports[5].refresh_preserve_extent()
            self.viewports[5].update_crosshairs()
        elif vp_id == 4:
            self.viewports[3].vertical_line_idx = slice_index
            # self.viewports[3].refresh_preserve_extent()
            self.viewports[3].update_crosshairs()
            self.viewports[5].vertical_line_idx = slice_index
            # self.viewports[5].refresh_preserve_extent()
            self.viewports[5].update_crosshairs()
        elif vp_id == 5:
            self.viewports[3].horizontal_line_idx = slice_index
            # self.viewports[3].refresh_preserve_extent()
            self.viewports[3].update_crosshairs()
            self.viewports[4].vertical_line_idx = slice_index
            # self.viewports[4].refresh_preserve_extent()
            self.viewports[4].update_crosshairs()
        else:
            pass

    # slots for menu actions ===========================================================================================
    def action_view_meta_triggered(self, which_image):
        if self.moving_volume is None:
            return
        if which_image == 0:
            volume = self.moving_volume
        elif which_image == 1:
            if self.fixed_volume is None:
                return
            volume = self.fixed_volume
        else:
            if self.registered_volume is None:
                return
            volume = self.registered_volume

        dialog = VolumeHeaderInfoDialog(volume, parent=self)
        self.open_volume_info_dialogs.append(dialog)
        # Connect to dialog's finished signal to remove it from the list when closed
        dialog.finished.connect(lambda: self.open_volume_info_dialogs.remove(dialog) if dialog in self.open_volume_info_dialogs else None)
        dialog.show()

    # def action_open_user_guide(self):
    #     try:
    #         webbrowser.open(self.path_to_user_guide, new=2)  # 'new=2' opens the URL in a new tab, if possible
    #     except Exception as e:
    #         error_msg = f"Error opening the user guide: {str(e)}"
    #         self.print_log_message(error_msg)
    #         print(error_msg)

    def action_open_user_guide(self):
        try:
            if not self.path_to_user_guide.is_file():
                raise FileNotFoundError(
                    f"User guide not found: {self.path_to_user_guide}"
                )

            webbrowser.open(self.path_to_user_guide.as_uri(), new=2)

        except Exception as e:
            error_msg = f"Error opening the user guide: {str(e)}"
            self.print_log_message(error_msg)
            print(error_msg)

    def _apply_font_to_widget(self, widget, font):
        """Recursively apply font to a widget and all its children, preserving style properties (bold, italic)."""
        if widget is None:
            return
        
        # Collect references to coordinates_label widgets from all viewports to exclude them
        coordinates_labels_to_exclude = []
        if hasattr(self, 'viewports'):
            for viewport in self.viewports:
                if viewport is not None and hasattr(viewport, 'coordinates_label'):
                    coordinates_labels_to_exclude.append(viewport.coordinates_label)
        
        def apply_font_preserving_style(target_widget, new_font):
            """Apply font to widget while preserving existing style properties (bold, italic, weight)."""
            if target_widget is None:
                return
            # Get the widget's current font to preserve style properties
            current_font = target_widget.font()
            # Create a new font with the new size and family, but preserve style
            preserved_font = QtGui.QFont(new_font)
            preserved_font.setBold(current_font.bold())
            preserved_font.setItalic(current_font.italic())
            preserved_font.setWeight(current_font.weight())
            preserved_font.setUnderline(current_font.underline())
            preserved_font.setStrikeOut(current_font.strikeOut())
            target_widget.setFont(preserved_font)
        
        # Apply font to the widget itself (unless it's a coordinates_label)
        if widget not in coordinates_labels_to_exclude:
            if not (hasattr(widget, 'objectName') and widget.objectName() == 'coordinates_label'):
                apply_font_preserving_style(widget, font)
        
        # Get all child widgets recursively (findChildren already does this)
        # We apply to the widget first, then to all children to ensure proper inheritance
        for child in widget.findChildren(QtWidgets.QWidget):
            # Skip coordinates_label widgets - they should never change font
            if child in coordinates_labels_to_exclude:
                continue
            if hasattr(child, 'objectName') and child.objectName() == 'coordinates_label':
                continue
            apply_font_preserving_style(child, font)
        
        # Also apply to menu bar and menu actions
        if hasattr(widget, 'menuBar') and widget.menuBar() is not None:
            apply_font_preserving_style(widget.menuBar(), font)
            for action in widget.menuBar().actions():
                if action is not None and hasattr(action, 'menu') and action.menu() is not None:
                    apply_font_preserving_style(action.menu(), font)
                    for sub_action in action.menu().actions():
                        if sub_action is not None:
                            # For actions, preserve their font style too
                            if hasattr(sub_action, 'font'):
                                current_action_font = sub_action.font()
                                preserved_action_font = QtGui.QFont(font)
                                preserved_action_font.setBold(current_action_font.bold())
                                preserved_action_font.setItalic(current_action_font.italic())
                                preserved_action_font.setWeight(current_action_font.weight())
                                sub_action.setFont(preserved_action_font)
                            else:
                                sub_action.setFont(font)
    
    def action_font_triggered(self):
        """Open dialog to allow user to select font size only (font family remains unchanged)."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        
        # Get current font from the main window (or application as fallback)
        current_font = self.font() if self.font() else app.font()
        current_size = current_font.pointSize()
        
        # Open input dialog to get font size only
        size, ok = QInputDialog.getInt(
            self, 
            "Font Size", 
            "Enter font size (in points):", 
            current_size, 
            min=6, 
            max=72, 
            step=1
        )
        
        if ok:
            # Create new font with same family but new size
            new_font = QtGui.QFont(current_font)
            new_font.setPointSize(size)
            
            # Apply the new font to the entire application (for new widgets)
            app.setFont(new_font)
            
            # Recursively apply font to all widgets in the main window
            # This overrides any explicit font settings in the UI file
            self._apply_font_to_widget(self, new_font)
            
            # Also apply font to the dfield_manager_window if it exists
            if hasattr(self, 'dfield_manager_window') and self.dfield_manager_window is not None:
                self._apply_font_to_widget(self.dfield_manager_window, new_font)
            
            # Apply font to registered_vol_main_frame whether it's in main window or separate window
            if hasattr(self.ui, 'registered_vol_main_frame') and self.ui.registered_vol_main_frame is not None:
                self._apply_font_to_widget(self.ui.registered_vol_main_frame, new_font)
            
            # Also apply font to reg_vol_window if it exists (covers the case when frame is in separate window)
            if hasattr(self, 'reg_vol_window') and self.reg_vol_window is not None:
                self._apply_font_to_widget(self.reg_vol_window, new_font)
            
            self.print_log_message(f"Font size changed to {size}pt (family: {new_font.family()})")

    def action_reset_font_triggered(self):
        """Reset font to original setting (Segoe UI 8pt)."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        
        # Use the original font that was stored during initialization
        original_font = self.original_font
        
        # Apply the original font to the entire application
        app.setFont(original_font)
        
        # Recursively apply font to all widgets in the main window
        self._apply_font_to_widget(self, original_font)
        
        # Also apply font to the dfield_manager_window if it exists
        if hasattr(self, 'dfield_manager_window') and self.dfield_manager_window is not None:
            self._apply_font_to_widget(self.dfield_manager_window, original_font)
        
        # Apply font to registered_vol_main_frame whether it's in main window or separate window
        if hasattr(self.ui, 'registered_vol_main_frame') and self.ui.registered_vol_main_frame is not None:
            self._apply_font_to_widget(self.ui.registered_vol_main_frame, original_font)
        
        # Also apply font to reg_vol_window if it exists (covers the case when frame is in separate window)
        if hasattr(self, 'reg_vol_window') and self.reg_vol_window is not None:
            self._apply_font_to_widget(self.reg_vol_window, original_font)
        
        self.print_log_message(f"Font reset to original: {original_font.family()}, size {original_font.pointSize()}pt")

    def action_set_device(self, device):
        """ Simply sets device option (cpu or gpu) from menubar, then calls refresh_plot to update the info string. """

        if device == "cuda":
            self.device = "cuda:0"
        elif device == 'cpu':  # CPU
            self.device = device
        else:
            # TODO: handle error
            return

        self.print_log_message("Device set to " + self.device)
        self.refresh(False)

    def action_set_algorithm(self, algorithm):
        """ Sets registration algorithm option from menubar, then calls register to update the registration, and
            refresh_plot to update the viewports and info string.
        """
        self.algorithm = algorithm
        self.print_log_message("Algorithm set to " + algorithm.algorithm_description)
        self.refresh()

    def action_load_moving_triggered(self, dataset_type):
        """
        :param dataset_type:
        :return:
        """
        if dataset_type == "dicom":
            error_msg = "DICOM format not currently supported."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Sorry.", error_msg, QMessageBox.Ok)
            return  # FIXME: temporary

        # select nifti file(s) or DICOM directory (with optional "Load as segmentation" checkbox)
        result = self.select_read_file_or_dir(dataset_type, "Select Moving Image File(s)", offer_segmentation_option=True)
        file_or_dir_name, load_as_segmentation = result
        if not file_or_dir_name:
            # user canceled or other selection problem
            return

        # Show loading dialog
        loading_dialog = WaitDialog(self, "Loading image, please wait...", title="Loading image")
        loading_dialog.show()
        QApplication.processEvents()

        try:
            # read the image from nifti or DICOM
            for fn in file_or_dir_name:
                # check to see if image with this name is already loaded
                if len(self.moving_volumes) > 0:
                    for mv in self.moving_volumes:
                        if mv.full_file_name == fn:
                            error_msg = "A moving image with the same name is already loaded."
                            self.print_log_message(error_msg)
                            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                            return
                dataset, data_class, uid_meta = self.read_data(dataset_type, fn, as_discrete=load_as_segmentation)
                if not dataset:
                    # TODO: handle this
                    # QMessageBox.warning(self, "Error", "Problem reading moving image data.", QMessageBox.Ok)
                    return
                if data_class is None:
                    # QMessageBox.warning(self, "Error", "Problem reading moving image data.", QMessageBox.Ok)
                    return
                

                # validate discrete: max unique values
                if data_class == "Discrete":
                    if len(np.unique(dataset.get_fdata())) > self.max_discrete_values:
                        error_msg = f"The image is classified as discrete, but has more than currently supported number ({self.max_discrete_values}) unique values."
                        self.print_log_message(error_msg)
                        QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                        return

                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=True)
                else:
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=False)

                new_vol.populate_with_nifti(dataset, fn, uid_meta=uid_meta)

                if new_vol.data is None:
                    error_msg = "Problem populating moving image with data."
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return

                sc_path = LMSidecarUtils.sidecar_path_for(fn)
                if not sc_path.exists():
                    LMSidecarUtils.write_sidecar(fn, uid_meta, landmarker_version=self.version, parent=self)

                # append the new image to list
                self.moving_volumes.append(new_vol)

                # populate the moving combo boxes
                self.moving_update_combo()

                # for convenience, set the current image to the newly loaded image
                self.ui.current_moving_combo.setCurrentIndex(len(self.moving_volumes)) # triggers slot
                # this calls refresh() and updates the current landmarks list

                self.print_log_message("Loaded moving image " + fn)
                if new_vol:
                    self.print_log_message(f"Image shape: {new_vol.shape}")
                    self.print_log_message(f"Image resolution: {new_vol.resolution}")
                    self.print_log_message(f"Image origin: {new_vol.origin}")
        finally:
            # Close loading dialog if it's still open
            # Check if dialog is visible before closing (user may have closed it manually)
            try:
                if loading_dialog is not None and loading_dialog.isVisible():
                    loading_dialog.allow_close()  # WaitDialog ignores close() until allow_close()
                    loading_dialog.close()
            except (RuntimeError, AttributeError):
                pass  # Dialog was already deleted (user closed it) or is invalid
            except Exception:
                pass  # If closing dialog fails for any other reason, continue

    def action_load_moving_overlay_triggered(self, dataset_type):
        """
        :param dataset_type:
        :return:
        """
        if dataset_type == "dicom":
            error_msg = "DICOM format not currently supported."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Nope, sorry.", error_msg, QMessageBox.Ok)
            return  # FIXME: temporary

        # select nifti file(s) or DICOM directory (with optional "Load as segmentation" checkbox)
        result = self.select_read_file_or_dir(dataset_type, "Select Moving Overlay Image File(s)", offer_segmentation_option=True)
        file_or_dir_name, load_as_segmentation = result
        if not file_or_dir_name:
            # user canceled or other selection problem
            return

        # Show loading dialog
        loading_dialog = WaitDialog(self, "Loading image, please wait...", title="Loading image")
        loading_dialog.show()
        QApplication.processEvents()

        new_vol = None

        try:
            # read the image from nifti or DICOM
            for fn in file_or_dir_name:
                # check to see if image with this name is already loaded
                if len(self.moving_overlay_volumes) > 0:
                    for mo in self.moving_overlay_volumes:
                        if mo.full_file_name == fn:
                            error_msg = "A moving overlay with the same name is already loaded."
                            self.print_log_message(error_msg)
                            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                            return
                dataset, data_class, uid_meta = self.read_data(dataset_type, fn, as_discrete=load_as_segmentation)
                if not dataset:
                    # TODO: handle this
                    # QMessageBox.warning(self, "Error", "Problem reading moving image data.", QMessageBox.Ok)
                    return
                if data_class is None:
                    # QMessageBox.warning(self, "Error", "Problem reading moving image data.", QMessageBox.Ok)
                    return

                # validate discrete: max unique values
                if data_class == "Discrete":
                    if len(np.unique(dataset.get_fdata())) > self.max_discrete_values:
                        error_msg = f"The image is classified as discrete, but has more than currently supported number ({self.max_discrete_values}) unique values."
                        self.print_log_message(error_msg)
                        QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                        return

                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=True)
                else:
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=False)

                new_vol.populate_with_nifti(dataset, fn, uid_meta=uid_meta)
                if new_vol.data is None:
                    error_msg = "Problem populating moving overlay with data."
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return

                LMSidecarUtils.write_sidecar(fn, uid_meta, landmarker_version=self.version, parent=self)

                # append the new image to list
                self.moving_overlay_volumes.append(new_vol)

                # populate the moving combo boxes
                self.moving_overlay_update_combo()

                # for convenience, set current image to the newly loaded image
                self.ui.moving_overlay_combo.setCurrentIndex(len(self.moving_overlay_volumes))  # triggers slot
                # this calls refresh() and updates the current landmarks list
                self.moving_overlay_combo_changed(len(self.moving_overlay_volumes))

                # update the display settings image list with the new image - this calls disp_refresh()
                # self.disp_update_images("moving")

                self.print_log_message("Loaded moving overlay " + fn)
                self.print_log_message(f"Image shape: {new_vol.shape}")
                self.print_log_message(f"Image resolution: {new_vol.resolution}")
                self.print_log_message(f"Image origin: {new_vol.origin}")
        finally:
            # Close loading dialog if it's still open
            # Check if dialog is visible before closing (user may have closed it manually)
            try:
                if loading_dialog is not None and loading_dialog.isVisible():
                    loading_dialog.allow_close()  # WaitDialog ignores close() until allow_close()
                    loading_dialog.close()
            except (RuntimeError, AttributeError):
                pass  # Dialog was already deleted (user closed it) or is invalid
            except Exception:
                pass  # If closing dialog fails for any other reason, continue

    def action_save_moving_triggered(self):
        """Save the moving image as a NIfTI (.nii or .nii.gz). Always behaves like 'Save As'."""
        if self.moving_volume is None or self.moving_volume.data is None:
            error_msg = "No moving image selected."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # --- Construct a suggested filename ---
        base = self.moving_volume.file_base_name or "moving"
        suggested_name = base + ".nii.gz"

        selected_file = self.select_write_file_or_dir("nifti", default_name=suggested_name)

        if not selected_file:
            return  # user cancelled

        out_path = selected_file
        low = out_path.lower()
        if not (low.endswith(".nii") or low.endswith(".nii.gz")):
            out_path += ".nii.gz"  # default to compressed NIfTI

        ok = self.write_image3d_as_nifti(self.moving_volume, out_path)
        if ok:
            self.print_log_message(f"Moving image saved to {out_path}.")
        else:
            self.print_log_message("Failed to save moving image (see logs).")

    def action_save_fixed_triggered(self):
        """Save the fixed volume as a NIfTI (.nii or .nii.gz). Always behaves like 'Save As'."""
        if self.fixed_volume is None or self.fixed_volume.data is None:
            error_msg = "No fixed image selected."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # --- Construct a suggested filename ---
        base = self.fixed_volume.file_base_name or "fixed"
        suggested_name = base + ".nii.gz"

        selected_file = self.select_write_file_or_dir("nifti", default_name=suggested_name)

        if not selected_file:
            return  # user cancelled

        out_path = selected_file
        low = out_path.lower()
        if not (low.endswith(".nii") or low.endswith(".nii.gz")):
            out_path += ".nii.gz"  # default to compressed NIfTI

        ok = self.write_image3d_as_nifti(self.fixed_volume, out_path)
        if ok:
            self.print_log_message(f"Fixed image saved to {out_path}.")
        else:
            self.print_log_message("Failed to save fixed image (see logs).")

    def action_load_fixed_triggered(self, dataset_type):
        """
        :param dataset_type:
        :return:
        """
        if dataset_type == "dicom":
            error_msg = "DICOM format not currently supported."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Nope, sorry.", error_msg, QMessageBox.Ok)
            return  # FIXME: temporary

        # select nifti file(s) or DICOM directory (with optional "Load as segmentation" checkbox)
        result = self.select_read_file_or_dir(dataset_type, "Select Fixed Image File(s)", offer_segmentation_option=True)
        file_or_dir_name, load_as_segmentation = result
        if not file_or_dir_name:
            # user canceled or other selection problem
            return

        # Show loading dialog
        loading_dialog = WaitDialog(self, "Loading image, please wait...", title="Loading image")
        loading_dialog.show()
        QApplication.processEvents()

        new_vol = None

        try:
            # read the image from nifti or DICOM
            for fn in file_or_dir_name:
                # check to see if image with this name is already loaded
                if len(self.fixed_volumes) > 0:
                    for fv in self.fixed_volumes:
                        if fv.full_file_name == fn:
                            error_msg = "A fixed image with the same name is already loaded."
                            self.print_log_message(error_msg)
                            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                            return
                dataset, data_class, uid_meta = self.read_data(dataset_type, fn, as_discrete=load_as_segmentation)
                if not dataset:
                    # QMessageBox.warning(self, "Error", "Problem reading fixed image data from file.", QMessageBox.Ok)
                    return
                if data_class is None:
                    # QMessageBox.warning(self, "Error", "Problem reading fixed image data from file.", QMessageBox.Ok)
                    return

                # validate discrete: max unique values
                if data_class == "Discrete":
                    if len(np.unique(dataset.get_fdata())) > self.max_discrete_values:
                        error_msg = f"The image is classified as discrete, but has more than currently supported number ({self.max_discrete_values}) unique values."
                        self.print_log_message(error_msg)
                        QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                        return
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=True)
                else:
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=False)

                new_vol.populate_with_nifti(dataset, fn, uid_meta=uid_meta)
                if new_vol.data is None:
                    error_msg = "Problem populating image with data."
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return

                sc_path = LMSidecarUtils.sidecar_path_for(fn)
                if not sc_path.exists():
                    LMSidecarUtils.write_sidecar(fn, uid_meta, landmarker_version=self.version, parent=self)

                # append the new image to list
                self.fixed_volumes.append(new_vol)

                # populate the moving combo boxes
                self.fixed_update_combo()

                # for convenience, set the current image to the newly loaded image
                # triggers slot fixed_combo_changed()
                self.ui.current_fixed_combo.setCurrentIndex(len(self.fixed_volumes))

                # update the display settings image list with the new image - this calls disp_refresh()
                # self.disp_update_images("fixed")

                self.print_log_message("Loaded fixed image " + fn)
                self.print_log_message(f"Image shape: {new_vol.shape}")
                self.print_log_message(f"Image resolution: {new_vol.resolution}")
                self.print_log_message(f"Image origin: {new_vol.origin}")
        finally:
            # Close loading dialog if it's still open
            # Check if dialog is visible before closing (user may have closed it manually)
            try:
                if loading_dialog is not None and loading_dialog.isVisible():
                    loading_dialog.allow_close()  # WaitDialog ignores close() until allow_close()
                    loading_dialog.close()
            except (RuntimeError, AttributeError):
                pass  # Dialog was already deleted (user closed it) or is invalid
            except Exception:
                pass  # If closing dialog fails for any other reason, continue

    def action_load_fixed_overlay_triggered(self, dataset_type):
        """
        :param dataset_type:
        :return:
        """
        if dataset_type == "dicom":
            error_msg = "DICOM format not currently supported."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Nope, sorry.", error_msg, QMessageBox.Ok)
            return  # FIXME: temporary

        # select nifti file(s) or DICOM directory (with optional "Load as segmentation" checkbox)
        result = self.select_read_file_or_dir(dataset_type, "Select Fixed Overlay Image File(s)", offer_segmentation_option=True)
        file_or_dir_name, load_as_segmentation = result
        if not file_or_dir_name:
            # user canceled or other selection problem
            return

        # Show loading dialog
        loading_dialog = WaitDialog(self, "Loading image, please wait...", title="Loading image")
        loading_dialog.show()
        QApplication.processEvents()

        new_vol = None
        try:
            # read the image from nifti or DICOM
            for fn in file_or_dir_name:
                # check to see if image with this name is already loaded
                if len(self.fixed_overlay_volumes) > 0:
                    for mo in self.fixed_overlay_volumes:
                        if mo.full_file_name == fn:
                            error_msg = "A fixed overlay with the same name is already loaded."
                            self.print_log_message(error_msg)
                            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                            return
                dataset, data_class, uid_meta = self.read_data(dataset_type, fn, as_discrete=load_as_segmentation)
                if not dataset:
                    # FIXME: message already handled in read_data?
                    # QMessageBox.warning(self, "Error", "Problem reading fixed image data.", QMessageBox.Ok)
                    return
                if data_class is None:
                    # FIXME: message already handled in read_data?
                    # QMessageBox.warning(self, "Error", "Problem reading fixed image data.", QMessageBox.Ok)
                    return

                # validate discrete: max unique values
                if data_class == "Discrete":
                    if len(np.unique(dataset.get_fdata())) > self.max_discrete_values:
                        error_msg = f"The image is classified as discrete, but has more than currently supported number ({self.max_discrete_values}) unique values."
                        self.print_log_message(error_msg)
                        QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                        return
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=True)
                else:
                    new_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=False)

                new_vol.populate_with_nifti(dataset, fn, uid_meta=uid_meta)

                if new_vol.data is None:
                    error_msg = "Problem populating fixed overlay with data."
                    self.print_log_message(error_msg)
                    QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                    return
                LMSidecarUtils.write_sidecar(fn, uid_meta, landmarker_version=self.version, parent=self)

                # append the new image to list
                self.fixed_overlay_volumes.append(new_vol)

                # populate the moving combo boxes
                self.fixed_overlay_update_combo()

                # for convenience, set current image to the newly loaded image
                self.ui.fixed_overlay_combo.setCurrentIndex(len(self.fixed_overlay_volumes))  # doesn't trigger slot
                # this calls refresh() and updates the current landmarks list
                self.fixed_overlay_combo_changed(len(self.fixed_overlay_volumes))

                # update the display settings image list with the new image - this calls disp_refresh()
                # self.disp_update_images('fixed')

                self.print_log_message("Loaded fixed overlay " + fn)
                self.print_log_message(f"Image shape: {new_vol.shape}")
                self.print_log_message(f"Image resolution: {new_vol.resolution}")
                self.print_log_message(f"Image origin: {new_vol.origin}")
        finally:
            # Close loading dialog if it's still open
            # Check if dialog is visible before closing (user may have closed it manually)
            try:
                if loading_dialog is not None and loading_dialog.isVisible():
                    loading_dialog.allow_close()  # WaitDialog ignores close() until allow_close()
                    loading_dialog.close()
            except (RuntimeError, AttributeError):
                pass  # Dialog was already deleted (user closed it) or is invalid
            except Exception:
                pass  # If closing dialog fails for any other reason, continue

    def action_add_lm_by_coords_triggered(self):
        """
        Create a landmark by entering patient coordinates.
        Opens a dialog to enter patient coordinates, then converts to voxel coordinates
        for both moving and fixed images and creates the landmark.
        """
        if self.moving_volume is None or self.fixed_volume is None:
            QMessageBox.warning(self, "Error", 
                              "Both moving and fixed images must be loaded to create a landmark.",
                              QMessageBox.Ok)
            return
        
        # Open dialog to get patient coordinates
        dialog = LMPatientCoordDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return  # User cancelled
        
        # Get patient coordinates from dialog (returns tuple: (moving_coords, fixed_coords))
        coords_result = dialog.get_coordinates()
        if coords_result is None:
            QMessageBox.warning(self, "Invalid Input", 
                              "Please enter valid numeric values for all coordinates.",
                              QMessageBox.Ok)
            return
        
        moving_patient_coords, fixed_patient_coords = coords_result
        
        # Convert patient coordinates to voxel coordinates for both images
        moving_patient_coords_homog = np.array([moving_patient_coords[0], moving_patient_coords[1], moving_patient_coords[2], 1])
        fixed_patient_coords_homog = np.array([fixed_patient_coords[0], fixed_patient_coords[1], fixed_patient_coords[2], 1])
        
        # Moving image voxel coordinates
        moving_voxel_coords_homog = np.dot(np.linalg.inv(self.moving_volume.transform), moving_patient_coords_homog)
        moving_voxel_coords = moving_voxel_coords_homog[:3].astype(int)
        
        # Fixed image voxel coordinates
        fixed_voxel_coords_homog = np.dot(np.linalg.inv(self.fixed_volume.transform), fixed_patient_coords_homog)
        fixed_voxel_coords = fixed_voxel_coords_homog[:3].astype(int)
        
        # Check if coordinates are within image bounds
        if (moving_voxel_coords[0] < 0 or moving_voxel_coords[0] >= self.moving_volume.num_cols or
            moving_voxel_coords[1] < 0 or moving_voxel_coords[1] >= self.moving_volume.num_rows or
            moving_voxel_coords[2] < 0 or moving_voxel_coords[2] >= self.moving_volume.num_slices):
            QMessageBox.warning(self, "Out of Bounds", 
                              f"Moving image voxel coordinates {moving_voxel_coords} are out of bounds.",
                              QMessageBox.Ok)
            return
        
        if (fixed_voxel_coords[0] < 0 or fixed_voxel_coords[0] >= self.fixed_volume.num_cols or
            fixed_voxel_coords[1] < 0 or fixed_voxel_coords[1] >= self.fixed_volume.num_rows or
            fixed_voxel_coords[2] < 0 or fixed_voxel_coords[2] >= self.fixed_volume.num_slices):
            QMessageBox.warning(self, "Out of Bounds", 
                              f"Fixed image voxel coordinates {fixed_voxel_coords} are out of bounds.",
                              QMessageBox.Ok)
            return
        
        # Get the landmark ID first so we can use it for the markers
        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        landmark_id = self.lm_mgr.id_mgr.next_id(pair)
        
        # Create markers in viewports using the landmark ID
        new_src_marker = None
        new_trg_marker = None
        
        # Add markers to moving image viewports (0, 1, 2)
        for vp_idx in (0, 1, 2):
            # self.viewports[vp_idx].add_layer(self.moving_volume, 0)
            new_src_marker = self.viewports[vp_idx].add_marker(
                int(moving_voxel_coords[0]), int(moving_voxel_coords[1]), int(moving_voxel_coords[2]), 0, landmark_id
            )
            if new_src_marker is None:
                QMessageBox.warning(self, "Error", "Problem creating moving marker.", QMessageBox.Ok)
                return
        
        # Add markers to fixed image viewports (3, 4, 5)
        for vp_idx in (3, 4, 5):
            # self.viewports[vp_idx].add_layer(self.fixed_volume, 0)
            new_trg_marker = self.viewports[vp_idx].add_marker(
                int(fixed_voxel_coords[0]), int(fixed_voxel_coords[1]), int(fixed_voxel_coords[2]), 0, landmark_id
            )
            if new_trg_marker is None:
                QMessageBox.warning(self, "Error", "Problem creating fixed marker.", QMessageBox.Ok)
                return
        
        # Create the landmark with the same ID
        self.lm_add_landmark(
            src_vox=moving_voxel_coords,
            src_pat=moving_patient_coords,
            src_mkr_id=new_src_marker["id"],
            trg_vox=fixed_voxel_coords,
            trg_pat=fixed_patient_coords,
            trg_mkr_id=new_trg_marker["id"],
            lm_id=landmark_id
        )
        
        # Refresh the UI
        self.refresh(False)
        for vp in self.viewports:
            vp.refresh_preserve_extent()
        
        self.print_log_message(f"Landmark created - Moving: [{moving_patient_coords[0]:.2f}, {moving_patient_coords[1]:.2f}, {moving_patient_coords[2]:.2f}], Fixed: [{fixed_patient_coords[0]:.2f}, {fixed_patient_coords[1]:.2f}, {fixed_patient_coords[2]:.2f}]")

    def action_load_lm_triggered(self):
        """
        Load landmarks from source.mrk.json and target.mrk.json.
        Validate that files belong to the CURRENT moving/fixed pair. If not, abort.
        Create viewport markers and LMLandmark objects via LMLandmarkManager.
        """
        import os, json
        from LMLandmarkManager import pair_key_from_vols

        # Confirm overwrite
        if self.current_landmarks:
            reply = QMessageBox.question(
                self, "Landmarks already exist.",
                "This will clear existing landmarks. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Choose folder
        selected_dir = self.select_read_file_or_dir("landmarks")
        if not selected_dir:
            return

        # File paths
        src_path = os.path.join(selected_dir, "source.mrk.json")
        trg_path = os.path.join(selected_dir, "target.mrk.json")
        if not os.path.exists(src_path):
            error_msg = "'source.mrk.json' not found."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return
        if not os.path.exists(trg_path):
            error_msg = "'target.mrk.json' not found."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # Must have active pair
        if self.moving_volume is None or self.fixed_volume is None:
            error_msg = "No active moving/fixed volumes."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # Read JSON and validate pair
        try:
            with open(src_path, "r") as f:
                src_json = json.load(f)
            with open(trg_path, "r") as f:
                trg_json = json.load(f)
        except Exception as e:
            error_msg = f"Failed to read landmark files: {e}"
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        def _pair_from_json(obj):
            meta = obj.get("_landmarkerMeta", {}) or obj.get("landmarkerMeta", {})
            p = (meta.get("pair") or {}).get("pair_key")
            # accept list or tuple
            if isinstance(p, (list, tuple)) and len(p) == 2:
                return (str(p[0]), str(p[1]))
            return None

        src_pair = _pair_from_json(src_json)
        trg_pair = _pair_from_json(trg_json)

        if src_pair is None or trg_pair is None or src_pair != trg_pair:
            error_msg = "Landmark files have inconsistent or missing pair metadata."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # --- helper to extract UIDs only ---
        def _uids_from_json(obj):
            """Return (moving_uid, fixed_uid) tuple from a LandMarker mrk.json file."""
            meta = obj.get("_landmarkerMeta") or obj.get("landmarkerMeta") or {}
            pair = meta.get("pair") or {}
            mv_uid = (pair.get("moving") or {}).get("uid")
            fx_uid = (pair.get("fixed") or {}).get("uid")
            if not mv_uid or not fx_uid:
                pk = pair.get("pair_key")
                if isinstance(pk, (list, tuple)) and len(pk) == 2:
                    mv_uid, fx_uid = pk[0], pk[1]
            if mv_uid is None or fx_uid is None:
                return None
            return (str(mv_uid), str(fx_uid))

        # --- now validate the pair ---
        current_pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        src_pair_uids = _uids_from_json(src_json)
        trg_pair_uids = _uids_from_json(trg_json)

        if src_pair_uids is None or trg_pair_uids is None or src_pair_uids != trg_pair_uids:
            error_msg = "Landmark files have inconsistent or missing pair metadata."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return
        if src_pair_uids != current_pair:
            error_msg = "Landmark files are for a different image pair than the currently active image pair."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            return

        # Extract points
        def _points(obj):
            m = obj.get("markups", [])
            if not m:
                return []
            cp = m[0].get("controlPoints", [])
            out = []
            for it in cp:
                pos = it.get("position")
                vox = (it.get("_landmarkerExtras") or {}).get("voxel")
                out.append({
                    "id": it.get("id"),
                    "pos": pos,  # patient coords [x,y,z]
                    "vox": vox,  # voxel [c,r,s]
                    "src_mkr_id": it.get("source_marker_id"),
                    "trg_mkr_id": it.get("target_marker_id"),
                })
            return out

        src_pts = _points(src_json)
        trg_pts = _points(trg_json)
        if not src_pts or not trg_pts or len(src_pts) != len(trg_pts):
            error_msg = "Problem reading control points from markup files."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # Clear existing markers and landmarks for this pair
        for vp in self.viewports:
            vp.marker_delete_all()
        self.lm_mgr.clear_pair(current_pair)
        self.current_landmarks = []

        # Create markers + landmarks
        for i in range(len(src_pts)):
            s = src_pts[i];
            t = trg_pts[i]

            # Add viewport markers
            # Create marker in first viewport to generate ID, then use same ID for all viewports
            new_src_marker = None
            new_trg_marker = None
            
            # Create source marker in first viewport (0) to generate ID
            self.viewports[0].add_layer(self.moving_volume, 0)
            new_src_marker = self.viewports[0].add_marker(
                int(s["vox"][0]), int(s["vox"][1]), int(s["vox"][2]), 0, None
            )
            if new_src_marker is None:
                error_msg = "Problem creating moving marker."
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return
            
            # Use the same marker ID for remaining source viewports (1, 2)
            src_marker_id = new_src_marker["id"]
            for vp_idx in (1, 2):  # moving
                self.viewports[vp_idx].add_layer(self.moving_volume, 0)
                self.viewports[vp_idx].add_marker(
                    int(s["vox"][0]), int(s["vox"][1]), int(s["vox"][2]), 0, src_marker_id
                )
            
            # Create target marker in first viewport (3) to generate ID
            self.viewports[3].add_layer(self.fixed_volume, 0)
            new_trg_marker = self.viewports[3].add_marker(
                int(t["vox"][0]), int(t["vox"][1]), int(t["vox"][2]), 0, None
            )
            if new_trg_marker is None:
                error_msg = "Problem creating fixed marker."
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return
            
            # Use the same marker ID for remaining target viewports (4, 5)
            trg_marker_id = new_trg_marker["id"]
            for vp_idx in (4, 5):  # fixed
                self.viewports[vp_idx].add_layer(self.fixed_volume, 0)
                self.viewports[vp_idx].add_marker(
                    int(t["vox"][0]), int(t["vox"][1]), int(t["vox"][2]), 0, trg_marker_id
                )

            # Add landmark via manager (use ID from file)
            self.lm_mgr.add(
                current_pair,
                parent=self,
                source_voxel_coords=s["vox"],
                source_patient_coords=s["pos"],
                source_marker_id=new_src_marker["id"],
                target_voxel_coords=t["vox"],
                target_patient_coords=t["pos"],
                target_marker_id=new_trg_marker["id"],
                lm_id=s["id"],  # Use the ID from the loaded file
            )

        # Sync marker counters in all viewports to prevent ID conflicts
        # This ensures new markers created after loading won't reuse existing IDs
        for vp in self.viewports:
            vp.marker_sync_counter()

        # Sync cache and refresh UI
        self.current_landmarks = self.lm_mgr.list(current_pair)
        self.refresh(False)
        for vp in self.viewports:
            vp.refresh_preserve_extent()
        self.print_log_message(f"Loaded landmarks from {selected_dir}")

    def action_import_json_triggered(self):
        """
        Import landmarks from individual json files. Generally expect Slicer3D format.
        These files do not contain image pair metadata, so the user imports at their own risk.
        Assumes source.mrk.json is for the current moving volume and target.mrk.json is for the current fixed volume.
        """
        import os, json
        from LMLandmarkManager import pair_key_from_vols

        # Must have active pair
        if self.moving_volume is None or self.fixed_volume is None:
            error_msg = "No active moving/fixed volumes."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # Confirm overwrite
        if self.current_landmarks:
            reply = QMessageBox.question(
                self, "Landmarks already exist.",
                "This will clear existing landmarks. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Select source file (for moving volume)
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        source_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Landmark File for MOVING Image",
            self.last_directory,
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if not source_file:
            return

        # Select target file (for fixed volume)
        target_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Landmark File for FIXED Image",
            os.path.dirname(source_file),
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if not target_file:
            return

        # Update last directory
        self.last_directory = QtCore.QFileInfo(source_file).dir().absolutePath()

        # Read JSON files
        try:
            with open(source_file, "r") as f:
                source_json = json.load(f)
            with open(target_file, "r") as f:
                target_json = json.load(f)
        except Exception as e:
            error_msg = f"Failed to read landmark files: {e}"
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # Extract points from Slicer3D format
        def _extract_slicer3d_points(obj):
            """Extract control points from Slicer3D format JSON."""
            markups = obj.get("markups", [])
            if not markups:
                return [], None
            
            markup = markups[0]
            control_points = markup.get("controlPoints", [])
            coordinate_system = markup.get("coordinateSystem", "RAS")
            
            points = []
            for cp in control_points:
                position = cp.get("position")
                cp_id = cp.get("id", "")
                if position is None or len(position) < 3:
                    continue
                points.append({
                    "id": cp_id,
                    "position": position[:3]  # Take first 3 elements [x, y, z]
                })
            
            return points, coordinate_system

        source_points, source_coord_sys = _extract_slicer3d_points(source_json)
        target_points, target_coord_sys = _extract_slicer3d_points(target_json)

        if not source_points or not target_points:
            error_msg = "No control points found in landmark files."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        if len(source_points) != len(target_points):
            error_msg = f"Number of control points mismatch: source has {len(source_points)}, target has {len(target_points)}."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return

        # Convert coordinates from LPS to RAS if needed
        def _convert_lps_to_ras(coords):
            """Convert coordinates from LPS to RAS by multiplying L and P by -1."""
            return [-coords[0], -coords[1], coords[2]]

        # Process source points
        processed_source_points = []
        for pt in source_points:
            pos = pt["position"]
            if source_coord_sys == "LPS":
                pos = _convert_lps_to_ras(pos)
            processed_source_points.append({
                "id": pt["id"],
                "position": pos
            })

        # Process target points
        processed_target_points = []
        for pt in target_points:
            pos = pt["position"]
            if target_coord_sys == "LPS":
                pos = _convert_lps_to_ras(pos)
            processed_target_points.append({
                "id": pt["id"],
                "position": pos
            })

        # Get current pair
        current_pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)

        # Clear existing markers and landmarks for this pair
        for vp in self.viewports:
            vp.marker_delete_all()
        self.lm_mgr.clear_pair(current_pair)
        self.current_landmarks = []

        # Validate all landmarks are within bounds before creating any
        landmarks_to_create = []
        for i in range(len(processed_source_points)):
            src_pt = processed_source_points[i]
            trg_pt = processed_target_points[i]

            # Convert patient coordinates to voxel coordinates
            src_patient_coords = np.array(src_pt["position"])
            trg_patient_coords = np.array(trg_pt["position"])

            # Convert to homogeneous coordinates
            src_patient_coords_homog = np.append(src_patient_coords, 1)
            trg_patient_coords_homog = np.append(trg_patient_coords, 1)

            # Convert to voxel coordinates
            src_voxel_coords_homog = np.dot(np.linalg.inv(self.moving_volume.transform), src_patient_coords_homog)
            src_voxel_coords = src_voxel_coords_homog[:3].astype(int)

            trg_voxel_coords_homog = np.dot(np.linalg.inv(self.fixed_volume.transform), trg_patient_coords_homog)
            trg_voxel_coords = trg_voxel_coords_homog[:3].astype(int)

            # Check if coordinates are within image bounds
            if (src_voxel_coords[0] < 0 or src_voxel_coords[0] >= self.moving_volume.num_cols or
                src_voxel_coords[1] < 0 or src_voxel_coords[1] >= self.moving_volume.num_rows or
                src_voxel_coords[2] < 0 or src_voxel_coords[2] >= self.moving_volume.num_slices):
                error_msg = f"Source landmark {i+1} (ID: {src_pt['id']}) is out of bounds. Voxel coordinates: {src_voxel_coords}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Out of Bounds", error_msg, QMessageBox.Ok)
                return

            if (trg_voxel_coords[0] < 0 or trg_voxel_coords[0] >= self.fixed_volume.num_cols or
                trg_voxel_coords[1] < 0 or trg_voxel_coords[1] >= self.fixed_volume.num_rows or
                trg_voxel_coords[2] < 0 or trg_voxel_coords[2] >= self.fixed_volume.num_slices):
                error_msg = f"Target landmark {i+1} (ID: {trg_pt['id']}) is out of bounds. Voxel coordinates: {trg_voxel_coords}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Out of Bounds", error_msg, QMessageBox.Ok)
                return

            # Store landmark data for creation
            landmarks_to_create.append({
                "src_patient": src_patient_coords,
                "src_voxel": src_voxel_coords,
                "trg_patient": trg_patient_coords,
                "trg_voxel": trg_voxel_coords,
                "id": src_pt["id"]
            })

        # All landmarks are valid, now create them
        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        
        for lm_data in landmarks_to_create:
            # Extract ID from Slicer3D file and append underscore to string representation
            # Since landmark IDs in LandMarker are integers, we'll extract a numeric ID
            # and use it, but conceptually the ID has an underscore appended (for display/logging)
            original_id = lm_data["id"]
            lm_id = None
            
            # Try to extract integer ID from the original ID
            try:
                if isinstance(original_id, (int, float)):
                    lm_id = int(original_id)
                elif isinstance(original_id, str):
                    # Try to extract number from string
                    numbers = re.findall(r'\d+', original_id)
                    if numbers:
                        lm_id = int(numbers[0])
                    else:
                        # No number found, auto-generate
                        lm_id = None
                else:
                    lm_id = None
            except (ValueError, TypeError):
                lm_id = None

            # If we couldn't get a valid ID, auto-generate one
            if lm_id is None:
                lm_id = self.lm_mgr.id_mgr.next_id(pair)
            else:
                # Try to register the ID, but if it conflicts, generate a new one
                try:
                    # Check if ID already exists
                    if self.lm_mgr.get(pair, lm_id) is not None:
                        # ID exists, generate new one
                        lm_id = self.lm_mgr.id_mgr.next_id(pair)
                    else:
                        # Register the ID
                        self.lm_mgr.id_mgr.register_existing(pair, lm_id)
                except:
                    # If registration fails, generate new ID
                    lm_id = self.lm_mgr.id_mgr.next_id(pair)
            
            # Create string representation with underscore for logging/display
            lm_id_str = f"{lm_id}_"

            # Create markers in viewports
            # Create marker in first viewport to generate ID, then use same ID for all viewports
            new_src_marker = None
            new_trg_marker = None

            # Create source marker in first viewport (0) to generate ID
            new_src_marker = self.viewports[0].add_marker(
                int(lm_data["src_voxel"][0]),
                int(lm_data["src_voxel"][1]),
                int(lm_data["src_voxel"][2]),
                0,
                None  # Let viewport generate marker ID
            )
            if new_src_marker is None:
                error_msg = f"Problem creating moving marker for landmark {lm_id_str}."
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return

            # Use the same marker ID for remaining source viewports (1, 2)
            src_marker_id = new_src_marker["id"]
            for vp_idx in (1, 2):
                self.viewports[vp_idx].add_marker(
                    int(lm_data["src_voxel"][0]),
                    int(lm_data["src_voxel"][1]),
                    int(lm_data["src_voxel"][2]),
                    0,
                    src_marker_id
                )

            # Create target marker in first viewport (3) to generate ID
            new_trg_marker = self.viewports[3].add_marker(
                int(lm_data["trg_voxel"][0]),
                int(lm_data["trg_voxel"][1]),
                int(lm_data["trg_voxel"][2]),
                0,
                None  # Let viewport generate marker ID
            )
            if new_trg_marker is None:
                error_msg = f"Problem creating fixed marker for landmark {lm_id_str}."
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return

            # Use the same marker ID for remaining target viewports (4, 5)
            trg_marker_id = new_trg_marker["id"]
            for vp_idx in (4, 5):
                self.viewports[vp_idx].add_marker(
                    int(lm_data["trg_voxel"][0]),
                    int(lm_data["trg_voxel"][1]),
                    int(lm_data["trg_voxel"][2]),
                    0,
                    trg_marker_id
                )

            # Add landmark via manager
            self.lm_mgr.add(
                current_pair,
                parent=self,
                source_voxel_coords=lm_data["src_voxel"],
                source_patient_coords=lm_data["src_patient"],
                source_marker_id=new_src_marker["id"],
                target_voxel_coords=lm_data["trg_voxel"],
                target_patient_coords=lm_data["trg_patient"],
                target_marker_id=new_trg_marker["id"],
                lm_id=lm_id,
            )

        # Sync marker counters in all viewports to prevent ID conflicts
        for vp in self.viewports:
            vp.marker_sync_counter()

        # Sync cache and refresh UI
        self.current_landmarks = self.lm_mgr.list(current_pair)
        self.refresh(False)
        for vp in self.viewports:
            vp.refresh_preserve_extent()
        self.print_log_message(f"Imported {len(landmarks_to_create)} landmarks from Slicer3D files")

    def _load_landmarks_from_dir(self, lm_dir):
        """
        Load landmarks from a directory (called automatically from __init__ if lm_dir is provided).
        Uses the same logic as action_load_lm_triggered but without user interaction.
        """
        import os, json
        from LMLandmarkManager import pair_key_from_vols

        # File paths
        src_path = os.path.join(lm_dir, "source.mrk.json")
        trg_path = os.path.join(lm_dir, "target.mrk.json")
        
        if not os.path.exists(src_path):
            self.print_log_message(f"Warning: 'source.mrk.json' not found in {lm_dir}")
            return
        if not os.path.exists(trg_path):
            self.print_log_message(f"Warning: 'target.mrk.json' not found in {lm_dir}")
            return

        # Must have active pair
        if self.moving_volume is None or self.fixed_volume is None:
            self.print_log_message(f"Warning: Cannot load landmarks - no active moving/fixed volumes.")
            return

        # Read JSON and validate pair
        try:
            with open(src_path, "r") as f:
                src_json = json.load(f)
            with open(trg_path, "r") as f:
                trg_json = json.load(f)
        except Exception as e:
            self.print_log_message(f"Error: Failed to read landmark files: {e}")
            return

        def _pair_from_json(obj):
            meta = obj.get("_landmarkerMeta", {}) or obj.get("landmarkerMeta", {})
            p = (meta.get("pair") or {}).get("pair_key")
            # accept list or tuple
            if isinstance(p, (list, tuple)) and len(p) == 2:
                return (str(p[0]), str(p[1]))
            return None

        def _uids_from_json(obj):
            """Return (moving_uid, fixed_uid) tuple from a LandMarker mrk.json file."""
            meta = obj.get("_landmarkerMeta") or obj.get("landmarkerMeta") or {}
            pair = meta.get("pair") or {}
            mv_uid = (pair.get("moving") or {}).get("uid")
            fx_uid = (pair.get("fixed") or {}).get("uid")
            if not mv_uid or not fx_uid:
                pk = pair.get("pair_key")
                if isinstance(pk, (list, tuple)) and len(pk) == 2:
                    mv_uid, fx_uid = pk[0], pk[1]
            if mv_uid is None or fx_uid is None:
                return None
            return (str(mv_uid), str(fx_uid))

        src_pair = _pair_from_json(src_json)
        trg_pair = _pair_from_json(trg_json)

        if src_pair is None or trg_pair is None or src_pair != trg_pair:
            self.print_log_message(f"Warning: Landmark files have inconsistent or missing pair metadata.")
            return

        # Validate the pair
        current_pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        src_pair_uids = _uids_from_json(src_json)
        trg_pair_uids = _uids_from_json(trg_json)

        if src_pair_uids is None or trg_pair_uids is None or src_pair_uids != trg_pair_uids:
            self.print_log_message(f"Warning: Landmark files have inconsistent or missing pair metadata.")
            return
        if src_pair_uids != current_pair:
            self.print_log_message(f"Warning: Landmark files are for a different image pair than the currently active image pair.")
            return

        # Extract points
        def _points(obj):
            m = obj.get("markups", [])
            if not m:
                return []
            cp = m[0].get("controlPoints", [])
            out = []
            for it in cp:
                pos = it.get("position")
                vox = (it.get("_landmarkerExtras") or {}).get("voxel")
                out.append({
                    "id": it.get("id"),
                    "pos": pos,  # patient coords [x,y,z]
                    "vox": vox,  # voxel [c,r,s]
                    "src_mkr_id": it.get("source_marker_id"),
                    "trg_mkr_id": it.get("target_marker_id"),
                })
            return out

        src_pts = _points(src_json)
        trg_pts = _points(trg_json)
        if not src_pts or not trg_pts or len(src_pts) != len(trg_pts):
            self.print_log_message(f"Warning: Problem reading control points from markup files.")
            return

        # Clear existing markers and landmarks for this pair
        for vp in self.viewports:
            vp.marker_delete_all()
        self.lm_mgr.clear_pair(current_pair)
        self.current_landmarks = []

        # Create markers + landmarks
        for i in range(len(src_pts)):
            s = src_pts[i]
            t = trg_pts[i]

            # Add viewport markers
            # Create marker in first viewport to generate ID, then use same ID for all viewports
            new_src_marker = None
            new_trg_marker = None
            
            # Create source marker in first viewport (0) to generate ID
            self.viewports[0].add_layer(self.moving_volume, 0)
            new_src_marker = self.viewports[0].add_marker(
                int(s["vox"][0]), int(s["vox"][1]), int(s["vox"][2]), 0, None
            )
            if new_src_marker is None:
                self.print_log_message(f"Warning: Problem creating moving marker.")
                return
            
            # Use the same marker ID for remaining source viewports (1, 2)
            src_marker_id = new_src_marker["id"]
            for vp_idx in (1, 2):  # moving
                self.viewports[vp_idx].add_layer(self.moving_volume, 0)
                self.viewports[vp_idx].add_marker(
                    int(s["vox"][0]), int(s["vox"][1]), int(s["vox"][2]), 0, src_marker_id
                )
            
            # Create target marker in first viewport (3) to generate ID
            self.viewports[3].add_layer(self.fixed_volume, 0)
            new_trg_marker = self.viewports[3].add_marker(
                int(t["vox"][0]), int(t["vox"][1]), int(t["vox"][2]), 0, None
            )
            if new_trg_marker is None:
                self.print_log_message(f"Warning: Problem creating fixed marker.")
                return
            
            # Use the same marker ID for remaining target viewports (4, 5)
            trg_marker_id = new_trg_marker["id"]
            for vp_idx in (4, 5):  # fixed
                self.viewports[vp_idx].add_layer(self.fixed_volume, 0)
                self.viewports[vp_idx].add_marker(
                    int(t["vox"][0]), int(t["vox"][1]), int(t["vox"][2]), 0, trg_marker_id
                )

            # Add landmark via manager (use ID from file)
            self.lm_mgr.add(
                current_pair,
                parent=self,
                source_voxel_coords=s["vox"],
                source_patient_coords=s["pos"],
                source_marker_id=new_src_marker["id"],
                target_voxel_coords=t["vox"],
                target_patient_coords=t["pos"],
                target_marker_id=new_trg_marker["id"],
                lm_id=s["id"],  # Use the ID from the loaded file
            )

        # Sync marker counters in all viewports to prevent ID conflicts
        # This ensures new markers created after loading won't reuse existing IDs
        for vp in self.viewports:
            vp.marker_sync_counter()

        # Sync cache and refresh UI
        self.current_landmarks = self.lm_mgr.list(current_pair)
        self.refresh(False)
        for vp in self.viewports:
            vp.refresh_preserve_extent()
        self.print_log_message(f"Loaded landmarks from {lm_dir}")

    def action_save_lm_triggered(self, mode):
        """ Saves landmarks to a json file using a 3D Slicer markup schema.
            If mode is "save" and user has already selected a save file, then this
            method writes to that file.
            If mode is "save" and user has not yet selected a save file, then
            they are prompted to select a file.
            If mode is "save_as", user is prompted to select a save file.
        """
        if len(self.current_landmarks) == 0:
            # no landmarks to save
            QMessageBox.information(self, 'Sorry.', 'No Landmarks to Save.', QMessageBox.Ok)
            return

        selected_dir = self.select_write_file_or_dir("landmarks")
        if selected_dir is None:
            # user cancelled - do nothing
            return
        else:
            self.landmarks_dir = selected_dir


        if self.lm_write_landmarks():
            self.print_log_message("Landmarks file pair has been written to folder " + self.landmarks_dir)

    def action_quit_triggered(self):
        """ Quit the application. This triggers closeEvent which handles cleanup. """
        # Use close() instead of quit() to ensure proper cleanup order
        self.close()

    def action_save_registered_triggered(self, type, mode="save_as"):
        """ Saves the registered image as a nifti or DICOM.
            If mode is "save" and user has already selected a save file,
            then this method writes to that file. If mode is "save" and user
            has not yet selected a save file, then they are prompted to select
            a file. If mode is "save_as", user is prompted to select a save file.
            Type is "dicom" or "nifti".
        """
        if self.registered_volume is None or self.registered_volume.data is None:
            return

        if mode == "save":
            # check to see if current save filename is set. if so, write image
            # to that file (overwrite whole file)
            if self.registered_save_filename is None:
                # --- Construct a suggested filename ---
                base = self.registered_volume.file_base_name or "registered"
                suggested_name = base + ".nii.gz"
                selected_file = self.select_write_file_or_dir(datatype=type, default_name=suggested_name)
                if selected_file is None:
                    # user cancelled - do nothing
                    return
                else:
                    self.registered_save_filename = selected_file
        elif mode == "save_as":
            # --- Construct a suggested filename ---
            base = self.registered_volume.file_base_name or "registered"
            suggested_name = base + ".nii.gz"
            selected_file = self.select_write_file_or_dir(datatype=type, default_name=suggested_name)
            if selected_file is None:
                # user cancelled - do nothing
                return
            else:
                self.registered_save_filename = selected_file

        # if self.registered_write(type):
        if self.write_image3d_as_nifti(self.registered_volume, self.registered_save_filename):
            self.print_log_message("Registered image " + self.registered_save_filename + " saved.")

    def open_dfield_manager_window(self):
        """Open the deformation field manager and bring it to the foreground."""
        win = self.dfield_manager_window

        # If it's minimized, restoring is necessary for raise/activate to work reliably on Windows.
        if win.isMinimized():
            win.showNormal()
        else:
            win.show()

        # Bring to front immediately…
        win.raise_()
        win.activateWindow()
        win.setFocus()

        # …and again after the button click finishes (Qt/Windows sometimes re-focuses the main window).
        def _reassert_focus():
            if not win.isVisible():
                return
            win.raise_()
            win.activateWindow()
            win.setFocus()

        QTimer.singleShot(0, _reassert_focus)

    # slots for image tool buttons ====================================================================================
    def settings_button_clicked(self, which_vol):
        if which_vol == 0:
            # moving image
            if self.ui.moving_disp_settings_frame.isVisible():
                self.ui.moving_disp_settings_frame.setVisible(False)
                self.ui.moving_disp_show_button.setToolTip("Show display settings panel")
                self.ui.moving_disp_show_button.setIcon(self.moving_disp_caret_down_icon)
                self.ui.moving_disp_show_button.setChecked(False)
            else:
                self.ui.moving_disp_settings_frame.setVisible(True)
                self.ui.moving_disp_show_button.setToolTip("Hide display settings panel")
                self.ui.moving_disp_show_button.setIcon(self.moving_disp_caret_right_icon)
                self.ui.moving_disp_show_button.setChecked(True)
        elif which_vol == 1:
            if self.ui.fixed_disp_settings_frame.isVisible():
                self.ui.fixed_disp_settings_frame.setVisible(False)
                self.ui.fixed_disp_show_button.setToolTip("Show display settings panel")
                self.ui.fixed_disp_show_button.setIcon(self.fixed_disp_caret_down_icon)
                self.ui.fixed_disp_show_button.setChecked(False)
            else:
                self.ui.fixed_disp_settings_frame.setVisible(True)
                self.ui.fixed_disp_show_button.setToolTip("Hide display settings panel")
                self.ui.fixed_disp_show_button.setIcon(self.fixed_disp_caret_right_icon)
                self.ui.fixed_disp_show_button.setChecked(True)
        else:  # registered image
            pass
            # if self.registered_display_settings_widget.isVisible():
            #     self.registered_display_settings_widget.setVisible(False)
            #     self.ui.registered_disp_show_button.setToolTip("Show display settings panel")
            # else:
            #     self.registered_display_settings_widget.setVisible(True)
            #     self.ui.registered_disp_show_button.setToolTip("Hide display settings panel")

        # self.adjustSize()

    def moving_active_button_clicked(self, button):
        """
        Slot for signal sent when user clicks a button in the active_moving_group.
        Sets the active display settings image based on which button was clicked.
        """
        if button is self.ui.moving_active:
            # Set moving image as active
            # Check if combo has an image selected (index > 0 means an image is selected)
            if self.ui.current_moving_combo.currentIndex() > 0 and getattr(self, "moving_volume", None) is not None:
                self.active_moving_settings_image = self.moving_volume
                # Show the display settings frames before setting the image
                self._show_display_settings_widgets(self.moving_display_settings)
                self.moving_display_settings.set_active_image(self.active_moving_settings_image)
                # Update colormap selection to match the image's current colormap
                if hasattr(self.moving_display_settings, 'colormap_combo') and self.moving_volume.colormap_index is not None:
                    self.moving_display_settings.colormap_combo.setCurrentIndex(self.moving_volume.colormap_index)
                self.moving_display_settings.refresh()
            else:
                # No image selected - clear and hide display settings
                self.active_moving_settings_image = None
                self.moving_display_settings.clear_active_image()
                self._hide_display_settings_widgets(self.moving_display_settings)
        elif button is self.ui.moving_overlay_active:
            # Set moving overlay as active
            # Check if combo has an image selected (index > 0 means an image is selected)
            if self.ui.moving_overlay_combo.currentIndex() > 0 and getattr(self, "moving_overlay", None) is not None:
                self.active_moving_settings_image = self.moving_overlay
                # Show the display settings frames before setting the image
                self._show_display_settings_widgets(self.moving_display_settings)
                self.moving_display_settings.set_active_image(self.active_moving_settings_image)
                # Update colormap selection to match the image's current colormap
                if hasattr(self.moving_display_settings, 'colormap_combo') and self.moving_overlay.colormap_index is not None:
                    self.moving_display_settings.colormap_combo.setCurrentIndex(self.moving_overlay.colormap_index)
                self.moving_display_settings.refresh()
            else:
                # No image selected - clear and hide display settings
                self.active_moving_settings_image = None
                self.moving_display_settings.clear_active_image()
                self._hide_display_settings_widgets(self.moving_display_settings)

    def fixed_active_button_clicked(self, button):
        """
        Slot for signal sent when user clicks a button in the active_fixed_group.
        Sets the active display settings image based on which button was clicked.
        """
        if button is self.ui.fixed_active:
            # Set fixed image as active
            # Check if combo has an image selected (index > 0 means an image is selected)
            if self.ui.current_fixed_combo.currentIndex() > 0 and getattr(self, "fixed_volume", None) is not None:
                self.active_fixed_settings_image = self.fixed_volume
                # Show the display settings frames before setting the image
                self._show_display_settings_widgets(self.fixed_display_settings)
                self.fixed_display_settings.set_active_image(self.active_fixed_settings_image)
                # Update colormap selection to match the image's current colormap
                if hasattr(self.fixed_display_settings, 'colormap_combo') and self.fixed_volume.colormap_index is not None:
                    self.fixed_display_settings.colormap_combo.setCurrentIndex(self.fixed_volume.colormap_index)
                self.fixed_display_settings.refresh()
            else:
                # No image selected - clear and hide display settings
                self.active_fixed_settings_image = None
                self.fixed_display_settings.clear_active_image()
                self._hide_display_settings_widgets(self.fixed_display_settings)
        elif button is self.ui.fixed_overlay_active:
            # Set fixed overlay as active
            # Check if combo has an image selected (index > 0 means an image is selected)
            if self.ui.fixed_overlay_combo.currentIndex() > 0 and getattr(self, "fixed_overlay", None) is not None:
                self.active_fixed_settings_image = self.fixed_overlay
                # Show the display settings frames before setting the image
                self._show_display_settings_widgets(self.fixed_display_settings)
                self.fixed_display_settings.set_active_image(self.active_fixed_settings_image)
                # Update colormap selection to match the image's current colormap
                if hasattr(self.fixed_display_settings, 'colormap_combo') and self.fixed_overlay.colormap_index is not None:
                    self.fixed_display_settings.colormap_combo.setCurrentIndex(self.fixed_overlay.colormap_index)
                self.fixed_display_settings.refresh()
            else:
                # No image selected - clear and hide display settings
                self.active_fixed_settings_image = None
                self.fixed_display_settings.clear_active_image()
                self._hide_display_settings_widgets(self.fixed_display_settings)

    def _hide_display_settings_widgets(self, display_settings):
        """
        Hide all display settings widgets: colormap combo, histogram, opacity slider, 
        discrete colors widget, and clear the active image label.
        """
        if display_settings is None:
            return
        
        # Hide the color settings frame (contains colormap combo and discrete colors widget in a stack)
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'color_settings_frame'):
            display_settings.ui.color_settings_frame.setVisible(False)
        
        # Hide histogram settings frame
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'histogram_settings_frame'):
            display_settings.ui.histogram_settings_frame.setVisible(False)
        
        # Hide opacity slider frame
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'alpha_slider_frame'):
            display_settings.ui.alpha_slider_frame.setVisible(False)
        
        # Clear the active image label (already done by clear_active_image, but ensure it's cleared)
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'curr_image_text'):
            display_settings.ui.curr_image_text.setText("")
    
    def _show_display_settings_widgets(self, display_settings):
        """
        Show all display settings widgets: colormap combo, histogram, opacity slider, 
        and discrete colors widget. The set_active_image() method will handle showing
        the appropriate widgets based on whether the image is discrete or continuous.
        """
        if display_settings is None:
            return
        
        # Show the color settings frame (contains colormap combo and discrete colors widget in a stack)
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'color_settings_frame'):
            display_settings.ui.color_settings_frame.setVisible(True)
        
        # Show histogram settings frame
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'histogram_settings_frame'):
            display_settings.ui.histogram_settings_frame.setVisible(True)
        
        # Show opacity slider frame
        if hasattr(display_settings, 'ui') and hasattr(display_settings.ui, 'alpha_slider_frame'):
            display_settings.ui.alpha_slider_frame.setVisible(True)

    def link_button_click(self, whichVol):
        self.is_linked_zoom = not self.is_linked_zoom
        if self.is_linked_zoom:
            self.ui.moving_link_button.setIcon(QIcon(str(icons_dir / 'link_icon_white.svg')))
            self.ui.moving_link_button.setToolTip("Unlink zooming between viewports")
            self.ui.fixed_link_button.setIcon(QIcon(str(icons_dir / 'link_icon_white.svg')))
            self.ui.fixed_link_button.setToolTip("Unlink zooming between viewports")
            self.ui.registered_link_button.setIcon(QIcon(str(icons_dir / 'link_icon_white.svg')))
            self.ui.registered_link_button.setToolTip("Unlink zooming between viewports")
        else:
            self.ui.moving_link_button.setIcon(QIcon(str(icons_dir / 'slash_link_icon.svg')))
            self.ui.moving_link_button.setToolTip("Link zooming between viewports")
            self.ui.fixed_link_button.setIcon(QIcon(str(icons_dir / 'slash_link_icon.svg')))
            self.ui.fixed_link_button.setToolTip("Link zooming between viewports")
            self.ui.registered_link_button.setIcon(QIcon(str(icons_dir / 'slash_link_icon.svg')))
            self.ui.registered_link_button.setToolTip("Link zooming between viewports")

    def home_button_clicked(self, whichVol):
        """
        Slot for signal sent when user clicks the home button in the toolbar.
        Resets extent of all viewports to initial extent.
        """
        pass
        # # TODO: only update the viewports that are displaying the image
        # if whichVol == 0:
        #     # moving image
        #     for vp in self.viewports:
        #         if vp.initial_extent is not None:
        #             vp.ax.set_position(vp.initial_extent)
        #             vp.canvas.draw()
        #
        # elif whichVol == 1:
        #     # fixed image
        #     for vp in self.viewports:
        #         if vp.initial_extent is not None:
        #             vp.ax.set_position(vp.initial_extent)
        #             vp.canvas.draw()
        # else:  # registered image
        #     for vp in self.viewports:
        #         if vp.initial_extent is not None:
        #             vp.ax.set_position(vp.initial_extent)
        #             vp.canvas.draw()

    def toggle_slice_guides(self, vol_type):
        """
        Toggle the visibility of the horizontal and vertical slice guides (crosshairs) in all viewports.

        :param vol_type: ('moving', 'fixed', 'reg') string indicating which image viewports to toggle slice guides
               visibility for.
        :return:
        """
        if vol_type in ['moving', 'fixed', 'reg']:
            if vol_type == 'moving':
                for vp_id in [0, 1, 2]:
                    self.viewports[vp_id].show_slice_guides = not self.viewports[vp_id].show_slice_guides
                    self.viewports[vp_id].horizontal_line.setVisible(self.viewports[vp_id].show_slice_guides)
                    self.viewports[vp_id].vertical_line.setVisible(self.viewports[vp_id].show_slice_guides)
            elif vol_type == 'fixed':
                for vp_id in [3, 4, 5]:
                    self.viewports[vp_id].show_slice_guides = not self.viewports[vp_id].show_slice_guides
                    self.viewports[vp_id].horizontal_line.setVisible(self.viewports[vp_id].show_slice_guides)
                    self.viewports[vp_id].vertical_line.setVisible(self.viewports[vp_id].show_slice_guides)
            elif vol_type == 'reg':
                for vp_id in [6, 7, 8]:
                    self.viewports[vp_id].show_slice_guides = not self.viewports[vp_id].show_slice_guides
                    self.viewports[vp_id].horizontal_line.setVisible(self.viewports[vp_id].show_slice_guides)
                    self.viewports[vp_id].vertical_line.setVisible(self.viewports[vp_id].show_slice_guides)

    # slots for moving, fixed, and registered combo boxes and close buttons ============================================
    def toggle_show_hide_button(self, state, which_image, which_layer=0):
        if state:
            if which_image == 'moving':
                if which_layer == 2:  # NOTE: a cheat -- 2 == landmarks table
                    # self.ui.show_hide_moving_lm_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
                    self.ui.moving_lm_table_frame.setVisible(True)
                    # self.ui.moving_lm_label_frame.setVisible(True)
                else:
                    if which_layer == 0:
                        self.ui.show_hide_moving_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
                    elif which_layer == 1:
                        self.ui.show_hide_mov_overlay_button.setIcon(QIcon(str(icons_dir / 'eye.png')))

                    self.viewports[0].show_layer(which_layer)
                    self.viewports[1].show_layer(which_layer)
                    self.viewports[2].show_layer(which_layer)

            elif which_image == 'fixed':
                if which_layer == 2:  # a cheat - 2 == landmarks table
                    # self.ui.show_hide_fixed_lm_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
                    self.ui.fixed_lm_table_frame.setVisible(True)
                    # self.ui.fixed_lm_label_frame.setVisible(True)
                else:
                    if which_layer == 0:
                        self.ui.show_hide_fixed_button.setIcon(QIcon(str(icons_dir / 'eye.png')))
                    elif which_layer == 1:
                        self.ui.show_hide_fix_overlay_button.setIcon(QIcon(str(icons_dir / 'eye.png')))

                    self.viewports[3].show_layer(which_layer)
                    self.viewports[4].show_layer(which_layer)
                    self.viewports[5].show_layer(which_layer)

            elif which_image == 'reg':
                self.viewports[6].show_layer(which_layer)
                self.viewports[7].show_layer(which_layer)
                self.viewports[8].show_layer(which_layer)
        else:
            if which_image == 'moving':
                if which_layer == 2:  # a cheat - 2 == landmarks table
                    # self.ui.show_hide_moving_lm_button.setIcon(QIcon(str(icons_dir / 'eye-close.png')))
                    self.ui.moving_lm_table_frame.setVisible(False)
                    # self.ui.moving_lm_label_frame.setVisible(False)
                else:
                    if which_layer == 0:
                        self.ui.show_hide_moving_button.setIcon(QIcon(str(icons_dir / 'eye-close.png')))
                    elif which_layer == 1:
                        self.ui.show_hide_mov_overlay_button.setIcon(QIcon(str(icons_dir / 'eye-close.png')))

                    self.viewports[0].hide_layer(which_layer)
                    self.viewports[1].hide_layer(which_layer)
                    self.viewports[2].hide_layer(which_layer)

            elif which_image == 'fixed':
                if which_layer == 2:  # a cheat - 2 == landmarks table
                    # self.ui.show_hide_fixed_lm_button.setIcon(QIcon(str(icons_dir / 'eye-close.png')))
                    self.ui.fixed_lm_table_frame.setVisible(False)
                    # self.ui.fixed_lm_label_frame.setVisible(False)
                else:
                    if which_layer == 0:
                        self.ui.show_hide_fixed_button.setIcon(QIcon(str(icons_dir / 'eye-close.png')))
                    elif which_layer == 1:
                        self.ui.show_hide_fix_overlay_button.setIcon(QIcon(str(icons_dir / 'eye-close.png')))

                    self.viewports[3].hide_layer(which_layer)
                    self.viewports[4].hide_layer(which_layer)
                    self.viewports[5].hide_layer(which_layer)

            elif which_image == 'reg':
                self.viewports[6].hide_layer(which_layer)
                self.viewports[7].hide_layer(which_layer)
                self.viewports[8].hide_layer(which_layer)

        self.refresh_all_viewports()

    def moving_combo_changed(self, combo_idx):
        """
        Updates the overlay image in the fixed viewports, updates current landmarks,
        and refreshes state of the main window
        """
        if len(self.moving_volumes) == 0:
            return

        # clear the markers
        for vp in self.viewports:
            vp.marker_delete_all()

        self.current_moving_volume_index = combo_idx - 1  # account for header line in combo box
        if combo_idx > 0:
            self.moving_volume = self.moving_volumes[self.current_moving_volume_index]
            self.ui.current_moving_combo.setToolTip(self.moving_volume.file_base_name)
            self.viewports[0].add_layer(self.moving_volume, 0)
            self.viewports[1].add_layer(self.moving_volume, 0)
            self.viewports[2].add_layer(self.moving_volume, 0)
            # Log volume selection
            vol_name = self.moving_volume.file_base_name if hasattr(self.moving_volume, 'file_base_name') else "moving"
            self.print_log_message(f"Selected moving image: {vol_name}")
        else:
            self.moving_volume = None
            self.ui.current_moving_combo.setToolTip(None)
            self.viewports[0].add_layer(None, 0)
            self.viewports[1].add_layer(None, 0)
            self.viewports[2].add_layer(None, 0)
            # Log volume cleared
            self.print_log_message("Cleared moving image selection")

        # --- update current landmarks using the (moving, fixed) pairing key ---
        if self.fixed_volume is not None and self.moving_volume is not None:
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            self.current_landmarks = self.lm_mgr.list(pair)
        else:
            self.current_landmarks = []

        # add corresponding markers to the viewports
        if len(self.current_landmarks) > 0:
            for lm in self.current_landmarks:
                for vp_idx in [0, 1, 2]:
                    self.viewports[vp_idx].add_marker(
                        lm.source_voxel_coords[0], lm.source_voxel_coords[1], lm.source_voxel_coords[2],
                        0, lm.source_marker_id
                    )
                for vp_idx in [3, 4, 5]:
                    self.viewports[vp_idx].add_marker(
                        lm.target_voxel_coords[0], lm.target_voxel_coords[1], lm.target_voxel_coords[2],
                        0, lm.target_marker_id
                    )

        # Set active display settings image and check corresponding button
        if combo_idx > 0 and self.moving_volume is not None:
            self.active_moving_settings_image = self.moving_volume
            # Show the display settings widgets before setting the image
            self._show_display_settings_widgets(self.moving_display_settings)
            self.moving_display_settings.set_active_image(self.active_moving_settings_image)
            # Update colormap selection to match the image's current colormap
            if hasattr(self.moving_display_settings, 'colormap_combo') and self.moving_volume.colormap_index is not None:
                self.moving_display_settings.colormap_combo.setCurrentIndex(self.moving_volume.colormap_index)
            self.moving_display_settings.refresh()
            # Block signals to avoid triggering buttonClicked when programmatically setting state
            self.ui.active_moving_group.blockSignals(True)
            self.ui.moving_active.setChecked(True)
            self.ui.moving_overlay_active.setChecked(False)
            self.ui.active_moving_group.blockSignals(False)

        # refresh the state of the main window widgets
        self.refresh(False)

    def moving_overlay_combo_changed(self, index):
        """
        Updates the overlay image in the fixed viewports, updates current landmarks,
        and refreshes state of the main window

        :param index:
        :return:
        """
        if len(self.moving_overlay_volumes) == 0:
            return

        self.current_moving_overlay_volume_index = index - 1  # account for header line in combo box
        if index > 0:
            self.moving_overlay = self.moving_overlay_volumes[self.current_moving_overlay_volume_index]
            self.viewports[0].add_layer(self.moving_overlay, 1)
            self.viewports[1].add_layer(self.moving_overlay, 1)
            self.viewports[2].add_layer(self.moving_overlay, 1)
            # Log overlay selection
            overlay_name = self.moving_overlay.file_base_name if hasattr(self.moving_overlay, 'file_base_name') else "moving overlay"
            self.print_log_message(f"Selected moving overlay: {overlay_name}")
        else:
            self.moving_overlay = None
            self.viewports[0].add_layer(None, 1)
            self.viewports[1].add_layer(None, 1)
            self.viewports[2].add_layer(None, 1)
            # Log overlay cleared
            self.print_log_message("Cleared moving overlay selection")

        # Set active display settings image and check corresponding button
        if index > 0 and self.moving_overlay is not None:
            self.active_moving_settings_image = self.moving_overlay
            # Show the display settings widgets before setting the image
            self._show_display_settings_widgets(self.moving_display_settings)
            self.moving_display_settings.set_active_image(self.active_moving_settings_image)
            # Update colormap selection to match the image's current colormap
            if hasattr(self.moving_display_settings, 'colormap_combo') and self.moving_overlay.colormap_index is not None:
                self.moving_display_settings.colormap_combo.setCurrentIndex(self.moving_overlay.colormap_index)
            self.moving_display_settings.refresh()
            # Block signals to avoid triggering buttonClicked when programmatically setting state
            self.ui.active_moving_group.blockSignals(True)
            self.ui.moving_overlay_active.setChecked(True)
            self.ui.moving_active.setChecked(False)
            self.ui.active_moving_group.blockSignals(False)

        self.refresh(False)

    def moving_close_button_clicked(self):
        if self.moving_volume is None:
            return
        if self.current_moving_volume_index < 0:
            return
        combo_idx = self.ui.current_moving_combo.currentIndex()
        if combo_idx == 0:
            return

        # Check if landmarks exist for the current image pair
        has_landmarks = False
        landmark_count = 0
        if self.moving_volume is not None and self.fixed_volume is not None:
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            landmarks = self.lm_mgr.list(pair)
            if landmarks:
                has_landmarks = True
                landmark_count = len(landmarks)

        # Show appropriate warning based on whether landmarks exist
        if has_landmarks:
            message = (f"Closing this moving image will delete {landmark_count} landmark(s) associated with "
                      f"the current image pair.\n\n"
                      f"Consider exporting your landmarks first using 'Landmarks > Save to File'.\n\n"
                      f"Do you want to proceed with closing the image?")
            warning_msg = f"Warning: Closing moving image will delete {landmark_count} landmark(s)"
            self.print_log_message(warning_msg)
            reply = QMessageBox.warning(self, "Close image?",
                                       message,
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            reply = QMessageBox.question(self, "Close image?",
                                         "Do you want to close the selected moving image?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        # Clear landmarks for the current pair if they exist (before closing the volume)
        if has_landmarks and self.moving_volume is not None and self.fixed_volume is not None:
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            # Exit edit mode if active
            if self.lm_edit_mode:
                for vp in self.viewports:
                    vp.marker_set_edit_mode(False)
                self.lm_edit_mode = False
            # Exit add mode if active
            for vp in self.viewports:
                vp.marker_set_add_mode(False)
            # Clear all markers from viewports
            for vp in self.viewports:
                vp.marker_delete_all()
                vp.marker_clear_selected()
            # Clear landmarks from manager
            self.lm_mgr.clear_pair(pair)
            self.current_landmarks = []
            self.print_log_message(f"Cleared {landmark_count} landmark(s) after closing moving image")

        name = self.moving_volume.full_file_name
        self.moving_volume = None

        # clear the moving vol
        del self.moving_volumes[self.current_moving_volume_index]
        # self.ui.current_moving_combo.removeItem(combo_idx) # accounts for header line in combo box

        self.viewports[0].add_layer(None, 0)
        self.viewports[1].add_layer(None, 0)
        self.viewports[2].add_layer(None, 0)

        # populate the moving combo boxes
        self.moving_update_combo()

        # as a convenience, set the current moving image to the newly loaded image
        self.ui.current_moving_combo.setCurrentIndex(len(self.moving_volumes))  # doesn't trigger slot
        self.moving_combo_changed(len(self.moving_volumes))

        self.refresh(False)

        self.print_log_message(f'Closed moving image {name}')

    def moving_overlay_close_button_clicked(self):
        if self.moving_overlay is None:
            return
        if self.current_moving_overlay_volume_index < 0:
            return
        combo_idx = self.ui.moving_overlay_combo.currentIndex()
        if combo_idx == 0:
            return

        reply = QMessageBox.question(self, "Close image?",
                                     "Do you want to close the selected moving image?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        # # clear the registered vols
        # self.ui.current_registered_combo.clear()
        # self.ui.current_registered_combo.addItem("No registered images")
        # self.registered_volumes = []
        # self.current_registered_volume_index = 0
        # self.registered_volume = None
        # self.viewports[6].add_layer(None, 0)
        # self.viewports[6].add_layer(None, 1)
        # self.viewports[7].add_layer(None, 0)
        # self.viewports[7].add_layer(None, 1)
        # self.viewports[8].add_layer(None, 0)
        # self.viewports[8].add_layer(None, 1)
        # # clear deformation fields
        # self.current_deformation_field = None
        # self.deformation_fields = []
        # # reset session stats
        # self.session_shape = None  # shape of the moving or fixed image at the time the session was started
        # self.session_resolution = None  # resolution of the moving or fixed image at the time the session was started
        # self.session_origin = []  # origin of the moving or fixed image at the time the session was started

        name = self.moving_overlay.full_file_name
        self.moving_overlay = None
        # clear the moving vol
        del self.moving_overlay_volumes[self.current_moving_overlay_volume_index]
        self.ui.moving_overlay_combo.removeItem(combo_idx) # accounts for header line in combo box

        self.viewports[0].add_layer(None, 1)
        self.viewports[1].add_layer(None, 1)
        self.viewports[2].add_layer(None, 1)

        # populate the moving combo boxes
        self.moving_overlay_update_combo()

        # as a convenience, set the current moving image to the newly loaded image
        self.ui.moving_overlay_combo.setCurrentIndex(len(self.moving_overlay_volumes))  # doesn't trigger slot
        self.moving_overlay_combo_changed(len(self.moving_overlay_volumes))

        # update list of display settings images (this calls disp_refresh())
        # self.disp_update_images('moving')

        self.print_log_message(f'Closed moving overlay image {name}')

    def moving_update_combo(self):
        # update the moving combo box
        self.ui.current_moving_combo.blockSignals(True)
        self.ui.current_moving_combo.clear()  # sets combo box index to -1
        if len(self.moving_volumes) == 0:
            self.ui.current_moving_combo.addItem('No moving images loaded.') # increments combo index by 1
        else:
            self.ui.current_moving_combo.addItem('Select moving image...') # increments combo index by 1
            for mov in self.moving_volumes:
                self.ui.current_moving_combo.addItem(mov.file_base_name)  # increments combo index by 1
        # deselect the current moving image
        self.current_moving_volume_index = -1
        self.ui.current_moving_combo.setCurrentIndex(0)
        self.ui.current_moving_combo.blockSignals(False)

    def moving_overlay_update_combo(self):
        # update the moving overlay combo box
        self.ui.moving_overlay_combo.currentIndexChanged.disconnect(self.moving_overlay_combo_changed)
        self.ui.moving_overlay_combo.clear()
        if len(self.moving_overlay_volumes) == 0:
            self.ui.moving_overlay_combo.addItem('No moving overlays loaded.') # increments combo index by 1
        else:
            self.ui.moving_overlay_combo.addItem('Select moving overlay...') # increments combo index by 1
            for mov in self.moving_overlay_volumes:
                self.ui.moving_overlay_combo.addItem(mov.file_base_name) # increments combo index by 1
        # deselect the current moving overlay image
        self.current_moving_overlay_volume_index = -1
        self.ui.moving_overlay_combo.setCurrentIndex(0)
        self.ui.moving_overlay_combo.currentIndexChanged.connect(self.moving_overlay_combo_changed)

    def moving_resample_button_clicked(self):
        """
        Resample the moving image to match the fixed image geometry. This is done by resampling the moving image
        to the fixed image's shape, origin, and resolution. The resampled image is added to the moving image list.

        :return:
        """
        if self.fixed_volume is None or self.moving_volume is None:
            return

        diffs = self.compare_image_geometry(self.moving_volume, self.fixed_volume)
        if diffs is not None:
            # current moving volume must be the same geometry as fixed volume
            # ask user if they want to resample the moving volume to match the fixed volume
            # build the differences string from the diffs dictionary (shape, origin, resolution)
            diff_str = ""
            for key, val in diffs.items():
                diff_str += f"{key}: {val}\n\n"
            choice = QMessageBox.question(self, 'Resample?',
                                          f"The geometry of the moving image does not match the fixed image:"
                                          f"\n{diff_str}\n"
                                          "Do you want to resample the moving image to match the fixed image?",
                                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
            if choice == QMessageBox.Cancel:
                return
            elif choice == QMessageBox.No:
                return
            elif choice == QMessageBox.Yes:
                resampled_image = self.resample_lm_nifti_image(self.moving_volume, self.fixed_volume)
                if resampled_image is not None:
                    # compute a fresh UID for the resampled volume
                    uid_meta = compute_image_uid_from_nifti(resampled_image)
                    
                    # Auto-generate a unique temporary name for the resampled image
                    base_name = f"{self.moving_volume.file_base_name}_resampled"
                    temp_name = base_name
                    moving_dir = Path(self.moving_volume.full_file_name).parent
                    temp_full_path = self.ensure_nii_path(moving_dir, temp_name)
                    
                    # Check if file already exists or if a volume with this name exists in memory
                    counter = 1
                    while temp_full_path.exists() or any(vol.file_base_name == temp_name for vol in self.moving_volumes):
                        temp_name = f"{base_name}_{counter}"
                        temp_full_path = self.ensure_nii_path(moving_dir, temp_name)
                        counter += 1
                    
                    new_lm_image = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                               continuous_palette_names=self.continous_palette,
                                            _is_discrete=self.moving_volume.is_discrete)
                    new_lm_image.populate_with_nifti(resampled_image, str(temp_full_path), uid_meta=uid_meta)

                    # Copy colormap and display levels from the original moving image
                    if self.moving_volume.is_discrete:
                        # For discrete images, copy the LUT directly (labels should be the same after resampling)
                        if self.moving_volume.lut is not None:
                            new_lm_image.lut = self.moving_volume.lut.copy()
                        if self.moving_volume.display_range is not None:
                            new_lm_image.display_range = self.moving_volume.display_range
                    else:
                        # For continuous images, copy colormap name and display range
                        if self.moving_volume.colormap_name is not None:
                            new_lm_image.set_continuous_colormap(self.moving_volume.colormap_name)
                        if self.moving_volume.display_range is not None:
                            new_lm_image.display_range = self.moving_volume.display_range

                    # Ask user if they want to save the resampled image using Save As dialog
                    save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                        self, 
                        'Save Resampled Image?',
                        str(temp_full_path),
                        'NIfTI files (*.nii.gz)'
                    )
                    
                    if save_path:
                        # Ensure the file path ends with .nii.gz
                        if not save_path.endswith('.nii.gz'):
                            save_path += '.nii.gz'
                        
                        # User chose to save - update the image with the chosen path
                        new_lm_image.full_file_name = save_path
                        # Extract the base name (filename without .nii.gz extension)
                        save_path_obj = Path(save_path)
                        # For .nii.gz files, stem gives us the filename with .nii, so we need to remove that too
                        new_lm_image.file_base_name = Path(save_path_obj.stem).stem
                        self.write_image3d_as_nifti(new_lm_image, save_path)
                        # LMSidecarUtils.write_sidecar(new_lm_image.full_file_name, uid_meta, landmarker_version=self.version, parent=self)
                        self.print_log_message("Resampled moving image saved as " + new_lm_image.file_base_name)
                    else:
                        # User cancelled the save dialog
                        self.print_log_message("Resampled image created but not saved")

                    # all set now, go ahead and set the fixed volume as the target
                    self.moving_volumes.append(new_lm_image)
                    self.moving_update_combo()
                    self.ui.current_moving_combo.setCurrentIndex(len(self.moving_volumes))
                    self.refresh()
        else:
            # no moving volume to compare, so ok to set target
            self.refresh()

    def ensure_nii_path(self, base_dir: Path, user_text: str) -> Path:
        """
        If user_text is absolute, respect it. Otherwise, place it in base_dir.
        Ensure extension is .nii or .nii.gz (default to .nii.gz).
        """
        name = user_text.strip()
        if not name:
            return None

        p = Path(name)
        # If relative, build under base_dir
        if not p.is_absolute():
            p = base_dir / p.name

        # Normalize extension:
        #   keep .nii.gz or .nii if already provided; otherwise append .nii.gz
        suffs = [s.lower() for s in p.suffixes]
        if suffs[-2:] == ['.nii', '.gz'] or suffs[-1:] == ['.nii']:
            return p
        if p.name.lower().endswith('.nii.gz'):
            return p  # (covers odd suffix parsing cases)
        if p.suffix.lower() == '.nii':
            return p
        return p.with_name(p.name + '.nii.gz')

    def fixed_combo_changed(self, index):
        """
        Slot. Updates the fixed image in the fixed viewports, updates current landmarks,
        and refreshes the state of the main window
        """
        if len(self.fixed_volumes) == 0:
            return

        # clear the markers
        for vp in self.viewports:
            vp.marker_delete_all()

        self.current_fixed_volume_index = index - 1  # accounts for combo box list header line
        if index > 0:
            self.fixed_volume = self.fixed_volumes[self.current_fixed_volume_index]
            self.ui.current_fixed_combo.setToolTip(self.fixed_volume.file_base_name)
            self.viewports[3].add_layer(self.fixed_volume, 0)
            self.viewports[4].add_layer(self.fixed_volume, 0)
            self.viewports[5].add_layer(self.fixed_volume, 0)
            # Log volume selection
            vol_name = self.fixed_volume.file_base_name if hasattr(self.fixed_volume, 'file_base_name') else "fixed"
            self.print_log_message(f"Selected fixed image: {vol_name}")
        else:
            self.fixed_volume = None
            self.ui.current_fixed_combo.setToolTip(None)
            self.viewports[3].add_layer(None, 0)
            self.viewports[4].add_layer(None, 0)
            self.viewports[5].add_layer(None, 0)
            # Log volume cleared
            self.print_log_message("Cleared fixed image selection")

        # --- update current landmarks using the (moving, fixed) pairing key ---
        if self.fixed_volume is not None and self.moving_volume is not None:
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            self.current_landmarks = self.lm_mgr.list(pair)
        else:
            self.current_landmarks = []

        # clear the registered image and overlay
        self.registered_volume = None
        self.viewports[6].add_layer(None, 0)
        self.viewports[7].add_layer(None, 0)
        self.viewports[8].add_layer(None, 0)
        self.viewports[6].add_layer(None, 1)
        self.viewports[7].add_layer(None, 1)
        self.viewports[8].add_layer(None, 1)

        # Set active display settings image and check corresponding button
        if index > 0 and self.fixed_volume is not None:
            self.active_fixed_settings_image = self.fixed_volume
            # Show the display settings widgets before setting the image
            self._show_display_settings_widgets(self.fixed_display_settings)
            self.fixed_display_settings.set_active_image(self.active_fixed_settings_image)
            # Update colormap selection to match the image's current colormap
            if hasattr(self.fixed_display_settings, 'colormap_combo') and self.fixed_volume.colormap_index is not None:
                self.fixed_display_settings.colormap_combo.setCurrentIndex(self.fixed_volume.colormap_index)
            self.fixed_display_settings.refresh()
            # Block signals to avoid triggering buttonClicked when programmatically setting state
            self.ui.active_fixed_group.blockSignals(True)
            self.ui.fixed_active.setChecked(True)
            self.ui.fixed_overlay_active.setChecked(False)
            self.ui.active_fixed_group.blockSignals(False)

        # refresh the state of the main window widgets
        self.refresh(False)

    def fixed_overlay_combo_changed(self, index):
        """
        Updates the overlay image in the fixed viewports, updates current landmarks,
        and refreshes state of the main window

        :param index:
        :return:
        """
        if len(self.fixed_overlay_volumes) == 0:
            return

        self.current_fixed_overlay_volume_index = index - 1  # accounts for combo box list header line
        if index > 0:
            self.fixed_overlay = self.fixed_overlay_volumes[self.current_fixed_overlay_volume_index]
            self.viewports[3].add_layer(self.fixed_overlay, 1)
            self.viewports[4].add_layer(self.fixed_overlay, 1)
            self.viewports[5].add_layer(self.fixed_overlay, 1)
            # Log overlay selection
            overlay_name = self.fixed_overlay.file_base_name if hasattr(self.fixed_overlay, 'file_base_name') else "fixed overlay"
            self.print_log_message(f"Selected fixed overlay: {overlay_name}")
        else:
            self.fixed_overlay = None
            self.viewports[3].add_layer(None, 1)
            self.viewports[4].add_layer(None, 1)
            self.viewports[5].add_layer(None, 1)
            # Log overlay cleared
            self.print_log_message("Cleared fixed overlay selection")

        # # if landmarks exist, update voxel coordinates
        # if self.current_landmarks:
        #     for lm in self.current_landmarks:
        #         lm.update_fixed_voxel_coords(self.fixed_volume)

        # # clear the registered image
        # self.registered_volume = None
        # self.viewports[6].set_current_bg_volume(None)
        # self.viewports[7].set_current_bg_volume(None)
        # self.viewports[8].set_current_bg_volume(None)

        # Set active display settings image and check corresponding button
        if index > 0 and self.fixed_overlay is not None:
            self.active_fixed_settings_image = self.fixed_overlay
            # Show the display settings widgets before setting the image
            self._show_display_settings_widgets(self.fixed_display_settings)
            self.fixed_display_settings.set_active_image(self.active_fixed_settings_image)
            # Update colormap selection to match the image's current colormap
            if hasattr(self.fixed_display_settings, 'colormap_combo') and self.fixed_overlay.colormap_index is not None:
                self.fixed_display_settings.colormap_combo.setCurrentIndex(self.fixed_overlay.colormap_index)
            self.fixed_display_settings.refresh()
            # Block signals to avoid triggering buttonClicked when programmatically setting state
            self.ui.active_fixed_group.blockSignals(True)
            self.ui.fixed_overlay_active.setChecked(True)
            self.ui.fixed_active.setChecked(False)
            self.ui.active_fixed_group.blockSignals(False)

        self.refresh(False)

        # # update list of display settings images (this calls disp_refresh())
        # self.disp_update_images("fixed")

    def fixed_close_button_clicked(self):
        if self.fixed_volume is None:
            return
        if self.current_fixed_volume_index < 0:
            return

        combo_idx = self.ui.current_fixed_combo.currentIndex()
        if combo_idx == 0:
            return

        # Check if landmarks exist for the current image pair
        has_landmarks = False
        landmark_count = 0
        if self.moving_volume is not None and self.fixed_volume is not None:
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            landmarks = self.lm_mgr.list(pair)
            if landmarks:
                has_landmarks = True
                landmark_count = len(landmarks)

        # Show appropriate warning based on whether landmarks exist
        if has_landmarks:
            message = (f"Closing this fixed image will delete {landmark_count} landmark(s) associated with "
                      f"the current image pair.\n\n"
                      f"Consider exporting your landmarks first using 'Landmarks > Save to File'.\n\n"
                      f"Do you want to proceed with closing the image?")
            warning_msg = f"Warning: Closing fixed image will delete {landmark_count} landmark(s)"
            self.print_log_message(warning_msg)
            reply = QMessageBox.warning(self, "Close image?",
                                       message,
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            reply = QMessageBox.question(self, "Close image?",
                                         "Do you want to close the selected fixed image?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        # Clear landmarks for the current pair if they exist (before closing the volume)
        if has_landmarks and self.moving_volume is not None and self.fixed_volume is not None:
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            # Exit edit mode if active
            if self.lm_edit_mode:
                for vp in self.viewports:
                    vp.marker_set_edit_mode(False)
                self.lm_edit_mode = False
            # Exit add mode if active
            for vp in self.viewports:
                vp.marker_set_add_mode(False)
            # Clear all markers from viewports
            for vp in self.viewports:
                vp.marker_delete_all()
                vp.marker_clear_selected()
            # Clear landmarks from manager
            self.lm_mgr.clear_pair(pair)
            self.current_landmarks = []
            self.print_log_message(f"Cleared {landmark_count} landmark(s) after closing fixed image")
            # Refresh the UI to update landmark tables

        name = self.fixed_volume.full_file_name
        self.fixed_volume = None

        del self.fixed_volumes[self.current_fixed_volume_index]
        self.ui.current_fixed_combo.removeItem(combo_idx)  # accounts for header line in combo box

        self.viewports[3].add_layer(None ,0)
        self.viewports[4].add_layer(None, 0)
        self.viewports[5].add_layer(None, 0)

        # populate the moving combo boxes
        self.fixed_update_combo()

        # as a convenience, set the current moving image to the newly loaded image
        self.ui.current_fixed_combo.setCurrentIndex(len(self.fixed_volumes))  # doesn't trigger slot
        self.fixed_combo_changed(len(self.fixed_volumes))

        self.refresh(False)

        self.print_log_message(f'Closed fixed image {name}')

    def fixed_overlay_close_button_clicked(self):
        if self.fixed_overlay is None:
            return
        if self.current_fixed_overlay_volume_index < 0:
            return

        combo_idx = self.ui.fixed_overlay_combo.currentIndex()
        if combo_idx == 0:
            return

        reply = QMessageBox.question(self, "Close image?",
                                     "Do you want to close the selected fixed image?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        # # clear the registered vols
        # self.ui.current_registered_combo.clear()
        # self.ui.current_registered_combo.addItem("No registered images")
        # self.registered_volumes = []
        # self.current_registered_volume_index = 0
        # self.registered_volume = None
        # self.viewports[6].add_layer(None, 0)
        # self.viewports[6].add_layer(None, 1)
        # self.viewports[7].add_layer(None, 0)
        # self.viewports[7].add_layer(None, 1)
        # self.viewports[8].add_layer(None, 0)
        # self.viewports[8].add_layer(None, 1)
        # # clear deformation fields
        # self.current_deformation_field = None
        # self.deformation_fields = []
        # # reset session stats
        # self.session_shape = None  # shape of the moving or fixed image at the time the session was started
        # self.session_resolution = None  # resolution of the moving or fixed image at the time the session was started
        # self.session_origin = []  # origin of the moving or fixed image at the time the session was started

        del self.fixed_overlay_volumes[self.current_fixed_overlay_volume_index]
        self.ui.fixed_overlay_combo.removeItem(combo_idx)

        name = self.fixed_overlay.full_file_name
        self.fixed_overlay = None

        self.viewports[3].add_layer(None ,1)
        self.viewports[4].add_layer(None, 1)
        self.viewports[5].add_layer(None, 1)

        # populate the moving combo boxes
        self.fixed_overlay_update_combo()

        # as a convenience, set the current moving image to the newly loaded image
        self.ui.fixed_overlay_combo.setCurrentIndex(len(self.fixed_overlay_volumes))  # doesn't trigger slot
        self.fixed_overlay_combo_changed(len(self.fixed_overlay_volumes))

        # update list of display settings images (this calls disp_refresh())
        # self.disp_update_images('fixed')

        self.print_log_message(f'Closed fixed overlay image  {name}')

    def fixed_update_combo(self):
        # update the fixed combo box
        self.ui.current_fixed_combo.currentIndexChanged.disconnect(self.fixed_combo_changed)
        self.ui.current_fixed_combo.clear()
        if len(self.fixed_volumes) == 0:
            self.ui.current_fixed_combo.addItem('No fixed images loaded.')
        else:
            self.ui.current_fixed_combo.addItem('Select fixed image...')
            for fv in self.fixed_volumes:
                self.ui.current_fixed_combo.addItem(fv.file_base_name)
        self.current_fixed_volume_index = -1
        self.ui.current_fixed_combo.setCurrentIndex(0)
        self.ui.current_fixed_combo.currentIndexChanged.connect(self.fixed_combo_changed)

    def fixed_overlay_update_combo(self):
        # update the fixed overlay combo box
        self.ui.fixed_overlay_combo.currentIndexChanged.disconnect(self.fixed_overlay_combo_changed)
        self.ui.fixed_overlay_combo.clear()
        if len(self.fixed_overlay_volumes) == 0:
            self.ui.fixed_overlay_combo.addItem('No fixed overlays loaded.')
        else:
            self.ui.fixed_overlay_combo.addItem('Select fixed overlay...')
            for fov in self.fixed_overlay_volumes:
                self.ui.fixed_overlay_combo.addItem(fov.file_base_name)
        self.current_fixed_overlay_volume_index = -1
        self.ui.fixed_overlay_combo.setCurrentIndex(0)
        self.ui.fixed_overlay_combo.currentIndexChanged.connect(self.fixed_overlay_combo_changed)

    def registered_close_button_clicked(self, vol_idx=None, confirm=True):
        if vol_idx is None:
            # use the currently selected image
            if self.registered_volume is None:
                return
            if self.current_registered_volume_index < 0:
                return

            vol_idx = self.ui.current_registered_combo.currentIndex() - 1
            if vol_idx < 0:
                return

            if confirm:
                reply = QMessageBox.question(self, "Delete registered image?",
                                             "Do you want to delete the selected registered image?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return

            name = self.registered_volume.full_file_name
        else:
            name = self.registered_volumes[vol_idx].full_file_name

            if confirm:
                reply = QMessageBox.question(self, "Delete registered image?",
                                             f"Do you want to delete the registered image {name}?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return

        self.registered_volume = None
        self.current_registered_volume_index = -1

        del self.registered_volumes[vol_idx]

        # re-populate the registered combo box
        self.registered_update_combo()

        # as a convenience, set the current moving image to the newly loaded image
        self.ui.current_registered_combo.setCurrentIndex(0)  # doesn't trigger slot
        self.registered_combo_changed(0)

        self.viewports[6].add_layer(None, 0)
        self.viewports[7].add_layer(None, 0)
        self.viewports[8].add_layer(None, 0)
        self.viewports[6].add_layer(None, 1)
        self.viewports[7].add_layer(None, 1)
        self.viewports[8].add_layer(None, 1)

        self.print_log_message(f'Closed registered image {name}')

    def registered_combo_changed(self, index):
        """ Slot for signal sent when the user changes the selection in the combo box for the moving image.
        Allows user to update which image is currently displayed as the moving image.
        :param index: integer automatically sent by the combo box widget
        :return: None
        """

        if len(self.registered_volumes) == 0:
            return

        self.current_registered_volume_index = index -1  # accounts for header line in combo box
        if index > 0:
            self.registered_volume = self.registered_volumes[self.current_registered_volume_index]
            self.ui.current_registered_combo.setToolTip(self.registered_volume.full_file_name)
            # self.fixed_volume = self.registered_volume.fixed_vol_reference
            self.viewports[6].add_layer(self.registered_volume, 0)
            self.viewports[7].add_layer(self.registered_volume, 0)
            self.viewports[8].add_layer(self.registered_volume, 0)
            self.viewports[6].add_layer(self.registered_volume.fixed_vol_reference, 1)
            self.viewports[7].add_layer(self.registered_volume.fixed_vol_reference, 1)
            self.viewports[8].add_layer(self.registered_volume.fixed_vol_reference, 1)
            # Apply current user alpha blending so it is preserved when switching registered images
            registered_blend_alpha = round(1.0 - self.alpha_slider_value_pct, 2)
            fixed_blend_alpha = round(self.alpha_slider_value_pct, 2)
            self.registered_volume.opacity = registered_blend_alpha
            self.fixed_volume.blend_opacity = fixed_blend_alpha
            for vp_idx in [6, 7, 8]:
                self.viewports[vp_idx].refresh_preserve_extent(use_blend_opacity=True)
            # Log volume selection
            vol_name = self.registered_volume.file_base_name if hasattr(self.registered_volume, 'file_base_name') else "registered"
            self.print_log_message(f"Selected registered image: {vol_name}")
        else:
            self.registered_volume = None
            self.viewports[6].add_layer(None, 0)
            self.viewports[7].add_layer(None, 0)
            self.viewports[8].add_layer(None, 0)
            self.viewports[6].add_layer(None, 1)
            self.viewports[7].add_layer(None, 1)
            self.viewports[8].add_layer(None, 1)
            # Log volume cleared
            self.print_log_message("Cleared registered image selection")

        # # if landmarks exist, update voxel coordinates
        # if self.current_landmarks:
        #     for lm in self.current_landmarks:
        #         lm.update_moving_voxel_coords(self.moving_volume)

        # refresh the state of the main window widgets
        self.refresh(False)

        # update list of display settings images (this calls disp_refresh())
        # self.disp_update_images('registered')

    def registered_update_combo(self):
        self.ui.current_registered_combo.currentIndexChanged.disconnect(self.registered_combo_changed)
        self.ui.current_registered_combo.clear()
        if len(self.registered_volumes) == 0:
            self.ui.current_registered_combo.addItem('No registered images.')
        else:
            self.ui.current_registered_combo.addItem('Select registered image...')
            for rv in self.registered_volumes:
                self.ui.current_registered_combo.addItem(rv.full_file_name)
        self.current_registered_volume_index = -1
        self.ui.current_registered_combo.setCurrentIndex(0)
        self.ui.current_registered_combo.currentIndexChanged.connect(self.registered_combo_changed)

    def registered_alpha_slider_change(self, value, vp_id):
        """
        This is the slot for the signal sent by the "alpha blending" slider in the registered viewports.
        slider-down is 100 opacity for registered volume (background)
        slider-up is 100 opacity for fixed volume (foreground)
        """
        if self.registered_volume is None or self.fixed_volume is None:
            return

        self.ui.reg_fix_label_0.setText("Fixed\n" + str(value) + "%")
        self.ui.reg_reg_label_0.setText("Reg.\n" + str(100 - value) + "%")
        # Update labels for the specific slider/viewport that was changed
        # if vp_id == 0:
        #     self.ui.reg_fix_label_0.setText("Fixed\n" + str(value) + "%")
        #     self.ui.reg_reg_label_0.setText("Reg.\n" + str(100 - value) + "%")
        # elif vp_id == 1:
        #     self.ui.reg_fix_label_1.setText("Fixed\n" + str(value) + "%")
        #     self.ui.reg_reg_label_1.setText("Reg.\n" + str(100 - value) + "%")
        # elif vp_id == 2:
        #     self.ui.reg_fix_label_2.setText("Fixed\n" + str(value) + "%")
        #     self.ui.reg_reg_label_2.setText("Reg.\n" + str(100 - value) + "%")
        
        self.alpha_slider_value_pct = value / 100  # scale to range 0.0 -> 1.0
        registered_blend_alpha = round(1.0 - self.alpha_slider_value_pct, 2)  # the registered image
        fixed_blend_alpha = round(self.alpha_slider_value_pct, 2)  # the fixed image

        self.registered_volume.opacity = registered_blend_alpha
        # Update blend_opacity for fixed volume (used in registered viewports)
        # Note: This is independent of self.fixed_volume.opacity (used in fixed viewports)
        self.fixed_volume.blend_opacity = fixed_blend_alpha
        # Use blend_opacity for registered viewports (6, 7, 8)
        for vp_idx in [6,7,8]:
            self.viewports[vp_idx].refresh_preserve_extent(use_blend_opacity=True)

    def registered_expand_button_clicked(self):
        if self.reg_vol_window is None:
            self.ui.centralwidget.layout().removeWidget(self.ui.registered_vol_main_frame)
            self.ui.expand_window_button.setIcon(QIcon(str(icons_dir / 'collapseIcon.svg')))
            self.ui.registered_vol_main_frame.setParent(None)
            self.reg_vol_window = RegisteredViewportsWindow(self.ui.registered_vol_main_frame, self)
            self.reg_vol_window.show()
        else:
            self.reg_vol_window.close()

    def registered_write(self, datatype):
        """ This method saves the registered image as a nifti or DICOM.
            No return type.

        """
        success = False
        if self.registered_volume is None:
            error_msg = "No registered image to save."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return success
        if datatype == "nifti":
            try:
                nifti_image = nib.Nifti1Image(self.registered_volume.data, self.registered_volume.transform)
                nifti_image.update_header()
                nifti_image.to_filename(self.registered_save_filename)
                success = True
            except Exception as e:
                error_msg = f"An error occurred while saving to nifti: {e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
        elif datatype == 'dicom':
            # loop through each 2D slice in the 3D array
            try:
                for slice_number, slice_data in enumerate(self.registered_volume.data):
                    # create a new DICOM dataset for each slice
                    ds = FileDataset(os.path.join(self.registered_save_filename, f'slice_{slice_number}.dcm'), {},
                                     file_meta={}, preamble=b"\0" * 128)

                    # set required DICOM attributes for each slice
                    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # example for Enhanced MR Image Storage SOP Class
                    ds.SOPInstanceUID = 'YOUR_INSTANCE_UID'  # TODO: generate a unique instance UID for each slice
                    ds.Modality = 'MR'
                    ds.PatientsName = 'Patient Name'
                    # TODO: ... set more attributes as needed

                    # set the pixel data (2D slice)
                    ds.PixelData = slice_data.tobytes()
                    ds.Rows, ds.Columns = slice_data.shape

                    # save the DICOM file for this slice
                    ds.save_as(os.path.join(self.registered_save_filename, f'slice_{slice_number}.dcm'))
                    success = True
            except Exception as e:
                error_msg = f"An error occurred while saving to DICOM: {e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
        else:
            # internal error
            error_msg = str(__name__ + " unsupported data type: " + str(datatype))
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Internal Error", error_msg, QMessageBox.Ok)

        return success

    def lm_new_cancel_button_clicked(self):
        """
        This push button has two functions: initializing the process of creating a new landmark, and canceling landmark
            creation. We control this by updating the string on the New button ("New" vs "Cancel").
            Initializing the process of creating a landmark means setting setting all viewports to marker add mode.
            This mode will make the viewports plot a point when there is a mouse click.
        :return:
        """
        if self.ui.moving_lm_new_cancel_button.text() == 'New' and self.ui.fixed_lm_new_cancel_button.text() == 'New':
            # set status for both images to 'creating waiting'
            self.lm_status = 2
            # Exit edit mode if active
            self.lm_edit_mode = False
            for vp in self.viewports:
                vp.marker_set_edit_mode(False)
            for vp in self.viewports:
                vp.marker_set_add_mode(True)
                vp.marker_clear_selected(notify=True)  # this clears any selected landmark if true
            
            # Log starting landmark creation
            if self.moving_volume is not None and self.fixed_volume is not None:
                moving_name = self.moving_volume.file_base_name if hasattr(self.moving_volume, 'file_base_name') else "moving"
                fixed_name = self.fixed_volume.file_base_name if hasattr(self.fixed_volume, 'file_base_name') else "fixed"
                self.print_log_message(f"Starting landmark creation for {moving_name} <-> {fixed_name}")
        else:  # canceling landmark creation
            # user is canceling landmark creation, clear our temp points
            for vp in self.viewports:
                vp.marker_delete(self.temp_source_marker_id)
                vp.marker_delete(self.temp_target_marker_id)
                vp.marker_set_add_mode(False)
                vp.marker_set_edit_mode(False)
            self.temp_source_point_voxel = None
            self.temp_source_point_patient = None
            self.temp_target_point_voxel = None
            self.temp_target_point_patient = None
            self.temp_source_marker_id = None
            self.temp_target_marker_id = None

            self.lm_status = -1  # idle
            
            # Log canceling landmark creation
            self.print_log_message("Landmark creation canceled")

        # refreshes the buttons and landmark tables in the main window
        self.refresh(False)

    def lm_add_landmark(self,
                        src_vox,
                        src_pat,
                        src_mkr_id,
                        trg_vox,
                        trg_pat,
                        trg_mkr_id,
                        lm_id=None):
        """
        Add a landmark to current landmarks, then update session landmarks.

        :param src_vox: voxel coordinates in source (moving) image
        :param src_pat: patient coordinates in source (moving) image
        :param src_mkr_id: marker id in source (moving) image (optional)
        :param trg_vox: voxel coordinates in target (fixed) image
        :param trg_pat: patient coordinates in target (fixed) image
        :param trg_mkr_id: marker id in target (fixed) image (optional)
        :param lm_id: Optional landmark ID. If None, will be auto-generated.
        :return:
        """

        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        if lm_id is None:
            lm_id = self.lm_mgr.id_mgr.next_id(pair)
        lm = self.lm_mgr.add(pair,
                             self,
                              source_voxel_coords=src_vox,
                              source_patient_coords=src_pat,
                              source_marker_id=src_mkr_id,
                              target_voxel_coords=trg_vox,
                              target_patient_coords=trg_pat,
                              target_marker_id=trg_mkr_id,
                              lm_id=lm_id)

        # keep UI cache in sync if you still use it
        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        self.current_landmarks = self.lm_mgr.list(pair)
        
        # Log landmark creation with volume names and coordinates
        moving_name = self.moving_volume.file_base_name if self.moving_volume and hasattr(self.moving_volume, 'file_base_name') else "moving"
        fixed_name = self.fixed_volume.file_base_name if self.fixed_volume and hasattr(self.fixed_volume, 'file_base_name') else "fixed"
        
        # Format coordinates
        src_vox_str = f"[{src_vox[0]:.2f}, {src_vox[1]:.2f}, {src_vox[2]:.2f}]" if src_vox is not None else "N/A"
        src_pat_str = f"[{src_pat[0]:.2f}, {src_pat[1]:.2f}, {src_pat[2]:.2f}]" if src_pat is not None else "N/A"
        trg_vox_str = f"[{trg_vox[0]:.2f}, {trg_vox[1]:.2f}, {trg_vox[2]:.2f}]" if trg_vox is not None else "N/A"
        trg_pat_str = f"[{trg_pat[0]:.2f}, {trg_pat[1]:.2f}, {trg_pat[2]:.2f}]" if trg_pat is not None else "N/A"
        
        self.print_log_message(
            f"Landmark {lm_id} created: {moving_name} <-> {fixed_name} | "
            f"Moving voxel: {src_vox_str}, patient: {src_pat_str} | "
            f"Fixed voxel: {trg_vox_str}, patient: {trg_pat_str}"
        )
        
        return lm

    def lm_clear_current_landmarks(self):
        """Delete all landmarks for the current (moving,fixed) pair."""
        if self.moving_volume is None or self.fixed_volume is None:
            return

        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        num_cleared = len(self.current_landmarks)
        self.lm_mgr.clear_pair(pair)
        self.current_landmarks = []
        
        # Log landmark clearing
        if num_cleared > 0:
            moving_name = self.moving_volume.file_base_name if hasattr(self.moving_volume, 'file_base_name') else "moving"
            fixed_name = self.fixed_volume.file_base_name if hasattr(self.fixed_volume, 'file_base_name') else "fixed"
            self.print_log_message(f"Cleared {num_cleared} landmark(s) for {moving_name} <-> {fixed_name}")

    def lm_done_delete_button_clicked(self):
        """
        This button should only be enabled if the user is creating a new landmark and has placed points in both images,
        or if a landmark is selected, or if in edit mode.

        :return:
        """
        if self.lm_edit_mode:
            # In edit mode, "Done" button exits edit mode and saves changes
            # Clear registered landmarks if any exist (editing invalidates them)
            if self.current_landmarks:
                for lm in self.current_landmarks:
                    lm.registered_voxel_coords = None
                    lm.registered_patient_coords = None
                    lm.registered_marker_id = None
            
            # Clear registered volume markers
            if self.registered_volume is not None:
                for vp_idx in (6, 7, 8):
                    self.viewports[vp_idx].marker_delete_all()
            
            # Exit edit mode
            for vp in self.viewports:
                vp.marker_set_edit_mode(False)
                vp.marker_clear_selected()
                vp.refresh_preserve_extent()
            
            # Clear edit mode flag - refresh() will handle all button states
            self.lm_edit_mode = False
            self.refresh(False)
            
            # Log landmark edit completion
            selected_lm = self.lm_get_selected_landmark()
            if selected_lm:
                self.print_log_message(f"Landmark {selected_lm.id} edited successfully")
        elif self.ui.moving_lm_delete_done_button.text() == 'Save' and self.ui.fixed_lm_delete_done_button.text() == 'Save':
            # we are ready to save new landmark. update list of landmarks with temp points
            # create an id for the landmark

            self.lm_add_landmark(self.temp_source_point_voxel,
                                 self.temp_source_point_patient,
                                 self.temp_source_marker_id,
                                 self.temp_target_point_voxel,
                                 self.temp_target_point_patient,
                                 self.temp_target_marker_id)

            # set status for both images to 'idle'
            self.lm_status = -1

            # Disable edit mode since we're finalizing the landmark
            for vp in self.viewports:
                vp.marker_set_edit_mode(False)

            # clear temp markers
            self.temp_source_point_voxel = None
            self.temp_source_point_patient = None
            self.temp_target_point_voxel = None
            self.temp_target_point_patient = None
            self.temp_source_marker_id = None
            self.temp_target_marker_id = None
        else:
            # delete the selected landmark
            pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
            for lm in self.current_landmarks:
                if lm.selected:
                    self.lm_mgr.remove(pair, lm.id)
                    for vp_id in range(3):
                        mkr = self.viewports[vp_id].marker_find_by_id(lm.source_marker_id)
                        if mkr is not None:
                            self.viewports[vp_id].marker_delete(mkr['id'])
                    for vp_id in range(3, 6):
                        mkr = self.viewports[vp_id].marker_find_by_id(lm.target_marker_id)
                        if mkr is not None:
                            self.viewports[vp_id].marker_delete(mkr['id'])

                    # refresh current list
                    self.current_landmarks = self.lm_mgr.list(pair)
                    
                    # Log landmark deletion
                    self.print_log_message(f"Landmark {lm.id} deleted successfully")
                    break

        # refresh_plot the main window, and optionally update the registered image
        if self.moving_volume is not None and self.fixed_volume is not None and self.current_landmarks:
            self.refresh(True)
        else:
            self.refresh(False)

        # refresh_plot the viewports
        for vp in self.viewports:
            vp.marker_set_add_mode(False)   # should already be set to False, but just in case
            vp.marker_clear_selected()
            vp.refresh_preserve_extent()

    def lm_get_selected_landmark(self):
        """Return the currently selected LMLandmark, or None."""
        if not self.current_landmarks:
            return None
        for lm in self.current_landmarks:
            if lm.selected:
                return lm
        return None

    def lm_edit_done_button_clicked(self):
        """
        Toggle edit mode for markers. When entering edit mode, warn user if registered landmarks exist
        that will be cleared after editing. Lock marker selection during edit mode until Done is clicked.
        """
        # Check button text to determine if we're entering or exiting edit mode
        if self.ui.moving_lm_edit_cancel_button.text() == 'Edit' and self.ui.fixed_lm_edit_cancel_button.text() == 'Edit':
            # Entering edit mode - need a selected landmark first
            selected_lm = self.lm_get_selected_landmark()
            if selected_lm is None:
                error_msg = "Please select a landmark to edit."
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return
            
            # Check if registered landmarks exist and warn user
            has_registered_landmarks = False
            if self.current_landmarks:
                for lm in self.current_landmarks:
                    if hasattr(lm, 'registered_voxel_coords') and lm.registered_voxel_coords is not None:
                        has_registered_landmarks = True
                        break
            
            if has_registered_landmarks:
                reply = QMessageBox.question(
                    self, "Edit Marker",
                    "Editing this marker will invalidate registered landmarks.\n"
                    "Registered landmarks will be cleared after you click Done.\n\n"
                    "Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            # Exit add mode if active
            for vp in self.viewports:
                vp.marker_set_add_mode(False)
            # Enter edit mode
            for vp in self.viewports:
                vp.marker_set_edit_mode(True)
            
            # Set edit mode flag - refresh() will handle all button states
            self.lm_edit_mode = True
            self.refresh(False)
            
            # Log entering edit mode
            self.print_log_message(f"Entered edit mode for landmark {selected_lm.id}")
            
        else:
            # Exiting edit mode (Cancel clicked) - restore marker to original position
            # Exit edit mode and restore position
            for vp in self.viewports:
                vp.marker_set_edit_mode(False, restore_position=True)
                vp.marker_clear_selected()
                vp.refresh_preserve_extent()
            
            # Clear edit mode flag - refresh() will handle all button states
            self.lm_edit_mode = False
            self.refresh(False)
            
            # Log canceling edit mode
            self.print_log_message("Edit mode canceled - landmark position restored")

    def lm_delete_all_button_clicked(self):
        """
        Delete all landmarks for the current image pair. If registered landmarks exist, they will also be deleted.
        Asks for user confirmation before proceeding.
        """
        # Check if we have volumes loaded
        if self.moving_volume is None or self.fixed_volume is None:
            error_msg = "Please load moving and fixed images first."
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return
        
        # Check if there are any landmarks to delete
        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        landmarks = self.lm_mgr.list(pair)
        if not landmarks:
            QMessageBox.information(self, "Information", "No landmarks to delete.", QMessageBox.Ok)
            return
        
        # Check if registered landmarks exist
        has_registered_landmarks = False
        if landmarks:
            for lm in landmarks:
                if hasattr(lm, 'registered_voxel_coords') and lm.registered_voxel_coords is not None:
                    has_registered_landmarks = True
                    break
        
        # Ask for confirmation
        message = f"This will delete all {len(landmarks)} landmark(s) for the current image pair."
        if has_registered_landmarks:
            message += "\n\nRegistered landmarks will also be deleted."
        message += "\n\nThis action cannot be undone.\n\nContinue?"
        
        reply = QMessageBox.question(
            self, "Delete All Landmarks",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Exit edit mode if active
        if self.lm_edit_mode:
            for vp in self.viewports:
                vp.marker_set_edit_mode(False)
            self.lm_edit_mode = False
        
        # Exit add mode if active
        for vp in self.viewports:
            vp.marker_set_add_mode(False)
        
        # Clear all markers from viewports (moving/fixed: 0-5, registered: 6-8)
        for vp in self.viewports:
            vp.marker_delete_all()
            vp.marker_clear_selected()
        
        # Clear landmarks from manager
        num_deleted = len(landmarks)
        self.lm_mgr.clear_pair(pair)
        self.current_landmarks = []
        
        # Log deletion of all landmarks
        self.print_log_message(f"All {num_deleted} landmark(s) deleted successfully")
        
        # Refresh the UI
        self.refresh(False)
        
        # Refresh viewports
        for vp in self.viewports:
            vp.refresh_preserve_extent()

    def lm_read_landmarks(self, src_file, trg_file):
        """ read the data from the .mrk.json file pair
            :param src_file:
            :param trg_file:
            :return: src_data, trg_data, or None
        """
        with open(src_file, 'r') as input_file:
            try:
                src_data = json.load(input_file)
            except Exception as e:
                error_msg = f"An error occurred while reading the source landmarks file {e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return None, None
        with open(trg_file, 'r') as input_file:
            try:
                trg_data = json.load(input_file)
            except Exception as e:
                error_msg = f"An error occurred while reading the target landmarks file {e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                return None, None

        # extract image names ('title')
        src_image_name = src_data.get("title", "")
        trg_image_name = trg_data.get("title", "")

        # are these images loaded into the session?
        src_image = None
        trg_image = None
        # for vol in self.moving_volumes:
        #     if vol.full_file_name == src_image_name:
        #         src_image = vol
        #         break
        # for vol in self.fixed_volumes:
        #     if vol.full_file_name == trg_image_name:
        #         trg_image = vol
        #         break
        if self.moving_volume is None or self.moving_volume.file_base_name != src_image_name:
            QMessageBox.warning(self, "Error",
                                f"{src_image_name} does not match current moving image name.", QMessageBox.Ok)
            return None, None
        if self.fixed_volume is None or self.fixed_volume.file_base_name != trg_image_name:
            QMessageBox.warning(self, "Error",
                                f"{trg_image_name} does not match current fixed image name.", QMessageBox.Ok)
            return None, None
        # parse the dictionaries containing the landmark data
        temp_src_lm_info = []
        temp_trg_lm_info = []
        sys_coords = src_data.get("coordinateSystem")  # "LPS" or "RAS"
        src_markups = src_data.get("markups", [])
        for markup_entry in src_markups:
            landmarks = markup_entry.get("controlPoints", [])
            # iterate through the landmarks for this markup entry
            for lm in landmarks:
                lm_id = lm.get("id", "Unknown")
                position = lm.get("position", {})
                patient_coords = np.array([position[0], position[1], position[2], 1])
                if sys_coords != self.display_convention:  # only accounts for "RAS" vs "LPS" !
                    patient_coords[:2] *= -1
                if self.moving_volume is not None:
                    coords = np.dot(np.linalg.inv(self.moving_volume.transform), patient_coords)
                    voxel_coords = coords.astype(int)  # FIXME: is this correct? Shouldn't we just be using the voxel coords?
                else:
                    voxel_coords = lm.get("voxel_coords")
                source_marker_id = lm.get("source_marker_id")
                temp_src_lm_info.append([voxel_coords, patient_coords, source_marker_id, lm_id, self.moving_volume])

        trg_markups = trg_data.get("markups", [])
        for markup_entry in trg_markups:
            landmarks = markup_entry.get("controlPoints", [])
            # iterate through the landmarks for this markup entry
            for lm in landmarks:
                lm_id = lm.get("id", "Unknown")
                position = lm.get("position", {})
                patient_coords = np.array([position[0], position[1], position[2], 1])
                if sys_coords != self.display_convention:
                    patient_coords[:2] *= -1
                if self.fixed_volume is not None:
                    coords = np.dot(np.linalg.inv(self.fixed_volume.transform), patient_coords)
                    voxel_coords = coords.astype(int)
                else:
                    voxel_coords = lm.get("voxel_coords")  # FIXME: is this correct? Shouldn't we just be using the voxel coords?
                target_marker_id = lm.get("target_marker_id")
                temp_trg_lm_info.append([voxel_coords, patient_coords, target_marker_id, lm_id, self.fixed_volume])

        return temp_src_lm_info, temp_trg_lm_info

    def lm_write_landmarks(self):
        """
        Write current pair's landmarks to two files (source.mrk.json, target.mrk.json),
        augmented with LandMarker pair metadata to link back to the exact moving/fixed images.
        """

        # Guards
        if self.moving_volume is None or self.fixed_volume is None:
            QMessageBox.warning(self, "Error", "No active moving/fixed volumes.", QMessageBox.Ok)
            return False

        pair = pair_key_from_vols(self.moving_volume, self.fixed_volume)
        current_landmarks = self.lm_mgr.list(pair)
        if not current_landmarks:
            return False

        # Schema file (optional)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, "..", "schema", "markups-schema-v1.0.3.json")
        schema = {}
        try:
            with open(json_path, "r") as schema_file:
                schema = json.load(schema_file)
        except Exception as e:
            self.print_log_message(f"Warning: Could not load schema file: {e}")
            pass  # proceed without external schema block

        # Helpers
        def _safe_getattr(v, name):
            try:
                return getattr(v, name)
            except Exception as e:
                # Silent fallback for missing attributes
                return None

        def _pos3(arr):
            if arr is None:
                return None
            a = list(arr)
            return [float(a[0]), float(a[1]), float(a[2])]

        def _vox3(arr):
            if arr is None:
                return None
            a = list(arr)
            return [int(a[0]), int(a[1]), int(a[2])]

        # Pair identity
        mv = self.moving_volume
        fx = self.fixed_volume
        moving_info = {
            "uid": str(_safe_getattr(mv, "uid")),
            "displayName": _safe_getattr(mv, "file_base_name"),
            "path": _safe_getattr(mv, "file_path"),
            "shape_zyx": list(getattr(mv, "data", []).shape) if getattr(mv, "data", None) is not None else None,
            "voxel_size_xyz_mm": [getattr(mv, "dx", None), getattr(mv, "dy", None), getattr(mv, "dz", None)],
            "affine_4x4": getattr(mv, "transform", None).tolist() if getattr(mv, "transform",
                                                                             None) is not None else None,
        }
        fixed_info = {
            "uid": str(_safe_getattr(fx, "uid")),
            "displayName": _safe_getattr(fx, "file_base_name"),
            "path": _safe_getattr(fx, "file_path"),
            "shape_zyx": list(getattr(fx, "data", []).shape) if getattr(fx, "data", None) is not None else None,
            "voxel_size_xyz_mm": [getattr(fx, "dx", None), getattr(fx, "dy", None), getattr(fx, "dz", None)],
            "affine_4x4": getattr(fx, "transform", None).tolist() if getattr(fx, "transform",
                                                                             None) is not None else None,
        }

        lm_meta = {
            "tool": "LandMarker",
            "version": "1.0",
            "createdUtc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "pair": {
                "pair_key": [moving_info["uid"], fixed_info["uid"]],
                "moving": moving_info,
                "fixed": fixed_info,
                "coordinateSystem": getattr(self, "display_convention", None),
            },
        }

        # Build source points (moving, patient coords)
        src_points = []
        for lm in current_landmarks:
            src_points.append({
                "id": str(lm.id),
                "position": _pos3(lm.source_patient_coords),
                "source_marker_id": getattr(lm, "source_marker_id", None),
                "_landmarkerExtras": {"voxel": _vox3(lm.source_voxel_coords)},
            })

        # Build target points (fixed, patient coords)
        trg_points = []
        for lm in current_landmarks:
            trg_points.append({
                "id": str(lm.id),
                "position": _pos3(lm.target_patient_coords),
                "target_marker_id": getattr(lm, "target_marker_id", None),
                "_landmarkerExtras": {"voxel": _vox3(lm.target_voxel_coords)},
            })

        src_data = {
            "schema": schema.get("$schema"),
            "title": moving_info.get("displayName") or "",
            "description": "source image",
            "coordinateSystem": getattr(self, "display_convention", None),
            "coordinateUnits": "mm",
            "_landmarkerMeta": lm_meta,
            "markups": [{
                "type": "Fiducial",
                "coordinateSystem": getattr(self, "display_convention", None),
                "controlPoints": src_points,
            }],
        }

        trg_data = {
            "schema": schema.get("$schema"),
            "title": fixed_info.get("displayName") or "",
            "description": "target image",
            "coordinateSystem": getattr(self, "display_convention", None),
            "coordinateUnits": "mm",
            "_landmarkerMeta": lm_meta,
            "markups": [{
                "type": "Fiducial",
                "coordinateSystem": getattr(self, "display_convention", None),
                "controlPoints": trg_points,
            }],
        }

        # Build registered points (if available)
        reg_points = []
        has_registered = False
        if getattr(self, "registered_volume", None) is not None and getattr(self.registered_volume, "data", None) is not None:
            for lm in current_landmarks:
                if getattr(lm, "registered_patient_coords", None) is not None:
                    has_registered = True
                    reg_points.append({
                        "id": str(lm.id),
                        "position": _pos3(lm.registered_patient_coords),
                        "registered_marker_id": getattr(lm, "registered_marker_id", None),
                        "_landmarkerExtras": {"voxel": _vox3(lm.registered_voxel_coords)},
                    })

        if has_registered:
            registered_info = {
                "uid": str(_safe_getattr(self.registered_volume, "uid")),
                "displayName": _safe_getattr(self.registered_volume, "file_base_name"),
                "path": _safe_getattr(self.registered_volume, "file_path"),
                "shape_zyx": list(getattr(self.registered_volume, "data", []).shape) if getattr(self.registered_volume, "data", None) is not None else None,
                "voxel_size_xyz_mm": [getattr(self.registered_volume, "dx", None), getattr(self.registered_volume, "dy", None), getattr(self.registered_volume, "dz", None)],
                "affine_4x4": getattr(self.registered_volume, "transform", None).tolist() if getattr(self.registered_volume, "transform", None) is not None else None,
            }
            reg_data = {
                "schema": schema.get("$schema"),
                "title": registered_info.get("displayName") or "",
                "description": "registered image",
                "coordinateSystem": getattr(self, "display_convention", None),
                "coordinateUnits": "mm",
                "_landmarkerMeta": lm_meta,
                "markups": [{
                    "type": "Fiducial",
                    "coordinateSystem": getattr(self, "display_convention", None),
                    "controlPoints": reg_points,
                }],
            }
        else:
            reg_data = None

        try:
            os.makedirs(self.landmarks_dir, exist_ok=True)
            with open(os.path.join(self.landmarks_dir, "source.mrk.json"), "w") as f:
                json.dump(src_data, f, indent=4)
            with open(os.path.join(self.landmarks_dir, "target.mrk.json"), "w") as f:
                json.dump(trg_data, f, indent=4)
            if reg_data is not None:
                with open(os.path.join(self.landmarks_dir, "registered.mrk.json"), "w") as f:
                    json.dump(reg_data, f, indent=4)
            
            # Log successful save
            moving_name = moving_info.get("displayName") or "moving"
            fixed_name = fixed_info.get("displayName") or "fixed"
            num_landmarks = len(current_landmarks)
            reg_msg = " (including registered landmarks)" if has_registered else ""
            self.print_log_message(f"Saved {num_landmarks} landmark(s) for {moving_name} <-> {fixed_name}{reg_msg} to {self.landmarks_dir}")
            return True
        except Exception as e:
            error_msg = f"An error occurred while saving landmarks: {e}"
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
            return False

    def lm_find_by_marker_id(self, marker_id):
        for lm in self.current_landmarks:
            if lm.source_marker_id == marker_id or lm.target_marker_id == marker_id:
                return lm

        return None

    def lm_select(self, lm, point=None, vp_id=None):
        """
        This method clears any previously selected landmark(s) then sets the 'selected' property of this landmark.
        Calls refresh() to update the main window.

        :param lm:
        :param point: optional
        :param vp_id: optional
        :return:
        """
        self.lm_clear_selected()
        lm.selected = True

        # if point is not None and vp_id is not None:
        #     # set the selected status of the corresponding points in the viewports
        #     all_src_vps = [0, 1, 2]
        #     all_trg_vps = [3, 4, 5]
        #
        #     for vp_id in all_src_vps:
        #         self.viewports[vp_id].marker_select(lm.source_point, False)  # False = don't notify parent
        #     for vp_id in all_trg_vps:
        #         self.viewports[vp_id].marker_select(lm.target_point, False)  # False = don't notify parent

        self.refresh()

    def lm_marker_selects(self, lm, ):
        """
        This method selects the source and target points associated with this landmark in all viewports.

        :param lm:
        :return:
        """
        for vp_id in [0, 1, 2]:
            src_pt = self.viewports[vp_id].marker_find_by_id(lm.source_marker_id)
            if src_pt is not None:
                self.viewports[vp_id].marker_select(src_pt, False)
            else:
                pass
                #TODO: notify? ignore?
        for vp_id in [3, 4, 5]:
            trg_pt = self.viewports[vp_id].marker_find_by_id(lm.target_marker_id)
            if trg_pt is not None:
                self.viewports[vp_id].marker_select(trg_pt, False)
            else:
                pass
            #TODO: notify? ignore?
        # select registered points in viewports 6, 7, 8 (and jump slices)
        try:
            if getattr(self, 'registered_volume', None) is not None and getattr(self.registered_volume, 'data', None) is not None:
                reg_id = getattr(lm, 'registered_marker_id', None)
                for vp_id in [6, 7, 8]:
                    selected = False
                    if reg_id is not None:
                        reg_pt = self.viewports[vp_id].marker_find_by_id(reg_id)
                        if reg_pt is not None:
                            self.viewports[vp_id].marker_select(reg_pt, False)
                            selected = True
                    if not selected:
                        # Fallback: use stored registered voxel coords to jump to slice
                        rvc = getattr(lm, 'registered_voxel_coords', None)
                        if rvc is None:
                            continue
                        # rvc = [col(x), row(y), slice(z)]
                        if vp_id == 6:
                            self.viewports[vp_id].goto_slice(int(rvc[2]))  # axial (z)
                        elif vp_id == 7:
                            self.viewports[vp_id].goto_slice(int(rvc[0]))  # sagittal (x)
                        else:  # 8
                            self.viewports[vp_id].goto_slice(int(rvc[1]))  # coronal (y)
        except Exception as e:
            self.print_log_message(f"Error navigating to registered landmark slice: {e}")
            pass

    def lm_any_selected(self):
        if any(lm.selected for lm in self.current_landmarks):
            return True
        else:
            return False

    def lm_clear_selected(self):
        """
        Deselects all landmarks in the landmark tables.
        Doesn't deselect corresponding markers

        """
        if len(self.current_landmarks) > 0:
            for lm in self.current_landmarks:
                lm.selected = False
            self.moving_lm_table.table.clearSelection()
            self.fixed_lm_table.table.clearSelection()

    def disp_handle_colormap_changed_signal(self, mov_fix_reg, name, index=None):
        """
        Respond to DisplaySettings.colormap_changed(name, index).

        mov_fix_reg: 'moving' | 'fixed'
        name: canonical colormap name (str)
        index: int or None (not strictly required here)
        """
        im = self.active_moving_settings_image if mov_fix_reg == 'moving' else self.active_fixed_settings_image
        if im is None:
            return
        # Only applies to continuous images
        if getattr(im, "is_discrete", False):
            return

        # If you keep a canonicalizer, use it; otherwise just pass the name through
        try:
            cname = self.canonical_cmap_name(name)
        except Exception as e:
            self.print_log_message(f"Warning: Could not get canonical colormap name for '{name}': {e}")
            cname = name

        # Rebuild LUT via LMVolume API
        try:
            im.set_continuous_colormap(cname)  # builds im.lut (uint8 Nx4) and updates name/index
        except Exception as e:
            error_msg = f"[colormap_changed] failed to set '{cname}': {e}"
            self.print_log_message(error_msg)
            print(error_msg)
            return

        # Refresh viewers (your existing pattern)
        for vp_id in ([0, 1, 2] if mov_fix_reg == 'moving' else [3, 4, 5]):
            self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=False)
        if mov_fix_reg == 'fixed' and getattr(self.registered_volume, 'data', None) is not None:
            # Registered viewports use blend_opacity for fixed volume overlay
            for vp_id in [6, 7, 8]:
                self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=True)

    def disp_handle_discrete_LUT_changed_signal(self, mov_fix_reg, payload):
        """
        Handle discrete LUT changes from the discrete colors widget.
        Updates the image's LUT and refreshes the viewports.
        """
        im = self.active_moving_settings_image if mov_fix_reg == 'moving' else self.active_fixed_settings_image
        if im is None:
            return
        
        # Ensure the image is discrete
        if not getattr(im, "is_discrete", False):
            return
        
        # Convert payload to numpy array if needed (should already be uint8 Nx4 from discrete colors widget)
        if hasattr(payload, 'getLookupTable'):
            # If it's a colormap object, extract the LUT
            lut = payload.getLookupTable(nPts=len(im.labels) if im.labels is not None else 256, alpha=True)
            lut = (lut * 255).astype(np.uint8) if lut.dtype != np.uint8 else lut
        else:
            # Should already be a numpy array (uint8 Nx4)
            lut = np.asarray(payload, dtype=np.uint8)
        
        # Ensure LUT has the correct shape (N, 4)
        if lut.ndim != 2 or lut.shape[1] != 4:
            self.print_log_message(f"Warning: Invalid LUT shape {lut.shape}, expected (N, 4)")
            return
        
        # Update the image's LUT directly
        im.lut = lut

        # Refresh the viewports to show the updated colors
        for vp_id in ([0, 1, 2] if mov_fix_reg == 'moving' else [3, 4, 5]):
            self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=False)
        if mov_fix_reg == 'fixed' and getattr(self.registered_volume, 'data', None) is not None:
            # Registered viewports use blend_opacity for fixed volume overlay
            for vp_id in [6, 7, 8]:
                self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=True)

    def disp_handle_histogram_changed_signal(self, mov_fix_reg, vals=None):
        if vals is None or len(vals) == 0:
            return
        lower_val = vals[0]
        upper_val = vals[1]
        if mov_fix_reg == 'moving':
            if self.active_moving_settings_image is None:
                return

            # update the image's display min and max
            self.active_moving_settings_image.display_min = lower_val
            self.active_moving_settings_image.display_max = upper_val
            # refresh the viewports
            for vp_id in [0, 1, 2]:
                self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=False)
        elif mov_fix_reg == 'fixed':
            if self.active_fixed_settings_image is None:
                return

            # update the image's display min and max
            self.active_fixed_settings_image.display_min = lower_val
            self.active_fixed_settings_image.display_max = upper_val
            # refresh the viewports
            for vp_id in [3, 4, 5]:
                self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=False)
            if self.registered_volume is not None and self.registered_volume.data is not None:
                # Registered viewports use blend_opacity for fixed volume overlay
                for vp_id in [6, 7, 8]:
                    self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=True)
        elif mov_fix_reg == 'registered':
            pass

    def disp_handle_opacity_changed_signal(self, mov_fix_reg, val=0):
        if mov_fix_reg == 'moving':
            if self.active_moving_settings_image is not None:
                self.active_moving_settings_image.opacity = val
                for vp_id in [0, 1, 2]:
                    self.viewports[vp_id].refresh_preserve_extent()
        elif mov_fix_reg == 'fixed':
            if self.active_fixed_settings_image is not None:
                # Update opacity (used in fixed viewports 3, 4, 5)
                # Note: blend_opacity (used in registered viewports 6, 7, 8) is not changed
                self.active_fixed_settings_image.opacity = val
                # Fixed viewports use opacity (not blend_opacity)
                for vp_id in [3, 4, 5]:
                    self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=False)
                if self.registered_volume is not None and self.registered_volume.data is not None:
                    # Registered viewports use blend_opacity for fixed volume overlay
                    for vp_id in [6, 7, 8]:
                        self.viewports[vp_id].refresh_preserve_extent(use_blend_opacity=True)

    def _on_registration_worker_finished(self, dfield, dmatrix, cancelled, err_msg, wait_dialog):
        """Handle result of intensity-based registration worker (success, canceled, or error)."""
        if wait_dialog is None or not wait_dialog.isVisible():
            return
        QApplication.processEvents()

        if cancelled:
            self.print_log_message("Registration cancelled by user.")
            wait_dialog.show_cancelled()
            wait_dialog.exec_()
            return
        if err_msg:
            self.print_log_message(f"Registration error: {err_msg}")
            wait_dialog.show_error(err_msg)
            wait_dialog.exec_()
            return
        if dfield is None:
            self.print_log_message("Deformation field is empty.")
            wait_dialog.show_error("Deformation field is empty.")
            wait_dialog.exec_()
            return

        # Reuse existing deformation field if same inputs (moving, fixed, algorithm, landmarks)
        potential_hash = compute_unique_id_hash(
            self.moving_volume, self.fixed_volume, self.algorithm, self.current_landmarks
        )
        existing_dfield = None
        for df in self.deformation_fields:
            if not getattr(df, "is_composed", True) and getattr(df, "unique_id_hash", None) == potential_hash:
                existing_dfield = df
                break
        if existing_dfield is not None:
            self.current_deformation_field = existing_dfield
            existing_dfield.data = dfield
            if dmatrix is not None:
                existing_dfield.rigid_matrix = dmatrix
            existing_dfield.landmarks = self.current_landmarks
            self.dfields_updated_signal.emit(self.deformation_fields)
        else:
            self.current_deformation_field = LMDeformationField(
                self, len(self.deformation_fields) + 1, dfield, False,
                self.moving_volume, self.fixed_volume, self.algorithm, self.current_landmarks,
                None, dmatrix
            )
            self.deformation_fields.append(self.current_deformation_field)
            self.dfields_updated_signal.emit(self.deformation_fields)

        try:
            reg_volume_data = self.register(self.moving_volume, self.current_deformation_field.data, self.device)
        except Exception as e:
            err_msg = f"An error occurred during image registration. {e}"
            self.print_log_message(err_msg)
            wait_dialog.show_error(err_msg)
            wait_dialog.exec_()
            return

        self.registered_volume = LMVolume(
            self, default_discrete_palette=self.default_discrete_palette,
            continuous_palette_names=self.continous_palette,
            _is_discrete=self.moving_volume.is_discrete
        )
        self.registered_volume.populate_with_registered_data(
            reg_volume_data, self.fixed_volume, self.current_deformation_field, self.moving_volume
        )

        new_combo_idx = 0
        if len(self.registered_volumes) == 0:
            self.registered_volumes.append(self.registered_volume)
            self.registered_update_combo()
            new_combo_idx = len(self.registered_volumes)
        else:
            match = False
            for rv_idx, rv in enumerate(self.registered_volumes):
                if rv.dfield.unique_id_hash == self.registered_volume.dfield.unique_id_hash:
                    self.registered_volumes[rv_idx] = self.registered_volume
                    match = True
                    new_combo_idx = rv_idx + 1
                    self.registered_update_combo()
                    break
            if not match:
                self.registered_volumes.append(self.registered_volume)
                self.registered_update_combo()
                new_combo_idx = len(self.registered_volumes)

        moving_name = getattr(self.moving_volume, 'file_name', 'moving volume')
        fixed_name = getattr(self.fixed_volume, 'file_name', 'fixed volume')
        self.print_log_message(f"Moving image '{moving_name}' registered to fixed image '{fixed_name}' successfully")

        self.ui.current_registered_combo.setCurrentIndex(new_combo_idx)
        if new_combo_idx > 0 and len(self.registered_volumes) > 0:
            self.current_registered_volume_index = new_combo_idx - 1
            self.registered_volume = self.registered_volumes[self.current_registered_volume_index]
            self.ui.current_registered_combo.setToolTip(self.registered_volume.full_file_name)
            self.viewports[6].add_layer(self.registered_volume, 0)
            self.viewports[7].add_layer(self.registered_volume, 0)
            self.viewports[8].add_layer(self.registered_volume, 0)
            self.viewports[6].add_layer(self.registered_volume.fixed_vol_reference, 1)
            self.viewports[7].add_layer(self.registered_volume.fixed_vol_reference, 1)
            self.viewports[8].add_layer(self.registered_volume.fixed_vol_reference, 1)
            self.refresh(False)

        if hasattr(self, 'registered_volume') and self.registered_volume is not None:
            self.print_log_message("Registered image " + self.registered_volume.file_base_name + " created.")
        algorithm_display = getattr(self.algorithm, 'algorithm_description', None) or getattr(self.algorithm, 'algorithm_name', str(self.algorithm))
        self.print_log_message(
            "Registration completed successfully. Moving image: {}, Fixed image: {}, Algorithm: {}, Device: {}"
            .format(moving_name, fixed_name, algorithm_display, self.device)
        )
        wait_dialog.show_success(moving_name, fixed_name, algorithm_display, self.device)
        wait_dialog.exec_()

    def register_button_clicked(self):
        if self.moving_volume is None or self.fixed_volume is None:
            return
        if self.algorithm.algorithm_class == 'LM' and len(self.current_landmarks) == 0:
            QMessageBox.information(
                self, "No Landmarks",
                "There are no landmarks. Please place landmarks on the moving and fixed images before registering with a landmark-based algorithm.",
                QMessageBox.Ok)
            return

        # Intensity-based registration runs in a worker thread so it can be canceled
        if self.algorithm.algorithm_class == 'IN':
            if self.algorithm.algorithm_name is None:
                return
            # Show parameter dialog for intensity-based algorithms
            params_dialog = LMRegistrationParams(self, self.algorithm.algorithm_name)
            if params_dialog.exec_() != QDialog.Accepted:
                # User canceled, don't proceed with registration
                return
            # Update parameters with user-selected values
            self.params = params_dialog.get_parameters()
            cancel_token = type('CancelToken', (), {'cancelled': False})()
            wait = WaitDialog(self, "Registering images. Please wait...", show_cancel_button=True, title="Registering images")
            wait.set_cancel_callback(lambda: setattr(cancel_token, 'cancelled', True))
            wait.setWindowModality(Qt.ApplicationModal)
            worker = RegistrationWorker(
                self, self.current_landmarks, self.moving_volume, self.fixed_volume,
                self.algorithm, self.params, self.device, cancel_token)
            worker.output_line.connect(wait.append_message)
            worker.finished.connect(
                lambda d, f, c, e: self._on_registration_worker_finished(d, f, c, e, wait))
            wait.show()
            QApplication.processEvents()
            worker.start()
            return

        # Landmark-based registration: log landmark coordinates being used
        if len(self.current_landmarks) > 0:
            print(f"\nProcessing {len(self.current_landmarks)} landmark(s)...\n")
            for i, landmark in enumerate(self.current_landmarks):
                if landmark.source_patient_coords is not None:
                    print(f"Landmark {i+1} (ID: {landmark.id}):")
                    moving_patient_coords = landmark.source_patient_coords[:3]
                    moving_voxel_coords = landmark.source_voxel_coords[:3] if landmark.source_voxel_coords is not None else None
                    fixed_patient_coords = landmark.target_patient_coords[:3] if landmark.target_patient_coords is not None else None
                    fixed_voxel_coords = landmark.target_voxel_coords[:3] if landmark.target_voxel_coords is not None else None
                    print(f"Moving landmark patient coordinates: {moving_patient_coords}")
                    print(f"Moving image transform:\n{self.moving_volume.transform}")
                    print(f"Moving landmark voxel coordinates: {moving_voxel_coords}")
                    print(f"Fixed landmark patient coords: {fixed_patient_coords}")
                    print(f"Fixed image transform:\n{self.fixed_volume.transform}")
                    print(f"Fixed landmark voxel coords: {fixed_voxel_coords}")
                    print()

        wait = WaitDialog(self, "Registering images. Please wait...", title="Registering images")
        wait.show()
        QApplication.processEvents()  # let it paint

        err_msg = None
        registration_succeeded = False
        success_info = None  # (moving_name, fixed_name, algorithm_name, device)
        try:
            # Capture stdout/stderr
            with OutputCapture(
                    wait_dialog=wait,
                    status_callback=self.print_log_message
            ):
                dfield, dmatrix = self.calc_deformation(
                    self.current_landmarks, self.moving_volume, self.fixed_volume,
                    self.algorithm, self.params, self.device)

            # if dmatrix is not None and len(self.current_landmarks) > 0:
            #     # Collect all moving landmark voxel coordinates
            #     moving_voxel_coords = []
            #     for lm in self.current_landmarks:
            #         if lm.source_voxel_coords is not None:
            #             moving_voxel_coords.append(lm.source_voxel_coords[:3])
            #
            #     if len(moving_voxel_coords) > 0:
            #         # Convert to numpy array (N x 3)
            #         moving_voxel_coords = np.array(moving_voxel_coords)
            #
            #         # Convert to homogeneous coordinates (N x 4) by adding a column of ones
            #         moving_voxel_homogeneous = np.concatenate([moving_voxel_coords, np.ones((moving_voxel_coords.shape[0], 1))], axis=1)
            #
            #         # Apply dmatrix transformation: result is (N x 4)
            #         # dmatrix is 4x4, moving_voxel_homogeneous is (N x 4), so we transpose for matrix multiplication
            #         transformed_homogeneous = np.dot(dmatrix, moving_voxel_homogeneous.T).T
            #
            #         # Extract 3D coordinates (first 3 columns)
            #         t_pts = transformed_homogeneous[:, :3]

            if dfield is None:
                err_msg = "Deformation field is empty."
            else:
                # Reuse existing deformation field if same inputs (moving, fixed, algorithm, landmarks)
                potential_hash = compute_unique_id_hash(
                    self.moving_volume, self.fixed_volume, self.algorithm, self.current_landmarks
                )
                existing_dfield = None
                for df in self.deformation_fields:
                    if not getattr(df, "is_composed", True) and getattr(df, "unique_id_hash", None) == potential_hash:
                        existing_dfield = df
                        break
                if existing_dfield is not None:
                    self.current_deformation_field = existing_dfield
                    existing_dfield.data = dfield
                    if dmatrix is not None:
                        existing_dfield.rigid_matrix = dmatrix
                    existing_dfield.landmarks = self.current_landmarks
                    self.dfields_updated_signal.emit(self.deformation_fields)
                else:
                    self.current_deformation_field = LMDeformationField(
                        self, len(self.deformation_fields) + 1, dfield, False,
                        self.moving_volume, self.fixed_volume, self.algorithm, self.current_landmarks,
                        None, dmatrix
                    )
                    self.deformation_fields.append(self.current_deformation_field)
                    self.dfields_updated_signal.emit(self.deformation_fields)

                try:
                    reg_volume_data = self.register(self.moving_volume, self.current_deformation_field.data, self.device)
                except Exception as e:
                    err_msg = f"An error occurred during image registration. {e}"
                    self.print_log_message(err_msg)
                else:
                    # Only create registered volume if registration succeeded
                    self.registered_volume = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                           continuous_palette_names=self.continous_palette,
                                                      _is_discrete=self.moving_volume.is_discrete)
                    self.registered_volume.populate_with_registered_data(
                        reg_volume_data, self.fixed_volume, self.current_deformation_field, self.moving_volume
                    )
                    
                    new_combo_idx = 0
                    if len(self.registered_volumes) == 0:
                        self.registered_volumes.append(self.registered_volume)
                        self.registered_update_combo()
                        new_combo_idx = len(self.registered_volumes)
                    else:
                        match = False
                        for rv_idx, rv in enumerate(self.registered_volumes):
                            if rv.dfield.unique_id_hash == self.registered_volume.dfield.unique_id_hash:
                                self.registered_volumes[rv_idx] = self.registered_volume
                                match = True
                                new_combo_idx = rv_idx + 1
                                # Update combo box to reflect the updated volume
                                self.registered_update_combo()
                                break
                        if not match:
                            self.registered_volumes.append(self.registered_volume)
                            self.registered_update_combo()
                            new_combo_idx = len(self.registered_volumes)
                    
                    # Log successful registration
                    moving_name = getattr(self.moving_volume, 'file_name', 'moving volume')
                    fixed_name = getattr(self.fixed_volume, 'file_name', 'fixed volume')
                    self.print_log_message(f"Moving image '{moving_name}' registered to fixed image '{fixed_name}' successfully")

                    # FIXME: temp fix - viewport add_marker expects an image to be displayed in the viewport
                    # Set the combo box index - this may not trigger the signal if index is unchanged
                    self.ui.current_registered_combo.setCurrentIndex(new_combo_idx)
                    
                    # Always update viewports explicitly to ensure they show the updated registered volume
                    # This is necessary because if the combo index doesn't change, the signal won't fire
                    # and registered_combo_changed won't be called to update the viewports
                    if new_combo_idx > 0 and len(self.registered_volumes) > 0:
                        self.current_registered_volume_index = new_combo_idx - 1
                        self.registered_volume = self.registered_volumes[self.current_registered_volume_index]
                        self.ui.current_registered_combo.setToolTip(self.registered_volume.full_file_name)
                        self.viewports[6].add_layer(self.registered_volume, 0)
                        self.viewports[7].add_layer(self.registered_volume, 0)
                        self.viewports[8].add_layer(self.registered_volume, 0)
                        self.viewports[6].add_layer(self.registered_volume.fixed_vol_reference, 1)
                        self.viewports[7].add_layer(self.registered_volume.fixed_vol_reference, 1)
                        self.viewports[8].add_layer(self.registered_volume.fixed_vol_reference, 1)
                        self.refresh(False)

                    # create new transformed landmarks
                    # if t_pts is not None:
                    #     try:
                    #         # Convert to numpy array if it's a list
                    #         t_pts = np.asarray(t_pts)
                    #         # Check if it has the expected shape for transformed points (N x 3)
                    #         # Transformed points should be 2D array with 3 columns (x, y, z coordinates)
                    #         if t_pts.ndim != 2 or t_pts.shape[1] != 3:
                    #             # Skip if not in expected format (N x 3) - likely energies from intensity-based registration
                    #             t_pts = None
                    #     except (AttributeError, ValueError, TypeError):
                    #         # If conversion fails or shape check fails, skip processing
                    #         t_pts = None
                        
                        # if t_pts is not None:
                        #     # round the transformed points to the nearest integer to get registered voxel coords
                        #     registered_voxel_coords = np.round(t_pts).astype(int)
                        #
                        #     # Convert t_pts to homogeneous coordinates (N×4) by adding a column of ones
                        #     t_pts_homogeneous = np.concatenate([t_pts, np.ones((t_pts.shape[0], 1))], axis=1)
                        #
                        #     # Apply transform to convert from voxel coords to patient coords: result is (N×4)
                        #     # transform maps voxel -> patient, so use transform directly (not inverse)
                        #     registered_patient_coords_hom = np.dot(self.registered_volume.transform, t_pts_homogeneous.T).T
                        #
                        #     # Convert back to 3D coordinates by taking the first 3 columns
                        #     # (or divide by 4th coordinate if using projective transforms, but typically it's just 1)
                        #     registered_patient_coords = registered_patient_coords_hom[:, :3]
                        #
                        #     # add registered landmark coordinates to current landmarks
                        #     for i in range(len(self.current_landmarks)):
                        #         self.current_landmarks[i].registered_voxel_coords = registered_voxel_coords[i]
                        #         self.current_landmarks[i].registered_patient_coords = registered_patient_coords[i]
                        #         # reuse source marker id to associate registered marker with original pair
                        #         self.current_landmarks[i].registered_marker_id = self.current_landmarks[i].source_marker_id
                        #
                        #     # create markers in registered viewports from registered landmark coordinates
                        #     for i, lm in enumerate(self.current_landmarks):
                        #         vx, vy, vz = map(int, lm.registered_voxel_coords[:3])
                        #         reg_id = getattr(lm, 'registered_marker_id', lm.source_marker_id)
                        #         for vp_idx in (6, 7, 8):
                        #             # replace existing registered marker with same id to avoid duplicates
                        #             mkr = self.viewports[vp_idx].marker_find_by_id(reg_id)
                        #             if mkr is not None:
                        #                 self.viewports[vp_idx].marker_delete(mkr['id'])
                        #             self.viewports[vp_idx].add_marker(vx, vy, vz, 0, reg_id)

                    if hasattr(self, 'registered_volume') and self.registered_volume is not None:
                        self.print_log_message("Registered image " + self.registered_volume.file_base_name + " created.")
                    algorithm_display = getattr(self.algorithm, 'algorithm_description', None) or getattr(self.algorithm, 'algorithm_name', str(self.algorithm))
                    registration_succeeded = True
                    success_info = (moving_name, fixed_name, algorithm_display, self.device)
                    self.print_log_message(
                        "Registration completed successfully. Moving image: {}, Fixed image: {}, Algorithm: {}, Device: {}"
                        .format(moving_name, fixed_name, algorithm_display, self.device)
                    )
        except Exception as e:
            # Capture the error message as a string
            err_msg = f"An error occurred during deformation field calculation: {str(e)}"
            self.print_log_message(err_msg)
        finally:
            # On success: show summary in dialog and wait for user to click OK; otherwise close dialog
            try:
                if wait is not None and hasattr(wait, 'isVisible') and wait.isVisible():
                    if registration_succeeded and success_info is not None:
                        wait.show_success(*success_info)
                        wait.exec_()  # block until user clicks OK
                    else:
                        wait.allow_close()
                        wait.accept()  # end modal cleanly
            except Exception:
                pass  # If closing dialog fails, continue
            QApplication.processEvents()  # flush the hide/paint now

        # only show message boxes after dialog is fully closed
        if err_msg is not None:
            try:
                QMessageBox.warning(self, "Error", err_msg, QMessageBox.Ok)
            except Exception:
                pass  # If showing message box fails, at least we tried

    def register(self, moving_vol, dfield, device) -> np.ndarray:
        try:
            m_data = moving_vol.data.astype(np.float64)
            moving_vol_data = torch.from_numpy(m_data.copy()).to(device)
            # Ensure the deformation grid is on the same device as the image tensor.
            # This avoids "Expected all tensors to be on the same device" errors when
            # e.g. an imported deformation field tensor remains on CPU but images are on CUDA.
            if isinstance(dfield, torch.Tensor):
                if dfield.device != moving_vol_data.device:
                    dfield = dfield.to(moving_vol_data.device)
            elif isinstance(dfield, np.ndarray):
                # Be defensive: allow passing a raw numpy grid.
                dfield = torch.from_numpy(dfield).to(moving_vol_data.device).to(torch.float32)
            reg_vol_data = interp(moving_vol_data.data, dfield)
            reg_vol_data = reg_vol_data.cpu().detach().numpy()
            if moving_vol.is_discrete:
                reg_vol_data = reg_vol_data.astype(np.int32)
        except Exception:
            raise

        return reg_vol_data

    def calc_deformation(self, landmarks, moving_vol, fixed_vol, algorithm, params, device, cancel_check=None):
        """ This method calls LandmarkBasedRegistration or IntensityBasedRegistration
        :param landmarks: list of LMLandmark objects
        :param moving_vol: LMVolume object
        :param fixed_vol: LMVolume object
        :param algorithm: Algorithm object
        :param params: dictionary of parameters
        :param device: string, 'cpu' or 'cuda'
        :param cancel_check: optional callable(); if it returns True, intensity-based registration will stop.
        :return: tuple of deformation field, transformed points, and rigid_matrix/affine
                 deformation field is a numpy array of shape (3, *moving_vol.shape)
                 transformed points is a numpy array of shape (len(landmarks), 3)
                 rigid_matrix/affine is a numpy array (or None if not applicable)
        """

        method = (algorithm.algorithm_name, deepcopy(params))

        if algorithm.algorithm_class == "LM":
            # reformat coords for registration
            fixed_lm = [lm.target_voxel_coords[:3] for lm in landmarks]
            fixed_lm = np.vstack(fixed_lm)
            moving_lm = [lm.source_voxel_coords[:3] for lm in landmarks]
            moving_lm = np.vstack(moving_lm)
            try:
                deformation_field, dmatrix = (
                    LandmarkBasedRegistration(fixed_lm, moving_lm, moving_vol.data.astype(np.float32), method, device))
            except Exception:
                raise
            

            return deformation_field, dmatrix

            # TODO: each time registration per algorithm, save deformation field and landmarks (not registered)
            #       in a folder specific to that algorithm?  Ask user first?
            # if self.working_directory is not None:
            #     pass
            # nifti_img = nib.Nifti1Image(self.current_deformation_field, affine=self.fixed_volume.transform,
            #                             header=self.fixed_volume.header)
            # file_path = os.path.join(self.working_directory, "dField.nii.gz")
            # nib.save(nifti_img, file_path)
            #
            # file_path = os.path.join(self.working_directory, "dField.npy")
            # with open(file_path, 'wb') as f:
            #     np.save(f, current_deformation_field)
        else:  # "IN"
            try:
                deformation_field, energies = (
                    IntensityBasedRegistration(fixed_vol.data.astype(np.float32), moving_vol.data.astype(np.float32),
                                               method, device, cancel_check=cancel_check))
            except Exception:
                raise

            return deformation_field, None  # No dmatrix for intensity-based registration

    def apply_dfield(self, selected):
        """
        Apply the CURRENT deformation field to the provided moving volumes.
        Accepts either:
          - Iterable of moving-volume OBJECTS (preferred), or
          - Iterable of integer indices into self.moving_volumes (backward compatible)

        Replaces the entire list of registered images upon success.
        """
        # Guard: deformation field present
        if self.current_deformation_field is None or getattr(self.current_deformation_field, "data", None) is None:
            QMessageBox.warning(self, "No deformation field",
                                "No deformation field is currently selected.", QMessageBox.Ok)
            return

        if not selected:
            # Nothing to do
            return

        # --- Normalize input to a list of moving volume objects ---
        vols = []
        try:
            iterable = list(selected)
        except TypeError:
            iterable = [selected]

        for item in iterable:
            if isinstance(item, int):
                # Treat as index into self.moving_volumes
                if 0 <= item < len(self.moving_volumes):
                    vols.append(self.moving_volumes[item])
                else:
                    QMessageBox.warning(self, "Invalid selection",
                                        f"Row index {item} is out of range.", QMessageBox.Ok)
                    return
            else:
                # Assume it's already a moving volume object
                vols.append(item)

        # De-duplicate while preserving order (important if table selection returns duplicates)
        seen_ids = set()
        normalized_vols = []
        for v in vols:
            vid = id(v)
            if vid not in seen_ids:
                seen_ids.add(vid)
                normalized_vols.append(v)

        if not normalized_vols:
            return

        # NOTE: this is commented out because LMdFieldManager only allows selection of moving volumes
        # whose geometry matches the current fixed volume
        # # Ensure each selected volume matches the currently selected fixed volume geometry
        # # (Your table is already filtered, but this keeps the method robust if called elsewhere.)
        # if getattr(self, "compare_image_geometry", None) and self.fixed_volume is not None:
        #     for mv in normalized_vols:
        #         diffs = self.compare_image_geometry(self.fixed_volume, mv)
        #         if diffs:
        #             fn = getattr(mv, "file_name", "unknown")
        #             QMessageBox.warning(self, "Geometry mismatch",
        #                                 f"Selected image does not match the fixed volume's geometry:\n{fn}",
        #                                 QMessageBox.Ok)
        #             return

        wait = WaitDialog(self, "Applying deformation field. Please wait...", title="Registering images")
        wait.show()
        QApplication.processEvents()

        err_msg = None
        apply_succeeded = False
        success_info = None
        temp_reg_vol_list = []

        try:
            with OutputCapture(
                    wait_dialog=wait,
                    status_callback=self.print_log_message
            ):
                n_vols = len(normalized_vols)
                self.print_log_message(f"Applying deformation field to {n_vols} image(s)...")

                # --- Perform registration for each selected moving volume ---
                for mv in normalized_vols:
                    try:
                        reg_volume_data = self.register(mv, self.current_deformation_field.data, self.device)
                    except Exception as e:
                        error_msg = f"An error occurred during registration of image {getattr(mv, 'file_name', mv)}:\n{e}"
                        self.print_log_message(error_msg)
                        err_msg = error_msg
                        break

                    if reg_volume_data is None:
                        err_msg = "Registration returned no data."
                        break

                    # Build LMVolume from registered data
                    temp_vol = LMVolume(self, default_discrete_palette=self.default_discrete_palette,
                                       continuous_palette_names=self.continous_palette,
                                       _is_discrete=getattr(mv, "is_discrete", False))
                    temp_vol.populate_with_registered_data(reg_volume_data,
                                                           self.fixed_volume,
                                                           self.current_deformation_field,
                                                           mv)
                    temp_reg_vol_list.append(temp_vol)

                if err_msg is None:
                    # --- Success: replace the registered volume list and update viewers ---
                    self.registered_volumes.extend(temp_reg_vol_list)
                    self.registered_update_combo()

                    fixed_name = getattr(self.fixed_volume, 'file_name', 'fixed volume')
                    if len(temp_reg_vol_list) == 1:
                        vol_name = getattr(normalized_vols[0], 'file_name', 'volume')
                        self.print_log_message(f"Image '{vol_name}' registered to '{fixed_name}' successfully")
                    else:
                        self.print_log_message(
                            f"{len(temp_reg_vol_list)} volumes registered to '{fixed_name}' successfully"
                        )
                    self.print_log_message(
                        "Apply deformation field completed successfully. Volumes: {}, Fixed image: {}, Device: {}"
                        .format(len(temp_reg_vol_list), fixed_name, self.device)
                    )
                    apply_succeeded = True
                    success_info = (
                        f"{len(temp_reg_vol_list)} image(s)",
                        fixed_name,
                        "Apply deformation field",
                        self.device
                    )

                    # Point current_registered_volume_index at the last-added volume
                    if temp_reg_vol_list:
                        self.current_registered_volume_index = len(self.registered_volumes)
                        self.registered_volume = temp_reg_vol_list[-1]

                        # Update the "registered vs fixed" viewports
                        self.viewports[6].add_layer(self.registered_volume, 0)
                        self.viewports[6].add_layer(self.fixed_volume, 1)
                        self.viewports[7].add_layer(self.registered_volume, 0)
                        self.viewports[7].add_layer(self.fixed_volume, 1)
                        self.viewports[8].add_layer(self.registered_volume, 0)
                        self.viewports[8].add_layer(self.fixed_volume, 1)

                    self.refresh(False)
        except Exception as e:
            err_msg = f"An error occurred while applying deformation field: {str(e)}"
            self.print_log_message(err_msg)
        finally:
            try:
                if wait is not None and hasattr(wait, 'isVisible') and wait.isVisible():
                    if apply_succeeded and success_info is not None:
                        wait.show_success(*success_info)
                        wait.exec_()
                    else:
                        wait.allow_close()
                        wait.accept()
            except Exception:
                pass
            QApplication.processEvents()

        if err_msg is not None:
            try:
                QMessageBox.warning(self, "Error", err_msg, QMessageBox.Ok)
            except Exception:
                pass
            return

        # close the selector window after applying
        # self.volume_selector_window.close()

    def resample_lm_nifti_image(self, from_im, to_im):
        # resample the moving image to match the fixed image
        try:
            # # Get the source image's data and affine
            # source_data = from_im.canonical_nifti.get_fdata()
            # source_affine = from_im.transform
            #
            # # Get the target image's affine
            # target_affine = to_im.transform
            #
            # # Replace the translation part of the source affine with that of the target
            # shifted_affine = source_affine.copy()
            # shifted_affine[:3, 3] = target_affine[:3, 3]
            #
            # # Create a new Nifti image with the shifted affine
            # shifted_source_img = nib.Nifti1Image(source_data, shifted_affine)
            #
            # # NOTE: nilearn package must be v. 0.11.0 (or earlier?) for copy_header to work with resample_img
            # #       nilearn v 0.11.0 requires nibabel 5.2.0
            # # FutureWarning: 'force_resample' will be set to 'True' by default in Nilearn 0.13.0.
            # # Use 'force_resample=True' to suppress this warning.
            # resampled_image = nlimg.resample_to_img(shifted_source_img,
            #                                         to_im.canonical_nifti,
            #                                         interpolation='linear' if not from_im.is_discrete else 'nearest',
            #                                         force_resample=True)

            # src = from_im.canonical_nifti
            # tgt = to_im.canonical_nifti
            #
            # # ---- shift/alignment math (e.g., center match) ----
            # def center_world(img):
            #     ijk = np.array([(s - 1) / 2.0 for s in img.shape] + [1.0])
            #     return (img.affine @ ijk)[:3]
            #
            # shift = center_world(tgt) - center_world(src)
            # new_aff = src.affine.copy()
            # new_aff[:3, 3] += shift
            #
            # # IMPORTANT: make the data float32 for Nilearn
            # src_data_f = src.get_fdata(dtype=np.float32)
            #
            # # Rebuild shifted image with float32 data
            # src_shifted = nib.Nifti1Image(src_data_f, new_aff, header=src.header)
            # src_shifted.header.set_qform(new_aff, code=1)
            # src_shifted.header.set_sform(new_aff, code=1)
            #
            # interp = 'nearest' if getattr(from_im, 'is_discrete', False) else 'linear'
            #
            # resampled_image = nlimg.resample_to_img(
            #     src_shifted, tgt, interpolation=interp, force_resample=True
            # )
            #
            # # integer output (e.g., labels), round & cast back:
            # if getattr(from_im, 'is_discrete', False):
            #     out = np.rint(resampled_image.get_fdata()).astype(from_im.data.dtype)
            #     resampled_image = nib.Nifti1Image(out, tgt.affine, header=tgt.header)
            #     resampled_image.header.set_qform(tgt.affine, code=1)
            #     resampled_image.header.set_sform(tgt.affine, code=1)

            src = from_im.canonical_nifti
            tgt = to_im.canonical_nifti

            def center_world(img):
                ijk = np.array([(s - 1) / 2.0 for s in img.shape] + [1.0])
                return (img.affine @ ijk)[:3]

            # ---- alignment (your shift method) ----
            shift = center_world(tgt) - center_world(src)
            new_aff = src.affine.copy()
            new_aff[:3, 3] += shift

            src_data_f = src.get_fdata(dtype=np.float32)
            src_shifted = nib.Nifti1Image(src_data_f, new_aff, header=src.header)

            interp = 'nearest' if getattr(from_im, 'is_discrete', False) else 'linear'
            # fill = -1024.0 if not getattr(from_im, 'is_discrete', False) else 0.0
            fill = 0

            resampled_image = nlimg.resample_to_img(
                src_shifted, tgt, interpolation=interp, force_resample=True, fill_value=fill, copy_header=True
            )

        except Exception as e:
            error_msg = f"An error occurred while resampling the image:\n{str(e)}"
            self.print_log_message(error_msg)
            QMessageBox.critical(None, "Resampling Error", error_msg,
                QMessageBox.StandardButton.Ok
            )
            return None

        return resampled_image

    def compare_image_geometry(self, im_1, im_2):
        """
        Check for differences in image shape, origin, and resolution.
        :param im_1:
        :param im_2:
        :return: dict of differences {shape [], origin [], resolution []} or None if no differences
        """

        # return differences so that calling method can decide what to do with them (like specific message to user)
        diffs_dict = None

        if im_1 is None or im_2 is None:
            # TODO: warn or error message or log
            return diffs_dict

        if im_1.data is None or im_2.data is None:
            # TODO: warn or error message or log
            return diffs_dict

        # check for matching image geometry (shape, origin, and resolution)
        if not np.array_equal(im_1.data.shape, im_2.data.shape):
            # image shapes do not match
            diffs_dict = {'shape': (im_1.shape, im_2.shape)}
        if not np.array_equal(np.round(im_1.origin, 1), np.round(im_2.origin, 1)):
            diffs_dict = {'origin': (np.round(im_1.origin, 1), np.round(im_2.origin, 1))}
        if not np.array_equal(np.round(im_1.resolution, 1), np.round(im_2.resolution, 1)):
            diffs_dict = {'resolution': (np.round(im_1.resolution, 1), np.round(im_2.resolution, 1))}

        return diffs_dict

    def save_image_to_file(self, im, file_type, mode="save_as"):
        """ Saves the image as a NIfTI file or DICOM series.
            If mode is "save" and user has already selected a save file,
            then this method writes to that file. If mode is "save" and user
            has not yet selected a save file, then they are prompted to select
            a file. If mode is "save_as", user is prompted to select a save file.
            Type is "dicom" or "nifti".
        """
        if im is not None and im.data is not None:
            return

        if mode == "save":
            # check to see if current save filename is set. if so, write image
            # to that file (overwrite whole file)
            if self.registered_save_filename is None:
                selected_file = self.select_write_file_or_dir(type)
                if selected_file is None:
                    # user cancelled - do nothing
                    return
                else:
                    self.registered_save_filename = selected_file
        elif mode == "save_as":
            selected_file = self.select_write_file_or_dir(type)
            if selected_file is None:
                # user cancelled - do nothing
                return
            else:
                self.registered_save_filename = selected_file

        if self.registered_write(type):
            self.print_log_message("Registered image " + self.registered_save_filename + " saved.")

    def find_image_by_base_name(self, mov_fix_reg, base_name1, base_name2=None):
        if mov_fix_reg == 'moving':
           for idx, mv in enumerate(self.moving_volumes):
                if mv.file_base_name == base_name1:
                     return mv, idx
        elif mov_fix_reg == 'fixed':
            for idx, fv in enumerate(self.fixed_volumes):
                if fv.file_base_name == base_name1:
                    return fv, idx
        elif mov_fix_reg == 'registered':
            for idx, rv in enumerate(self.registered_volumes):
                if rv.moving_vol_reference.file_base_name == base_name1 and rv.fixed_vol_reference.file_base_name == base_name2:
                    return rv, idx
        return None, None

    def select_read_file_or_dir(self, dataset_type, _caption=None, offer_segmentation_option=False):
        """ Allow the user to interactively select a landmark or nifti file or DICOM
            directory. Returns the full path and name of file or dir, or None.

            When dataset_type is 'nifti' and offer_segmentation_option is True, uses a
            file dialog with a "Load as segmentation (label map)" checkbox and returns
            (selected_files_or_None, load_as_segmentation_bool).
        """
        # The native directory selection dialog (windows) does not allow you to see the contents of the directories,
        # nor does it allow you to select multiple directories. Therefore, we are using the PyQt selection dialog.
        # The tradeoff is that the PyQT dialog is SLOW.  See this post:
        # https://stackoverflow.com/questions/13459294/pyqt-qfiledialog-exec-is-slow
        selected_dataset = None

        # open up the selection dialog in the last directory opened
        # if self.last_directory:
        #     initial_directory = self.last_directory
        # else:
        #     initial_directory = ""

        if dataset_type == 'dicom':
            pass
            # selection_dialog = QtWidgets.QFileDialog(None, "Select DICOM Folder")
            # selected_dataset = selection_dialog.getExistingDirectory(self, "Select DICOM Folder", initial_directory)
        elif dataset_type == 'nifti':
            if _caption is None:
                caption = 'Select NIfTI File(s)'
            else:
                caption = _caption
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog  # important for multi-file selection

            if offer_segmentation_option:
                # Use dialog instance so we can add a "Load as segmentation" checkbox
                dlg = QtWidgets.QFileDialog(self, caption, self.last_directory, "NIfTI Files (*.nii.gz)")
                dlg.setOptions(options)
                dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
                load_as_seg_cb = QCheckBox(" Load as segmentation (label map)  ")
                load_as_seg_cb.setChecked(False)
                load_as_seg_cb.setStyleSheet(
                    "QCheckBox { background-color: #626568; padding: 6px 8px; border-radius: 2px; }"
                )
                load_as_seg_cb.setAutoFillBackground(True)
                load_as_seg_cb.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Fixed,
                )
                layout = dlg.layout()
                if layout is not None:
                    if isinstance(layout, QtWidgets.QGridLayout):
                        layout.addWidget(
                            load_as_seg_cb,
                            layout.rowCount(),
                            0,
                            1,
                            layout.columnCount(),
                        )
                    else:
                        layout.addWidget(load_as_seg_cb)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    files = dlg.selectedFiles()
                    if files:
                        self.last_directory = QtCore.QFileInfo(files[0]).dir().absolutePath()
                        return (files, load_as_seg_cb.isChecked())
                return (None, False)
            else:
                files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,
                                                                  caption,
                                                                  self.last_directory,
                                                                  "NIfTI Files (*.nii.gz)", options=options)
                if files:
                    selected_dataset = files
                else:
                    selected_dataset = None
                if selected_dataset is not None:
                    self.last_directory = QtCore.QFileInfo(selected_dataset[0]).dir().absolutePath()

        elif dataset_type == 'landmarks':
            # selection_dialog = QtWidgets.QFileDialog(None, "Select Folder that Contains Landmarks")
            # selected_dataset = selection_dialog.getExistingDirectory(self, "Select Landmark Folder", initial_directory)
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            if _caption is None:
                caption = "Select Landmark Files Directory"
            else:
                caption = _caption
            directory = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                   caption,
                                                                   self.last_directory, options=options)

            if directory:
                print("Selected directory:", directory)
                selected_dataset = directory
            else:
                selected_dataset = None

            # save the user selected directory to use next time
            if selected_dataset is not None:
                self.last_directory = QtCore.QFileInfo(selected_dataset).dir().absolutePath()
        elif dataset_type == 'dfield':
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            if _caption is None:
                caption = "Select Deformation Field File"
            else:
                caption = _caption
            filter_str = (
                "All supported (*.pt *.npy *.nii *.nii.gz *.h5);;"
                "LandMarker / PyTorch (*.pt);;"
                "NumPy (*.npy);;"
                "NIfTI (*.nii *.nii.gz);;"
                "HDF5 (*.h5)"
            )
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,
                                                              caption,
                                                              self.last_directory,
                                                              filter_str, options=options)
            if files:
                selected_dataset = files
            else:
                selected_dataset = None
                # save the user selected directory to use next time
            if selected_dataset is not None:
                self.last_directory = QtCore.QFileInfo(selected_dataset[0]).dir().absolutePath()

        return selected_dataset

    def read_data(self, dataset_type, dataset_name, as_discrete=False):
        """
        Read data from DICOM dir or NIfTI file, returns resulting dataset or None and also whether data is continuous or
        discrete.

        :param dataset_type: string, "dicom" or "nifti"
        :param dataset_name: string, full path and name of dataset
        :param as_discrete: if True, treat as label/segmentation (discrete LUT); if False, treat as intensity (continuous)
        :return: dataset: object, the dataset
        :return: data_class: string, "Continuous" or "Discrete"
        :return: uid_meta: dict or None
        """
        dataset = None
        data_class = None
        uid_meta = None

        if dataset_type == 'nifti':
            try:
                dataset = nib.load(dataset_name)
                # Compute fresh meta (for shape/affine/etc.)
                recomputed = compute_image_uid_from_nifti(dataset)
                # If a sidecar exists, use its UID (as requested)
                sc = LMSidecarUtils.read_sidecar_if_exists(dataset_name)
                if sc and isinstance(sc.get("image_uid"), str):
                    if sc["image_uid"] != recomputed["image_uid"]:
                        self.print_log_message("Sidecar UID does not match the file’s computed UID. Using UID from sidecar")
                    # Keep recomputed metadata but override UID (and space if present)
                    recomputed["image_uid"] = sc["image_uid"]
                    if "space" in sc:
                        recomputed["space"] = sc["space"]
                uid_meta = recomputed
            except Exception as e:
                error_msg = f"An error occurred while reading this nifti: {e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
                dataset = None  # FIXME: does this line execute?
        elif dataset_type == 'dicom':
            file_list = []
            for dirpath, dirnames, filenames in os.walk(dataset_name):
                if len(filenames) == 0:
                    QMessageBox.warning(self, "Error",
                                        "Couldn't find any files in this folder.", QMessageBox.Ok)
                    break
                else:
                    for file in filenames:
                        if file.lower().endswith('.dcm') or file.lower().endswith('ima'):
                            file_list.append(os.path.join(dirpath, file))
                    file_list.sort()
                    if len(file_list) == 0:
                        QMessageBox.warning(self, "Error",
                                            "Couldn't find any *.ima or *.dcm files in this folder.", QMessageBox.Ok)
                    else:
                        try:
                            # read all the files into pydicom datasets
                            dataset = [pydicom.dcmread(f) for f in file_list]
                        except Exception as e:
                            error_msg = f"An error occurred while reading this DICOM: {e}"
                            self.print_log_message(error_msg)
                            QMessageBox.warning(self, "Error", error_msg, QMessageBox.Ok)
        else:
            # internal error
            error_msg = str(__name__ + " unsupported data type: " + str(dataset_type))
            self.print_log_message(error_msg)
            QMessageBox.warning(self, "Internal Error", error_msg, QMessageBox.Ok)
            dataset = None

        if dataset is not None:
            data_class = "Discrete" if as_discrete else "Continuous"

        return dataset, data_class, uid_meta

    def classify_image(self, data, original_dtype=None):
        """
        Classifies image as discrete or continuous based on data characteristics.
        For integer data, asks the user to choose.

        :param data: numpy array of image data
        :param original_dtype: optional, original dtype before conversion (e.g., from get_fdata())
        :return: "Discrete" or "Continuous"
        """
        # Check original dtype if provided (before get_fdata() conversion to float)
        if original_dtype is not None:
            is_integer_type = issubclass(original_dtype.type, np.integer)
        else:
            # Fallback to checking data dtype (may be float if converted)
            is_integer_type = issubclass(data.dtype.type, np.integer)
        unique_values = np.unique(data)

        # If data is integer type, ask the user
        if is_integer_type:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Image Classification")
            msg_box.setText("This image contains integer data. How should it be classified?")
            msg_box.setInformativeText(
                "• Discrete: For label/segmentation/mask images with distinct categories\n"
                "• Continuous: For intensity images (e.g., CT, MRI) with integer values"
            )
            discrete_button = msg_box.addButton("Discrete", QMessageBox.YesRole)
            continuous_button = msg_box.addButton("Continuous", QMessageBox.NoRole)
            msg_box.setDefaultButton(discrete_button)
            
            reply = msg_box.exec_()
            
            if msg_box.clickedButton() == discrete_button:
                return "Discrete"
            else:
                return "Continuous"
        
        # For non-integer data, use the existing logic based on unique values
        if len(unique_values) < 12:
            return "Discrete"
        else:
            return "Continuous"

    # def select_write_file_or_dir(self, datatype):
    #     """ Allow the user to interactively select a landmark or nifti file or
    #         DICOM directory for saving. Returns the full path and name of file or dir, or None.
    #     """
    #     # for convenience, open up the selection dialog in the last directory opened
    #     # if self.last_directory is not None:
    #     #     initial_directory = self.last_directory
    #     # else:
    #     #     initial_directory = ""
    #
    #     if datatype == 'dicom':
    #         selection_dialog = QtWidgets.QFileDialog(None, "Save as DICOM")
    #         options = QtWidgets.QFileDialog.Options()
    #         options |= QtWidgets.QFileDialog.ShowDirsOnly
    #         # use the native file dialog and allow directory creation
    #         selected_directory = QtWidgets.QFileDialog.getExistingDirectory(self,
    #                                                                         "Select or Create DICOM Folder",
    #                                                                         self.last_directory, options=options)
    #         if selected_directory:
    #             file_or_dir = selected_directory
    #         else:
    #             file_or_dir = None
    #     elif datatype == 'nifti':
    #         selection_dialog = QtWidgets.QFileDialog(None, "Save as nifti")
    #         selection_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    #         selection_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
    #         selection_dialog.setNameFilter("nifti Files (*.nii.gz)")
    #         selection_dialog.setDirectory(self.last_directory)
    #         result = selection_dialog.exec_()
    #         if result == QtWidgets.QFileDialog.Rejected:
    #             file_or_dir = None
    #         else:
    #             file_or_dir = selection_dialog.selectedFiles()[0]
    #     elif datatype == 'landmarks':
    #         selection_dialog = QtWidgets.QFileDialog(None,
    #                                                  "Select Folder to Save source.mrk.json and target.mrk.json Files")
    #         options = QtWidgets.QFileDialog.Options()
    #         options |= QtWidgets.QFileDialog.ShowDirsOnly
    #         # use the native file dialog and allow directory creation
    #         selected_directory = QtWidgets.QFileDialog.getExistingDirectory(self,
    #                                                                         "Select or Create Landmarks Folder",
    #                                                                         self.last_directory, options=options)
    #         if selected_directory:
    #             file_or_dir = selected_directory
    #         else:
    #             file_or_dir = None
    #     elif datatype == 'dfield':
    #         selection_dialog = QtWidgets.QFileDialog(None, "Save as deformation field file")
    #         selection_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    #         selection_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
    #         selection_dialog.setNameFilter("dfield Files (*.pt)")
    #         selection_dialog.setDirectory(self.last_directory)
    #         result = selection_dialog.exec_()
    #         if result == QtWidgets.QFileDialog.Rejected:
    #             file_or_dir = None
    #         else:
    #             file_or_dir = selection_dialog.selectedFiles()[0]
    #     elif datatype == 'svg':
    #         selection_dialog = QtWidgets.QFileDialog(None, "Save as vector graphics (*.svg) file")
    #         selection_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    #         selection_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
    #         selection_dialog.setNameFilter("(*.svg)")
    #         selection_dialog.setDirectory(self.last_directory)
    #         result = selection_dialog.exec_()
    #         if result == QtWidgets.QFileDialog.Rejected:
    #             file_or_dir = None
    #         else:
    #             file_or_dir = selection_dialog.selectedFiles()[0]
    #     else:
    #         # internal error
    #         QMessageBox.warning(self, "Internal Error", str(__name__ + " unsupported data type: " + str(datatype)),
    #                             QMessageBox.Ok)
    #         file_or_dir = None
    #
    #     # save the user selected directory to use next time
    #     self.last_directory = QtCore.QFileInfo(file_or_dir).dir().absolutePath()
    #
    #     return file_or_dir
    def select_write_file_or_dir(self, datatype, default_name: str = "", default_dir: str | None = None):
        """Allow the user to interactively select a landmark or nifti file or
           DICOM directory for saving. Returns the full path and name of file or dir, or None.

           default_name: suggested filename to pre-fill the name field (files only)
           default_dir:  initial directory (falls back to self.last_directory)
        """
        # Resolve initial directory
        initial_dir = default_dir if default_dir else (self.last_directory or "")

        file_or_dir = None  # ensure defined for final block

        if datatype == 'dicom':
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.ShowDirsOnly
            selected_directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select or Create DICOM Folder",
                initial_dir,
                options=options
            )
            file_or_dir = selected_directory if selected_directory else None

        elif datatype == 'nifti':
            selection_dialog = QtWidgets.QFileDialog(self, "Save as NIfTI")
            selection_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            selection_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
            # Offer both .nii.gz and .nii; default to .nii.gz
            selection_dialog.setNameFilters([
                "NIfTI compressed (*.nii.gz)",
                "NIfTI (*.nii)",
                "All files (*)"
            ])
            selection_dialog.setDefaultSuffix("nii.gz")  # applies if user omits extension
            selection_dialog.setDirectory(initial_dir)
            if default_name:
                # Pre-fill the filename field
                # If default_name is just a basename, Qt will join it with the current directory
                selection_dialog.selectFile(default_name)

            result = selection_dialog.exec_()
            if result == QtWidgets.QFileDialog.Accepted:
                file_or_dir = selection_dialog.selectedFiles()[0]
            else:
                file_or_dir = None

        elif datatype == 'landmarks':
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.ShowDirsOnly
            selected_directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select or Create Landmarks Folder",
                initial_dir,
                options=options
            )
            file_or_dir = selected_directory if selected_directory else None

        elif datatype == 'dfield':
            selection_dialog = QtWidgets.QFileDialog(self, "Save as deformation field file")
            selection_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            selection_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
            # TEMPORARILY DISABLED DURING REFACTORING: Only .pt format with full metadata
            selection_dialog.setNameFilter("LandMarker PyTorch (*.pt)")
            # selection_dialog.setNameFilter(
            #     "PyTorch (*.pt, includes LandMarker metadata);;"
            #     "NumPy (*.npy);;"
            #     "NIfTI (*.nii *.nii.gz);;"
            #     "HDF5 (*.h5)"
            # )
            selection_dialog.setDefaultSuffix("pt")
            selection_dialog.setDirectory(initial_dir)
            if default_name:
                selection_dialog.selectFile(default_name)

            result = selection_dialog.exec_()
            file_or_dir = selection_dialog.selectedFiles()[0] if result == QtWidgets.QFileDialog.Accepted else None

        elif datatype == 'svg':
            selection_dialog = QtWidgets.QFileDialog(self, "Save as vector graphics (*.svg) file")
            selection_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            selection_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
            selection_dialog.setNameFilter("SVG Files (*.svg)")
            selection_dialog.setDefaultSuffix("svg")
            selection_dialog.setDirectory(initial_dir)
            if default_name:
                selection_dialog.selectFile(default_name)

            result = selection_dialog.exec_()
            file_or_dir = selection_dialog.selectedFiles()[0] if result == QtWidgets.QFileDialog.Accepted else None

        else:
            QMessageBox.warning(
                self, "Internal Error",
                str(__name__ + " unsupported data type: " + str(datatype)),
                QMessageBox.Ok
            )
            file_or_dir = None

        # Update last_directory only if user actually selected something
        if file_or_dir:
            self.last_directory = QtCore.QFileInfo(file_or_dir).dir().absolutePath()

        return file_or_dir

    import os
    import json
    from PyQt5.QtWidgets import QMessageBox


    # def sync_link_zoom(self, from_viewport_id, link_unlink):
    #     if from_viewport_id == 0:
    #         # sync other source vol viewports to z slice viewport
    #         self.viewports[1].is_linked_zooming = link_unlink
    #         # toggle link zoom button on
    #         #             for action in vp.toolbar.actions():
    #         #                 if action.text() == "Pan":  # (Home Back Forward Pan Zoom)
    #         #                     action.trigger()
    #         self.viewports[2].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 1:
    #         # sync other source vol viewports to x slice viewport
    #         self.viewports[0].is_linked_zooming = link_unlink
    #         self.viewports[2].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 2:
    #         # sync other source vol viewports to y slice viewport
    #         self.viewports[0].is_linked_zooming = link_unlink
    #         self.viewports[1].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 3:
    #         # sync other target vol viewports to z slice viewport
    #         self.viewports[4].is_linked_zooming = link_unlink
    #         self.viewports[5].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 4:
    #         # sync other target vol viewports to x slice viewport
    #         self.viewports[3].is_linked_zooming = link_unlink
    #         self.viewports[5].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 5:
    #         # sync other target vol viewports to y slice viewport
    #         self.viewports[3].is_linked_zooming = link_unlink
    #         self.viewports[4].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 6:
    #         # sync other target vol viewports to z slice viewport
    #         self.viewports[7].is_linked_zooming = link_unlink
    #         self.viewports[8].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 7:
    #         # sync other target vol viewports to x slice viewport
    #         self.viewports[6].is_linked_zooming = link_unlink
    #         self.viewports[8].is_linked_zooming = link_unlink
    #     elif from_viewport_id == 8:
    #         # sync other target vol viewports to y slice viewport
    #         self.viewports[6].is_linked_zooming = link_unlink
    #         self.viewports[7].is_linked_zooming = link_unlink
    #
    def refresh_all_viewports(self):
        """ Just refreshes all the viewports. """
        for vp in self.viewports:
            # Registered viewports (6, 7, 8) use blend_opacity, others use opacity
            use_blend = vp.id in [6, 7, 8]
            vp.refresh_preserve_extent(use_blend_opacity=use_blend)

    def sync_viewport_slices(self, from_viewport_id, voxel_coords):
        """
        Use slice number of specified viewport to set slice number of other viewports for the same image.

        :param from_viewport_id:
        :param voxel_coords:
        :return:
        """

        # voxel_coords are [col, row, slice] indices to image
        if from_viewport_id == 0:
            # sync other source vol viewports to z slice viewport
            self.viewports[1].goto_slice(voxel_coords[0])
            self.viewports[2].goto_slice(voxel_coords[1])
        elif from_viewport_id == 1:
            # sync other source vol viewports to x slice viewport
            self.viewports[0].goto_slice(voxel_coords[2])
            self.viewports[2].goto_slice(voxel_coords[1])
        elif from_viewport_id == 2:
            # sync other source vol viewports to y slice viewport
            self.viewports[0].goto_slice(voxel_coords[2])
            self.viewports[1].goto_slice(voxel_coords[0])
        elif from_viewport_id == 3:
            # sync other target vol viewports to z slice viewport
            self.viewports[4].goto_slice(voxel_coords[0])
            self.viewports[5].goto_slice(voxel_coords[1])
        elif from_viewport_id == 4:
            # sync other target vol viewports to x slice viewport
            self.viewports[3].goto_slice(voxel_coords[2])
            self.viewports[5].goto_slice(voxel_coords[1])
        elif from_viewport_id == 5:
            # sync other target vol viewports to y slice viewport
            self.viewports[3].goto_slice(voxel_coords[2])
            self.viewports[4].goto_slice(voxel_coords[0])
        else:
            pass

    def sync_extents(self, from_viewport_id, zoom_factor):
        """ Refreshes the other viewports for the other two orthogonal views of the same image.
            This method is called when the user zooms in or out on one of the viewports.
            """
        if not self.is_linked_zoom:
            return
        # # voxel_coords are [col, row, slice] indices to image
        # if from_viewport_id in [0, 1, 2]:
        #     new_ext = []
        #     if zoom_factor is None:
        #         for vp in range(3):
        #             new_ext.append(self.viewports[vp].initial_extent)
        #     else:
        #         for vp in range(3):
        #             # get the current axes position
        #             ax_pos = self.viewports[vp].ax.get_position()
        #             # calculate the change in size based on the zoom factor
        #             dx = (ax_pos.width - ax_pos.width / zoom_factor) / 2
        #             dy = (ax_pos.height - ax_pos.height / zoom_factor) / 2
        #             # update the new axes position
        #             new_ext.append(
        #                 [ax_pos.x0 + dx, ax_pos.y0 + dy, ax_pos.width / zoom_factor, ax_pos.height / zoom_factor])
        #
        #     if from_viewport_id == 0:
        #         # sync other source vol viewports to z slice viewport
        #         self.viewports[1].ax.set_position(new_ext[1])
        #         self.viewports[1].canvas.draw()
        #         self.viewports[2].ax.set_position(new_ext[2])
        #         self.viewports[2].canvas.draw()
        #     elif from_viewport_id == 1:
        #         # sync other source vol viewports to x slice viewport
        #         self.viewports[0].ax.set_position(new_ext[0])
        #         self.viewports[0].canvas.draw()
        #         self.viewports[2].ax.set_position(new_ext[2])
        #         self.viewports[2].canvas.draw()
        #     elif from_viewport_id == 2:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[0].ax.set_position(new_ext[0])
        #         self.viewports[0].canvas.draw()
        #         self.viewports[1].ax.set_position(new_ext[1])
        #         self.viewports[1].canvas.draw()
        # elif from_viewport_id in [3, 4, 5]:
        #     new_ext = []
        #     if zoom_factor is None:
        #         for vp in [3, 4, 5]:
        #             new_ext.append(self.viewports[vp].initial_extent)
        #     else:
        #         for vp in [3, 4, 5]:
        #             # get the current axes position
        #             ax_pos = self.viewports[vp].ax.get_position()
        #             # calculate the change in size based on the zoom factor
        #             dx = (ax_pos.width - ax_pos.width / zoom_factor) / 2
        #             dy = (ax_pos.height - ax_pos.height / zoom_factor) / 2
        #             # update the new axes position
        #             new_ext.append(
        #                 [ax_pos.x0 + dx, ax_pos.y0 + dy, ax_pos.width / zoom_factor, ax_pos.height / zoom_factor])
        #     if from_viewport_id == 3:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[4].ax.set_position(new_ext[0])
        #         self.viewports[4].canvas.draw()
        #         self.viewports[5].ax.set_position(new_ext[1])
        #         self.viewports[5].canvas.draw()
        #     elif from_viewport_id == 4:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[3].ax.set_position(new_ext[0])
        #         self.viewports[3].canvas.draw()
        #         self.viewports[5].ax.set_position(new_ext[1])
        #         self.viewports[5].canvas.draw()
        #     elif from_viewport_id == 5:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[3].ax.set_position(new_ext[0])
        #         self.viewports[3].canvas.draw()
        #         self.viewports[4].ax.set_position(new_ext[1])
        #         self.viewports[4].canvas.draw()
        # elif from_viewport_id in [6, 7, 8]:
        #     new_ext = []
        #     if zoom_factor is None:
        #         for vp in [6, 7, 8]:
        #             new_ext.append(self.viewports[vp].initial_extent)
        #     else:
        #         for vp in [6, 7, 8]:
        #             # get the current axes position
        #             ax_pos = self.viewports[vp].ax.get_position()
        #             # calculate the change in size based on the zoom factor
        #             dx = (ax_pos.width - ax_pos.width / zoom_factor) / 2
        #             dy = (ax_pos.height - ax_pos.height / zoom_factor) / 2
        #             # update the new axes position
        #             new_ext.append(
        #                 [ax_pos.x0 + dx, ax_pos.y0 + dy, ax_pos.width / zoom_factor, ax_pos.height / zoom_factor])
        #     if from_viewport_id == 6:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[7].ax.set_position(new_ext[0])
        #         self.viewports[7].canvas.draw()
        #         self.viewports[8].ax.set_position(new_ext[1])
        #         self.viewports[8].canvas.draw()
        #     elif from_viewport_id == 7:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[6].ax.set_position(new_ext[0])
        #         self.viewports[6].canvas.draw()
        #         self.viewports[8].ax.set_position(new_ext[1])
        #         self.viewports[8].canvas.draw()
        #     elif from_viewport_id == 8:
        #         # sync other source vol viewports to y slice viewport
        #         self.viewports[6].ax.set_position(new_ext[0])
        #         self.viewports[6].canvas.draw()
        #         self.viewports[7].ax.set_position(new_ext[1])
        #         self.viewports[7].canvas.draw()

    def print_log_message(self, message):
        """
        Post a message to the rotating log file.

        :param message:
        :return:
        """
        if hasattr(self, 'logger'):
            self.logger.info(message)

    def refresh(self, update_registration=False):
        """ This method refreshes the state of the widgets in the main window.
            - landmark tables are updated
            - landmark toolbar buttons enabled/disabled
            - all image tool buttons enabled/disabled
        """
        # update the state of the fixed and moving landmark table buttons and the resample button and matching
        # geometry indicator frame
        # a fixed image and a moving image must be selected for the landmark tool buttons to be enabled
        # their geometries must also match
        if (self.moving_volume is None) or (self.fixed_volume is None):
            self.ui.moving_lm_new_cancel_button.setText('New')
            self.ui.fixed_lm_new_cancel_button.setText('New')
            self.ui.moving_lm_new_cancel_button.setEnabled(False)
            self.ui.fixed_lm_new_cancel_button.setEnabled(False)
            self.ui.moving_lm_delete_done_button.setText('Delete')
            self.ui.fixed_lm_delete_done_button.setText('Delete')
            self.ui.moving_lm_delete_done_button.setEnabled(False)
            self.ui.fixed_lm_delete_done_button.setEnabled(False)
            self.ui.moving_lm_edit_cancel_button.setText('Edit')
            self.ui.fixed_lm_edit_cancel_button.setText('Edit')
            self.ui.moving_lm_edit_cancel_button.setEnabled(False)
            self.ui.fixed_lm_edit_cancel_button.setEnabled(False)
            self.ui.moving_lm_delete_all_button.setEnabled(False)
            self.ui.fixed_lm_delete_all_button.setEnabled(False)

            self.moving_lm_table.table.clearContents()
            self.moving_lm_table.setEnabled(False)
            self.fixed_lm_table.table.clearContents()
            self.fixed_lm_table.setEnabled(False)
            # self.registered_lm_table.table.clearContents()
            # self.registered_lm_table.setEnabled(False)

            self.ui.moving_resamp_button.setIcon(QIcon(str(icons_dir / 'black_resamp.svg')))
            self.ui.moving_resamp_button.setStyleSheet(
                "QPushButton#moving_resamp_button {background-color: #FFC90E; border: 0.04em solid #FFC90E;}")  # gold yellow
            self.ui.moving_resamp_button.setEnabled(False)
            # self.ui.frame.setStyleSheet("QFrame#frame {border: none; }")

            # update the state of the register and register multiple buttons -------------------------------------------
            self.ui.register_button.setEnabled(False)
            self.ui.register_button.setStyleSheet("QPushButton#register_button {background-color: #31363B; border: 0.04em solid #76797c;}") # gray
            # self.ui.apply_to_button.setEnabled(False)
            # self.ui.apply_to_button.setStyleSheet("QPushButton#apply_to_button {background-color: #31363B; border: 0.04em solid #76797c;}") # gray

        else:  # both images are loaded
            diffs = self.compare_image_geometry(self.fixed_volume, self.moving_volume)
            if diffs:
                # if the two volumes do not have the same geometry, all landmark buttons are disabled
                self.ui.moving_resamp_button.setIcon(QIcon(str(icons_dir / 'black_resamp.svg')))
                self.ui.moving_resamp_button.setStyleSheet(
                    "QPushButton#moving_resamp_button {background-color: #FFC90E; border: 0.04em solid #FFC90E;}")  # gold yellow
                self.ui.moving_resamp_button.setEnabled(True)
                self.ui.moving_resamp_button.setToolTip('Resample moving image to match fixed image geometry')
                self.ui.moving_vol_main_frame.setStyleSheet("""
                                            QFrame#moving_vol_main_frame {
                                                border-style: inset;
                                                border-width: 2px;
                                                border-color: #1D2023;
                                            }
                                            """)
                self.ui.fixed_vol_main_frame.setStyleSheet("""
                                            QFrame#fixed_vol_main_frame {
                                                border-style: inset;
                                                border-width: 2px;
                                                border-color: #1D2023;
                                            }
                                            """)


                self.ui.moving_lm_new_cancel_button.setEnabled(False)
                self.ui.fixed_lm_new_cancel_button.setEnabled(False)
                self.ui.moving_lm_delete_done_button.setEnabled(False)
                self.ui.fixed_lm_delete_done_button.setEnabled(False)
                self.ui.moving_lm_edit_cancel_button.setEnabled(False)
                self.ui.fixed_lm_edit_cancel_button.setEnabled(False)
                self.ui.moving_lm_delete_all_button.setEnabled(False)
                self.ui.fixed_lm_delete_all_button.setEnabled(False)
                self.moving_lm_table.table.clearContents()
                self.moving_lm_table.setEnabled(False)
                self.fixed_lm_table.table.clearContents()
                self.fixed_lm_table.setEnabled(False)
                # self.registered_lm_table.table.clearContents()
                # self.registered_lm_table.setEnabled(False)
                self.ui.register_button.setEnabled(False)
                self.ui.register_button.setStyleSheet(
                    "QPushButton#register_button {background-color: #31363B; border: 0.04em solid #76797c;}")  # gray
                # self.ui.apply_to_button.setEnabled(False)
                # self.ui.apply_to_button.setStyleSheet(
                #     "QPushButton#apply_to_button {background-color: #31363B; border: 0.04em solid #76797c;}")  # gray
            else:
                self.ui.moving_resamp_button.setIcon(QIcon(str(icons_dir / 'white_check.svg')))
                self.ui.moving_resamp_button.setStyleSheet(
                    "QPushButton#moving_resamp_button {background-color: #549159; border: 0.04em solid #549159;}")  # green
                self.ui.moving_resamp_button.setEnabled(True)
                self.ui.moving_resamp_button.setToolTip('Images have matching geometry')
                # self.ui.moving_vol_main_frame.setStyleSheet("QFrame#moving_vol_main_frame {border: 1.5px solid #549159}")  # green
                # self.ui.fixed_vol_main_frame.setStyleSheet("QFrame#fixed_vol_main_frame {border: 1.5px solid #549159}")  # green
                self.ui.moving_vol_main_frame.setStyleSheet("""
                                                            QFrame#moving_vol_main_frame {
                                                                border-style: inset;
                                                                border-width: 1px;
                                                                border-color: #549159;
                                                            }
                                                            """)
                self.ui.fixed_vol_main_frame.setStyleSheet("""
                                                            QFrame#fixed_vol_main_frame {
                                                                border-style: inset;
                                                                border-width: 1px;
                                                                border-color: #549159;
                                                            }
                                                            """)

                self.moving_lm_table.setEnabled(True)
                self.fixed_lm_table.setEnabled(True)
                # self.registered_lm_table.setEnabled(True)

                self.ui.register_button.setEnabled(True)
                # self.ui.register_button.setStyleSheet("QPushButton#register_button {background-color: #447CF9; border: 0.04em solid #447CF9;}") # theme blue
                self.ui.register_button.setStyleSheet("QPushButton#register_button {background-color: #549159; border: 0.04em solid #549159;}")  # green

                # if self.current_deformation_field is None:
                #     self.ui.apply_to_button.setEnabled(False)
                #     self.ui.apply_to_button.setStyleSheet(
                #         "QPushButton#apply_to_button {background-color: #31363B; border: 0.04em solid #76797c;}")  # gray
                # else:
                #     self.ui.apply_to_button.setEnabled(True)
                #     self.ui.apply_to_button.setStyleSheet(
                #         "QPushButton#apply_to_button {background-color: #447CF9; border: 0.04em solid #447CF9;}")  # theme blue

                # landmarking status ---------
                # possible cases to consider:
                # case -1: idle, not creating a landmark
                # case 0: creating a landmark, both points placed
                # case 1: creating landmark and one of the images is waiting for the placement of a marker
                # case 2: creating a landmark and neither image is waiting for the placement of a marker
                if self.lm_status == -1:
                    # status idle. we are not creating a landmark
                    self.ui.moving_lm_new_cancel_button.setText('New')
                    self.ui.fixed_lm_new_cancel_button.setText('New')
                    self.ui.moving_lm_new_cancel_button.setEnabled(True)
                    self.ui.fixed_lm_new_cancel_button.setEnabled(True)
                    self.ui.moving_lm_delete_done_button.setText('Delete')
                    self.ui.fixed_lm_delete_done_button.setText('Delete')
                    self.ui.moving_lm_delete_done_button.setEnabled(True)
                    self.ui.fixed_lm_delete_done_button.setEnabled(True)
                    # Check if we're in edit mode - if so, show "Cancel", otherwise "Edit"
                    if self.lm_edit_mode:
                        self.ui.moving_lm_edit_cancel_button.setText('Cancel')
                        self.ui.fixed_lm_edit_cancel_button.setText('Cancel')
                        self.ui.moving_lm_edit_cancel_button.setEnabled(True)
                        self.ui.fixed_lm_edit_cancel_button.setEnabled(True)
                        # In edit mode, show "Done" button (similar to new landmark creation)
                        self.ui.moving_lm_delete_done_button.setText('Save')
                        self.ui.fixed_lm_delete_done_button.setText('Save')
                        self.ui.moving_lm_delete_done_button.setEnabled(True)
                        self.ui.fixed_lm_delete_done_button.setEnabled(True)
                        # Disable New button while editing
                        self.ui.moving_lm_new_cancel_button.setEnabled(False)
                        self.ui.fixed_lm_new_cancel_button.setEnabled(False)
                        self.ui.moving_lm_delete_all_button.setEnabled(False)
                        self.ui.fixed_lm_delete_all_button.setEnabled(False)
                    else:
                        self.ui.moving_lm_delete_all_button.setEnabled(True)
                        self.ui.fixed_lm_delete_all_button.setEnabled(True)
                        self.ui.moving_lm_edit_cancel_button.setText('Edit')
                        self.ui.fixed_lm_edit_cancel_button.setText('Edit')
                        if self.lm_any_selected():
                            self.ui.moving_lm_delete_done_button.setEnabled(True)
                            self.ui.fixed_lm_delete_done_button.setEnabled(True)
                            self.ui.moving_lm_edit_cancel_button.setEnabled(True)
                            self.ui.fixed_lm_edit_cancel_button.setEnabled(True)
                        else:
                            self.ui.moving_lm_delete_done_button.setEnabled(False)
                            self.ui.fixed_lm_delete_done_button.setEnabled(False)
                            self.ui.moving_lm_edit_cancel_button.setEnabled(False)
                            self.ui.fixed_lm_edit_cancel_button.setEnabled(False)
                elif self.lm_status == 0:
                    # we are creating a landmark and both points have been placed. The delete/done button is enabled
                    self.ui.moving_lm_new_cancel_button.setText('Cancel')
                    self.ui.fixed_lm_new_cancel_button.setText('Cancel')
                    self.ui.moving_lm_new_cancel_button.setEnabled(True)
                    self.ui.fixed_lm_new_cancel_button.setEnabled(True)
                    self.ui.moving_lm_delete_done_button.setText('Save')
                    self.ui.fixed_lm_delete_done_button.setText('Save')
                    self.ui.moving_lm_delete_done_button.setEnabled(True)
                    self.ui.fixed_lm_delete_done_button.setEnabled(True)
                    self.ui.moving_lm_delete_all_button.setEnabled(False)
                    self.ui.fixed_lm_delete_all_button.setEnabled(False)
                elif self.lm_status == 1 or self.lm_status == 2:
                    # Creating a landmark, one of the images is waiting for the placement of a point.
                    # The delete/done button won't be enabled until markers have been placed in both images
                    self.ui.moving_lm_new_cancel_button.setText('Cancel')
                    self.ui.fixed_lm_new_cancel_button.setText('Cancel')
                    self.ui.moving_lm_new_cancel_button.setEnabled(True)
                    self.ui.fixed_lm_new_cancel_button.setEnabled(True)
                    self.ui.moving_lm_delete_done_button.setText('Save')
                    self.ui.fixed_lm_delete_done_button.setText('Save')
                    self.ui.moving_lm_delete_done_button.setEnabled(False)
                    self.ui.fixed_lm_delete_done_button.setEnabled(False)
                    self.ui.moving_lm_delete_all_button.setEnabled(False)
                    self.ui.fixed_lm_delete_all_button.setEnabled(False)
                    self.ui.moving_lm_edit_cancel_button.setText('Edit')
                    self.ui.fixed_lm_edit_cancel_button.setText('Edit')
                    self.ui.moving_lm_edit_cancel_button.setEnabled(False)
                    self.ui.fixed_lm_edit_cancel_button.setEnabled(False)

                # update the moving image landmark table using the list of landmarks -----------------------------------
                num_rows = len(self.current_landmarks)
                self.moving_lm_table.set_num_rows(num_rows + 1)  # add a row for the next landmark
                for row_num in range(len(self.current_landmarks)):  # from 0 to num landmarks - 1

                    # FIXME: during testing and dev
                    # print(self.current_landmarks[row_num].source_patient_coords)

                    self.moving_lm_table.set_item(row_num, 0, QTableWidgetItem(str(self.current_landmarks[row_num].id)))
                    self.moving_lm_table.set_item(row_num, 1, QTableWidgetItem(
                        str(f'{self.current_landmarks[row_num].source_patient_coords[0]:.1f}')))
                    self.moving_lm_table.set_item(row_num, 2, QTableWidgetItem(
                        str(f'{self.current_landmarks[row_num].source_patient_coords[1]:.1f}')))
                    self.moving_lm_table.set_item(row_num, 3, QTableWidgetItem(
                        str(f'{self.current_landmarks[row_num].source_patient_coords[2]:.1f}')))
                    if self.current_landmarks[row_num].selected:
                        self.moving_lm_table.table.selectRow(row_num)


                # FIXME: this is a weird way of tracking the status of the images.
                # fill in row for the landmark in progress if we are not editing and existing lm
                if self.temp_source_point_patient is not None:
                    self.moving_lm_table.set_item(num_rows, 1,
                                                  QTableWidgetItem(str(f'{self.temp_source_point_patient[0]:.1f}')),
                                                  (256, 0, 0))
                    self.moving_lm_table.set_item(num_rows, 2,
                                                  QTableWidgetItem(str(f'{self.temp_source_point_patient[1]:.1f}')),
                                                  (256, 0, 0))
                    self.moving_lm_table.set_item(num_rows, 3,
                                                  QTableWidgetItem(str(f'{self.temp_source_point_patient[2]:.1f}')),
                                                  (256, 0, 0))
                else:
                    # clear last line if no temp landmark
                    self.moving_lm_table.set_item(num_rows, 0, QTableWidgetItem(None))
                    self.moving_lm_table.set_item(num_rows, 1, QTableWidgetItem(None))
                    self.moving_lm_table.set_item(num_rows, 2, QTableWidgetItem(None))
                    self.moving_lm_table.set_item(num_rows, 3, QTableWidgetItem(None))

                # self.moving_lm_table.table.resizeColumnsToContents()
                # self.moving_lm_table.table.resizeRowsToContents()

                # update the fixed image landmark table using the list of landmarks --------------------------------------------
                self.fixed_lm_table.set_num_rows(num_rows + 1)
                for row_num in range(len(self.current_landmarks)):
                    self.fixed_lm_table.set_item(row_num, 0, QTableWidgetItem(str(self.current_landmarks[row_num].id)))
                    self.fixed_lm_table.set_item(row_num, 1, QTableWidgetItem(
                        str(f'{self.current_landmarks[row_num].target_patient_coords[0]:.1f}')))
                    self.fixed_lm_table.set_item(row_num, 2, QTableWidgetItem(
                        str(f'{self.current_landmarks[row_num].target_patient_coords[1]:.1f}')))
                    self.fixed_lm_table.set_item(row_num, 3, QTableWidgetItem(
                        str(f'{self.current_landmarks[row_num].target_patient_coords[2]:.1f}')))
                    if self.current_landmarks[row_num].selected:
                        self.fixed_lm_table.table.selectRow(row_num)

                # if this is the first landmark added, resize the columns to fit the header and data
                if num_rows == 1:
                    self.moving_lm_table.table.resizeColumnsToContents()
                    self.fixed_lm_table.table.resizeColumnsToContents()

                # fill in row for the landmark in progress if not editing an existing lm
                if self.temp_target_point_patient is not None:
                    self.fixed_lm_table.set_item(num_rows, 1,
                                                 QTableWidgetItem(str(f'{self.temp_target_point_patient[0]:.1f}')))
                    self.fixed_lm_table.set_item(num_rows, 2,
                                                 QTableWidgetItem(str(f'{self.temp_target_point_patient[1]:.1f}')))
                    self.fixed_lm_table.set_item(num_rows, 3,
                                                 QTableWidgetItem(str(f'{self.temp_target_point_patient[2]:.1f}')))
                else:
                    # clear last line if no temp landmark
                    self.fixed_lm_table.set_item(num_rows, 0, QTableWidgetItem(None))
                    self.fixed_lm_table.set_item(num_rows, 1, QTableWidgetItem(None))
                    self.fixed_lm_table.set_item(num_rows, 2, QTableWidgetItem(None))
                    self.fixed_lm_table.set_item(num_rows, 3, QTableWidgetItem(None))

                # update the registered landmark table
                # self.registered_lm_table.set_num_rows(num_rows + 1)
                # for row_num in range(len(self.current_landmarks)):
                #     self.registered_lm_table.set_item(row_num, 0, QTableWidgetItem(str(self.current_landmarks[row_num].id)))
                #     if self.current_landmarks[row_num].registered_patient_coords is not None:
                #         self.registered_lm_table.set_item(row_num, 1, QTableWidgetItem(
                #             str(f'{self.current_landmarks[row_num].registered_patient_coords[0]:.1f}')))
                #         self.registered_lm_table.set_item(row_num, 2, QTableWidgetItem(
                #             str(f'{self.current_landmarks[row_num].registered_patient_coords[1]:.1f}')))
                #         self.registered_lm_table.set_item(row_num, 3, QTableWidgetItem(
                #             str(f'{self.current_landmarks[row_num].registered_patient_coords[2]:.1f}')))
                #     else:
                #         # Clear cells if no registered coordinates
                #         self.registered_lm_table.set_item(row_num, 1, QTableWidgetItem(""))
                #         self.registered_lm_table.set_item(row_num, 2, QTableWidgetItem(""))
                #         self.registered_lm_table.set_item(row_num, 3, QTableWidgetItem(""))
                #     if self.current_landmarks[row_num].selected:
                #         self.registered_lm_table.table.selectRow(row_num)
                #
                # # if this is the first landmark added, resize the columns to fit the header and data
                # if num_rows == 1:
                #     self.registered_lm_table.table.resizeColumnsToContents()
        # if self.ui.moving_lm_table_frame.isVisible():
        #     # self.ui.show_hide_moving_lm_button.setToolTip('Hide landmarks lable')
        # else:
        #     # self.ui.show_hide_moving_lm_button.setToolTip('Show landmarks table')
        # if self.ui.fixed_lm_table_frame.isVisible():
        #     # self.ui.show_hide_fixed_lm_button.setToolTip('Hide landmarks table')
        # else:
        #     # self.ui.show_hide_fixed_lm_button.setToolTip('Show landmarks table')

        # update the session label widgets -----------------------------------------------------------------------------
        self.ui.device_label.setText(self.device.upper())
        self.ui.algorithm_label.setText(self.algorithm.algorithm_description)
        self.ui.algorithm_label.setToolTip(self.algorithm.algorithm_description)
        if self.current_deformation_field is not None:
            self.dfield_manager_window.ui.current_dfield_label.setText(self.current_deformation_field.name)
            self.dfield_manager_window.ui.current_dfield_label.setToolTip(self.current_deformation_field.name)
        else:
            self.dfield_manager_window.ui.current_dfield_label.setText("")
            self.dfield_manager_window.ui.current_dfield_label.setToolTip("")

        # update the state of menu action items ------------------------------------------------------------------------
        if self.registered_volume is None:
            self.ui.action_save_registered_nifti.setEnabled(False)
        else:
            self.ui.action_save_registered_nifti.setEnabled(True)
        # TODO

        # refresh the standalone vol selections window -----------------------------------------------------------------
        # self.volume_selector_window.refresh()

        # refresh dfield manager window widgets ------------------------------------------------------------------------
        self.dfield_manager_window.refresh()

        # refresh the moving image widgets -----------------------------------------------------------------------------
        if self.moving_volume is None:
            self.ui.show_hide_moving_button.setEnabled(False)
            self.ui.close_moving_button.setEnabled(False)
            if self.moving_overlay is None:
                # self.ui.moving_disp_show_button.setEnabled(False)
                self.ui.moving_slice_guides_button.setEnabled(False)
            else:
                # self.ui.moving_disp_show_button.setEnabled(True)
                self.ui.moving_slice_guides_button.setEnabled(True)
        else:
            self.ui.show_hide_moving_button.setEnabled(True)
            self.ui.close_moving_button.setEnabled(True)
            # self.ui.moving_disp_show_button.setEnabled(True)
            self.ui.moving_slice_guides_button.setEnabled(True)
        if self.moving_overlay is None:
            self.ui.show_hide_mov_overlay_button.setEnabled(False)
            self.ui.close_moving_overlay_button.setEnabled(False)
        else:
            self.ui.show_hide_mov_overlay_button.setEnabled(True)
            self.ui.close_moving_overlay_button.setEnabled(True)

        # refresh the fixed image widgets ------------------------------------------------------------------------------
        if self.fixed_volume is None:
            self.ui.show_hide_fixed_button.setEnabled(False)
            self.ui.close_fixed_button.setEnabled(False)
            if self.fixed_overlay is None:
                # self.ui.fixed_disp_show_button.setEnabled(False)
                self.ui.fixed_slice_guides_button.setEnabled(False)
            else:
                # self.ui.fixed_disp_show_button.setEnabled(True)
                self.ui.fixed_slice_guides_button.setEnabled(True)
        else:
            self.ui.show_hide_fixed_button.setEnabled(True)
            self.ui.close_fixed_button.setEnabled(True)
            # self.ui.fixed_disp_show_button.setEnabled(True)
            self.ui.fixed_slice_guides_button.setEnabled(True)
        if self.fixed_overlay is None:
            self.ui.show_hide_fix_overlay_button.setEnabled(False)
            self.ui.close_fixed_overlay_button.setEnabled(False)
        else:
            self.ui.show_hide_fix_overlay_button.setEnabled(True)
            self.ui.close_fixed_overlay_button.setEnabled(False)
        # fixed display settings widgets
        # if self.active_fixed_settings_image is None:
        #     self.ui.fixed_log_checkbox.setEnabled(False)
        #     self.ui.fixed_min_edit.setEnabled(False)
        #     self.ui.fixed_max_edit.setEnabled(False)
        #     self.ui.fixed_transparency_slider.setEnabled(False)
        #     # self.fixed_colormap_widget_stack.setEnabled(False)
        # else:
        #     self.ui.fixed_log_checkbox.setEnabled(True)
        #     self.ui.fixed_min_edit.setEnabled(True)
        #     self.ui.fixed_max_edit.setEnabled(True)
        #     self.ui.fixed_transparency_slider.setEnabled(True)
        #     # self.fixed_colormap_widget_stack.setEnabled(True)

        # self.disp_refresh('fixed')

        # refresh the registered image widgets -------------------------------------------------------------------------
        if self.registered_volume is None:
            self.ui.reg_alpha_slider.setEnabled(False)
            # self.ui.reg_alpha_slider_1.setEnabled(False)
            # self.ui.reg_alpha_slider_2.setEnabled(False)
            self.ui.reg_slice_guides_button.setEnabled(False)
            self.ui.show_hide_reg_button.setEnabled(False)
            self.ui.close_registered_button.setEnabled(False)
        else:
            self.ui.reg_alpha_slider.setEnabled(True)
            # self.ui.reg_alpha_slider_1.setEnabled(True)
            # self.ui.reg_alpha_slider_2.setEnabled(True)
            self.ui.reg_slice_guides_button.setEnabled(True)
            self.ui.show_hide_reg_button.setEnabled(True)
            self.ui.close_registered_button.setEnabled(True)

    def refresh_throttled(self):
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self.do_refresh_once)

    def do_refresh_once(self):
        self._refresh_pending = False
        self.refresh()

    def canonical_cmap_name(self, name: str) -> str:
        return self._CMAP_ALIASES.get(name.strip(), name.strip())

    def write_image3d_as_nifti(self, im3d, filename: str) -> bool:
        """
        Write an Image3D to a NIfTI file and embed LandMarker UID/provenance.

        Uses:
          - im3d.data          : ndarray, 3D or 4D (T, Z, Y, X) also supported
          - im3d.transform     : 4x4 affine (ijk -> world), if present
          - im3d.dx,dy,dz      : voxel spacing (mm)
          - im3d.canonical_nifti: optional nib.Nifti1Image to copy header from

        Returns True on success.
        """
        try:
            # ---- Data ----
            data = np.asarray(im3d.data)
            if getattr(im3d, "data_type", None) is not None:
                try:
                    data = data.astype(im3d.data_type, copy=False)
                except Exception as e:
                    # Non-critical: data type conversion failed, continue with original type
                    pass

            # Normalize unsupported integer dtypes (e.g. int64/uint64) to NIfTI-compatible ones
            dt = data.dtype
            if np.issubdtype(dt, np.signedinteger) and dt.itemsize >= 8:
                self.print_log_message(
                    f"write_image3d_as_nifti: casting data from {dt} to int32 for NIfTI compatibility."
                )
                data = data.astype(np.int32, copy=False)
                dt = data.dtype
            elif np.issubdtype(dt, np.unsignedinteger) and dt.itemsize >= 8:
                self.print_log_message(
                    f"write_image3d_as_nifti: casting data from {dt} to uint32 for NIfTI compatibility."
                )
                data = data.astype(np.uint32, copy=False)
                dt = data.dtype

            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)

            # Log final dtype being written
            try:
                self.print_log_message(
                    f"write_image3d_as_nifti saving {filename} with dtype {data.dtype}"
                )
            except Exception:
                # Logging is non-critical
                pass

            # ---- Affine (ijk -> world) ----
            affine = None
            if getattr(im3d, "transform", None) is not None:
                A = np.asarray(im3d.transform)
                if A.shape == (4, 4):
                    affine = A
            if affine is None:
                dx = float(getattr(im3d, "dx", 1.0) or 1.0)
                dy = float(getattr(im3d, "dy", 1.0) or 1.0)
                dz = float(getattr(im3d, "dz", 1.0) or 1.0)
                ox = oy = oz = 0.0
                if getattr(im3d, "origin", None) is not None:
                    try:
                        ox, oy, oz = [float(v) for v in im3d.origin[:3]]
                    except Exception as e:
                        # Non-critical: origin parsing failed, use defaults
                        pass
                affine = np.array([[dx, 0.0, 0.0, ox],
                                   [0.0, dy, 0.0, oy],
                                   [0.0, 0.0, dz, oz],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=float)

            # ---- Header (optionally copy canonical) ----
            header = None
            canon = getattr(im3d, "canonical_nifti", None)
            if isinstance(canon, nib.Nifti1Image):
                header = canon.header.copy()

            # Create image
            img = nib.Nifti1Image(data, affine, header=header)

            # ---- Set zooms (voxel sizes) ----
            try:
                dx = float(getattr(im3d, "dx", 1.0) or 1.0)
                dy = float(getattr(im3d, "dy", 1.0) or 1.0)
                dz = float(getattr(im3d, "dz", 1.0) or 1.0)
                if data.ndim == 3:
                    img.header.set_zooms((dx, dy, dz))
                elif data.ndim == 4:
                    dt = float(getattr(im3d, "time_spacing", 1.0) or 1.0)
                    img.header.set_zooms((dx, dy, dz, dt))
            except Exception as e:
                # Non-critical: zoom setting failed
                pass

            # Units
            try:
                img.header.set_xyzt_units("mm", "sec")
            except Exception as e:
                # Non-critical: unit setting failed
                pass

            # ---- sform/qform codes ----
            try:
                img.set_sform(affine, code=1)  # 1 = scanner_anat
                img.set_qform(affine, code=1)
            except Exception as e:
                # Non-critical: sform/qform setting failed
                pass

            # Optional display range
            try:
                dmin = float(im3d.display_min) if getattr(im3d, "display_min", None) is not None else None
                dmax = float(im3d.display_max) if getattr(im3d, "display_max", None) is not None else None
                if dmin is not None and dmax is not None and dmin < dmax:
                    img.header["cal_min"] = dmin
                    img.header["cal_max"] = dmax
            except Exception as e:
                # Non-critical: display range setting failed
                pass

            # ---- Save NIfTI ----
            nib.save(img, filename)

            # ---- LandMarker sidecar ----
            try:
                # 1. Compute UID from the NIfTI object that's being saved
                # This ensures the UID matches what will be computed when the file is loaded
                uid_meta = compute_image_uid_from_nifti(img)
                # Update the volume's uid to match the saved file
                if hasattr(im3d, "uid"):
                    im3d.uid = uid_meta["image_uid"]

                # 2. Write sidecar next to NIfTI
                LMSidecarUtils.write_sidecar(filename, uid_meta, landmarker_version=self.version, parent=self)
            except Exception as e:
                error_msg = f"Image saved successfully, but sidecar could not be written:\n\n{e}"
                self.print_log_message(error_msg)
                QMessageBox.warning(self, "Sidecar not written", error_msg, QMessageBox.Ok)

            QMessageBox.information(
                self, "Saved", f"Image saved successfully:\n{filename}", QMessageBox.Ok
            )
            return True

        except Exception as e:
            error_msg = f"write_image3d_as_nifti error saving {filename}: {e}"
            self.print_log_message(error_msg)
            print(error_msg)
            return False

    def landmarker_sidecar_path(self, out_path: str) -> str:
        """Return a sidecar path next to the NIfTI (e.g., foo.landmarker.json)."""
        low = out_path.lower()
        if low.endswith(".nii.gz"):
            stem = out_path[:-7]
        elif low.endswith(".nii"):
            stem = out_path[:-4]
        else:
            stem = out_path
        return f"{stem}.landmarker.json"


# globals
# FIXME: also defined in LMdFieldManager.py!
pkg_root = Path(__file__).resolve().parents[1]
icons_dir = pkg_root / "ui" / "icons"

def main():
    # set up argument parser
    parser = argparse.ArgumentParser(description="Launch the PyQt application with optional arguments.")
    parser.add_argument("--moving_nifti", type=str, help="Full path to the moving image NIfTI file.")
    parser.add_argument("--fixed_nifti", type=str, help="Full path to the fixed image NIfTI file.")
    parser.add_argument("--landmarks_dir", type=str, help="Directory to read or write landmark files.")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark", help="Choose the UI theme.")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    set_hidpi_safely()

    # initialize the application
    app = QtWidgets.QApplication(sys.argv)
    
    # set the application icon (for taskbar, etc.)
    icon_path = icons_dir / "landmarker_icon.ico"
    if not icon_path.exists():
        # Try alternative formats if .ico doesn't exist
        icon_path = icons_dir / "landmarker_icon.png"
    if not icon_path.exists():
        icon_path = icons_dir / "landmarker_icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # TODO: other stylesheet options
    stylesheet_file = "breezeDarkStylesheet.qss"   # if args.theme == "dark" else "breezeLightStylesheet.qss"
    qss_path = pkg_root / "ui" / "stylesheets" / stylesheet_file

    # load and clean up the stylesheet
    with open(qss_path, "r") as f:
        stylesheet = f.read()
    stylesheet = stylesheet.replace("../icons", str(icons_dir).replace("\\", "/"))
    app.setStyleSheet(stylesheet)
    if sys.platform == "darwin":
        app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    # app.setPalette(QtGui.QPalette(QtGui.QColor(240, 240, 240)))
    # app.setStyleSheet("""
    #     QWidget {
    #         background-color: #f0f0f0;
    #         color: #333;
    #     }
    # """)

    if args.debug_mode:
        print(f"Moving NIfTI file: {args.moving_nifti}")
        print(f"Fixed NIfTI file: {args.fixed_nifti}")
        print(f"Landmarks directory: {args.landmarks_dir}")
        print(f"Using theme: {args.theme}")
        print(f"Stylesheet path: {qss_path}")
        print("Debug mode is enabled.")


    mainWindow = LandMarker(
        source_file=args.moving_nifti,
        target_file=args.fixed_nifti,
        lm_dir=args.landmarks_dir,
        theme=args.theme,
        debug_mode=args.debug_mode
    )
    # launch as maximized
    # mainWindow.showMaximized()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
