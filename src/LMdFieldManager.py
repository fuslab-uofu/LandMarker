"""
LMdFieldManager.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from PyQt5.QtWidgets import QMainWindow, QAbstractItemView, QMenu, QInputDialog, \
    QMessageBox, QDialog, QVBoxLayout, QTextEdit, QHeaderView
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

import numpy as np
import torch
from pathlib import Path

from LMDeformationField import LMDeformationField
from ui.Ui_LMdFieldManager import Ui_LMdFieldManager
from Common.composeHfields import composeHfields

# globals
# FIXME: also defined in LandMarker.py!!
pkg_root = Path(__file__).resolve().parents[1]
icons_dir = pkg_root / "ui" / "icons"

class TableModel(QAbstractTableModel):
    def __init__(self, data, headers, parent=None):
        super(TableModel, self).__init__(parent)
        self._data = data
        self._headers = headers

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        if self.rowCount() > 0:
            return len(self._data[0])
        return 0

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and self._headers and 0 <= section < len(self._headers):
            return self._headers[section]
        return None

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        self._data.sort(key=lambda x: x[column], reverse=order == Qt.DescendingOrder)
        self.layoutChanged.emit()


class LMdFieldManager(QMainWindow):
    """ This class is a QMainWindow that allows the user to manage deformation fields.
    The user can select deformation fields to compose by moving them from the left table to the right table,
    order them as desired, and compose them.
    The composed deformation field is added to the list of available deformation fields.
    This is also where the user can apply a deformation field to one or more moving volumes.
    """
    dfield_selected_signal = pyqtSignal(int)  # signal emitted when range changes (min, max)
    save_dfield_signal = pyqtSignal(int)  # signal emitted when a deformation field is saved
    rename_dfield_signal = pyqtSignal(int, str)  # signal emitted when a deformation field is renamed
    load_dfield_signal = pyqtSignal()  # signal emitted when a deformation field is loaded

    images_selected_signal = pyqtSignal(int)  # signal emitted user selects image and clicks "Apply"
    apply_dfield_signal = pyqtSignal(object)  # (list of moving volumes) emitted when user clicks Apply

    def __init__(self, parent=None):
        super().__init__()

        self.parent = parent

        self.ui = Ui_LMdFieldManager()
        self.ui.setupUi(self)

        # sync looks and feel of text box with the parent window
        theme_font = QFont("Segoe UI", 8)
        self.ui.current_dfield_label.setStyleSheet("""
                            QLabel#current_dfield_label {
                                background-color: #626568;
                                border-color: #626568;
                            }
                            """)
        self.ui.current_dfield_label.setFont(theme_font)
        self.ui.curr_df_label_label.setFont(theme_font)
        self.ui.volumes_frame.setStyleSheet("""
                            QFrame#volumes_frame {
                                border-style: inset;
                                border-width: 2px;
                                border-color: #1D2023;
                            }
                            """)
        self.ui.current_dfield_frame.setStyleSheet("""
                            QFrame#current_dfield_frame {
                                border-style: inset;
                                border-width: 2px;
                                border-color: #1D2023;
                            }
                            """)

        # left table (deformation fields)
        self.left_table_data = []
        self.left_table_model = TableModel(self.left_table_data, ["Index", "Name"])
        self.ui.left_table_view.setModel(self.left_table_model)
        # right-click context menu for the left table (for renaming dfield)
        self.ui.left_table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.left_table_view.customContextMenuRequested.connect(self.show_context_menu)
        self.ui.set_current_left_button.clicked.connect(self.set_current)
        self.ui.save_dfield_button.clicked.connect(self.save_selected_dfield)
        self.ui.import_dfield_button.clicked.connect(self.import_dfield)
        self.ui.view_matrix_left_button.clicked.connect(self.view_matrix_left)
        self.ui.view_matrix_right_button.clicked.connect(self.view_matrix_right)

        # arrange deformation fields buttons <-- --> ^ v
        self.ui.move_right_button.setIcon(QIcon(str(icons_dir / 'right_arrow.svg')))
        self.ui.move_right_button.clicked.connect(self.move_item_right)
        self.ui.remove_button.setIcon(QIcon(str(icons_dir / 'left_arrow.svg')))
        self.ui.remove_button.clicked.connect(self.remove_item)
        self.ui.move_up_button.setIcon(QIcon(str(icons_dir / 'up_arrow.svg')))
        self.ui.move_up_button.clicked.connect(self.move_item_up)
        self.ui.move_down_button.setIcon(QIcon(str(icons_dir / 'down_arrow.svg')))
        self.ui.move_down_button.clicked.connect(self.move_item_down)

        # right table (deformation fields to compose)
        self.right_table_data = []
        self.right_table_model = TableModel(self.right_table_data, ["Index", "Name"])
        self.ui.right_table_view.setModel(self.right_table_model)
        self.ui.compose_button.clicked.connect(self.compose)

        # bottom table (moving volumes)
        self.moving_vol_table_data = []
        self.moving_vol_objects = []  # row index -> volume object, for retrieval when applying dfield
        self.moving_vol_table_model = TableModel(self.moving_vol_table_data, ["Name"])
        self.ui.moving_vol_table_view.setModel(self.moving_vol_table_model)

        for table in (self.ui.left_table_view, self.ui.right_table_view, self.ui.moving_vol_table_view):
            table.horizontalHeader().setFont(theme_font)
            table.verticalHeader().setFont(theme_font)
            table.horizontalHeader().setFixedHeight(20)
            table.verticalHeader().setDefaultSectionSize(20)

        self.ui.apply_button.clicked.connect(self.apply)

        # when deformation fields are updated in the main window (during registration), update the left table
        self.parent.dfields_updated_signal.connect(self.refresh)

        self.refresh()

    def show_context_menu(self, pos):
        selected_row = self.ui.left_table_view.selectedIndexes()[0].row()
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self.rename_item(selected_row))
        menu.exec_(self.ui.left_table_view.viewport().mapToGlobal(pos))

    def rename_item(self, selected_row):
        dfield_idx = self.left_table_data[selected_row][0]  # index into list of parent deformation fields
        old_value = self.left_table_data[selected_row][1]
        new_value, ok = QInputDialog.getText(self, "Rename Item", "New value:", text=old_value)
        if ok and new_value:
            self.rename_dfield_signal.emit(dfield_idx, new_value)
            self.refresh()

    def set_current(self):
        """ Set the selected deformation field as the current deformation field.
        """
        if len(self.left_table_data) == 0:
            return  # no data
        selected_indexes = self.ui.left_table_view.selectedIndexes()
        if selected_indexes:
            selected_row = selected_indexes[0].row()
            dfield_idx = self.left_table_data[selected_row][0]
            self.dfield_selected_signal.emit(dfield_idx)

    def save_selected_dfield(self):
        """ Save the selected deformation field to a file. Calls LandMarker::save_dfield().
        """
        if len(self.left_table_data) == 0:
            return  # no data
        selected_indexes = self.ui.left_table_view.selectedIndexes()
        if selected_indexes:
            # something is selected
            selected_row = selected_indexes[0].row()
            dfield_idx = self.left_table_data[selected_row][0]
            self.save_dfield_signal.emit(dfield_idx)

    def import_dfield(self):
        """ Import a deformation field from a file. Calls LandMarker::load_dfield().
            Adds the imported deformation field to the list of availabel deformation fields.
        """
        self.load_dfield_signal.emit()
        self.refresh()

    def refresh(self):
        """ Refresh the available deformation fields and moving volumes in the left and right tables.
        """
        # populate/refresh the left table with the list of deformation fields
        dfield_list = []
        for idx, df in enumerate(self.parent.deformation_fields):
            dfield_list.append([idx, df.name])
        self.left_table_data = dfield_list
        self.left_table_model.beginResetModel()  # Notify view that model data will change
        self.left_table_model._data = dfield_list
        self.left_table_model._headers = ["Index", "Name"]  # FIXME: doesn't work
        self.left_table_model.endResetModel()  # Notify view that model data has changed
        self.left_table_model.sort(0, Qt.AscendingOrder)  # sort left table by the first column
        self.ui.left_table_view.resizeColumnsToContents()

        # populate/refresh the moving volumes table
        # first, determine which moving volumes match the geometry of the current fixed volume
        matches, mismatches = self._matching_moving_volumes()
        # fill table with ONLY matches (model expects list of rows, each row is list of column values)
        self.moving_vol_table_data = [[str(vol.file_name)] for vol in matches]
        self.moving_vol_objects = list(matches)
        self.moving_vol_table_model.beginResetModel()
        self.moving_vol_table_model._data = self.moving_vol_table_data
        self.moving_vol_table_model._headers = ["Name"]
        self.moving_vol_table_model.endResetModel()
        self.ui.moving_vol_table_view.resizeColumnsToContents()

    def move_item_right(self):
        if len(self.left_table_data) == 0:
            return  # No items to move
        selected_indexes = self.ui.left_table_view.selectedIndexes()
        if not selected_indexes:
            return  # No selected indexes, do nothing
        selected_rows = {index.row() for index in selected_indexes}  # Get unique row indices
        for row in sorted(selected_rows, reverse=True):  # Iterate in reverse order to avoid index issues
            item_data = self.left_table_data[row]
            self.right_table_data.append(item_data)

        self.right_table_model.layoutChanged.emit()

    def remove_item(self):
        if len(self.right_table_data) == 0:
            return
        selected_indexes = self.ui.right_table_view.selectedIndexes()
        if not selected_indexes:
            return  # No selected indexes, do nothing
        selected_rows = {index.row() for index in selected_indexes}  # Get unique row indices
        for row in sorted(selected_rows, reverse=True):  # Iterate in reverse order to avoid index issues
            del self.right_table_data[row]
        self.right_table_model.layoutChanged.emit()

    def move_item_up(self):
        rows = self.ui.right_table_view.selectionModel().selectedRows()
        if len(rows) == 0:
            return
        row = rows[0].row()
        if row > 0:
            # Swap the selected row with the row above it
            self.right_table_data[row], self.right_table_data[row - 1] = (
                self.right_table_data[row - 1],
                self.right_table_data[row],
            )
        self.right_table_model.layoutChanged.emit()

    def move_item_down(self):
        rows = self.ui.right_table_view.selectionModel().selectedRows()
        if len(rows) == 0:
            return
        row = rows[0].row()
        if row < len(self.right_table_data) - 1:
            # Swap the selected row with the row below it
            self.right_table_data[row], self.right_table_data[row + 1] = (
                self.right_table_data[row + 1],
                self.right_table_data[row],
            )
        self.right_table_model.layoutChanged.emit()

    def compose(self):
        """
        compose the selected deformation fields
        """
        if len(self.right_table_data) == 0:
            return

        # start with the first deformation field in the list
        composed_idx = self.right_table_data[0][0]
        composed_df_data = self.parent.deformation_fields[composed_idx].data

        # compose the rest of the deformation fields
        for df_name in self.right_table_data[1:]:
            idx = df_name[0]
            df_data = self.parent.deformation_fields[idx].data
            composed_df_data = composeHfields(composed_df_data, df_data, None, None)

        if composed_df_data is None:
            return
        # FIXME: temporary fix to add the composed dfield to the list of dfields
        new_df = LMDeformationField(self, len(self.parent.deformation_fields)+1, composed_df_data, True)
        self.parent.deformation_fields.append(new_df)
        self.refresh()

    def view_matrix_left(self):
        """ View matrix for the selected deformation field in the left table.
        """
        if len(self.left_table_data) == 0:
            QMessageBox.information(self, "No Data", "No deformation fields available.")
            return
        selected_indexes = self.ui.left_table_view.selectedIndexes()
        if not selected_indexes:
            QMessageBox.information(self, "No Selection", "Please select a deformation field to view its matrix.")
            return
        selected_row = selected_indexes[0].row()
        dfield_idx = self.left_table_data[selected_row][0]
        self._show_matrix_dialog(dfield_idx)

    def view_matrix_right(self):
        """ View matrix for the selected deformation field in the right table.
        """
        if len(self.right_table_data) == 0:
            QMessageBox.information(self, "No Data", "No deformation fields in the compose list.")
            return
        selected_indexes = self.ui.right_table_view.selectedIndexes()
        if not selected_indexes:
            QMessageBox.information(self, "No Selection", "Please select a deformation field to view its matrix.")
            return
        selected_row = selected_indexes[0].row()
        dfield_idx = self.right_table_data[selected_row][0]
        self._show_matrix_dialog(dfield_idx)

    def _show_matrix_dialog(self, dfield_idx):
        """ Show a dialog displaying matrix information for the deformation field.
        """
        if dfield_idx >= len(self.parent.deformation_fields):
            QMessageBox.warning(self, "Error", "Invalid deformation field index.")
            return

        dfield = self.parent.deformation_fields[dfield_idx]
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Matrix View - {dfield.name}")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier")
        
        # Build matrix information string
        info_text = f"Deformation Field: {dfield.name}\n"
        info_text += f"Description: {dfield.description}\n"
        info_text += f"Is Composed: {dfield.is_composed}\n"
        info_text += f"Is Imported: {getattr(dfield, 'is_imported', False)}\n"
        info_text += f"Algorithm: {dfield.algorithm}\n"
        info_text += f"Moving Volume: {dfield.moving_vol}\n"
        info_text += f"Fixed Volume: {dfield.fixed_vol}\n\n"
        
        # Display rigid/affine transformation matrix if available
        info_text += "Transformation Matrix:\n"
        if dfield.rigid_matrix is not None:
            if isinstance(dfield.rigid_matrix, torch.Tensor):
                matrix_np = dfield.rigid_matrix.cpu().numpy() if dfield.rigid_matrix.is_cuda else dfield.rigid_matrix.numpy()
            else:
                matrix_np = np.array(dfield.rigid_matrix)
            
            info_text += f"  Shape: {matrix_np.shape}\n"
            info_text += f"  Data Type: {matrix_np.dtype}\n\n"
            
            # Format the matrix nicely
            if len(matrix_np.shape) == 2:
                info_text += "  Matrix values:\n"
                for i in range(matrix_np.shape[0]):
                    row_str = "    ["
                    for j in range(matrix_np.shape[1]):
                        row_str += f"{matrix_np[i, j]:12.6f}"
                        if j < matrix_np.shape[1] - 1:
                            row_str += ", "
                    row_str += "]\n"
                    info_text += row_str
            else:
                info_text += f"  Matrix data: {matrix_np}\n"
        else:
            info_text += "  Not applicable for this registration method.\n"
        
        info_text += "\n"
        
        # Display deformation field data information
        if dfield.data is not None:
            if isinstance(dfield.data, torch.Tensor):
                data_np = dfield.data.cpu().numpy() if dfield.data.is_cuda else dfield.data.numpy()
            else:
                data_np = np.array(dfield.data)
            
            info_text += "Deformation Field Data:\n"
            info_text += f"  Shape: {data_np.shape}\n"
            info_text += f"  Data Type: {data_np.dtype}\n"
            
            # If it's a 3D deformation field (shape like [3, Z, Y, X] or [Z, Y, X, 3])
            if len(data_np.shape) == 4 and data_np.shape[0] == 3:
                # Extract sample values from center of each dimension
                z_center = data_np.shape[1] // 2
                y_center = data_np.shape[2] // 2
                x_center = data_np.shape[3] // 2
                
                info_text += f"\nSample displacement at center ({z_center}, {y_center}, {x_center}):\n"
                info_text += f"  X displacement: {data_np[0, z_center, y_center, x_center]:.6f}\n"
                info_text += f"  Y displacement: {data_np[1, z_center, y_center, x_center]:.6f}\n"
                info_text += f"  Z displacement: {data_np[2, z_center, y_center, x_center]:.6f}\n"
                
                # Statistics
                info_text += f"\nDisplacement Statistics:\n"
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    axis_data = data_np[i, :, :, :]
                    info_text += f"  {axis}: min={axis_data.min():.6f}, max={axis_data.max():.6f}, mean={axis_data.mean():.6f}\n"
            else:
                info_text += f"\nData preview (first 100 elements):\n"
                flat_data = data_np.flatten()[:100]
                info_text += str(flat_data) + "\n"
        else:
            info_text += "No deformation field data available.\n"
        
        text_edit.setPlainText(info_text)
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        dialog.exec_()

    def _selected_moving_volumes(self):
        """Return the list of Image3D objects currently selected in the table."""
        table = self.ui.moving_vol_table_view
        if table.selectionModel() is None:
            return []
        rows = sorted({idx.row() for idx in table.selectionModel().selectedIndexes()})
        return [self.moving_vol_objects[r] for r in rows if 0 <= r < len(self.moving_vol_objects)]

    def apply(self):
        """ Apply the selected deformation field to the selected moving volumes.
            Emits apply_dfield_signal(list of volumes) for the parent to handle.
        """
        vols = self._selected_moving_volumes()
        if not vols:
            QMessageBox.information(self, "Nothing selected",
                                              "Please select one or more moving volumes.")
            return
        self.apply_dfield_signal.emit(vols)

    def _matching_moving_volumes(self):
        """ Determine which moving volumes match the geometry of the current fixed volume.
            Returns two lists: (matches, mismatches)
        """
        fixed = getattr(self.parent, "fixed_volume", None)
        if fixed is None:
            return [], []

        moving_list = getattr(self.parent, "moving_volumes", []) or []
        if not moving_list:
            return [], []

        matches, mismatches = [], []
        for mv in moving_list:
            diffs = self.parent.compare_image_geometry(fixed, mv)
            if diffs:
                mismatches.append((mv, diffs))
            else:
                matches.append(mv)
        return matches, mismatches