""" LMTable.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidget, QVBoxLayout, QHeaderView, QSizePolicy, QAbstractItemView, QStyledItemDelegate
from PyQt5.QtGui import QFont, QBrush, QColor, QPen, QPainter
from PyQt5.QtCore import Qt, pyqtSignal, QEvent

class HighlightDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighted_row = -1  # No row is highlighted initially

    def set_highlighted_row(self, row):
        """Set the row to be highlighted."""
        self.highlighted_row = row
        parent_table = self.parent()
        if parent_table:
            parent_table.viewport().update()  # Trigger a repaint of the table

    def clear_highlighted_row(self):
        """Clear the highlighted row."""
        self.set_highlighted_row(-1)  # Reset the highlighted row

    def paint(self, painter, option, index):
        """Custom painting for the table."""
        super().paint(painter, option, index)  # Draw the default appearance

        # Check if this cell belongs to the highlighted row
        if index.row() == self.highlighted_row:
            pen = QPen(QColor("#447CF9"), 0.5)  # Define the outline color and thickness
            painter.save()
            painter.setPen(pen)

            # Draw the outline around the cell
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))
            painter.restore()


class LMTable(QtWidgets.QWidget):
    """ Class for managing an interactive (selectable) table of LMLandmarks. Instead of deriving from QTableWidget,
    LMTable includes a QTableWidet object.
    """
    row_clicked_signal = pyqtSignal(object, object)

    def __init__(self, parent, table_id):
        super(LMTable, self).__init__(parent)

        self.parent = parent  # main window
        self.id = table_id

        # create the table widget and layout
        self.table = QTableWidget(self)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.table)

        # allow LMTable itself and the table to expand
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)
        # table should fill available space, not resize to contents
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)

        # ---- table setup ------------------------------------------------------
        self.table.setColumnCount(4)
        self.selected_row_num = None
        self.highlighted_row_num = None

        # custom delegate
        self.delegate = HighlightDelegate(self.table)
        self.table.setItemDelegate(self.delegate)

        # headers
        if self.parent.display_convention == "LPS":
            headers = ['ID', 'L', 'P', 'S']
        else:  # self.parent.coord_system == "RAS":
            headers = ['ID', 'R', 'A', 'S']
        self.table.setHorizontalHeaderLabels(headers)

        horizontal_header = self.table.horizontalHeader()
        horizontal_header.setFont(QFont("Segoe UI", 8))  # font for headers
        # Initially, resize sections to fit contents (the header text)
        self.table.resizeColumnsToContents()
        # wwitch to Interactive so the user can resize manually
        horizontal_header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)  # QtWidgets.QHeaderView.Stretch
        self.table.verticalHeader().setVisible(False)
        
        # Note: Initial column width adjustment will happen in showEvent after frame is available

        # font for cells
        self.table.setFont(QFont("Courier New", ))

        # selection behavior
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.cellClicked.connect(self.on_cell_clicked)

        # disable corner button
        self.table.setCornerButtonEnabled(False)
        
        # Column width constraint management
        self._adjusting_columns = False  # Flag to prevent recursive adjustments
        self._parent_frame = None  # Will be set after widget is added to layout
        
        # Connect to header resize signal to maintain constraints
        horizontal_header.sectionResized.connect(self._on_section_resized)
        
        # Connect to widget resize to update column widths when frame resizes
        # We'll set this up after the widget is shown

    def highlight_row(self, row):
        """
        Highlight the specified row in the table by outlining it. Intent is to look different than selected row.

        :param row:
        """
        # clear any previous highlights
        if self.highlighted_row_num is not None:
            self.clear_highlighted_row()
            # self.reset_row_highlight(self.highlighted_row_num)

        # highlight the new row
        self.delegate.set_highlighted_row(row)

    def unhighlight_row(self):
        # clear any previous highlights
        if self.highlighted_row_num is not None:
            self.clear_highlighted_row()


    def clear_highlighted_row(self):
        self.delegate.clear_highlighted_row()

    def on_cell_clicked(self, row, column):
        """
        :param row:
        :param column:
        :return:
        Slot for signal sent when any cell in the table is clicked. Selects the landmark displayed in that row.
             Table property is already set to make the entire row selected when any cell clicked.
        """
        if self.parent.temp_source_point_voxel is not None or self.parent.temp_target_point_voxel is not None:   #self.parent.source_status != 0 or self.parent.target_status != 0:
            # landmark is being created or edited already - do nothing
            return

        # get the item in this row
        item = self.table.item(row, 0)

        # was already selected - deselect
        if (not item) or (self.selected_row_num == row):
            self.selected_row_num = None

        else:
            self.selected_row_num = row

        self.row_clicked_signal.emit(self.id, item)

        # this updates all three landmark tables
        # self.parent.refresh()

    def find_row_by_id(self, lm_id):
        """
        Find the row index where the first column matches the specified id.

        :param lm_id: Value to search for
        :return: Row index if found, otherwise -1
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)  # Get the item in the first column
            if item and item.text() == str(lm_id):  # Compare with the value
                return row
        return -1  # Return -1 if not found

    def set_num_rows(self, num_rows):
        self.table.setRowCount(num_rows)

    def get_num_rows(self):
        return self.table.rowCount()

    def set_item(self, row, col, item, color=(256, 256, 256)):
        self.table.setItem(row, col, item)
        item = self.table.item(row, col)

    def add_row(self):
        new_row_index = self.table.rowCount()
        self.table.insertRow(new_row_index)
    
    def _find_parent_frame(self):
        """Find the parent frame (mv_tbl_frame, fx_tbl_frame, or reg_tbl_frame) that contains this table."""
        if self._parent_frame is not None:
            return self._parent_frame
        
        # First try: find via main window's UI (most reliable)
        if hasattr(self.parent, 'ui'):
            if self.id == 0 and hasattr(self.parent.ui, 'mv_tbl_frame'):
                self._parent_frame = self.parent.ui.mv_tbl_frame
                return self._parent_frame
            elif self.id == 1:
                # Check both possible names (fx_tbl_frame and fx_tb_frame - there's a typo in the code)
                if hasattr(self.parent.ui, 'fx_tbl_frame'):
                    self._parent_frame = self.parent.ui.fx_tbl_frame
                    return self._parent_frame
                elif hasattr(self.parent.ui, 'fx_tb_frame'):
                    self._parent_frame = self.parent.ui.fx_tb_frame
                    return self._parent_frame
            elif self.id == 2 and hasattr(self.parent.ui, 'reg_tbl_frame'):
                self._parent_frame = self.parent.ui.reg_tbl_frame
                return self._parent_frame
        
        # Second try: walk up the widget hierarchy to find the frame
        widget = self.parent()
        while widget is not None:
            if hasattr(widget, 'objectName'):
                obj_name = widget.objectName()
                if obj_name in ['mv_tbl_frame', 'fx_tbl_frame', 'fx_tb_frame', 'reg_tbl_frame']:
                    self._parent_frame = widget
                    return widget
            widget = widget.parent()
        
        return None
    
    def _get_frame_width(self):
        """Get the current width of the parent frame."""
        frame = self._find_parent_frame()
        if frame is not None:
            return frame.width()
        # Fallback to table width if frame not found
        return self.table.width()
    
    def _on_section_resized(self, logical_index, old_size, new_size):
        """Handle column resize to maintain constraints:
        - Total width = frame width
        - Columns 1, 2, 3 (R, A, S) are always equal
        - Column 0 (ID) can be resized, which adjusts columns 1-3 proportionally
        """
        if self._adjusting_columns:
            return
        
        self._adjusting_columns = True
        
        try:
            frame_width = self._get_frame_width()
            if frame_width <= 0:
                # Frame not yet sized, skip adjustment
                self._adjusting_columns = False
                return
            
            horizontal_header = self.table.horizontalHeader()
            
            if logical_index == 0:
                # ID column was resized
                # Get the new ID column width (clamped to reasonable bounds)
                id_width = max(30, min(new_size, frame_width - 90))  # Min 30, leave at least 90 for coord columns
                
                # Calculate width for each coordinate column (columns 1, 2, 3)
                remaining_width = frame_width - id_width
                coord_column_width = remaining_width / 3
                
                # Set all column widths
                horizontal_header.blockSignals(True)
                horizontal_header.resizeSection(0, id_width)
                horizontal_header.resizeSection(1, int(coord_column_width))
                horizontal_header.resizeSection(2, int(coord_column_width))
                # Last column gets any remainder to ensure total equals frame width
                last_col_width = frame_width - id_width - 2 * int(coord_column_width)
                horizontal_header.resizeSection(3, last_col_width)
                horizontal_header.blockSignals(False)
                
            elif logical_index in [1, 2, 3]:
                # One of the coordinate columns was resized
                # All coordinate columns should remain equal, so we need to adjust based on ID column
                id_width = horizontal_header.sectionSize(0)
                
                # Calculate new coordinate column width from the resized column
                # But we want all three to be equal, so use average
                coord_width_1 = horizontal_header.sectionSize(1)
                coord_width_2 = horizontal_header.sectionSize(2)
                coord_width_3 = horizontal_header.sectionSize(3)
                
                # Calculate remaining width for coordinate columns
                remaining_width = frame_width - id_width
                coord_column_width = remaining_width / 3
                
                # Set all coordinate columns to equal width
                horizontal_header.blockSignals(True)
                horizontal_header.resizeSection(1, int(coord_column_width))
                horizontal_header.resizeSection(2, int(coord_column_width))
                # Last column gets any remainder
                last_col_width = frame_width - id_width - 2 * int(coord_column_width)
                horizontal_header.resizeSection(3, last_col_width)
                horizontal_header.blockSignals(False)
                
        finally:
            self._adjusting_columns = False
    
    def _update_column_widths_to_frame(self):
        """Update column widths to match the current frame width.
        Called when the frame is resized.
        """
        if self._adjusting_columns:
            return
        
        frame_width = self._get_frame_width()
        if frame_width <= 0:
            return
        
        self._adjusting_columns = True
        try:
            horizontal_header = self.table.horizontalHeader()
            
            # Get current ID column width (or use default if not set)
            current_id_width = horizontal_header.sectionSize(0)
            if current_id_width <= 0:
                # Initialize with default proportions: ID = 40%, coord columns = 20% each
                id_width = int(frame_width * 0.4)
            else:
                # Keep current ID width, but clamp it
                id_width = max(30, min(current_id_width, frame_width - 90))
            
            # Calculate coordinate column widths
            remaining_width = frame_width - id_width
            coord_column_width = remaining_width / 3
            
            # Set all column widths
            horizontal_header.blockSignals(True)
            horizontal_header.resizeSection(0, id_width)
            horizontal_header.resizeSection(1, int(coord_column_width))
            horizontal_header.resizeSection(2, int(coord_column_width))
            # Last column gets any remainder
            last_col_width = frame_width - id_width - 2 * int(coord_column_width)
            horizontal_header.resizeSection(3, last_col_width)
            horizontal_header.blockSignals(False)
            
        finally:
            self._adjusting_columns = False
    
    def eventFilter(self, obj, event):
        """Event filter to catch frame resize events."""
        frame = self._find_parent_frame()
        if obj == frame and event.type() == QEvent.Resize:
            # Frame was resized, update column widths
            self._update_column_widths_to_frame()
        return super().eventFilter(obj, event)
    
    def showEvent(self, event):
        """Override showEvent to set up frame resize monitoring after widget is shown."""
        super().showEvent(event)
        # Install event filter on parent frame to catch resize events
        frame = self._find_parent_frame()
        if frame is not None:
            frame.installEventFilter(self)
            # Initial update
            self._update_column_widths_to_frame()
