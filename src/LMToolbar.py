"""
LMToolbar.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QToolButton


class LMToolbar(NavigationToolbar):
    """ Custom class derived from Matplotlib's NavigationToolbar. Customized to limit the number of tools
    available on the toolbar, and to capture and customize signals sent from the toolbuttons. An LMToolbar
    object is included in each LMViewport.
    """

    def __init__(self, figure_canvas, parent):
        self.parent = parent

        self.toolitems = (('Home', 'Reset Extent', 'home', 'home'),
                          ('Back', 'Back to Extent', 'back', 'back'),
                          ('Forward', 'Forward to Extent', 'forward', 'forward'),
                          (None, None, None, None),
                          ('Pan', 'Interactive Pan', 'move', 'pan'),
                          ('Zoom', 'Interactive Zoom', 'zoom_to_rect', 'zoom'),
                          (None, None, None, None),
                          )

        NavigationToolbar.__init__(self, figure_canvas, parent, False)

        # TODO: maybe add a button to link zoom? Or other custom button?
        # path_to_svg = os.path.join("..", "ui", "link-solid.svg")
        # self.link_zoom_action = self.addAction(QIcon(path_to_svg), "Link Zoom", self.parent.link_zoom)
        # self.link_zoom_action.setToolTip("Link Zoom")
        # self.link_zoom_action.setCheckable(True)

        # grab the button objects so we can set our own background color
        buttons = self.findChildren(QToolButton)
        self.home_button = buttons[1]
        self.pan_button = buttons[4]
        self.zoom_button = buttons[5]

        # grab the actions so we can capture the signal
        actions = self.actions()
        self.home_action = actions[0]
        self.pan_action = actions[4]
        self.zoom_action = actions[5]
        self.home_action.triggered.connect(self.on_home_clicked)
        self.pan_action.triggered.connect(self.on_pan_clicked)
        self.zoom_action.triggered.connect(self.on_zoom_clicked)

        self.zooming = False
        self.panning = False

    def on_home_clicked(self):
        """ Slot for signal sent when user presses the "Home" button on the LMToolbar. This slot gets executed in
        addition to the parent class slot. LMViewport needs to keep track of plot extent ("zoom_lims") in order to
        maintain extent as user scrolls through slices.
        """
        self.parent.bg_volume.display_settings[self.parent.view]["zoom_lims"] = None

    def on_pan_clicked(self):
        """ Slot for signal sent when user clicks the "Pan" button on the LMToolbar. This slot gets executed in
        addition to the parent class slot. The purpose here is simply to set the background color to orange, to
        make it clearer that the pan button is currently toggled on or off.
        """
        if self.panning:
            self.pan_button.setStyleSheet("background-color: #323232;")  # grey
            self.panning = False
        else:
            self.pan_button.setStyleSheet("background-color: #ffa02f;")  # orange
            self.panning = True
            self.zoom_button.setStyleSheet("background-color: #323232;")  # grey
            self.zooming = False

    def on_zoom_clicked(self):
        """ Slot for signal sent when user clicks the "Zoom" button on the LMToolbar. This slot gets executed in
        addition to the parent class slot. The purpose here is simply to set the background color to orange, to
        make it clearer that the zoom button is currently toggled on or off.
        """
        if self.zooming:
            self.zoom_button.setStyleSheet("background-color: #323232;")  # grey
            self.zooming = False
        else:
            self.zoom_button.setStyleSheet("background-color: #ffa02f;")  # orange
            self.zooming = True
            self.pan_button.setStyleSheet("background-color: #323232;")  # grey
            self.panning = False

    def uncheck_all(self):
        """ Unchecks the checkable toolbuttons in the navigation toolbar. (Zoom and Pan)
        We are attempting to programatically "uncheck" the zoom and pan buttons, to get the plot out of zoom or pan
        mode. This would be necessary if the user was currently in zoom mode, then chose to hide the LMToolbar.
        """
        # FIXME: this doesn't work as expected. The "rubber band" or "rectangle" still is active in the plot, even
        #        after supposedly getting out of zoom or pan mode.
        if self.pan_button.isChecked():
            self.pan_action.trigger()
            # this doesn't get the plot out of pan mode as expected. see:
            # https://github.com/matplotlib/matplotlib/issues/25627
            # self.pan()

        if self.zoom_button.isChecked():
            self.zoom_action.trigger()
            # this doesn't get the plot out of zoom mode as expected. see:
            # https://github.com/matplotlib/matplotlib/issues/25627
            # self.zoom()

    def is_checked(self, toolbutton):
        """
        :param toolbutton: string with name of toolbutton
        :return: bool True if button is checked, False if not
        """
        if toolbutton.lower() == "pan":
            return self.pan_button.isChecked()
        elif toolbutton.lower() == "zoom":
            return self.pan_button.isChecked()

        else:
            # FIXME: log internal error?
            return False
