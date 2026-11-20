"""
LMPlatformUtils.py
Platform detection utilities for LandMarker


LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""
import sys
import platform


def is_macos() -> bool:
    """Check if running on macOS"""
    return sys.platform == "darwin"


def is_windows() -> bool:
    """Check if running on Windows"""
    return sys.platform == "win32"


def is_linux() -> bool:
    """Check if running on Linux"""
    return sys.platform.startswith("linux")


def get_platform_name() -> str:
    """Get the current platform name"""
    if is_macos():
        return "macos"
    elif is_windows():
        return "windows"
    elif is_linux():
        return "linux"
    else:
        return "unknown"


def get_macos_regular_slider_stylesheet() -> str:
    """
    Get macOS-specific stylesheet for regular QSlider widgets.
    This addresses visual issues with QSlider on macOS.
    """
    return """
    QSlider {
        background: transparent;
        border: none;
    }
    
    QSlider::groove:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #4a4a4a, stop:1 #2a2a2a);
        height: 6px;
        border-radius: 3px;
        border: 1px solid #1a1a1a;
    }
    
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #6a6a6a, stop:1 #4a4a4a);
        border: 1px solid #2a2a2a;
        width: 16px;
        height: 16px;
        border-radius: 8px;
        margin: -5px 0;
    }
    
    QSlider::handle:horizontal:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #7a7a7a, stop:1 #5a5a5a);
        border: 1px solid #3a3a3a;
    }
    
    QSlider::handle:horizontal:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #8a8a8a, stop:1 #6a6a6a);
        border: 1px solid #4a4a4a;
    }
    
    QSlider::sub-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #3a5a7a, stop:1 #2a4a6a);
        border-radius: 3px;
    }
    
    QSlider::add-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #4a4a4a, stop:1 #2a2a2a);
        border-radius: 3px;
    }
    """


def get_windows_regular_slider_stylesheet() -> str:
    """
    Get Windows-specific stylesheet for regular QSlider widgets.
    This provides a consistent look with the rest of the application.
    """
    return """
    QSlider {
        background: transparent;
        border: none;
    }
    
    QSlider::groove:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #4a4a4a, stop:1 #2a2a2a);
        height: 6px;
        border-radius: 3px;
        border: 1px solid #1a1a1a;
    }
    
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #6a6a6a, stop:1 #4a4a4a);
        border: 1px solid #2a2a2a;
        width: 16px;
        height: 16px;
        border-radius: 8px;
        margin: -5px 0;
    }
    
    QSlider::handle:horizontal:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #7a7a7a, stop:1 #5a5a5a);
        border: 1px solid #3a3a3a;
    }
    
    QSlider::handle:horizontal:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #8a8a8a, stop:1 #6a6a6a);
        border: 1px solid #4a4a4a;
    }
    
    QSlider::sub-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #3a5a7a, stop:1 #2a4a6a);
        border-radius: 3px;
    }
    
    QSlider::add-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
            stop:0 #4a4a4a, stop:1 #2a2a2a);
        border-radius: 3px;
    }
    """


def get_platform_regular_slider_stylesheet() -> str:
    """
    Get the appropriate stylesheet for regular QSlider widgets on the current platform.
    """
    if is_macos():
        return get_macos_regular_slider_stylesheet()
    elif is_windows():
        return get_windows_regular_slider_stylesheet()
    else:
        # Default stylesheet for Linux and other platforms
        return get_windows_regular_slider_stylesheet()
