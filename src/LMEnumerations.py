"""
LMEnumerations.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from enum import Enum


class Algorithm(Enum):
    """ This is an enumeration of the available registration algorithms.  """
    LMR = ("LM", "Landmarks-Based Rigid", "Rigid")
    LMA = ("LM", "Landmarks-Based Affine", "Affine")
    LMT = ("LM", "Landmarks-Based Thin Plate Splines", "TPS")
    LMK = ("LM", "Landmarks-Based Kernel Splines Flow", "SplinesFlow")
    LMF = ("LM", "Landmarks-Based FFT Projection Splines Flow", "FFTSplinesFlow")
    RTS = ("LM", "Landmarks-Based FFT Projection Rigid then Splines", "FFTRigidThenSplinesFlow")
    INR = ("IN", "Intensity-Based Rigid", "Rigid")
    INA = ("IN", "Intensity-Based Affine", "Affine")
    INE = ("IN", "Intensity-Based Elastic", "Elastic")
    INS = ("IN", "Intensity-Based Simple Diffeomorphic", "Diffeomorphic")
    INL = ("IN", "Intensity-Based LDMM", "LDMM")
    INF = ("IN", "Intensity-Based FFT Projection", "FFT")
    IND = ("IN", "Intensity-Based SDIFF", "sdiff")

    def __init__(self, algorithm_class, algorithm_description, algorithm_name):
        self.algorithm_class = algorithm_class
        self.algorithm_description = algorithm_description  # for user information
        self.algorithm_name = algorithm_name  # the name for the registration API

class ViewDir(Enum):
    """ This is an enumeration of the three orthogonal view orientations. Assumes RAS display convention. Axes chars
     are clockwise from the top. """
    AX = ("axial", ['A', 'L', 'P', 'R'])
    SAG = ("sagittal", ['S', 'P', 'I', 'A'])
    COR = ("coronal", ['S', 'L', 'I', 'R'])

    def __init__(self, dir_, chars_):
        self.dir = dir_
        self.chars = chars_

