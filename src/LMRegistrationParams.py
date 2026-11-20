"""
LMRegistrationParams.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from PyQt5.QtWidgets import QDialog, QMessageBox
from ui.Ui_LMRegistrationParams import Ui_LMRegistrationParams


class LMRegistrationParams(QDialog):
    """
    Dialog for setting intensity-based registration parameters.
    Supports Elastic, Diffeomorphic, and sdiff algorithms.
    """

    def __init__(self, parent=None, algorithm_name=None):
        super().__init__(parent)
        self.ui = Ui_LMRegistrationParams()
        self.ui.setupUi(self)
        
        self.algorithm_name = algorithm_name
        self.ui.algorithm_label.setText(f"Algorithm: {algorithm_name}")
        
        # Set up default values based on algorithm
        self._setup_defaults()
        
        # Show/hide parameters based on algorithm
        self._configure_ui_for_algorithm()
        
        # Connect OK button to validate
        self.ui.ok_button.clicked.disconnect()
        self.ui.ok_button.clicked.connect(self._validate_and_accept)
    
    def _setup_defaults(self):
        """Set default parameter values based on algorithm type."""
        if self.algorithm_name == "Elastic":
            # Default values for Elastic
            self.ui.sigma_level1.setText("500")
            self.ui.sigma_level2.setText("500")
            self.ui.sigma_level3.setText("500")
            self.ui.lambda_level1.setText("1")
            self.ui.lambda_level2.setText("1")
            self.ui.lambda_level3.setText("1")
            self.ui.eps_level1.setText("100")
            self.ui.eps_level2.setText("100")
            self.ui.eps_level3.setText("100")
            self.ui.niters_level1.setText("10")
            self.ui.niters_level2.setText("10")
            self.ui.niters_level3.setText("10")
        elif self.algorithm_name == "Diffeomorphic":
            self.ui.sigma_level1.setText("1")
            self.ui.sigma_level2.setText("1")
            self.ui.sigma_level3.setText("1")
            self.ui.lambda_level1.setText("1")
            self.ui.lambda_level2.setText("1")
            self.ui.lambda_level3.setText("1")
            self.ui.eps_level1.setText("10")
            self.ui.eps_level2.setText("10")
            self.ui.eps_level3.setText("10")
            self.ui.niters_level1.setText("100")
            self.ui.niters_level2.setText("1")
            self.ui.niters_level3.setText("1")
        elif self.algorithm_name == "sdiff":
            # Default values for sdiff
            self.ui.sigma_level1.setText("1")
            self.ui.sigma_level2.setText("1")
            self.ui.sigma_level3.setText("1")
            self.ui.eps_level1.setText("1")
            self.ui.eps_level2.setText("1")
            self.ui.eps_level3.setText("1")
            self.ui.niters_level1.setText("10")
            self.ui.niters_level2.setText("10")
            self.ui.niters_level3.setText("10")
        else:
            # Default values for unknown algorithms (use Elastic defaults)
            self.ui.sigma_level1.setText("500")
            self.ui.sigma_level2.setText("500")
            self.ui.sigma_level3.setText("500")
            self.ui.lambda_level1.setText("1")
            self.ui.lambda_level2.setText("1")
            self.ui.lambda_level3.setText("1")
            self.ui.eps_level1.setText("100")
            self.ui.eps_level2.setText("100")
            self.ui.eps_level3.setText("100")
            self.ui.niters_level1.setText("10")
            self.ui.niters_level2.setText("10")
            self.ui.niters_level3.setText("10")
    
    def _configure_ui_for_algorithm(self):
        """Show/hide and configure UI elements based on algorithm type."""
        if self.algorithm_name == "sdiff":
            # sdiff doesn't use lambda
            self.ui.lambda_label.setVisible(False)
            self.ui.lambda_level1.setVisible(False)
            self.ui.lambda_level2.setVisible(False)
            self.ui.lambda_level3.setVisible(False)
        else:
            # Elastic, Diffeomorphic, and unknown algorithms use all parameters
            self.ui.lambda_label.setVisible(True)
            self.ui.lambda_level1.setVisible(True)
            self.ui.lambda_level2.setVisible(True)
            self.ui.lambda_level3.setVisible(True)
    
    def _validate_and_accept(self):
        """Validate input values before accepting the dialog."""
        try:
            # Validate sigma values - must be positive floats
            sigma1 = float(self.ui.sigma_level1.text())
            sigma2 = float(self.ui.sigma_level2.text())
            sigma3 = float(self.ui.sigma_level3.text())
            if sigma1 <= 0 or sigma2 <= 0 or sigma3 <= 0:
                raise ValueError("Sigma values must be positive numbers (greater than 0)")
            
            # Validate eps (epsilon) values - must be positive floats
            eps1 = float(self.ui.eps_level1.text())
            eps2 = float(self.ui.eps_level2.text())
            eps3 = float(self.ui.eps_level3.text())
            if eps1 <= 0 or eps2 <= 0 or eps3 <= 0:
                raise ValueError("Epsilon values must be positive numbers (greater than 0)")
            
            # Validate niters (iterations) values - must be positive integers
            try:
                niters1 = int(self.ui.niters_level1.text())
                niters2 = int(self.ui.niters_level2.text())
                niters3 = int(self.ui.niters_level3.text())
            except ValueError:
                raise ValueError("Iterations must be integers (whole numbers)")
            
            if niters1 < 1 or niters2 < 1 or niters3 < 1:
                raise ValueError("Number of iterations must be at least 1")
            
            # Validate lambda values if needed (all algorithms except sdiff use lambda)
            # Lambda must be positive floats
            if self.algorithm_name != "sdiff":
                lambda1 = float(self.ui.lambda_level1.text())
                lambda2 = float(self.ui.lambda_level2.text())
                lambda3 = float(self.ui.lambda_level3.text())
                if lambda1 <= 0 or lambda2 <= 0 or lambda3 <= 0:
                    raise ValueError("Lambda values must be positive numbers (greater than 0)")
            
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", 
                              f"Please enter valid numeric values.\n{str(e)}")
    
    def get_parameters(self):
        """Return a dictionary of parameters based on the algorithm type."""
        params = {}
        
        # All algorithms use sigma, eps, and niters
        params["sigma"] = [
            float(self.ui.sigma_level1.text()),
            float(self.ui.sigma_level2.text()),
            float(self.ui.sigma_level3.text())
        ]
        params["eps"] = [
            float(self.ui.eps_level1.text()),
            float(self.ui.eps_level2.text()),
            float(self.ui.eps_level3.text())
        ]
        params["niters"] = [
            int(self.ui.niters_level1.text()),
            int(self.ui.niters_level2.text()),
            int(self.ui.niters_level3.text())
        ]
        
        # Elastic, Diffeomorphic, and other algorithms (except sdiff) use lambda
        if self.algorithm_name != "sdiff":
            params["lambda"] = [
                float(self.ui.lambda_level1.text()),
                float(self.ui.lambda_level2.text()),
                float(self.ui.lambda_level3.text())
            ]
        
        return params
