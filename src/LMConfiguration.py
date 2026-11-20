"""
LMConfiguration.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

import json
import os


class LMConfiguration:
    CONFIG_FILE = "../schema\\LMConfig.json"  # Configuration file name

    def __init__(self):
        """ Initialize and load existing configuration if available. """
        self.config_data = self._load_config()

    def _load_config(self):
        """ Load configuration from JSON file. """
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Corrupted schema file. Resetting.")
        return {}  # Return empty dict if file is missing or corrupted

    def _save_config(self):
        """ Save the current configuration to JSON file. """
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(self.config_data, f, indent=4)

    def get_last_directory(self):
        """ Retrieve the last working directory. Default to current directory. """
        return self.config_data.get("last_directory", os.getcwd())

    def set_last_directory(self, directory):
        """ Update and save the last working directory. """
        self.config_data["last_directory"] = directory
        self._save_config()