# LandMarker

Desktop application for registering 3D medical images using a variety of automated, landmarks-based, or intensity-based algorithms. For landmarks-based registration, LandMarker allows the user to interactively place landmarks on fixed and moving volumes.

---

## Requirements
This Python application was developed on **Windows**. However, because Python is cross-platform, it can also run on **macOS** and **Linux** with the appropriate environment setup.

---

## Installation

### 1. Prerequisites
- **Python version:** 3.10.x (LandMarker was developed with Python 3.10.8)
- **Recommended:** Install [Miniconda](https://repo.anaconda.com/miniconda/) v. miniconda3-py310_22.11.1-1-[your OS] for Python environment management
  - On Windows: `C:\Users\<name>\miniconda3`
  - On macOS: `/Users/<name>/miniconda`
- **Windows only: for optional GPU support,** CUDA 12.8 compatible driver

---

### 2. Clone the repository or download the .zip file, then navigate to the LandMarker directory
```bash
git clone https://github.com/fuslab-uofu/LandMarker.git
   -- or --
download and extract the LandMarker .zip file  

cd LandMarker
```

---

### 3. Create and activate a virtual environment
```bash
python -m venv venv  
```
#### Windows
```bash 
venv\Scripts\activate
````  
#### macOS
```bash
source venv/bin/activate
```

---

### 4. Install PyTorch
#### Windows (includes CUDA for GPU)
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

#### macOS
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

---

### 5. Install PyQt5
```bash
pip install PyQt5==5.15.11 PyQt5-Qt5 PyQt5_sip==12.17.1
````
---

### 6. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

### 7. Run LandMarker
```bash
cd src
python LandMarker.py

```  
For developers:  
```bash
usage: LandMarker.py [-h] [--moving_nifti MOVING_NIFTI] [--fixed_nifti FIXED_NIFTI] [--landmarks_dir LANDMARKS_DIR] [--theme {dark,light}] [--debug_mode]

options:
  -h, --help                      show this help message and exit
  --moving_nifti MOVING_NIFTI     Full path to the moving volume NIfTI file.
  --fixed_nifti FIXED_NIFTI       Full path to the fixed volume NIfTI file.
  --landmarks_dir LANDMARKS_DIR   Directory to read or write landmark files.
```

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Credits
- **Michelle Kline**, Department of Radiology and Imaging Sciences, University of Utah
- **Amanpreet Singh**, The Scientific Computing and Imaging Institute, University of Utah

