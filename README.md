# U-Net architecture
This repository has several different files. `main.py` is the core file which runs the training algorithm. 
This branch is copied on servers for downloading

## Installing the repository
The repository must first be cloned before usage.

### Installing Python 3.12
 - Python 3.12: https://www.python.org/downloads/release/python-3120/ --> Windows 64 bit version

### Cloning repository and making virtual environment
 1) Clone the repository
 2) Create virtual environment
 3) `python -m venv .venv`
 4) `.venv/Scripts/activate`

### Installing PyTorch
 - Check the correct version required for the GPU, the Cuda version of the GPU is checked in CMD with `nvidia-smi`.
 - `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

### Installing remaining packages
 - Install requirements.txt: `pip install -r requirements.txt`

## Running the algorithm
 - Specify the different paths for the loading and configure hyper-parameters before running.