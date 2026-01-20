# SOFI-MISRGRU Release Repository
Release repository accompanying our publication.
Tested on: Linux + NVIDIA GPU + CUDA 12.1 
CPU may work but is not validated

## Install - (Adjust versions/CUDA to match what you used.)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```
# Data Preparation
“Training expects .pt tensors with naming convention …”

“If you start from TIFF simulations, see README in examples/simulations and use scripts/prepare_pt_from_tiff.py”

“MATLAB simulation generator lives in <other repo link>”