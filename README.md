# SOFI-MISRGRU Release Repository
Release repository accompanying our publication.
Tested on: Linux + NVIDIA GPU + CUDA 12.1 
CPU may work but is not validated

## Install - (Adjust versions/CUDA to match what you used.)
```bash
python -m venv .venv_resurf
source .venv_resurf/bin/activate
pip install -U pip

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```
# Data Preparation
Training expects .pt tensors with naming convention (check simulation_examples) 
(simulationID)(PatternID)_input_(settingID).pt, 
(simulationID)(PatternID)_target_(settingID).pt

If you start from TIFF simulations the scripts/prepare_pt_from_tiff.py with our public dataset or generate your simulations using our public MATLAB tool.


## Citation

If you use this code in your research, please cite our paper: 
**[Enabling Real-Time Fluctuation-Based Super Resolution Imaging](https://doi.org/10.1101/2025.06.05.658028)**

## License & Attribution

This repository contains original code and uses the modified MISR-GRU design derived from:
- MISR-GRU (Element AI Inc. and MILA, 2019): https://github.com/rarefin/MISR-GRU

This repository is distributed under the cumulative terms of:
- Apache License 2.0
- Do No Harm License (modified)

See `LICENSE`, `LICENSE-APACHE-2.0`, `LICENSE-DO-NO-HARM`, and `NOTICE`.
