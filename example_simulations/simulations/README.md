# Example simulations (demo dataset)

This folder contains a **small example dataset** generated from simulated microscopy movies.
It is provided to help users:

- understand the expected **data format**
- test that the training / evaluation pipeline runs
- inspect example input/target tensors without generating simulations themselves

> ✅ The full simulation generation pipeline (MATLAB) is maintained in a separate repository:  
> **<ADD_SIMULATION_REPO_LINK_HERE>**

---

## Contents

we include a small subset of the **raw TIFF simulation movies** to show the original structure.
Typical structure in tiff_files:
tiff_files/
<SettingID>_TrainingSet/<PatternID>_Training/
Noisy/
<SNRID>_noisyvid_*.tif
_SOFI_Results_fwhm*/*/
<SNRID>_noisyvid_*.tif

Where:
- `<SettingID>_TrainingSet/` groups simulation settings (e.g. PSF, density, range of SNR and background)
- `<PatternID>_Training/` groups all movies belonging to the same underlying pattern
- `Noisy/` contains noisy simulated blinking movies of the same pattern (TIFF stacks)
    - each has different SNR and SBR setting
    - <SNRID>, 01 is the movie with highest quality  (high SNR , low BG); 04 has the worst quality, 
        and other movies have SNR and SBR, in the range in between 01 and 04

### Converting TIFF → PT
The `.pt` tensors can be generated from this structure using:

```bash
python scripts/prepare_pt_from_tiff.py \
  --tiff-root examples/simulations/tiff_files \
  --out-root  examples/simulations/pt_files \
  --subname Noisy \
  --pt-foldername input \
  --pt-filename input \
  --frames 20 \
  --workers 8

```

Typical structure in pt_files:

pt_files/<SettingID>_TrainingSet/
input/
<SettingID><PatternID>input<SNRID>.pt
target/
<SettingID><PatternID>target<SNRID>.pt


Example filenames:

- `01001_input_01.pt`
- `01001_target_01.pt`

Where:
- `PID` = pattern identifier (used to group samples: <SettingID><PatternID>) 
- first two digits of `PID` = setting ID (e.g. `01`)
- last three digits of `PID` = Pattern ID (e.g. `001`)
- `SNR` = noise / SNR identifier (e.g. `01`, `02`, ...)

---

## Tensor format

Each `.pt` file stores a `torch.Tensor`:

- **shape:** `(T, H, W)`
- **dtype:** `torch.float32`
- **range:** typically normalized to `[0, 1]` 
(normalized by using the bit-depth: for 16-bit movie, the tiff files are normalized by 65535)

Where:
- `T` = number of frames (time dimension)
- `H, W` = spatial dimensions

### Loading one example tensor

```python
import torch

x = torch.load("examples/simulations/input/01001_input_01.pt")
y = torch.load("examples/simulations/target/01001_target_01.pt")

print("Input shape:", x.shape, x.dtype, x.min().item(), x.max().item())
print("Target shape:", y.shape, y.dtype, y.min().item(), y.max().item())
