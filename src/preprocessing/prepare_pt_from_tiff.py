#!/usr/bin/env python3
"""
prepare_pt_from_tiff.py

Convert simulated TIFF stacks into normalized PyTorch tensors (.pt) in parallel.

Typical use-case in this repo:
- You already generated simulations as TIFF movies (e.g., noisy stacks).
- You want to convert them into `.pt` tensors so training/eval loads fast.

This script:
1) Collects TIFF files from a structured folder hierarchy
2) Crops frames (optional)
3) Applies optional Li-threshold mask (optional)
4) Normalizes by TIFF bit depth
5) Saves tensors to an output directory as .pt files

IMPORTANT: Multiprocessing guard
--------------------------------
The `if __name__ == "__main__": main()` guard is required for safe multiprocessing
(especially on Windows/macOS). It prevents infinite process spawning when child
processes re-import this file.

Dependencies:
- numpy, torch, tifffile, scikit-image, tqdm
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
from skimage.filters import threshold_li
from tqdm import tqdm


# ---------------------------------------------------------------------
# Job collection
# ---------------------------------------------------------------------
def collect_jobs(
    tiff_root: Path,
    out_root: Path,
    subname: str,
    setting_filter: Optional[Sequence[str]] = None,
    snr_filter: Optional[Sequence[str]] = None,
    pt_foldername: str = "input",
    pt_filename: str = "input",
    input_glob: str = "*.tif*",
    li_thr: bool = False,
    li_subname: Optional[str] = None,
    li_snr_prefix: str = "01_",
) -> List[Tuple[Path, Path, Optional[Path]]]:
    """
    Collect jobs as tuples:
        (movie_file_path, pt_output_path, li_mask_path_or_None)

    Expected-ish directory pattern (flexible):
        tiff_root/
          <setting>_*/
            <patternDir>/
              <subname>/
                <files matching input_glob>

    If li_thr=True, mask is searched under:
        <patternDir>/<li_subname>/<files matching input_glob>
    and selects the first file whose name starts with li_snr_prefix.

    In sofi simulations, the target with highest SNR is binarized(Li) and used as a mask
    """
    jobs: List[Tuple[Path, Path, Optional[Path]]] = []

    if not tiff_root.exists():
        raise FileNotFoundError(f"TIFF root not found: {tiff_root}")

    setting_dirs = [
        d for d in tiff_root.iterdir()
        if d.is_dir() and (setting_filter is None or any(d.name.startswith(s) for s in setting_filter))
    ]

    for setting_dir in setting_dirs:
        m = re.match(r"^(\d+)", setting_dir.name)
        setting_id = m.group(1) if m else "99"

        # Output mirrors top-level setting folder name
        setting_out = out_root / setting_dir.name
        (setting_out / pt_foldername).mkdir(parents=True, exist_ok=True)

        pattern_dirs = [p for p in setting_dir.iterdir() if p.is_dir()]
        for pattern_dir in pattern_dirs:
            pm = re.match(r"^(\d+)", pattern_dir.name)
            pattern_id = pm.group(1) if pm else "999"

            noisy_dir = pattern_dir / subname
            movie_list = sorted(glob.glob(f'{noisy_dir}'))
            # if not noisy_dir.exists():
            #     continue

            movie_files = sorted(Path(p).resolve() for p in glob.glob(str(noisy_dir / input_glob)))

            li_mask_path: Optional[Path] = None
            if li_thr:
                if not li_subname:
                    raise ValueError("li_subname must be provided when li_thr is enabled.")
                li_dir = pattern_dir / li_subname
                li_candidates = sorted(Path(p).resolve() for p in glob.glob(str(li_dir / input_glob)))
                li_mask_path = next((p for p in li_candidates if p.name.startswith(li_snr_prefix)), None)

            for movie_file in movie_files:
                # movie_file.stem might start with SNR token like "01_..." or "01..."
                sm = re.match(r"^(\d+)(.*)", movie_file.stem)
                snr_id = sm.group(1) if sm else "99"

                if snr_filter is not None and snr_id not in snr_filter:
                    continue

                # Construct PID-like prefix: <setting_id><pattern_id>_
                prefix = f"{setting_id}{pattern_id}_"
                pt_name = f"{prefix}{pt_filename}_{snr_id}.pt"
                pt_output = setting_out / pt_foldername / pt_name

                jobs.append((movie_file, pt_output, li_mask_path))

    return jobs


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------
def process_tiff_to_pt(args):
    """
    Process one TIFF -> normalized tensor -> save as .pt

    args tuple:
      (movie_file, pt_output_path, li_mask_path, framenum, start, roix, roiy, li_thr)
    """
    (movie_file, pt_output_path, li_mask_path, framenum, start, roix, roiy, li_thr) = args

    movie_file = Path(movie_file)
    pt_output_path = Path(pt_output_path)

    try:
        raw = tifffile.imread(movie_file).astype("float32")
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]  # add time dim
        frames, rows, cols = raw.shape

        # Safe slicing (crop ROI)
        sx = max(0, roix[0])
        ex = min(rows, rows + roix[1])  # roix[1] may be negative
        sy = max(0, roiy[0])
        ey = min(cols, cols + roiy[1])  # roiy[1] may be negative

        t0 = max(0, start)
        t1 = min(frames, start + framenum) if framenum is not None else frames

        raw = raw[t0:t1, sx:ex, sy:ey]

    except Exception as e:
        return f"[ERROR] Read/crop failed for {movie_file}: {e}"

    if li_thr:
        if li_mask_path is None:
            return f"[ERROR] Li mask requested but not found for {movie_file}"
        try:
            mask_raw = tifffile.imread(li_mask_path).astype("float32")
            thr = threshold_li(mask_raw.flatten())
            bin_mask = (mask_raw > thr).astype(np.float32)

            if bin_mask.shape != raw.shape[1:]:
                return f"[ERROR] Mask shape {bin_mask.shape} != image shape {raw.shape[1:]} for {movie_file}"

            raw = raw * bin_mask  # broadcast over time
        except Exception as e:
            return f"[ERROR] Li threshold failed for mask {li_mask_path}: {e}"

    # Normalize by bit depth
    try:
        with tifffile.TiffFile(movie_file) as tif:
            bit_depth = tif.pages[0].bitspersample
        max_value = float(2 ** bit_depth - 1)
        if max_value > 0:
            raw = raw / max_value
    except Exception as e:
        return f"[ERROR] Normalization failed for {movie_file}: {e}"

    tensor = torch.tensor(raw, dtype=torch.float32)

    try:
        pt_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, pt_output_path)
    except Exception as e:
        return f"[ERROR] Saving failed for {pt_output_path}: {e}"

    return f"[OK] Saved: {pt_output_path}"


# ---------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------
def execute_parallel(
    jobs: List[Tuple[Path, Path, Optional[Path]]],
    max_workers: int,
    framenum: Optional[int],
    start: int = 0,
    roix: Sequence[int] = (0, 0),
    roiy: Sequence[int] = (0, 0),
    li_thr: bool = False,
):
    # Compose args for each job
    job_args = [
        (movie, out, mask, framenum, start, list(roix), list(roiy), li_thr)
        for (movie, out, mask) in jobs
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(
            executor.map(process_tiff_to_pt, job_args),
            total=len(job_args),
            desc="TIFF -> PT",
        ):
            print(result)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert TIFF simulation movies into .pt tensors (parallel)."
    )
    p.add_argument("--tiff-root", type=Path, required=True, help="Root folder containing TIFF simulations.")
    p.add_argument("--out-root", type=Path, required=True, help="Root folder for output .pt tensors.")
    p.add_argument("--subname", type=str, required=True, help="Subfolder name under each pattern containing TIFFs (e.g. 'noisy').")

    p.add_argument("--pt-foldername", type=str, default="input", help="Subfolder under out-root/setting for .pt files (e.g. 'input').")
    p.add_argument("--pt-filename", type=str, default="input", help="Filename token used in saved .pt names (e.g. 'input' => *_input_<snr>.pt).")
    p.add_argument("--input-glob", type=str, default="*.tif*", help="Glob pattern inside subname folder (default: *.tif*).")

    p.add_argument("--setting-filter", nargs="*", default=None, help="Optional list of setting prefixes to include (e.g. 01 02).")
    p.add_argument("--snr-filter", nargs="*", default=None, help="Optional list of SNR prefixes to include (e.g. 01 02).")

    p.add_argument("--frames", type=int, default=None, help="Number of frames to keep (default: all).")
    p.add_argument("--start", type=int, default=0, help="Start frame index (default: 0).")

    p.add_argument("--roix", nargs=2, type=int, default=[0, 0], help="Crop in x: start_pad end_pad (end_pad can be negative).")
    p.add_argument("--roiy", nargs=2, type=int, default=[0, 0], help="Crop in y: start_pad end_pad (end_pad can be negative).")

    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    p.add_argument("--li-thr", action="store_true", help="Enable Li-threshold masking (requires li-subname).")
    p.add_argument("--li-subname", type=str, default=None, help="Subfolder name containing mask TIFFs (e.g. 'mask').")
    p.add_argument("--li-snr-prefix", type=str, default="01_", help="Prefix used to choose the mask file (default: 01_).")

    return p.parse_args()


def main():
    args = parse_args()

    jobs = collect_jobs(
        tiff_root=args.tiff_root,
        out_root=args.out_root,
        subname=args.subname,
        setting_filter=args.setting_filter,
        snr_filter=args.snr_filter,
        pt_foldername=args.pt_foldername,
        pt_filename=args.pt_filename,
        input_glob=args.input_glob,
        li_thr=args.li_thr,
        li_subname=args.li_subname,
        li_snr_prefix=args.li_snr_prefix,
    )

    print(f"Collected {len(jobs)} jobs.")
    if len(jobs) == 0:
        return

    execute_parallel(
        jobs=jobs,
        max_workers=args.workers,
        framenum=args.frames,
        start=args.start,
        roix=args.roix,
        roiy=args.roiy,
        li_thr=args.li_thr,
    )


if __name__ == "__main__":
    main()
