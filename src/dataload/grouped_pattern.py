"""
grouped_pattern.py

GroupedPatternDataset
---------------------

Purpose
^^^^^^^
This Dataset loads **pre-saved PyTorch tensors** (`.pt`) and groups samples by a shared
**Pattern ID (PID)** so you can build pattern-aware batches (e.g., batches containing
multiple different patterns instead of many samples from one pattern).

Contract (assumptions / naming)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We expect filenames to follow this convention:

    <PID>_input_<SNR>.pt
    <PID>_target_<SNR>.pt

Examples:
    01001_input_01.pt
    01001_target_01.pt

Where:
- PID: a string identifier. By convention, `PID[:2]` is treated as the **SettingID**
  (used for folder mapping when multiple settings exist). And `PID[2:]` is the **PatternID**
  of the pattern group in tiff files. However in .pt conversion PID is defined as : <SettingID><PatternID>
- SNR: an identifier token (often 2 digits). Used for filtering and optional pairing.

Folder layout is flexible; you pass folders via `input_paths` and `target_paths`.
The Dataset will scan those folders for `.pt` files.

Key features
^^^^^^^^^^^^
- Builds `self.samples`: list of dicts with {"input","target","pattern_id","snr"}.
- Builds `self.pattern_to_indices`: mapping {pattern_id -> [sample indices]} for samplers.
- Optional filters: `setting_filter`, `pattern_filter`, `snr_filter`.
- Optional frame slicing: `num_frames` keeps first T frames.
- Optional transform: applied to both input and target.

Pairing behavior
^^^^^^^^^^^^^^^^
If `pair_by_snr=False` (default):
    - All inputs for a PID will use the **first** available target for that PID.

If `pair_by_snr=True`:
    - Inputs and targets are paired by matching `snr` tokens. This is safer than `zip()`.
    - If a given input SNR has no matching target SNR, that input is skipped.

Determinism
^^^^^^^^^^^
All file lists are sorted to ensure deterministic ordering across machines.

"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _FileRecord:
    pid: str
    kind: str  # "input" or "target"
    snr: str
    path: Path


class GroupedPatternDataset(Dataset):
    def __init__(
        self,
        input_paths: Sequence[str],
        target_paths: Sequence[str],
        *,
        input_token: str = "_input_",
        target_token: str = "_target_",
        setting_filter: Optional[Sequence[str]] = None,
        pattern_filter: Optional[Sequence[str]] = None,
        snr_filter: Optional[Sequence[str]] = None,
        num_frames: Optional[int] = None,
        transform=None,
        pair_by_snr: bool = True,
        strict: bool = True,
    ):
        """
        Parameters
        ----------
        input_paths / target_paths:
            Folder paths containing .pt files.

        input_token / target_token:
            Tokens used in filenames to identify inputs and targets.

        setting_filter:
            Optional allowed setting IDs (strings). Setting ID is `pid[:2]`.

        pattern_filter:
            Optional allowed PIDs.

        snr_filter:
            Optional allowed SNR identifiers.
            Accepts entries like ["01","02"] OR ["01.pt","02.pt"].
            Matching is done on the extracted SNR token, not filename suffix.

        num_frames:
            If provided, keep only the first `num_frames` frames from input tensor.

        transform:
            Optional callable applied to BOTH input and target tensors.

        pair_by_snr:
            If True, pair input/target by matching SNR tokens.
            If False, use the first target for all inputs per PID.

        strict:
            If True, raise helpful errors when no files are found or no valid pairs exist.
            If False, silently create an empty dataset (not recommended for release repos).
        """
        self.transform = transform
        self.num_frames = num_frames
        self.pair_by_snr = pair_by_snr
        self.strict = strict

        self.setting_filter = set(setting_filter) if setting_filter else None
        self.pattern_filter = set(pattern_filter) if pattern_filter else None
        self.snr_filter = self._normalize_snr_filter(snr_filter)

        self.samples: List[Dict[str, Any]] = []
        self.pattern_to_indices: Dict[str, List[int]] = defaultdict(list)

        # 1) Scan folders
        input_records = self._scan_records(input_paths, kind="input", token=input_token)
        target_records = self._scan_records(target_paths, kind="target", token=target_token)

        if strict and not input_records:
            raise FileNotFoundError(
                f"No input .pt files found in input_paths={list(input_paths)} "
                f"with token='{input_token}'."
            )
        if strict and not target_records:
            raise FileNotFoundError(
                f"No target .pt files found in target_paths={list(target_paths)} "
                f"with token='{target_token}'."
            )

        # 2) Apply filters
        input_records = self._apply_filters(input_records)
        target_records = self._apply_filters(target_records)

        if strict and not input_records:
            raise FileNotFoundError(
                "After applying filters, no INPUT files remain. "
                f"setting_filter={setting_filter}, pattern_filter={pattern_filter}, snr_filter={snr_filter}"
            )
        if strict and not target_records:
            raise FileNotFoundError(
                "After applying filters, no TARGET files remain. "
                f"setting_filter={setting_filter}, pattern_filter={pattern_filter}, snr_filter={snr_filter}"
            )

        # 3) Build mapping pid -> inputs/targets
        pid_to_inputs: Dict[str, List[_FileRecord]] = defaultdict(list)
        pid_to_targets: Dict[str, List[_FileRecord]] = defaultdict(list)

        for r in input_records:
            pid_to_inputs[r.pid].append(r)
        for r in target_records:
            pid_to_targets[r.pid].append(r)

        # Sort deterministically
        for pid in pid_to_inputs:
            pid_to_inputs[pid].sort(key=lambda x: (x.snr, x.path.name))
        for pid in pid_to_targets:
            pid_to_targets[pid].sort(key=lambda x: (x.snr, x.path.name))

        # 4) Build samples with clear pairing behavior
        self._build_samples(pid_to_inputs, pid_to_targets)

        if strict and len(self.samples) == 0:
            raise RuntimeError(
                "No valid (input,target) samples could be constructed. "
                "Check that PIDs and SNR tokens match between inputs and targets."
            )

        # Deterministic pattern_ids list
        self.pattern_ids = sorted(self.pattern_to_indices.keys())

    # ----------------------------
    # Public helper: summary
    # ----------------------------
    def summary(self) -> Dict[str, Any]:
        """Return quick dataset statistics (useful for sanity checks)."""
        if len(self.samples) == 0:
            return {
                "num_samples": 0,
                "num_patterns": 0,
                "min_samples_per_pattern": 0,
                "max_samples_per_pattern": 0,
            }

        counts = [len(idxs) for idxs in self.pattern_to_indices.values()]
        return {
            "num_samples": len(self.samples),
            "num_patterns": len(self.pattern_to_indices),
            "min_samples_per_pattern": min(counts),
            "max_samples_per_pattern": max(counts),
        }

    # ----------------------------
    # Dataset protocol
    # ----------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        x = torch.load(sample["input"])
        y = torch.load(sample["target"])

        if self.num_frames is not None:
            x = x[: self.num_frames]

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return {
            "input": x,
            "target": y,
            "pattern_id": sample["pattern_id"],
            "snr": sample["snr"],
        }

    # ----------------------------
    # Internals
    # ----------------------------
    @staticmethod
    def _normalize_snr_filter(snr_filter: Optional[Sequence[str]]) -> Optional[set]:
        if not snr_filter:
            return None
        normalized = set()
        for s in snr_filter:
            s = str(s)
            s = s.replace(".pt", "")
            # Some users might pass "01_" from their TIFF naming; normalize that too
            s = s[:-1] if s.endswith("_") else s
            normalized.add(s)
        return normalized

    def _passes_filter(self, pid: str) -> bool:
        # Validate PID shape early for clearer errors
        if self.strict and (pid is None or len(pid) < 2):
            raise ValueError(f"Invalid PID '{pid}'. Expected PID with at least 2 characters (for SettingID).")

        setting_id = pid[:2]

        if self.setting_filter and setting_id not in self.setting_filter:
            return False
        if self.pattern_filter and pid not in self.pattern_filter:
            return False
        return True

    def _apply_filters(self, records: List[_FileRecord]) -> List[_FileRecord]:
        out: List[_FileRecord] = []
        for r in records:
            if not self._passes_filter(r.pid):
                continue
            if self.snr_filter and r.snr not in self.snr_filter:
                continue
            out.append(r)
        # deterministic
        out.sort(key=lambda x: (x.pid, x.kind, x.snr, x.path.name))
        return out

    @staticmethod
    def _iter_pt_files(paths: Sequence[str]) -> Iterable[Path]:
        for p in paths:
            pp = Path(p)
            if not pp.exists():
                continue
            # Only direct children *.pt (not recursive by design)
            for f in sorted(pp.glob("*.pt")):
                if f.is_file():
                    yield f

    @staticmethod
    def _parse_name(name: str, token: str) -> Optional[Tuple[str, str]]:
        """
        Parse PID and SNR from a filename containing a token:
            <PID><token><SNR>.pt

        Returns: (pid, snr) or None if token not found.
        """
        if token not in name:
            return None

        # Remove extension for parsing
        stem = name[:-3] if name.endswith(".pt") else name

        left, right = stem.split(token, 1)
        pid = left
        snr = right  # everything after token

        # Normalize snr (strip trailing underscores from some naming)
        snr = snr[:-1] if snr.endswith("_") else snr

        return pid, snr

    def _scan_records(self, paths: Sequence[str], *, kind: str, token: str) -> List[_FileRecord]:
        records: List[_FileRecord] = []
        for f in self._iter_pt_files(paths):
            parsed = self._parse_name(f.name, token=token)
            if parsed is None:
                continue
            pid, snr = parsed
            records.append(_FileRecord(pid=pid, kind=kind, snr=snr, path=f))
        # deterministic
        records.sort(key=lambda x: (x.pid, x.snr, x.path.name))
        return records

    def _build_samples(
        self,
        pid_to_inputs: Dict[str, List[_FileRecord]],
        pid_to_targets: Dict[str, List[_FileRecord]],
    ) -> None:
        idx = 0

        # Track skipped counts for debugging
        skipped_no_target = 0
        skipped_no_input = 0
        skipped_pair_mismatch = 0

        for pid in sorted(pid_to_inputs.keys()):
            inputs = pid_to_inputs.get(pid, [])
            targets = pid_to_targets.get(pid, [])

            if not inputs:
                skipped_no_input += 1
                continue
            if not targets:
                skipped_no_target += 1
                continue

            if not self.pair_by_snr:
                # Many inputs share first target
                target_path = targets[0].path
                # Use target snr for metadata; inputs retain their own snr
                for inp in inputs:
                    self.samples.append({
                        "input": str(inp.path),
                        "target": str(target_path),
                        "pattern_id": pid,
                        "snr": inp.snr,
                    })
                    self.pattern_to_indices[pid].append(idx)
                    idx += 1
            else:
                # Pair by matching snr
                target_by_snr: Dict[str, Path] = {t.snr: t.path for t in targets}
                for inp in inputs:
                    tpath = target_by_snr.get(inp.snr)
                    if tpath is None:
                        skipped_pair_mismatch += 1
                        continue
                    self.samples.append({
                        "input": str(inp.path),
                        "target": str(tpath),
                        "pattern_id": pid,
                        "snr": inp.snr,
                    })
                    self.pattern_to_indices[pid].append(idx)
                    idx += 1

        # Optional: store counters for debugging / reporting
        self._skipped = {
            "no_input": skipped_no_input,
            "no_target": skipped_no_target,
            "pair_mismatch": skipped_pair_mismatch,
        }
