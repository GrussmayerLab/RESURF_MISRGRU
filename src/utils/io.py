import json
from pathlib import Path
import os
import glob
from pathlib import Path
from typing import List, Optional, Sequence


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)




def find_setting_subfolders(
    main_path: str | Path,
    subname: str,
    setting_filter: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Find subfolders inside a structured dataset root.

    This is useful when your dataset is stored like:

        pt_files/
          01_TrainingSet/
            input/
            target/
          02_TrainingSet/
            input/
            target/

    Parameters
    ----------
    main_path:
        Root folder (e.g. "examples/simulations/pt_files")

    subname:
        Name of the subfolder to collect (e.g. "input" or "target")

    setting_filter:
        Optional list like ["01", "02"].
        If None, scans all settings.

    Returns
    -------
    List[str]
        List of matched folder paths.
    """
    main_path = str(main_path)
    results: List[str] = []

    if setting_filter is None:
        pattern = os.path.join(main_path, "*", subname)
        results = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    else:
        for setting in setting_filter:
            pattern = os.path.join(main_path, f"{setting}_*", subname)
            matched = [p for p in glob.glob(pattern) if os.path.isdir(p)]
            results.extend(matched)

    return sorted(set(results))
