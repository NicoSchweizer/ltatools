"""Loading and column normalization for HighFinesse wavemeter .lta files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from scipy.constants import c


def _normalize_column(name: str) -> str:
    """Normalize a raw .lta column header to a clean snake_case name."""
    stripped = re.sub(r"\s+", " ", name.strip())
    if re.search(r"\btime\b", stripped, re.IGNORECASE) and "[ms]" in stripped:
        return "time_ms"
    if re.search(r"wavelength", stripped, re.IGNORECASE):
        return "wavelength_nm"
    if re.search(r"power", stripped, re.IGNORECASE):
        return "power_uW"
    generic = re.sub(r"\[([^\]]+)\]", r"_\1", stripped)
    generic = re.sub(r"[^0-9a-zA-Z]+", "_", generic).strip("_").lower()
    return generic


def load_lta_file(file_path: str | Path, cleanup: bool = False) -> pd.DataFrame:
    """Load a HighFinesse wavemeter .lta log file into a DataFrame.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the .lta file (tab-separated, cp1252 encoding, comma decimal).
    cleanup : bool, default False
        If True, drop rows where ``wavelength_nm <= 0`` (invalid readings).

    Returns
    -------
    pandas.DataFrame
        Normalized columns (``time_ms``, ``wavelength_nm``, ``power_uW``, plus
        any unknown extra columns, stripped and snake_cased) with two derived
        columns appended: ``time_s`` (``time_ms * 1e-3``) and
        ``frequency_THz`` (``c / (wavelength_nm * 1e-9) * 1e-12``).

    Raises
    ------
    ValueError
        If no line containing "Time" is found (no table header).
    """
    path = Path(file_path)
    with open(path, encoding="cp1252", errors="ignore") as f:
        lines = f.readlines()

    header_row = next((i for i, line in enumerate(lines) if "Time" in line), None)
    if header_row is None:
        raise ValueError("Could not find the table header containing 'Time' in the file.")

    df = pd.read_csv(
        path,
        skiprows=header_row,
        sep="\t",
        decimal=",",
        skipinitialspace=True,
        encoding="cp1252",
    )
    df.columns = [_normalize_column(col) for col in df.columns]

    if cleanup:
        df = df[df["wavelength_nm"] > 0].reset_index(drop=True)

    df["time_s"] = df["time_ms"] * 1e-3
    df["frequency_THz"] = c / (df["wavelength_nm"] * 1e-9) * 1e-12

    return df
