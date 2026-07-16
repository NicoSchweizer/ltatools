"""Loading and column normalization for HighFinesse wavemeter .lta files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from scipy.constants import c


_SIGNAL_RE = re.compile(r"^Signal\s+(\d+)\b\s*", re.IGNORECASE)


def _normalize_column(name: str, canonical_signal: int = 1) -> str:
    """Normalize a raw .lta column header to a clean snake_case name.

    Columns prefixed with a "Signal N" other than `canonical_signal` are
    kept distinct (``signal{N}_wavelength_nm`` etc.) instead of colliding
    with the canonical ``wavelength_nm``/``power_uW`` names, since a .lta
    file can log more than one wavemeter channel at once. `canonical_signal`
    is chosen by `load_lta_file` as the lowest-numbered signal that actually
    has wavelength data, since a channel can be present in the header but
    left entirely empty if it wasn't connected during that recording.
    """
    stripped = re.sub(r"\s+", " ", name.strip())
    if re.search(r"\btime\b", stripped, re.IGNORECASE) and "[ms]" in stripped:
        return "time_ms"

    signal_match = _SIGNAL_RE.match(stripped)
    signal_num = int(signal_match.group(1)) if signal_match else None
    rest = stripped[signal_match.end():] if signal_match else stripped

    if re.search(r"wavelength", rest, re.IGNORECASE):
        base = "wavelength_nm"
    elif re.search(r"power", rest, re.IGNORECASE):
        base = "power_uW"
    else:
        base = None

    if base is not None:
        if signal_num is None or signal_num == canonical_signal:
            return base
        return f"signal{signal_num}_{base}"

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
        header_row = next((i for i, line in enumerate(f) if "Time" in line), None)
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
    wavelength_cols_by_signal = {
        int(m.group(1)): col
        for col in df.columns
        if (m := _SIGNAL_RE.match(col.strip())) and re.search(r"wavelength", col, re.IGNORECASE)
    }
    if wavelength_cols_by_signal:
        populated = [n for n, col in sorted(wavelength_cols_by_signal.items()) if df[col].notna().any()]
        canonical_signal = populated[0] if populated else min(wavelength_cols_by_signal)
    else:
        canonical_signal = 1
    df.columns = [_normalize_column(col, canonical_signal) for col in df.columns]

    if "wavelength_nm" not in df.columns:
        raise ValueError(f"No wavelength column found in {path}")

    if cleanup:
        df = df[df["wavelength_nm"] > 0].reset_index(drop=True)

    df["time_s"] = df["time_ms"] * 1e-3
    df["frequency_THz"] = c / (df["wavelength_nm"] * 1e-9) * 1e-12

    return df
