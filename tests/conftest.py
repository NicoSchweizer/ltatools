"""Shared pytest fixtures: synthetic .lta file writer and figure cleanup."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

_PREAMBLE = [
    "HighFinesse Wavelength Meter - Log File\n",
    "Software Version: 6.20\n",
    "Measurement started: 2026-01-01 00:00:00\n",
]

_HEADER = "Time  [ms]\tSignal 1  Wavelength, vac.  [nm]\tSignal 1 Power  [µW]\n"


def _format_number(value: float) -> str:
    return f"{value:.6f}".replace(".", ",")


def _format_row(time_ms: float, wavelength_nm: float, power_uW: float) -> str:
    return (
        f"{_format_number(time_ms)}\t"
        f"{_format_number(wavelength_nm)}\t"
        f"{_format_number(power_uW)}\n"
    )


@pytest.fixture
def write_lta(tmp_path):
    """Factory fixture writing a synthetic HighFinesse .lta file to tmp_path."""

    def _write(
        n_points=200,
        dt_ms=10.0,
        wavelength_nm=1064.0,
        power_uW=500.0,
        invalid_rows=0,
        mode_hop=False,
        hop_size_nm=0.05,
        include_header=True,
        filename="synthetic.lta",
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        times = np.arange(n_points) * dt_ms
        wavelengths = wavelength_nm + rng.normal(0, 1e-6, n_points)
        powers = power_uW + rng.normal(0, 1.0, n_points)

        if mode_hop:
            half = n_points // 2
            wavelengths[half:] += hop_size_nm

        if invalid_rows:
            idx = rng.choice(n_points, size=invalid_rows, replace=False)
            wavelengths[idx] = -1.0

        lines = list(_PREAMBLE)
        if include_header:
            lines.append(_HEADER)
        lines.extend(
            _format_row(t, wl, p) for t, wl, p in zip(times, wavelengths, powers)
        )

        path = tmp_path / filename
        with open(path, "w", encoding="cp1252", errors="ignore") as f:
            f.writelines(lines)
        return path

    return _write


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
