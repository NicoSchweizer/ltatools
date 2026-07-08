import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba

from ltatools.io import load_lta_file
from ltatools.plotting import (
    lta_overview,
    overview_figure,
    plot_adev,
    plot_psd,
    plot_timeseries,
    psd_figure,
)
from ltatools.style import COLORS


def _load_df(write_lta, **kwargs):
    path = write_lta(**kwargs)
    return load_lta_file(path, cleanup=True)


def test_plot_timeseries_freq(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, ax2 = plot_timeseries(df, kind="freq")
    assert ax1 is not None and ax2 is not None


def test_plot_timeseries_wl(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, ax2 = plot_timeseries(df, kind="wl")
    assert ax1 is not None and ax2 is not None


def test_plot_adev_with_and_without_error():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])

    ax_with = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency")
    ax_without = plot_adev(tau, dev, None, unit="MHz", quantity="frequency")

    assert ax_with is not None
    assert ax_without is not None


def test_plot_adev_ci_bounds_overrides_dev_err():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])
    lower = dev - np.array([0.02, 0.015, 0.01])
    upper = dev + np.array([0.03, 0.025, 0.02])

    ax = plot_adev(tau, dev, dev_err, ci_bounds=(lower, upper), unit="MHz", quantity="frequency")

    assert ax is not None
    line = ax.get_lines()[0]
    assert to_rgba(line.get_color()) == to_rgba(COLORS["frequency"])


def test_plot_psd_invalid_scaling_raises():
    with pytest.raises(ValueError):
        plot_psd([1.0, 2.0], [0.1, 0.2], scaling="bogus")


def test_overview_figure_axes_count(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave")
    assert len(fig.axes) == 4
    assert len(axes) == 3


def test_overview_figure_save_creates_missing_dir(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=200)
    save_path = tmp_path / "sub" / "overview.png"
    overview_figure(df, taus="octave", save=save_path)
    assert save_path.exists()


def test_overview_figure_colors(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave")
    ax_ts, _, ax_power = axes

    freq_line = ax_ts.get_lines()[0]
    assert to_rgba(freq_line.get_color()) == to_rgba(COLORS["frequency"])

    power_line = ax_power.get_lines()[0]
    assert to_rgba(power_line.get_color()) == to_rgba(COLORS["power"])


def test_lta_overview_segments(write_lta):
    path = write_lta(n_points=200, mode_hop=True, hop_size_nm=0.05)
    results = lta_overview(path, cleanup=True, segments=True, n_segments=2, taus="octave")
    assert len(results) == 2
    for fig, axes in results:
        assert len(axes) == 3


def test_overview_figure_ci_smoke(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave", ci=0.683)
    assert len(fig.axes) == 4
    assert len(axes) == 3


def test_psd_figure_psd_and_asd(write_lta):
    df = _load_df(write_lta, n_points=2000)

    fig_psd, axes_psd = psd_figure(df, scaling="psd")
    assert len(axes_psd) == 2
    assert len(fig_psd.axes) == 2

    fig_asd, axes_asd = psd_figure(df, scaling="asd")
    assert len(axes_asd) == 2
    assert len(fig_asd.axes) == 2


def test_psd_figure_colors(write_lta):
    df = _load_df(write_lta, n_points=2000)
    fig, axes = psd_figure(df)
    ax_freq, ax_power = axes

    freq_line = ax_freq.get_lines()[0]
    assert to_rgba(freq_line.get_color()) == to_rgba(COLORS["frequency"])

    power_line = ax_power.get_lines()[0]
    assert to_rgba(power_line.get_color()) == to_rgba(COLORS["power"])


def test_psd_figure_ci_band(write_lta):
    df = _load_df(write_lta, n_points=2000)
    fig, axes = psd_figure(df, ci=0.95)

    for ax in axes:
        assert any(isinstance(c, PolyCollection) for c in ax.collections)


def test_psd_figure_save_creates_missing_dir(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=2000)
    save_path = tmp_path / "sub" / "psd.png"
    psd_figure(df, save=save_path)
    assert save_path.exists()
