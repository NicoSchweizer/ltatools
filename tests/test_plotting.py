import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba

from ltatools.io import load_lta_file
from ltatools.plotting import (
    lta_overview,
    overview_figure,
    plot,
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
    assert ax1.get_lines()[0].get_markersize() == 4


def test_plot_timeseries_wl(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, ax2 = plot_timeseries(df, kind="wl")
    assert ax1 is not None and ax2 is not None


def test_plot_timeseries_units_and_markersize(write_lta):
    df = _load_df(write_lta, n_points=50, wavelength_nm=1064.0, power_uW=500.0)
    ax1, ax2 = plot_timeseries(df, kind="freq", freq_unit="GHz", power_unit="mW", markersize=7)

    freq_line = ax1.get_lines()[0]
    power_line = ax2.get_lines()[0]

    assert freq_line.get_markersize() == 7
    assert power_line.get_markersize() == 7
    np.testing.assert_allclose(freq_line.get_ydata(), df["frequency_THz"] * 1e3)
    np.testing.assert_allclose(power_line.get_ydata(), df["power_uW"] * 1e-3)
    assert "GHz" in ax1.get_ylabel()
    assert "mW" in ax2.get_ylabel()


def test_plot_adev_with_and_without_error():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])

    ax_with = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency")
    ax_without = plot_adev(tau, dev, None, unit="MHz", quantity="frequency")

    assert ax_with is not None
    assert ax_without is not None


def test_plot_adev_errorbars_false_suppresses_bars():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])

    ax_on = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", errorbars=True)
    ax_off = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", errorbars=False)

    assert len(ax_on.containers[0].lines[2]) > 0
    assert len(ax_off.containers[0].lines[2]) == 0


def test_plot_adev_title():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])

    ax = plot_adev(tau, dev, unit="MHz", quantity="frequency", title="Frequency Allan Deviation")
    assert ax.get_title() == "Frequency Allan Deviation"


def test_plot_adev_save_creates_missing_dir(tmp_path):
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])
    save_path = tmp_path / "sub" / "adev.png"

    plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", save=save_path)
    assert save_path.exists()


def test_plot_adev_save_with_external_ax(tmp_path):
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    save_path = tmp_path / "sub" / "adev_external.png"

    fig, ax = plt.subplots()
    plot_adev(tau, dev, unit="MHz", quantity="frequency", ax=ax, save=save_path)
    assert save_path.exists()


def test_plot_timeseries_save_creates_missing_dir(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=50)
    save_path = tmp_path / "sub" / "timeseries.png"

    plot_timeseries(df, kind="freq", save=save_path)
    assert save_path.exists()


def test_plot_psd_invalid_scaling_raises():
    with pytest.raises(ValueError):
        plot_psd([1.0, 2.0], [0.1, 0.2], scaling="bogus")


def test_plot_psd_save_creates_missing_dir(tmp_path):
    f = np.array([1.0, 2.0, 3.0, 4.0])
    Pxx = np.array([0.4, 0.3, 0.2, 0.1])
    save_path = tmp_path / "sub" / "psd_single.png"

    plot_psd(f, Pxx, quantity="frequency", save=save_path)
    assert save_path.exists()


def test_overview_figure_axes_count(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave")
    assert len(fig.axes) == 4
    assert len(axes) == 3


def test_overview_figure_adev_titles(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave")
    _, ax_freq, ax_power = axes

    assert ax_freq.get_title() == "Frequency Allan Deviation"
    assert ax_power.get_title() == "Power Allan Deviation"


def test_overview_figure_errorbars_false(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave", errorbars=False)
    _, ax_freq, ax_power = axes

    assert len(ax_freq.containers[0].lines[2]) == 0
    assert len(ax_power.containers[0].lines[2]) == 0


def test_overview_figure_timeseries_power_unit_matches_adev_default(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave")
    ax_ts, ax_freq, ax_power = axes

    # the twin (power) axis of the timeseries panel is the one remaining axes
    ts_power_ax = next(ax for ax in fig.axes if ax not in (ax_ts, ax_freq, ax_power))

    def _unit(label):
        return "uW" in label or "µW" in label

    assert _unit(ts_power_ax.get_ylabel())
    assert _unit(ax_power.get_ylabel())


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


_WRAPPER_KIND_KWARGS = {
    "overview": {"taus": "octave"},
    "psd": {},
    "timeseries": {},
    "adev": {"taus": "octave"},
    "spectrum": {},
}


@pytest.mark.parametrize("kind", ["overview", "psd", "timeseries", "adev", "spectrum"])
def test_plot_wrapper_dispatch_dataframe(write_lta, kind):
    df = _load_df(write_lta, n_points=2000)
    n_before = len(plt.get_fignums())

    result = plot(df, kind=kind, **_WRAPPER_KIND_KWARGS[kind])

    assert result is None
    assert len(plt.get_fignums()) > n_before


@pytest.mark.parametrize("kind", ["overview", "psd", "timeseries", "adev", "spectrum"])
def test_plot_wrapper_dispatch_lta_path(write_lta, kind):
    path = write_lta(n_points=2000)
    n_before = len(plt.get_fignums())

    result = plot(path, kind=kind, cleanup=True, **_WRAPPER_KIND_KWARGS[kind])

    assert result is None
    assert len(plt.get_fignums()) > n_before


def test_plot_wrapper_returns_none(write_lta):
    df = _load_df(write_lta, n_points=200)
    assert plot(df, kind="overview", taus="octave") is None


def test_plot_wrapper_save_name_appends_png(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=200)
    plot(df, kind="overview", taus="octave", save=str(tmp_path / "myfig"))
    assert (tmp_path / "myfig.png").exists()


def test_plot_wrapper_save_nested_dir(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=200)
    plot(df, kind="overview", taus="octave", save=tmp_path / "sub" / "run1")
    assert (tmp_path / "sub" / "run1.png").exists()


def test_plot_wrapper_adev_quantity_power(write_lta):
    df = _load_df(write_lta, n_points=200)
    assert plot(df, kind="adev", quantity="power", taus="octave") is None


def test_plot_wrapper_unknown_kind_raises(write_lta):
    df = _load_df(write_lta, n_points=200)
    with pytest.raises(ValueError):
        plot(df, kind="bogus")
