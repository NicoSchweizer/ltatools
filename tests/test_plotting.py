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
    plot_histogram,
    plot_psd,
    plot_timeseries,
    psd_figure,
)
from ltatools.style import COLORS, axis_label, darken_color, finer_unit


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


def test_plot_timeseries_label_replaces_title_instead_of_stacking(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="freq", label="Scenario A")
    assert ax1.get_figure()._suptitle.get_text() == "Scenario A"
    assert ax1.get_title() == ""


def test_plot_timeseries_no_label_keeps_default_title(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="freq")
    assert ax1.get_title() == "Frequency and Power over time"
    assert ax1.get_figure()._suptitle is None


def test_plot_timeseries_relative_centers_on_mean(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="freq", freq_unit="MHz", relative=True)
    baseline = df["frequency_THz"].mean()
    expected = (df["frequency_THz"] - baseline) * 1e6
    np.testing.assert_allclose(ax1.get_lines()[0].get_ydata(), expected)


def test_plot_timeseries_relative_ylabel_is_delta(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="freq", freq_unit="MHz", relative=True)
    assert ax1.get_ylabel() == "Frequency deviation Δν (MHz)"


def test_plot_timeseries_relative_shows_baseline_thz(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="freq", freq_unit="MHz", relative=True)
    baseline = df["frequency_THz"].mean()
    integer_part, _, frac_part = f"{baseline:.7f}".partition(".")
    grouped_frac = "'".join(frac_part[i : i + 3] for i in range(0, len(frac_part), 3))
    texts = [t.get_text() for t in ax1.texts]
    assert any(f"{integer_part}.{grouped_frac}" in t and "THz" in t for t in texts)


def test_plot_timeseries_relative_default_false_unchanged(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="freq", freq_unit="MHz")
    np.testing.assert_allclose(ax1.get_lines()[0].get_ydata(), df["frequency_THz"] * 1e6)
    assert ax1.get_ylabel() == "Frequency (MHz)"
    assert all("THz" not in t.get_text() for t in ax1.texts)


def test_plot_timeseries_relative_wl_leaves_left_axis_absolute(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax1, _ = plot_timeseries(df, kind="wl", relative=True)
    np.testing.assert_allclose(ax1.get_lines()[0].get_ydata(), df["wavelength_nm"])
    assert ax1.get_ylabel() == "Wavelength (nm)"


def test_plot_timeseries_relative_centers_power_on_mean(write_lta):
    df = _load_df(write_lta, n_points=50)
    _, ax2 = plot_timeseries(df, kind="freq", power_unit="uW", relative=True)
    baseline = df["power_uW"].mean()
    expected = df["power_uW"] - baseline
    np.testing.assert_allclose(ax2.get_lines()[0].get_ydata(), expected)
    assert ax2.get_ylabel() == "Power deviation ΔP (uW)"


def test_plot_timeseries_relative_shows_baseline_power(write_lta):
    df = _load_df(write_lta, n_points=50)
    _, ax2 = plot_timeseries(df, kind="freq", power_unit="uW", relative=True)
    baseline = df["power_uW"].mean()
    texts = [t.get_text() for t in ax2.texts]
    assert any(f"{baseline:.6g}" in t and "uW" in t for t in texts)


def test_plot_timeseries_relative_baseline_power_matches_power_unit(write_lta):
    df = _load_df(write_lta, n_points=50)
    _, ax2 = plot_timeseries(df, kind="freq", power_unit="mW", relative=True)
    baseline_mW = df["power_uW"].mean() * 1e-3
    integer_part, _, frac_part = f"{baseline_mW:.6g}".partition(".")
    grouped_frac = "'".join(frac_part[i : i + 3] for i in range(0, len(frac_part), 3))
    expected = f"{integer_part}.{grouped_frac}" if frac_part else integer_part
    texts = [t.get_text() for t in ax2.texts]
    assert any(expected in t and "mW" in t for t in texts)
    assert all("uW" not in t for t in texts)


def test_plot_timeseries_relative_power_applies_even_for_wl(write_lta):
    df = _load_df(write_lta, n_points=50)
    _, ax2 = plot_timeseries(df, kind="wl", power_unit="uW", relative=True)
    baseline = df["power_uW"].mean()
    expected = df["power_uW"] - baseline
    np.testing.assert_allclose(ax2.get_lines()[0].get_ydata(), expected)
    assert ax2.get_ylabel() == "Power deviation ΔP (uW)"


def test_axis_label_delta():
    assert axis_label("frequency", "MHz", delta=True) == "Frequency deviation Δν (MHz)"
    assert axis_label("frequency", "MHz") == "Frequency (MHz)"


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


def test_plot_adev_errorbar_default_has_darker_color_and_no_endcaps():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency")
    container = ax.containers[0]
    data_line, caplines, barlinecols = container.lines

    assert len(caplines) == 0
    assert to_rgba(barlinecols[0].get_color()[0]) == to_rgba(darken_color(COLORS["frequency"]))
    assert to_rgba(barlinecols[0].get_color()[0]) != to_rgba(COLORS["frequency"])


def test_plot_adev_errorbar_capsize_adds_endcaps():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", capsize=3)
    _, caplines, _ = ax.containers[0].lines
    assert len(caplines) > 0
    assert to_rgba(caplines[0].get_markeredgecolor()) == to_rgba(darken_color(COLORS["frequency"]))


def test_plot_adev_errorbar_color_override():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    dev_err = np.array([0.01, 0.007, 0.005])

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", errorbar_color="red")
    _, _, barlinecols = ax.containers[0].lines
    assert to_rgba(barlinecols[0].get_color()[0]) == to_rgba("red")


def test_plot_adev_title():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])

    ax = plot_adev(tau, dev, unit="MHz", quantity="frequency", title="Frequency Allan Deviation")
    assert ax.get_title() == "Frequency Allan Deviation"


def test_plot_adev_label_adds_figure_suptitle():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    ax = plot_adev(tau, dev, unit="MHz", quantity="frequency", label="Scenario A")
    fig = ax.get_figure()
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Scenario A"
    assert fig._suptitle.get_fontsize() > 14
    assert fig._suptitle.get_fontweight() == "bold"


def test_plot_adev_no_label_leaves_no_suptitle():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    ax = plot_adev(tau, dev, unit="MHz", quantity="frequency")
    assert ax.get_figure()._suptitle is None


def test_plot_adev_label_replaces_title_instead_of_stacking():
    tau = np.array([1.0, 2.0, 4.0])
    dev = np.array([0.1, 0.07, 0.05])
    ax = plot_adev(tau, dev, unit="MHz", quantity="frequency",
                    title="Frequency Allan Deviation", label="Scenario A")
    assert ax.get_figure()._suptitle.get_text() == "Scenario A"
    assert ax.get_title() == ""


def test_plot_adev_label_coexists_with_regions():
    tau = np.array([0.1, 0.2, 1.0, 1.5, 5.0, 10.0])
    dev = np.array([10.0, 20.0, 100.0, 200.0, 1000.0, 3000.0])
    dev_err = np.array([1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency",
                   regions=True, label="Scenario A")
    assert ax.get_figure()._suptitle.get_text() == "Scenario A"
    assert len(ax.texts) == 3  # region annotations unaffected — guards the rename


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


def _region_test_data():
    tau = np.array([0.1, 0.2, 1.0, 1.5, 5.0, 10.0])
    dev = np.array([10.0, 20.0, 100.0, 200.0, 1000.0, 3000.0])
    dev_err = np.array([1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    return tau, dev, dev_err


def test_plot_adev_regions_true_uses_default_boundaries():
    tau, dev, dev_err = _region_test_data()

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)

    vlines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(vlines) == 2
    assert len(ax.texts) == 3


def test_plot_adev_regions_custom_boundaries():
    tau, dev, dev_err = _region_test_data()

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=[0.5])

    vlines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(vlines) == 1
    assert len(ax.texts) == 2


def test_plot_adev_regions_requires_dev_err():
    tau, dev, _ = _region_test_data()
    with pytest.raises(ValueError):
        plot_adev(tau, dev, None, unit="MHz", quantity="frequency", regions=True)


def test_plot_adev_regions_median_agg_differs_from_mean():
    from ltatools.analysis import summarize_adev_regions

    # skewed first region (tau < 0.25) so mean and median genuinely differ
    tau = np.array([0.1, 0.15, 0.2, 1.0, 5.0])
    dev = np.array([10.0, 12.0, 100.0, 100.0, 1000.0])
    dev_err = np.array([1.0, 1.0, 1.0, 5.0, 50.0])

    mean_regions = summarize_adev_regions(tau, dev, dev_err, boundaries=(0.25, 2.0), agg="mean")
    median_regions = summarize_adev_regions(tau, dev, dev_err, boundaries=(0.25, 2.0), agg="median")

    assert mean_regions[0]["value"] != pytest.approx(median_regions[0]["value"])

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True, region_agg="median")
    assert len(ax.texts) == 3


def test_finer_unit_frequency_steps_down():
    assert finer_unit("MHz", "frequency") == "kHz"
    assert finer_unit("THz", "frequency") == "GHz"
    assert finer_unit("Hz", "frequency") == "Hz"  # already finest


def test_finer_unit_power_already_finest():
    assert finer_unit("uW", "power") == "uW"
    assert finer_unit("mW", "power") == "µW"
    assert finer_unit("W", "power") == "mW"


def test_finer_unit_unknown_quantity_unchanged():
    assert finer_unit("nm", "wavelength") == "nm"


def test_plot_adev_regions_labels_use_finer_unit():
    tau, dev, dev_err = _region_test_data()

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)

    assert ax.get_ylabel().endswith("MHz")
    for text in ax.texts:
        assert "kHz" in text.get_text()
        assert "MHz" not in text.get_text()


def test_plot_adev_regions_labels_share_the_same_height():
    tau, dev, dev_err = _region_test_data()

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)

    heights = {text.xy[1] for text in ax.texts}
    assert len(heights) == 1


def test_plot_adev_regions_labels_centered_in_log_space():
    tau, dev, dev_err = _region_test_data()

    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)

    xs = sorted(text.xy[0] for text in ax.texts)
    assert xs[0] == pytest.approx(np.sqrt(0.1 * 0.25))   # data-min .. first boundary
    assert xs[1] == pytest.approx(np.sqrt(0.25 * 2.0))   # boundary .. boundary
    assert xs[2] == pytest.approx(np.sqrt(2.0 * 10.0))   # last boundary .. data-max


def test_plot_adev_regions_boundary_minor_ticks():
    tau, dev, dev_err = _region_test_data()
    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)
    minor = ax.get_xticks(minor=True)
    assert all(any(np.isclose(b, m) for m in minor) for b in (0.25, 2.0))
    assert any(np.isclose(0.3, m) for m in minor)  # a default sub-tick, retained
    assert len(minor) == 33  # 32 default sub-ticks + the added 0.25 boundary


def test_plot_adev_regions_boundary_minor_tick_labels():
    tau, dev, dev_err = _region_test_data()
    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels(minor=True) if t.get_text()]
    assert labels == ["0.25", "2"]


def test_plot_adev_regions_boundary_ticks_styled_distinctly_from_default():
    tau, dev, dev_err = _region_test_data()
    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=True)
    ax.figure.canvas.draw()
    positions = ax.get_xticks(minor=True)
    boundary_lengths, default_lengths = set(), set()
    for tick, pos in zip(ax.xaxis.get_minor_ticks(), positions):
        is_boundary = any(np.isclose(pos, b) for b in (0.25, 2.0))
        (boundary_lengths if is_boundary else default_lengths).add(tick.tick1line.get_markersize())
    assert max(default_lengths) < min(boundary_lengths)  # boundary ticks are taller


def test_plot_adev_regions_custom_boundary_ticks():
    tau, dev, dev_err = _region_test_data()
    ax = plot_adev(tau, dev, dev_err, unit="MHz", quantity="frequency", regions=[0.5])
    minor = ax.get_xticks(minor=True)
    assert any(np.isclose(0.5, m) for m in minor)
    assert len(minor) > 1  # default sub-ticks retained, not just the boundary
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels(minor=True) if t.get_text()]
    assert labels == ["0.5"]


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


@pytest.mark.parametrize("quantity", ["frequency", "power", "wavelength"])
def test_plot_histogram_quantities(write_lta, quantity):
    df = _load_df(write_lta, n_points=200)
    ax = plot_histogram(df, quantity=quantity)
    assert len(ax.patches) > 0
    assert ax.get_ylabel() == "Frequency (n)"


def test_plot_histogram_label_replaces_title_instead_of_stacking(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", label="Scenario A")
    assert ax.get_figure()._suptitle.get_text() == "Scenario A"
    assert ax.get_title() == ""


def test_plot_histogram_no_label_keeps_default_title(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency")
    assert ax.get_title() == "Frequency Histogram"
    assert ax.get_figure()._suptitle is None


def test_plot_histogram_custom_unit_and_bins(write_lta):
    df = _load_df(write_lta, n_points=200)
    ax = plot_histogram(df, quantity="frequency", unit="GHz", bins=10)
    assert ax.get_xlabel() == "Frequency (GHz)"
    assert len(ax.patches) == 10


def test_plot_histogram_relative_centers_on_mean(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", unit="MHz", relative=True, bins=10)
    baseline = df["frequency_THz"].mean()
    expected_min = float((df["frequency_THz"] - baseline).min()) * 1e6
    expected_max = float((df["frequency_THz"] - baseline).max()) * 1e6
    xlim_low, xlim_high = ax.get_xlim()
    assert xlim_low < expected_max
    assert xlim_high > expected_min
    assert xlim_low < 1000  # nowhere near the un-centered ~2.68e8 MHz absolute value


def test_plot_histogram_relative_xlabel_is_delta(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", unit="MHz", relative=True)
    assert ax.get_xlabel() == "Frequency deviation Δν (MHz)"


def test_plot_histogram_relative_shows_baseline_thz(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", relative=True)
    baseline = df["frequency_THz"].mean()
    integer_part, _, frac_part = f"{baseline:.7f}".partition(".")
    grouped_frac = "'".join(frac_part[i : i + 3] for i in range(0, len(frac_part), 3))
    texts = [t.get_text() for t in ax.texts]
    assert any(f"{integer_part}.{grouped_frac}" in t and "THz" in t for t in texts)


def test_plot_histogram_relative_power_baseline_matches_unit(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="power", unit="mW", relative=True)
    baseline_mW = df["power_uW"].mean() * 1e-3
    texts = [t.get_text() for t in ax.texts]
    assert any(f"{baseline_mW:.6g}" in t.replace(" ", "").replace("'", "") and "mW" in t for t in texts)


def test_plot_histogram_relative_default_false_unchanged(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", unit="MHz")
    assert ax.get_xlabel() == "Frequency (MHz)"


def test_plot_histogram_absolute_frequency_replaces_matplotlib_offset(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", unit="MHz")
    assert ax.xaxis.get_offset_text().get_visible() is False
    texts = [t.get_text() for t in ax.texts]
    assert any("THz" in t for t in texts)


def test_plot_histogram_absolute_frequency_offset_matches_formatter(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="frequency", unit="MHz")
    offset_MHz = float(ax.xaxis.get_major_formatter().offset)
    integer_part, _, frac_part = f"{offset_MHz * 1e-6:.7f}".partition(".")
    grouped_frac = "'".join(frac_part[i : i + 3] for i in range(0, len(frac_part), 3))
    texts = [t.get_text() for t in ax.texts]
    assert any(f"{integer_part}.{grouped_frac}" in t and "THz" in t for t in texts)


def test_plot_histogram_absolute_power_unaffected(write_lta):
    df = _load_df(write_lta, n_points=50)
    ax = plot_histogram(df, quantity="power", unit="uW")
    assert all("THz" not in t.get_text() for t in ax.texts)


def test_plot_histogram_hist_kwargs_forwarded(write_lta):
    df = _load_df(write_lta, n_points=200)
    ax = plot_histogram(df, quantity="power", density=True)
    total_width = sum(patch.get_width() * patch.get_height() for patch in ax.patches)
    assert total_width == pytest.approx(1.0)


def test_plot_histogram_invalid_quantity_raises(write_lta):
    df = _load_df(write_lta, n_points=50)
    with pytest.raises(ValueError):
        plot_histogram(df, quantity="bogus")


def test_plot_histogram_save_creates_missing_dir(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=50)
    save_path = tmp_path / "sub" / "hist.png"

    plot_histogram(df, quantity="frequency", save=save_path)
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


def test_overview_figure_adev_panels_have_darker_errorbars_and_no_endcaps_by_default(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave")
    _, ax_freq, ax_power = axes

    for ax, quantity in [(ax_freq, "frequency"), (ax_power, "power")]:
        _, caplines, barlinecols = ax.containers[0].lines
        assert len(caplines) == 0
        assert to_rgba(barlinecols[0].get_color()[0]) == to_rgba(darken_color(COLORS[quantity]))


def test_overview_figure_capsize_adds_endcaps(write_lta):
    df = _load_df(write_lta, n_points=200)
    fig, axes = overview_figure(df, taus="octave", capsize=3)
    _, ax_freq, ax_power = axes

    assert len(ax_freq.containers[0].lines[1]) > 0
    assert len(ax_power.containers[0].lines[1]) > 0


def test_overview_figure_regions_forwarded(write_lta):
    df = _load_df(write_lta, n_points=2000)
    fig, axes = overview_figure(df, taus="octave", regions=True)
    _, ax_freq, ax_power = axes

    for ax in (ax_freq, ax_power):
        vlines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(vlines) == 2
        assert len(ax.texts) == 3


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


def test_overview_figure_label_stamps_multipanel(write_lta):
    df = _load_df(write_lta, n_points=50)
    fig, axes = overview_figure(df, taus="octave", label="2")
    assert fig._suptitle.get_text() == "2"
    assert axes[1].get_title() == "Frequency Allan Deviation"  # panel title unaffected


def test_plot_dispatcher_forwards_label(write_lta, tmp_path):
    df = _load_df(write_lta, n_points=50)
    save_path = tmp_path / "adev.png"
    plot(df, kind="adev", label="Scenario A", save=save_path)
    assert save_path.exists()


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
    "hist": {},
}


@pytest.mark.parametrize("kind", ["overview", "psd", "timeseries", "adev", "spectrum", "hist"])
def test_plot_wrapper_dispatch_dataframe(write_lta, kind):
    df = _load_df(write_lta, n_points=2000)
    n_before = len(plt.get_fignums())

    result = plot(df, kind=kind, **_WRAPPER_KIND_KWARGS[kind])

    assert result is None
    assert len(plt.get_fignums()) > n_before


@pytest.mark.parametrize("kind", ["overview", "psd", "timeseries", "adev", "spectrum", "hist"])
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


def test_plot_wrapper_adev_regions(write_lta):
    df = _load_df(write_lta, n_points=2000)
    result = plot(df, kind="adev", regions=True, taus="octave")

    assert result is None
    ax = plt.gcf().axes[0]
    assert len(ax.texts) == 3


def test_plot_wrapper_unknown_kind_raises(write_lta):
    df = _load_df(write_lta, n_points=200)
    with pytest.raises(ValueError):
        plot(df, kind="bogus")
