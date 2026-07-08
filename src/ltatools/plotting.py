"""Plotting functions: timeseries, Allan deviation, PSD, and the combined overview figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .analysis import compute_oadev, compute_psd, find_stable_segments
from .io import load_lta_file
from .style import COLORS, adev_label, axis_label, psd_label, scale_frequency, scale_power

_PSD_QUANTITY_UNITS = {"frequency": "Hz", "power": "uW"}


def _quantity_scaler(quantity):
    if quantity == "power":
        return scale_power
    if quantity == "frequency":
        return scale_frequency
    if quantity == "wavelength":
        def _scale_wavelength(values, unit):
            if unit != "nm":
                raise ValueError(f"Unknown wavelength unit {unit!r}; expected 'nm'")
            return np.asarray(values)

        return _scale_wavelength
    raise ValueError(f"Unknown quantity {quantity!r}; expected one of {sorted(COLORS)}")


def plot_timeseries(df, kind="freq", ax=None, lines=False):
    """Dual-axis time-series plot: frequency or wavelength (left) and power (right).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from ``load_lta_file`` or ``find_stable_segments``. Must
        contain ``time_s``, ``power_uW``, and either ``frequency_THz`` or
        ``wavelength_nm``.
    kind : {"freq", "wl"}, default "freq"
        Quantity to plot on the left axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the left series into. If omitted, a new figure is
        created. The right (power) axis is always created via ``twinx``.
    lines : bool, default False
        If True, connect data points with lines (marker ``'x-'``).

    Returns
    -------
    ax_left, ax_right : matplotlib.axes.Axes
        The frequency/wavelength axis and the power axis.
    """
    fmt = "x-" if lines else "x"

    if ax is None:
        _, ax1 = plt.subplots(figsize=(12, 5))
    else:
        ax1 = ax
    ax2 = ax1.twinx()

    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)

    power_mW = scale_power(df["power_uW"], "mW")
    ax2.plot(df["time_s"], power_mW, fmt, color=COLORS["power"], label=axis_label("power", "mW"))
    ax2.set_ylabel(axis_label("power", "mW"), color=COLORS["power"])
    ax2.tick_params(axis="y", labelcolor=COLORS["power"])
    ax2.margins(y=0.1)

    if kind == "wl":
        quantity, unit, title = "wavelength", "nm", "Wavelength and Power over time"
        left_data = df["wavelength_nm"]
    elif kind == "freq":
        quantity, unit, title = "frequency", "THz", "Frequency and Power over time"
        left_data = df["frequency_THz"]
    else:
        raise ValueError(f"Unknown kind {kind!r}; expected 'freq' or 'wl'")

    ax1.plot(
        df["time_s"], left_data, fmt, color=COLORS[quantity], label=axis_label(quantity, unit)
    )
    ax1.set_title(title)
    ax1.set_ylabel(axis_label(quantity, unit), color=COLORS[quantity])
    ax1.tick_params(axis="y", labelcolor=COLORS[quantity])
    ax1.margins(y=0.1)
    ax1.set_xlabel("Time (s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2 + lines1, labels2 + labels1, loc="best")

    ax1.grid(True, which="both", ls="--", alpha=0.5)

    return ax1, ax2


def plot_adev(tau, dev, dev_err=None, *, ci_bounds=None, unit="MHz", quantity="frequency", ax=None):
    """Log-log Allan deviation plot with error bars.

    Parameters
    ----------
    tau : array_like
        Averaging times in seconds.
    dev : array_like
        Allan deviation values, in the same native unit as `compute_oadev`
        was given (THz for frequency, uW for power, nm for wavelength).
    dev_err : array_like, optional
        Naive symmetric error of `dev` (e.g. from ``compute_oadev``'s
        `dev_err` return value). Ignored if `ci_bounds` is given.
    ci_bounds : tuple of array_like, optional
        ``(lower, upper)`` confidence bounds for `dev` (e.g. from
        ``compute_oadev``'s `ci` return value), drawn as asymmetric error
        bars. Takes precedence over `dev_err`.
    unit : str, default "MHz"
        Target unit for `dev`/`dev_err`/`ci_bounds`; see
        `scale_frequency`/`scale_power`.
    quantity : {"frequency", "wavelength", "power"}, default "frequency"
        Physical quantity `dev` represents. Determines color and scaling.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    scale = _quantity_scaler(quantity)
    dev_scaled = scale(dev, unit)

    if ci_bounds is not None:
        lower, upper = ci_bounds
        lower_scaled = scale(lower, unit)
        upper_scaled = scale(upper, unit)
        yerr = [
            np.clip(dev_scaled - lower_scaled, 0, None),
            np.clip(upper_scaled - dev_scaled, 0, None),
        ]
    elif dev_err is not None:
        yerr = scale(dev_err, unit)
    else:
        yerr = None

    ax.errorbar(tau, dev_scaled, yerr=yerr, fmt="x", color=COLORS[quantity])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ in s")
    ax.set_ylabel(adev_label(quantity, unit))
    ax.grid(True, which="both", ls="--", alpha=0.5)

    return ax


def plot_psd(f, Pxx, *, ci_bounds=None, quantity="frequency", scaling="psd", ax=None):
    """Log-log power (or amplitude) spectral density plot with a confidence band.

    Parameters
    ----------
    f : array_like
        Frequency bins in Hz (e.g. from ``compute_psd``).
    Pxx : array_like
        Power spectral density in ``[unit]**2 / Hz``.
    ci_bounds : tuple of array_like, optional
        ``(lower, upper)`` confidence bounds for `Pxx`, drawn as a shaded
        band via ``fill_between``.
    quantity : {"frequency", "power"}, default "frequency"
        Physical quantity `Pxx` represents. Determines color and the fixed
        native unit used for the axis label (Hz for frequency, uW for
        power).
    scaling : {"psd", "asd"}, default "psd"
        Plot the power spectral density directly, or its square root (the
        amplitude spectral density).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if quantity not in _PSD_QUANTITY_UNITS:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(_PSD_QUANTITY_UNITS)}"
        )
    if scaling not in ("psd", "asd"):
        raise ValueError(f"Unknown scaling {scaling!r}; expected 'psd' or 'asd'")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    f = np.asarray(f)
    Pxx = np.asarray(Pxx)
    mask = f > 0  # drop the DC bin: a log-frequency axis can't show f=0
    f_plot = f[mask]
    y = Pxx[mask]
    if scaling == "asd":
        y = np.sqrt(y)

    if ci_bounds is not None:
        lower, upper = ci_bounds
        lower = np.asarray(lower)[mask]
        upper = np.asarray(upper)[mask]
        if scaling == "asd":
            lower, upper = np.sqrt(lower), np.sqrt(upper)
        ax.fill_between(f_plot, lower, upper, alpha=0.3, color=COLORS[quantity])

    ax.plot(f_plot, y, color=COLORS[quantity])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(psd_label(quantity, _PSD_QUANTITY_UNITS[quantity], scaling=scaling))
    ax.grid(True, which="both", ls="--", alpha=0.5)

    return ax


def overview_figure(
    df,
    *,
    kind="freq",
    freq_unit="MHz",
    power_unit="uW",
    taus="all",
    ci=None,
    lines=False,
    save=None,
):
    """Combined overview figure: timeseries on top, frequency and power ADEV below.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from ``load_lta_file`` or ``find_stable_segments``.
    kind : {"freq", "wl"}, default "freq"
        Quantity for the timeseries left axis.
    freq_unit : str, default "MHz"
        Unit for the frequency ADEV panel.
    power_unit : str, default "uW"
        Unit for the power ADEV panel.
    taus : str or numpy.ndarray, default "all"
        Averaging times passed to ``compute_oadev``.
    ci : float, optional
        If given, a chi-squared confidence interval (see ``compute_oadev``)
        is computed and drawn for both ADEV panels instead of the naive
        `dev_err` bars.
    lines : bool, default False
        Passed to ``plot_timeseries``.
    save : str or pathlib.Path, optional
        If given, the figure is saved as a 300 dpi PNG; the parent
        directory is created if it does not exist.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        ``[ax_timeseries, ax_freq_adev, ax_power_adev]``.
    """
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax_ts = fig.add_subplot(gs[0, :])
    plot_timeseries(df, kind=kind, ax=ax_ts, lines=lines)

    ax_freq = fig.add_subplot(gs[1, 0])
    if ci is None:
        tau_f, dev_f, dev_f_err, _ = compute_oadev(df["frequency_THz"], time_s=df["time_s"], taus=taus)
        plot_adev(tau_f, dev_f, dev_f_err, unit=freq_unit, quantity="frequency", ax=ax_freq)
    else:
        tau_f, dev_f, dev_f_err, _, bounds_f = compute_oadev(
            df["frequency_THz"], time_s=df["time_s"], taus=taus, ci=ci
        )
        plot_adev(
            tau_f, dev_f, dev_f_err, ci_bounds=bounds_f, unit=freq_unit, quantity="frequency", ax=ax_freq
        )

    ax_power = fig.add_subplot(gs[1, 1])
    if ci is None:
        tau_p, dev_p, dev_p_err, _ = compute_oadev(df["power_uW"], time_s=df["time_s"], taus=taus)
        plot_adev(tau_p, dev_p, dev_p_err, unit=power_unit, quantity="power", ax=ax_power)
    else:
        tau_p, dev_p, dev_p_err, _, bounds_p = compute_oadev(
            df["power_uW"], time_s=df["time_s"], taus=taus, ci=ci
        )
        plot_adev(
            tau_p, dev_p, dev_p_err, ci_bounds=bounds_p, unit=power_unit, quantity="power", ax=ax_power
        )

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, [ax_ts, ax_freq, ax_power]


def psd_figure(df, *, scaling="psd", ci=None, nperseg=None, save=None):
    """Separate PSD/ASD overview figure: frequency panel and power panel side by side.

    Not part of ``overview_figure`` — call this explicitly when a spectral
    view is wanted in addition to the timeseries/ADEV overview.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from ``load_lta_file`` or ``find_stable_segments``.
    scaling : {"psd", "asd"}, default "psd"
        Passed to ``plot_psd`` for both panels.
    ci : float, optional
        If given, a chi-squared confidence band (see ``compute_psd``) is
        computed and drawn for both panels.
    nperseg : int, optional
        Passed to ``compute_psd`` for both panels.
    save : str or pathlib.Path, optional
        If given, the figure is saved as a 300 dpi PNG; the parent
        directory is created if it does not exist.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        ``[ax_freq_psd, ax_power_psd]``.
    """
    fig, (ax_freq, ax_power) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    freq_hz = df["frequency_THz"] * 1e12
    freq_result = compute_psd(freq_hz, time_s=df["time_s"], ci=ci, nperseg=nperseg)
    f_f, Pxx_f, *rest_f = freq_result
    bounds_f = rest_f[0] if rest_f else None
    plot_psd(f_f, Pxx_f, ci_bounds=bounds_f, quantity="frequency", scaling=scaling, ax=ax_freq)

    power_result = compute_psd(df["power_uW"], time_s=df["time_s"], ci=ci, nperseg=nperseg)
    f_p, Pxx_p, *rest_p = power_result
    bounds_p = rest_p[0] if rest_p else None
    plot_psd(f_p, Pxx_p, ci_bounds=bounds_p, quantity="power", scaling=scaling, ax=ax_power)

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, [ax_freq, ax_power]


def lta_overview(file_path, *, cleanup=False, segments=False, n_segments=2, **kwargs):
    """Load an .lta file and produce its overview figure(s).

    Convenience wrapper around ``load_lta_file`` + ``overview_figure``
    (and ``find_stable_segments`` when `segments` is True). For a PSD/ASD
    view, call ``psd_figure`` directly — it is a separate figure type and
    intentionally has no wrapper here.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the .lta file.
    cleanup : bool, default False
        Passed to ``load_lta_file``.
    segments : bool, default False
        If True, split the data into stable segments first and produce one
        overview figure per segment.
    n_segments : int, default 2
        Number of segments to use when `segments` is True.
    **kwargs
        Passed through to ``overview_figure``.

    Returns
    -------
    (fig, axes) or list of (fig, axes)
        A single overview-figure result, or one per stable segment.
    """
    df = load_lta_file(file_path, cleanup=cleanup)
    if segments:
        segs = find_stable_segments(df, n=n_segments)
        return [overview_figure(seg, **kwargs) for seg in segs]
    return overview_figure(df, **kwargs)
