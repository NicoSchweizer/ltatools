"""Plotting functions: timeseries, Allan deviation, PSD, and the combined overview figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from .analysis import (
    DEFAULT_ADEV_REGION_BOUNDARIES,
    compute_oadev,
    compute_psd,
    find_stable_segments,
    summarize_adev_regions,
)
from .io import load_lta_file
from .style import COLORS, adev_label, axis_label, darken_color, finer_unit, psd_label, scale_frequency, scale_power

_PSD_QUANTITY_UNITS = {"frequency": "Hz", "power": "µW"}
_REGION_LABEL_Y = 0.5
_THIN_SPACE = " "


def _format_grouped(value, decimals, group=3, sep=_THIN_SPACE):
    """Format `value` to `decimals` fractional digits, with `sep` every
    `group` digits after the decimal point for readability, e.g.
    ``"268.123 456 7"``. A thin space (not a comma) avoids collision with
    the comma-as-decimal-separator convention used in some locales.
    """
    integer_part, _, frac_part = f"{value:.{decimals}f}".partition(".")
    grouped_frac = sep.join(frac_part[i : i + group] for i in range(0, len(frac_part), group))
    return f"{integer_part}.{grouped_frac}"


def _add_figure_label(fig, label):
    """Stamp a large scenario label across the top of ``fig`` as a suptitle.

    No-op when ``label`` is None (the default), so existing figures are
    unchanged. When given, drawn via ``fig.suptitle`` at 22pt bold —
    noticeably larger/heavier than the 14pt axes titles (``axes.titlesize``
    in ``style.py``) so it reads as a stamp, not another title. matplotlib
    reserves no suptitle space by default on either kind of figure this
    module creates, so headroom is reserved explicitly: plain
    ``plt.subplots`` figures (the single-panel functions) via
    ``fig.subplots_adjust``; ``constrained_layout=True`` figures (the two
    multi-panel functions) via their layout engine's ``rect``, since
    calling ``subplots_adjust`` directly on a figure with a layout engine
    is ignored. Either way this keeps the stamp clear of each panel's own
    title and of the in-axes ``relative=True`` baseline text (drawn at
    axes-fraction y=1.0).
    """
    if label is None:
        return
    fig.suptitle(str(label), fontsize=22, fontweight="bold", y=0.99, va="top")
    engine = fig.get_layout_engine()
    if engine is None:
        fig.subplots_adjust(top=0.85)
    else:
        engine.set(rect=(0, 0, 1, 0.90))


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


def plot_timeseries(df, kind="freq", ax=None, lines=False, freq_unit="THz", power_unit="µW", markersize=4, save=None, relative=False, label=None):
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
    freq_unit : str, default "THz"
        Target unit for the left axis when ``kind="freq"``; see
        `scale_frequency`. Ignored when ``kind="wl"``.
    power_unit : str, default "µW"
        Target unit for the right (power) axis; see `scale_power`.
    markersize : float, default 4
        Marker size for both series.
    save : str or pathlib.Path, optional
        If given, the figure containing `ax1` is saved as a 300 dpi PNG;
        the parent directory is created if it does not exist. When an
        existing `ax` was passed in, this saves the entire containing
        figure.
    relative : bool, default False
        If True, plot the power axis (and the frequency axis, when
        ``kind="freq"``) as *deviation from its own mean* instead of its
        absolute value: ``mean(power_uW)`` (and ``mean(frequency_THz)``)
        is subtracted in each quantity's native unit (µW / THz) before
        scaling to `power_unit`/`freq_unit`, so each axis reads small ±
        deviations around zero — useful for inspecting relative stability.
        Each subtracted baseline is shown as a plain value/unit label near
        the top of its axis (frequency top-left, power top-right),
        replacing matplotlib's unitless scientific-notation offset with a
        physically meaningful one, and the corresponding y-axis label
        becomes a deviation label using the quantity's physics symbol
        (e.g. ``"Frequency deviation Δν (MHz)"``, ``"Power deviation ΔP
        (µW)"``). The frequency baseline is always shown in THz, its
        fractional digits grouped in triples with a thin space (e.g.
        ``"268.123 456 7 THz"``) for readability — a thin space rather
        than a comma, since a comma there would collide with the
        comma-as-decimal-separator convention used in some locales —
        regardless of `freq_unit`, since frequency units span many
        orders of magnitude (THz vs. kHz) and that mismatch is exactly
        the unitless-offset problem being fixed; the power baseline is
        shown in `power_unit` itself, matching the power axis, since
        power units (µW/mW/W) don't have that same huge-offset problem
        and showing a different unit than the axis would just be
        confusing. Ignored for the left axis when ``kind="wl"``
        (wavelength has no deviation view — only the power axis reacts).
        Defaults to False, leaving existing behavior unchanged.
    label : str, optional
        If given, drawn as a large bold stamp across the top of the
        figure (via ``fig.suptitle``) — used to mark which measurement
        scenario a saved figure belongs to. Replaces the plot's own
        descriptive title (e.g. ``"Frequency and Power over time"`` is
        not set) rather than stacking above it. Applies to the whole
        containing figure, including one passed via `ax`. Defaults to
        None (no stamp; descriptive title shown as usual).

    Returns
    -------
    ax_left, ax_right : matplotlib.axes.Axes
        The frequency/wavelength axis and the power axis.

    Raises
    ------
    ValueError
        If `kind` is not ``"freq"`` or ``"wl"``.

    Examples
    --------
    >>> df = load_lta_file("scan.lta")
    >>> plot_timeseries(df, kind="freq", freq_unit="MHz", save="timeseries")
    (<Axes: title={'center': 'Frequency and Power over time'}, xlabel='Time (s)', ylabel='Frequency (MHz)'>, <Axes: ylabel='Power (µW)'>)
    >>> plot_timeseries(df, kind="freq", freq_unit="kHz", relative=True)  # relative stability
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

    if relative:
        # subtract the mean in native uW *before* scaling (mirrors the frequency
        # baseline below), but — unlike frequency, which is always shown in the
        # fixed native THz — display the baseline in `power_unit` itself: power
        # units don't span the many orders of magnitude frequency units do, so
        # there's no huge-offset problem to fix, and matching the axis unit
        # keeps the baseline label directly comparable to the plotted values.
        baseline_uW = float(df["power_uW"].mean())
        power_scaled = scale_power(df["power_uW"] - baseline_uW, power_unit)
    else:
        power_scaled = scale_power(df["power_uW"], power_unit)
    ax2.plot(
        df["time_s"], power_scaled, fmt, color=COLORS["power"],
        markersize=markersize, label=axis_label("power", power_unit, delta=relative),
    )
    ax2.set_ylabel(axis_label("power", power_unit, delta=relative), color=COLORS["power"])
    ax2.tick_params(axis="y", labelcolor=COLORS["power"])
    ax2.margins(y=0.1)
    if relative:
        baseline_power_display = float(scale_power(baseline_uW, power_unit))
        ax2.text(
            1.0, 1.0, f"{baseline_power_display:.6g} {power_unit}",
            transform=ax2.transAxes, ha="right", va="bottom", color="0.2",
        )

    if kind == "wl":
        quantity, unit, title = "wavelength", "nm", "Wavelength and Power over time"
        left_data = df["wavelength_nm"]
    elif kind == "freq":
        quantity, unit, title = "frequency", freq_unit, "Frequency and Power over time"
        if relative:
            # subtract the mean in native THz *before* scaling — subtracting a
            # scaled ~1e8 MHz mean afterwards would lose precision to cancellation.
            baseline_THz = float(df["frequency_THz"].mean())
            left_data = scale_frequency(df["frequency_THz"] - baseline_THz, freq_unit)
        else:
            left_data = scale_frequency(df["frequency_THz"], freq_unit)
    else:
        raise ValueError(f"Unknown kind {kind!r}; expected 'freq' or 'wl'")

    delta = kind == "freq" and relative
    ax1.plot(
        df["time_s"], left_data, fmt, color=COLORS[quantity],
        markersize=markersize, label=axis_label(quantity, unit, delta=delta),
    )
    if label is None:
        ax1.set_title(title)
    ax1.set_ylabel(axis_label(quantity, unit, delta=delta), color=COLORS[quantity])
    ax1.tick_params(axis="y", labelcolor=COLORS[quantity])
    ax1.margins(y=0.1)
    ax1.set_xlabel("Time (s)")
    if delta:
        ax1.text(
            0.0, 1.0, f"{_format_grouped(baseline_THz, decimals=7)} THz",
            transform=ax1.transAxes, ha="left", va="bottom", color="0.2",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2 + lines1, labels2 + labels1, loc="best")

    ax1.grid(True, which="both", ls="--", alpha=0.5)

    _add_figure_label(ax1.get_figure(), label)

    if save is not None:
        fig = ax1.get_figure()
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return ax1, ax2


def plot_adev(
    tau, dev, dev_err=None, *, unit="MHz", quantity="frequency", ax=None,
    errorbars=True, title=None, save=None, label=None, capsize=0, errorbar_color=None,
    regions=None, region_agg="mean",
):
    """Log-log Allan deviation plot with error bars.

    Parameters
    ----------
    tau : array_like
        Averaging times in seconds.
    dev : array_like
        Allan deviation values, in the same native unit as `compute_oadev`
        was given (THz for frequency, uW for power, nm for wavelength).
    dev_err : array_like, optional
        Error of `dev` (e.g. from ``compute_oadev``'s `dev_err` return
        value). Required if `regions` is given.
    unit : str, default "MHz"
        Target unit for `dev`/`dev_err`; see
        `scale_frequency`/`scale_power`.
    quantity : {"frequency", "wavelength", "power"}, default "frequency"
        Physical quantity `dev` represents. Determines color and scaling.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new figure is created.
    errorbars : bool, default True
        If False, `dev_err` is ignored and no error bars are drawn.
    title : str, optional
        If given, set as the axes title — unless `label` is also given,
        in which case `label` takes over as the figure's suptitle and
        `title` is not set (avoids a redundant double heading).
    save : str or pathlib.Path, optional
        If given, the figure containing `ax` is saved as a 300 dpi PNG; the
        parent directory is created if it does not exist. When an existing
        `ax` was passed in, this saves the entire containing figure.
    label : str, optional
        If given, drawn as a large bold stamp across the top of the
        figure (via ``fig.suptitle``) — used to mark which measurement
        scenario a saved figure belongs to. Replaces `title` (see above)
        rather than stacking above it. Applies to the whole containing
        figure, including one passed via `ax`. Defaults to None (no
        stamp; `title`, if given, shown as usual).
    capsize : float, default 0
        Size of the error bar end caps, in points. ``0`` draws error bars
        without caps (the default); set e.g. ``capsize=3`` to add caps.
    errorbar_color : str or tuple, optional
        Color for the error bars. Defaults to a darkened version of the
        marker color (see `style.darken_color`) so error bars stand out
        against the data points. Pass the same color as the marker (e.g.
        ``COLORS["frequency"]``) to make them match again.
    regions : bool or sequence of float, optional
        If given, split `tau` into regions and annotate each with an
        aggregated value (see `region_agg`) plus its error, following
        ``analysis.summarize_adev_regions`` — vertical dotted lines mark
        the region boundaries, and each region gets a text annotation at a
        fixed height (see `_REGION_LABEL_Y` in this module, a fraction of
        the axes height from the bottom, so all region labels line up
        regardless of each region's data values — e.g. a "short/mid/long
        term" summary, as in a typical frequency-stability plot).
        Horizontally, each label is centered in log-space (geometric mean)
        between its two region boundaries — falling back to the region's
        extreme data τ on the outer side of the first/last region, where
        no boundary line is drawn. ``True`` uses
        ``analysis.DEFAULT_ADEV_REGION_BOUNDARIES``
        (0.25 s, 2 s); a sequence gives custom τ boundaries in seconds.
        Requires `dev_err` (used for error propagation — see below).

        The annotated error is the **propagated** error of each region's
        aggregate (quadrature sum of the region's `dev_err` values divided
        by the point count), not the standard deviation of the `dev`
        values in the region. The Allan deviation typically has a real
        trend across a region rather than scattering around a constant
        value, so using the raw spread as an error bar would conflate
        that trend with measurement uncertainty; propagating the
        already-computed per-tau errors avoids this. See
        `analysis.summarize_adev_regions` for details.

        The annotated value/error are shown in a unit one step finer than
        `unit` (see `style.finer_unit`), e.g. in kHz when ``unit="MHz"`` —
        the axis itself stays in `unit`.

        Each region boundary τ is additionally drawn as a labeled minor
        tick on the x-axis (e.g. ``0.25``, ``2``) so its exact value can be
        read directly off the axis rather than inferred against the decade
        gridlines. These boundary ticks are styled distinctly (taller and
        gray, with a label) and are added alongside matplotlib's default
        unlabeled log-scale minor ticks (the 2-9 sub-ticks per decade),
        which remain in their normal short, unlabeled style.
    region_agg : {"mean", "median"}, default "mean"
        Aggregation statistic used within each region when `regions` is
        given. Ignored otherwise.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> tau, dev, dev_err, _ = compute_oadev(df["frequency_THz"], time_s=df["time_s"])
    >>> plot_adev(tau, dev, dev_err, unit="MHz", save="adev_freq")
    <Axes: xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in MHz'>
    >>> plot_adev(tau, dev, dev_err, unit="MHz", capsize=3)  # with end caps
    <Axes: xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in MHz'>
    >>> plot_adev(tau, dev, dev_err, unit="MHz", regions=True)  # short/mid/long-term summary
    <Axes: xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in MHz'>
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    scale = _quantity_scaler(quantity)
    dev_scaled = scale(dev, unit)

    yerr = scale(dev_err, unit) if (errorbars and dev_err is not None) else None
    color = COLORS[quantity]
    ecolor = errorbar_color if errorbar_color is not None else darken_color(color)

    ax.errorbar(tau, dev_scaled, yerr=yerr, fmt="x", color=color, ecolor=ecolor, capsize=capsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ in s")
    ax.set_ylabel(adev_label(quantity, unit))
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if title is not None and label is None:
        ax.set_title(title)

    if regions is not None:
        if dev_err is None:
            raise ValueError("regions requires dev_err for error propagation")
        boundaries = DEFAULT_ADEV_REGION_BOUNDARIES if regions is True else regions
        region_unit = finer_unit(unit, quantity)
        dev_region_scaled = scale(dev, region_unit)
        dev_err_region_scaled = scale(dev_err, region_unit)
        tau_arr = np.asarray(tau)
        sorted_boundaries = sorted(boundaries)
        for boundary in sorted_boundaries:
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=1.5)
        # Label each region boundary directly on the x-axis so its exact τ can
        # be read off the axis, *without* discarding matplotlib's default
        # unlabeled log-scale minor ticks (the 2-9 sub-ticks per decade). There
        # is only one minor locator per axis, so capture the default minor-tick
        # positions first (the errorbar() above already set real x-limits, so
        # the log locator returns sensible sub-ticks), then install a combined
        # fixed set — default sub-ticks plus the boundaries — labeling only the
        # boundaries. tick_params would style the whole minor level at once, so
        # instead style each boundary tick individually (taller/gray/labeled) so
        # it reads as an intentional annotation, leaving the default sub-ticks in
        # matplotlib's normal style (short, default color, unlabeled).
        xlim = ax.get_xlim()
        default_minor = list(ax.get_xticks(minor=True))
        combined_minor = sorted(set(default_minor) | set(sorted_boundaries))
        minor_labels = [
            f"{pos:g}" if any(np.isclose(pos, b) for b in sorted_boundaries) else ""
            for pos in combined_minor
        ]
        ax.set_xticks(combined_minor, labels=minor_labels, minor=True)
        # set_xticks widens the view to fit every tick it's given (the default
        # sub-ticks span full decades touching the data, not just the tight
        # autoscaled range) — restore the original x-range so the plot doesn't
        # zoom out to show mostly-empty decades.
        ax.set_xlim(xlim)
        for tick, pos in zip(ax.xaxis.get_minor_ticks(), combined_minor):
            if any(np.isclose(pos, b) for b in sorted_boundaries):
                for line in (tick.tick1line, tick.tick2line):
                    line.set_markersize(6)
                    line.set_markeredgewidth(1.5)
                    line.set_color("gray")
                tick.label1.set_color("gray")
                tick.label2.set_color("gray")
        for region in summarize_adev_regions(
            tau_arr, dev_region_scaled, dev_err_region_scaled, boundaries=boundaries, agg=region_agg
        ):
            mask = (tau_arr >= region["tau_min"]) & (tau_arr < region["tau_max"])
            left = region["tau_min"] if region["tau_min"] > 0 else float(np.min(tau_arr[mask]))
            right = region["tau_max"] if np.isfinite(region["tau_max"]) else float(np.max(tau_arr[mask]))
            # log-scale x-axis: center in log-space (geometric mean), not linearly.
            tau_center = float(np.sqrt(left * right))
            region_label = ax.annotate(
                f"{region['value']:.2f} {region_unit}\n±{region['error']:.2f} {region_unit}",
                xy=(tau_center, _REGION_LABEL_Y), xycoords=("data", "axes fraction"),
                ha="center", va="center",
            )
            region_label.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    _add_figure_label(ax.get_figure(), label)

    if save is not None:
        fig = ax.get_figure()
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return ax


def plot_psd(f, Pxx, *, ci_bounds=None, quantity="frequency", scaling="psd", ax=None, save=None, label=None):
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
    save : str or pathlib.Path, optional
        If given, the figure containing `ax` is saved as a 300 dpi PNG; the
        parent directory is created if it does not exist. When an existing
        `ax` was passed in, this saves the entire containing figure.
    label : str, optional
        If given, drawn as a large bold stamp across the top of the
        figure (via ``fig.suptitle``) — used to mark which measurement
        scenario a saved figure belongs to. Applies to the whole
        containing figure, including one passed via `ax`. Defaults to
        None (no stamp).

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

    _add_figure_label(ax.get_figure(), label)

    if save is not None:
        fig = ax.get_figure()
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return ax


_HIST_QUANTITY_DEFAULTS = {
    "frequency": {"column": "frequency_THz", "unit": "MHz"},
    "power": {"column": "power_uW", "unit": "µW"},
    "wavelength": {"column": "wavelength_nm", "unit": "nm"},
}


def plot_histogram(df, quantity="frequency", *, unit=None, bins=50, ax=None, save=None, label=None, **hist_kwargs):
    """Histogram of a single quantity's distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from ``load_lta_file`` or ``find_stable_segments``. Must
        contain the column for `quantity` (``frequency_THz``,
        ``power_uW``, or ``wavelength_nm``).
    quantity : {"frequency", "power", "wavelength"}, default "frequency"
        Physical quantity to histogram. Determines color, source column,
        and default unit.
    unit : str, optional
        Target unit for the data; see `scale_frequency`/`scale_power`.
        Defaults to ``"MHz"`` for frequency, ``"µW"`` for power, and
        ``"nm"`` for wavelength.
    bins : int or sequence or str, default 50
        Passed to ``ax.hist``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new figure is created.
    save : str or pathlib.Path, optional
        If given, the figure containing `ax` is saved as a 300 dpi PNG; the
        parent directory is created if it does not exist. When an existing
        `ax` was passed in, this saves the entire containing figure.
    label : str, optional
        If given, drawn as a large bold stamp across the top of the
        figure (via ``fig.suptitle``) — used to mark which measurement
        scenario a saved figure belongs to. Replaces the plot's own
        descriptive title (e.g. ``"Frequency Histogram"`` is not set)
        rather than stacking above it. Applies to the whole containing
        figure, including one passed via `ax`. Defaults to None (no
        stamp; descriptive title shown as usual).
    **hist_kwargs
        Forwarded to ``ax.hist`` (e.g. ``density``, ``cumulative``,
        ``histtype``, ``alpha``).

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If `quantity` is not recognized.

    Examples
    --------
    >>> df = load_lta_file("scan.lta")
    >>> plot_histogram(df, quantity="frequency", unit="MHz", save="hist_freq")
    <Axes: title={'center': 'Frequency Histogram'}, xlabel='Frequency (MHz)', ylabel='Count'>
    >>> plot_histogram(df, quantity="power", bins=100, density=True)
    <Axes: title={'center': 'Power Histogram'}, xlabel='Power (µW)', ylabel='Count'>
    """
    if quantity not in _HIST_QUANTITY_DEFAULTS:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(_HIST_QUANTITY_DEFAULTS)}"
        )
    defaults = _HIST_QUANTITY_DEFAULTS[quantity]
    unit = unit if unit is not None else defaults["unit"]
    scale = _quantity_scaler(quantity)
    data = scale(df[defaults["column"]], unit)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(data, bins=bins, color=COLORS[quantity], **hist_kwargs)
    if label is None:
        ax.set_title(f"{quantity.capitalize()} Histogram")
    ax.set_xlabel(axis_label(quantity, unit))
    ax.set_ylabel("Count")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    _add_figure_label(ax.get_figure(), label)

    if save is not None:
        fig = ax.get_figure()
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return ax


def overview_figure(
    df,
    *,
    kind="freq",
    freq_unit="MHz",
    power_unit="µW",
    taus="octave",
    lines=False,
    errorbars=True,
    markersize=4,
    save=None,
    label=None,
    capsize=0,
    errorbar_color=None,
    regions=None,
    region_agg="mean",
):
    """Combined overview figure: timeseries on top, frequency and power ADEV below.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from ``load_lta_file`` or ``find_stable_segments``.
    kind : {"freq", "wl"}, default "freq"
        Quantity for the timeseries left axis.
    freq_unit : str, default "MHz"
        Unit for the timeseries frequency axis (when ``kind="freq"``) and
        the frequency ADEV panel.
    power_unit : str, default "µW"
        Unit for the timeseries power axis and the power ADEV panel.
    taus : str or numpy.ndarray, default "octave"
        Averaging times passed to ``compute_oadev``. ``"octave"`` (powers of
        two) is dramatically faster than ``"all"`` on large files — e.g.
        ~450x on a 145k-row file — with visually indistinguishable results
        on the usual log-log ADEV plot; pass ``taus="all"`` for the
        exhaustive (much slower) computation.
    lines : bool, default False
        Passed to ``plot_timeseries``.
    errorbars : bool, default True
        If False, no error bars are drawn on either ADEV panel.
    markersize : float, default 4
        Marker size for the timeseries panel; passed to ``plot_timeseries``.
    save : str or pathlib.Path, optional
        If given, the figure is saved as a 300 dpi PNG; the parent
        directory is created if it does not exist.
    label : str, optional
        If given, drawn as a large bold stamp across the top of the
        whole figure (via ``fig.suptitle``) — used to mark which
        measurement scenario a saved figure belongs to. Rendered above
        and clear of the panel titles. Defaults to None (no stamp).
    capsize : float, default 0
        Passed to ``plot_adev`` for both ADEV panels; ``0`` draws error
        bars without caps (the default), set e.g. ``capsize=3`` to add caps.
    errorbar_color : str or tuple, optional
        Passed to ``plot_adev`` for both ADEV panels. Defaults to a
        darkened version of each panel's marker color.
    regions : bool or sequence of float, optional
        Passed to ``plot_adev`` for both ADEV panels — annotates each
        panel with a short/mid/long-term (or custom) τ-region summary; see
        `plot_adev` for details. Requires `errorbars` data internally
        (each panel's own `dev_err` from `compute_oadev` is used, so this
        works regardless of the `errorbars` display toggle above).
    region_agg : {"mean", "median"}, default "mean"
        Passed to ``plot_adev`` for both ADEV panels. Ignored unless
        `regions` is given.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        ``[ax_timeseries, ax_freq_adev, ax_power_adev]``.

    Examples
    --------
    >>> df = load_lta_file("scan.lta")
    >>> fig, axes = overview_figure(df, freq_unit="kHz", errorbars=False, save="overview")
    >>> axes
    [<Axes: title={'center': 'Frequency and Power over time'}, xlabel='Time (s)', ylabel='Frequency (kHz)'>, <Axes: title={'center': 'Frequency Allan Deviation'}, xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in kHz'>, <Axes: title={'center': 'Power Allan Deviation'}, xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in µW'>]
    >>> fig, axes = overview_figure(df, capsize=3)  # with end caps
    >>> fig, axes = overview_figure(df, regions=True)  # short/mid/long-term summary
    """
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax_ts = fig.add_subplot(gs[0, :])
    plot_timeseries(
        df, kind=kind, ax=ax_ts, lines=lines,
        freq_unit=freq_unit, power_unit=power_unit, markersize=markersize,
    )

    ax_freq = fig.add_subplot(gs[1, 0])
    tau_f, dev_f, dev_f_err, _ = compute_oadev(df["frequency_THz"], time_s=df["time_s"], taus=taus)
    plot_adev(
        tau_f, dev_f, dev_f_err, unit=freq_unit, quantity="frequency", ax=ax_freq,
        errorbars=errorbars, title="Frequency Allan Deviation",
        capsize=capsize, errorbar_color=errorbar_color,
        regions=regions, region_agg=region_agg,
    )

    ax_power = fig.add_subplot(gs[1, 1])
    tau_p, dev_p, dev_p_err, _ = compute_oadev(df["power_uW"], time_s=df["time_s"], taus=taus)
    plot_adev(
        tau_p, dev_p, dev_p_err, unit=power_unit, quantity="power", ax=ax_power,
        errorbars=errorbars, title="Power Allan Deviation",
        capsize=capsize, errorbar_color=errorbar_color,
        regions=regions, region_agg=region_agg,
    )

    _add_figure_label(fig, label)

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, [ax_ts, ax_freq, ax_power]


def psd_figure(df, *, scaling="psd", ci=None, nperseg=None, save=None, label=None):
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
    label : str, optional
        If given, drawn as a large bold stamp across the top of the
        whole figure (via ``fig.suptitle``) — used to mark which
        measurement scenario a saved figure belongs to. Defaults to
        None (no stamp).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        ``[ax_freq_psd, ax_power_psd]``.

    Examples
    --------
    >>> df = load_lta_file("scan.lta")
    >>> fig, axes = psd_figure(df, scaling="asd", ci=0.95, save="psd")
    >>> axes
    [<Axes: xlabel='Frequency (Hz)', ylabel='ASD in Hz/$\\sqrt{\\mathrm{Hz}}$'>, <Axes: xlabel='Frequency (Hz)', ylabel='ASD in µW/$\\sqrt{\\mathrm{Hz}}$'>]
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

    _add_figure_label(fig, label)

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, [ax_freq, ax_power]


def lta_overview(file_path, *, cleanup=False, segments=False, n_segments=2, **kwargs):
    """Load an .lta file and produce its overview figure(s).

    Convenience wrapper around ``load_lta_file`` + ``overview_figure``
    (and ``find_stable_segments`` when `segments` is True). For any other
    plot kind (PSD, standalone timeseries or ADEV), see ``plot``, which
    also accepts an .lta path directly.

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

    Examples
    --------
    >>> fig, axes = lta_overview("scan.lta")
    >>> axes
    [<Axes: title={'center': 'Frequency and Power over time'}, xlabel='Time (s)', ylabel='Frequency (MHz)'>, <Axes: title={'center': 'Frequency Allan Deviation'}, xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in MHz'>, <Axes: title={'center': 'Power Allan Deviation'}, xlabel='$\\tau$ in s', ylabel='$\\sigma(\\tau)$ in µW'>]
    >>> results = lta_overview("scan.lta", segments=True, n_segments=2)
    >>> len(results)
    2
    """
    df = load_lta_file(file_path, cleanup=cleanup)
    if segments:
        segs = find_stable_segments(df, n=n_segments)
        return [overview_figure(seg, **kwargs) for seg in segs]
    return overview_figure(df, **kwargs)


_ADEV_QUANTITY_DEFAULTS = {
    "frequency": {"column": "frequency_THz", "unit": "MHz", "title": "Frequency Allan Deviation"},
    "power": {"column": "power_uW", "unit": "µW", "title": "Power Allan Deviation"},
}

_SPECTRUM_QUANTITY_COLUMNS = {
    "frequency": "frequency_THz",
    "power": "power_uW",
}


def plot(data, kind="overview", *, quantity="frequency", save=None, cleanup=False, **kwargs):
    """Build (and optionally save) any ltatools plot from a DataFrame or an .lta file.

    Single entry point covering every plot kind in this module — the
    intended default way to make a plot interactively, since it returns
    ``None`` (so nothing is echoed at the end of a notebook cell) and
    accepts either an already-loaded DataFrame or a raw ``.lta`` path.

    Parameters
    ----------
    data : pandas.DataFrame or str or pathlib.Path
        Either a DataFrame from ``load_lta_file``/``find_stable_segments``,
        or a path to an ``.lta`` file (loaded internally via
        ``load_lta_file(data, cleanup=cleanup)``).
    kind : {"overview", "psd", "timeseries", "adev", "spectrum", "hist"}, default "overview"
        Which plot to build:

        - ``"overview"`` — ``overview_figure``.
        - ``"psd"`` — ``psd_figure`` (frequency + power PSD/ASD panels).
        - ``"timeseries"`` — ``plot_timeseries``. Pass ``ts_kind="wl"`` in
          `kwargs` to plot wavelength instead of frequency (default
          ``"freq"``).
        - ``"adev"`` — Allan deviation of a single column, computed via
          ``compute_oadev`` and drawn via ``plot_adev``; `quantity`
          selects the column. Defaults to ``taus="all"`` (pass
          ``taus="octave"`` in `kwargs` for the faster, sparser variant
          used by default in ``kind="overview"``).
        - ``"spectrum"`` — PSD/ASD of a single column, computed via
          ``compute_psd`` and drawn via ``plot_psd``; `quantity` selects
          the column.
        - ``"hist"`` — Histogram of a single column's distribution, drawn
          via ``plot_histogram``; `quantity` selects the column
          (frequency, power, or wavelength).
    quantity : {"frequency", "power", "wavelength"}, default "frequency"
        Column to use for ``kind="adev"``/``kind="spectrum"`` (frequency
        or power only) or ``kind="hist"`` (frequency, power, or
        wavelength). Ignored for the other kinds.
    save : str or pathlib.Path, optional
        If given, the resulting figure is saved as a 300 dpi PNG. Treated
        as a name rather than a full path: if it has no file suffix,
        ``.png`` is appended. Parent directories are created as needed, so
        e.g. ``save="figs/run1"`` saves to ``figs/run1.png``.
    cleanup : bool, default False
        Passed to ``load_lta_file`` when `data` is a path. Ignored when
        `data` is already a DataFrame.
    **kwargs
        Forwarded to the underlying plotting function for the chosen
        `kind` (e.g. `freq_unit`, `errorbars`, `lines`, `scaling`, `taus`,
        `unit`, `title`, `ax`, `capsize`, `errorbar_color`, `regions`,
        `region_agg`, `label`, ...). For ``kind="adev"``/``kind="overview"``,
        `regions`/`region_agg` add a short/mid/long-term (or custom)
        τ-region summary to the ADEV panel(s) — see `plot_adev`. `label`
        stamps a large bold scenario label across the top of the figure
        — see any of the underlying plotting functions.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `kind` is not one of the supported values.

    Examples
    --------
    >>> plot("scan.lta", kind="overview", save="overview")
    >>> df = load_lta_file("scan.lta")
    >>> plot(df, kind="adev", quantity="power")
    >>> plot(df, kind="psd", scaling="asd", save="figs/asd")
    >>> plot(df, kind="adev", regions=True)  # short/mid/long-term summary
    >>> plot(df, kind="hist", quantity="wavelength", bins=100)
    >>> plot(df, kind="adev", quantity="power", label="Scenario A", save="fig/adev_A")
    """
    df = data if isinstance(data, pd.DataFrame) else load_lta_file(data, cleanup=cleanup)

    save_path = None
    if save is not None:
        save_path = Path(save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")

    if kind == "overview":
        overview_figure(df, save=save_path, **kwargs)
    elif kind == "psd":
        psd_figure(df, save=save_path, **kwargs)
    elif kind == "timeseries":
        ts_kind = kwargs.pop("ts_kind", "freq")
        plot_timeseries(df, kind=ts_kind, save=save_path, **kwargs)
    elif kind == "adev":
        if quantity not in _ADEV_QUANTITY_DEFAULTS:
            raise ValueError(f"Unknown quantity {quantity!r}; expected 'frequency' or 'power'")
        defaults = _ADEV_QUANTITY_DEFAULTS[quantity]
        taus = kwargs.pop("taus", "all")
        unit = kwargs.pop("unit", defaults["unit"])
        title = kwargs.pop("title", defaults["title"])
        tau, dev, dev_err, _ = compute_oadev(df[defaults["column"]], time_s=df["time_s"], taus=taus)
        plot_adev(tau, dev, dev_err, unit=unit, quantity=quantity, title=title, save=save_path, **kwargs)
    elif kind == "spectrum":
        if quantity not in _SPECTRUM_QUANTITY_COLUMNS:
            raise ValueError(f"Unknown quantity {quantity!r}; expected 'frequency' or 'power'")
        column = _SPECTRUM_QUANTITY_COLUMNS[quantity]
        series = df[column] * 1e12 if quantity == "frequency" else df[column]
        nperseg = kwargs.pop("nperseg", None)
        ci = kwargs.pop("ci", None)
        f, Pxx, *rest = compute_psd(series, time_s=df["time_s"], ci=ci, nperseg=nperseg)
        bounds = rest[0] if rest else None
        plot_psd(f, Pxx, ci_bounds=bounds, quantity=quantity, save=save_path, **kwargs)
    elif kind == "hist":
        plot_histogram(df, quantity=quantity, save=save_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown kind {kind!r}; expected one of "
            "'overview', 'psd', 'timeseries', 'adev', 'spectrum', 'hist'"
        )

    return None
