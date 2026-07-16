"""Central color scheme and unit scaling/labeling helpers for ltatools plots."""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors

# Slightly larger default text across every ltatools plot (titles, axis
# labels, ticks, legend). Applied globally at import time since font size
# isn't threaded through every plotting function individually.
mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

COLORS = {
    "frequency": "tab:blue",
    "wavelength": "tab:blue",  # same visual identity as frequency
    "power": "tab:orange",
}


def darken_color(color, factor=0.7):
    """Return a darker variant of `color`.

    Parameters
    ----------
    color : str or tuple
        Any matplotlib-recognized color (e.g. ``"tab:blue"``).
    factor : float, default 0.7
        Multiplier applied to each RGB channel; smaller values are darker.

    Returns
    -------
    tuple of float
        RGB triple in ``[0, 1]``.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)

_FREQUENCY_FACTORS = {
    "THz": 1.0,
    "GHz": 1e3,
    "MHz": 1e6,
    "kHz": 1e9,
    "Hz": 1e12,
}

_POWER_FACTORS = {
    "uW": 1.0,
    "µW": 1.0,
    "mW": 1e-3,
    "W": 1e-6,
}

_QUANTITY_NAMES = {
    "frequency": "Frequency",
    "wavelength": "Wavelength",
    "power": "Power",
}

# Coarse -> fine, so index + 1 is always "one step finer".
_FREQUENCY_UNIT_ORDER = ["THz", "GHz", "MHz", "kHz", "Hz"]
_POWER_UNIT_ORDER = ["W", "mW", "uW"]

_UNIT_ORDERS = {
    "frequency": _FREQUENCY_UNIT_ORDER,
    "power": _POWER_UNIT_ORDER,
}


def finer_unit(unit, quantity):
    """Return the next-finer unit than `unit` for `quantity`.

    E.g. ``finer_unit("MHz", "frequency") == "kHz"``. Returns `unit`
    unchanged if it is already the finest defined unit for `quantity`
    (e.g. ``"uW"`` for power), or if `quantity` has no defined unit
    ordering (e.g. ``"wavelength"``, which only ever uses ``"nm"``).

    Parameters
    ----------
    unit : str
        A unit recognized by `scale_frequency`/`scale_power`.
    quantity : {"frequency", "wavelength", "power"}
        Physical quantity `unit` belongs to.

    Returns
    -------
    str
    """
    order = _UNIT_ORDERS.get(quantity)
    if order is None:
        return unit
    normalized = "uW" if unit == "µW" else unit
    if normalized not in order:
        return unit
    idx = order.index(normalized)
    if idx + 1 >= len(order):
        return unit
    return order[idx + 1]


def scale_frequency(values_THz, unit="THz"):
    """Scale frequency values given in THz to another frequency unit.

    Parameters
    ----------
    values_THz : array_like
        Frequency values in THz.
    unit : str, default "THz"
        Target unit, one of ``{"THz", "GHz", "MHz", "kHz", "Hz"}``.

    Returns
    -------
    numpy.ndarray
        Scaled frequency values.

    Raises
    ------
    ValueError
        If `unit` is not a recognized frequency unit.
    """
    if unit not in _FREQUENCY_FACTORS:
        raise ValueError(
            f"Unknown frequency unit {unit!r}; expected one of {sorted(_FREQUENCY_FACTORS)}"
        )
    return np.asarray(values_THz) * _FREQUENCY_FACTORS[unit]


def scale_power(values_uW, unit="uW"):
    """Scale power values given in microwatts to another power unit.

    Parameters
    ----------
    values_uW : array_like
        Power values in microwatts.
    unit : str, default "uW"
        Target unit, one of ``{"uW", "µW", "mW", "W"}``.

    Returns
    -------
    numpy.ndarray
        Scaled power values.

    Raises
    ------
    ValueError
        If `unit` is not a recognized power unit.
    """
    if unit not in _POWER_FACTORS:
        raise ValueError(
            f"Unknown power unit {unit!r}; expected one of {sorted(_POWER_FACTORS)}"
        )
    return np.asarray(values_uW) * _POWER_FACTORS[unit]


def axis_label(quantity, unit, delta=False):
    """Build an axis label for a quantity/unit pair.

    Parameters
    ----------
    quantity : {"frequency", "wavelength", "power"}
        Physical quantity being labeled.
    unit : str
        Unit string, e.g. ``"MHz"`` or ``"nm"``.
    delta : bool, default False
        If True, prefix the label with ``Δ`` to denote a deviation from a
        reference (e.g. ``"Δ Frequency (MHz)"``) rather than an absolute
        value. Used by `plotting.plot_timeseries` with ``relative=True``.

    Returns
    -------
    str
        Human-readable axis label, e.g. ``"Frequency (MHz)"``.

    Raises
    ------
    ValueError
        If `quantity` is not recognized.
    """
    if quantity not in _QUANTITY_NAMES:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(_QUANTITY_NAMES)}"
        )
    name = _QUANTITY_NAMES[quantity]
    if delta:
        return f"Δ {name} ({unit})"
    return f"{name} ({unit})"


def adev_label(quantity, unit):
    """Build a y-axis label for an Allan deviation plot.

    Parameters
    ----------
    quantity : {"frequency", "wavelength", "power"}
        Physical quantity the deviation was computed from.
    unit : str
        Unit the deviation values are expressed in.

    Returns
    -------
    str
        Label of the form ``r"$\\sigma(\\tau)$ in {unit}"``.

    Raises
    ------
    ValueError
        If `quantity` is not recognized.
    """
    if quantity not in _QUANTITY_NAMES:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(_QUANTITY_NAMES)}"
        )
    return rf"$\sigma(\tau)$ in {unit}"


def psd_label(quantity, unit, scaling="psd"):
    """Build a y-axis label for a power spectral density plot.

    Parameters
    ----------
    quantity : {"frequency", "wavelength", "power"}
        Physical quantity the spectrum was computed from.
    unit : str
        Unit of the underlying signal (e.g. ``"Hz"`` or ``"uW"``); the PSD
        label appends ``²/Hz``, the ASD label appends ``/√Hz``.
    scaling : {"psd", "asd"}, default "psd"
        Whether to label a power spectral density or an amplitude spectral
        density (``sqrt(PSD)``).

    Returns
    -------
    str
        Axis label, e.g. ``"PSD in Hz²/Hz"`` or ``"ASD in µW/√Hz"``.

    Raises
    ------
    ValueError
        If `quantity` or `scaling` is not recognized.
    """
    if quantity not in _QUANTITY_NAMES:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(_QUANTITY_NAMES)}"
        )
    if scaling == "psd":
        return rf"PSD in {unit}$^2$/Hz"
    if scaling == "asd":
        return rf"ASD in {unit}/$\sqrt{{\mathrm{{Hz}}}}$"
    raise ValueError(f"Unknown scaling {scaling!r}; expected 'psd' or 'asd'")
