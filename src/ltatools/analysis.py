"""Pure computation: Allan deviation, PSD and stable-segment extraction (no plotting)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import allantools as at
from scipy.signal import welch
from scipy.stats import chi2


def _estimate_rate(time_s):
    """Estimate a sampling rate in Hz from timestamps in seconds.

    Uses ``1 / median(positive time differences)`` — robust against gaps.
    """
    dt = pd.Series(time_s).diff().dropna()
    dt = dt[dt > 0]
    return 1.0 / np.median(dt)


def compute_oadev(data, *, rate=None, time_s=None, data_type="freq", taus="all"):
    """Compute the overlap Allan deviation (OADEV) of a data series.

    Parameters
    ----------
    data : array_like
        Frequency or power samples.
    rate : float, optional
        Sampling rate in Hz. If omitted, estimated from `time_s`.
    time_s : array_like, optional
        Timestamps in seconds, used to estimate `rate` when it is not given.
        The rate is ``1 / median(positive time differences)``.
    data_type : str, default "freq"
        Passed through to ``allantools.oadev``.
    taus : str or array_like, default "all"
        Averaging times, passed through to ``allantools.oadev``.

    Returns
    -------
    tau : numpy.ndarray
        Averaging times in seconds.
    dev : numpy.ndarray
        Overlap Allan deviation values.
    dev_err : numpy.ndarray
        Error of `dev`, as returned directly by ``allantools.oadev``
        (``dev / sqrt(n)``). This is the only error estimate this function
        provides; it does not account for the actual noise type or the
        correlation introduced by overlapping samples.
    n : numpy.ndarray
        Number of pairs used at each `tau`.

    Raises
    ------
    ValueError
        If neither `rate` nor `time_s` is given.
    """
    if rate is None:
        if time_s is None:
            raise ValueError("Either 'rate' or 'time_s' must be given.")
        rate = _estimate_rate(time_s)

    data = np.asarray(data)
    tau, dev, dev_err, n = at.oadev(data, rate=rate, data_type=data_type, taus=taus)

    return tau, dev, dev_err, n


DEFAULT_ADEV_REGION_BOUNDARIES = (0.25, 2.0)


def summarize_adev_regions(tau, dev, dev_err, boundaries=DEFAULT_ADEV_REGION_BOUNDARIES, agg="mean"):
    """Aggregate Allan deviation values into τ regions (e.g. short/mid/long term).

    Splits `tau` into ``len(boundaries) + 1`` half-open regions
    ``[0, b_0), [b_0, b_1), ..., [b_n, inf)`` and reduces the `dev` values
    falling into each region to a single value plus an error estimate.

    The error is computed by propagating the individual `dev_err` values
    of a region through quadrature, ``sqrt(sum(dev_err_i**2)) / n``,
    rather than the sample standard deviation of the `dev` values in the
    region. The Allan deviation typically has a real trend across a
    region (e.g. it falls as ``tau**-0.5`` for white frequency noise), so
    the spread of `dev` values there partly reflects that trend, not
    measurement uncertainty — using it as an error bar would overstate
    the true uncertainty. Propagating the already-computed per-tau
    `dev_err` avoids that conflation. This propagated value is used as
    the error estimate for both ``agg="mean"`` and ``agg="median"``; for
    the median it is an approximation (the formally correct median error
    would require e.g. a bootstrap estimate), but is used here for
    simplicity.

    Parameters
    ----------
    tau : array_like
        Averaging times in seconds (e.g. from `compute_oadev`).
    dev : array_like
        Allan deviation values, same length as `tau`.
    dev_err : array_like
        Per-tau error of `dev` (e.g. `compute_oadev`'s `dev_err`), same
        length as `tau`. Used for the error propagation described above.
    boundaries : sequence of float, default (0.25, 2.0)
        τ boundaries in seconds (need not be pre-sorted). ``k`` boundaries
        produce ``k + 1`` regions.
    agg : {"mean", "median"}, default "mean"
        Aggregation statistic applied to `dev` within each region.

    Returns
    -------
    list of dict
        One dict per non-empty region, in ascending τ order, with keys
        ``tau_min``, ``tau_max`` (floats in seconds; ``tau_max`` is
        ``inf`` for the last region), ``value`` (the aggregated `dev`),
        ``error`` (the propagated error of `value`), and ``n`` (number of
        points in the region). Regions with no points are omitted.

    Raises
    ------
    ValueError
        If `agg` is not ``"mean"``/``"median"``, or if any `boundaries`
        entry is not positive.

    Examples
    --------
    >>> tau, dev, dev_err, _ = compute_oadev(df["frequency_THz"], time_s=df["time_s"])
    >>> summarize_adev_regions(tau, dev, dev_err)
    [{'tau_min': 0.0, 'tau_max': 0.25, 'value': 9.370167593610577e-08, 'error': 9.825484202459966e-10, 'n': 24}, {'tau_min': 0.25, 'tau_max': 2.0, 'value': 2.5384383336447446e-08, 'error': 1.18386558723204e-10, 'n': 175}, {'tau_min': 2.0, 'tau_max': inf, 'value': 9.804624969733248e-09, 'error': 2.463951327085816e-10, 'n': 50}]
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"Unknown agg {agg!r}; expected 'mean' or 'median'")

    tau = np.asarray(tau)
    dev = np.asarray(dev)
    dev_err = np.asarray(dev_err)

    boundaries = sorted(boundaries)
    if any(b <= 0 for b in boundaries):
        raise ValueError("boundaries must be positive")

    edges = [0.0, *boundaries, np.inf]
    agg_func = np.mean if agg == "mean" else np.median

    regions = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (tau >= lo) & (tau < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        value = float(agg_func(dev[mask]))
        error = float(np.sqrt(np.sum(dev_err[mask] ** 2)) / n)
        regions.append({"tau_min": lo, "tau_max": hi, "value": value, "error": error, "n": n})

    return regions


def compute_psd(data, *, rate=None, time_s=None, ci=None, nperseg=None, resample=True, jitter_tol=0.01):
    """Compute the one-sided Welch power spectral density of a data series.

    Parameters
    ----------
    data : array_like
        Samples in their native unit (e.g. Hz for frequency, uW for power).
        The result is in ``[unit]**2 / Hz``; conversion to an amplitude
        spectral density (``sqrt(PSD)``) happens at the plotting layer.
    rate : float, optional
        Sampling rate in Hz. If omitted, estimated from `time_s`.
    time_s : array_like, optional
        Timestamps in seconds, used to estimate `rate` and to detect
        irregular sampling.
    ci : float, optional
        If given, also compute a chi-squared confidence band (e.g.
        ``0.95``) using the equivalent degrees of freedom of a Welch
        estimate with a Hann window and 50% overlap,
        ``nu = 36 * K**2 / (19 * K - 1)`` for `K` segments [1]_.
    nperseg : int, optional
        Segment length passed to ``scipy.signal.welch``. Defaults to
        ``min(len(data), max(256, len(data) // 8))``.
    resample : bool, default True
        Welch's method assumes uniform sampling. If the timestamps in
        `time_s` are irregular beyond `jitter_tol`, a warning is always
        raised; if `resample` is True, `data` is additionally linearly
        interpolated onto a uniform grid with spacing
        ``median(positive time differences)`` before computing the PSD.
    jitter_tol : float, default 0.01
        Relative tolerance on ``max|dt - median(dt)| / median(dt)`` before
        the sampling is considered irregular.

    Returns
    -------
    f : numpy.ndarray
        Frequency bins in Hz.
    Pxx : numpy.ndarray
        One-sided power spectral density.
    ci_bounds : tuple of numpy.ndarray, only if `ci` is given
        ``(lower, upper)`` chi-squared confidence bounds for `Pxx`.

    Raises
    ------
    ValueError
        If neither `rate` nor `time_s` is given.

    References
    ----------
    .. [1] D. B. Percival & A. T. Walden, "Spectral Analysis for Physical
       Applications", Cambridge University Press (1993), chapter 6.
    """
    data = np.asarray(data, dtype=float)

    if rate is None:
        if time_s is None:
            raise ValueError("Either 'rate' or 'time_s' must be given.")
        rate = _estimate_rate(time_s)

    if time_s is not None:
        time_s = np.asarray(time_s, dtype=float)
        dt = np.diff(time_s)
        dt_pos = dt[dt > 0]
        dt_med = np.median(dt_pos)
        jitter = np.max(np.abs(dt - dt_med)) / dt_med
        if jitter > jitter_tol:
            warnings.warn(
                f"Timestamps are irregularly spaced (max relative jitter "
                f"{jitter:.2%} exceeds tolerance {jitter_tol:.2%}); Welch's "
                "method assumes uniform sampling.",
                stacklevel=2,
            )
            if resample:
                t_uniform = np.arange(time_s[0], time_s[-1], dt_med)
                data = np.interp(t_uniform, time_s, data)

    if nperseg is None:
        nperseg = min(len(data), max(256, len(data) // 8))

    f, Pxx = welch(data, fs=rate, window="hann", detrend="constant", nperseg=nperseg)

    if ci is None:
        return f, Pxx

    noverlap = nperseg // 2
    step = nperseg - noverlap
    K = max(1, 1 + (len(data) - nperseg) // step)
    nu = 36 * K**2 / (19 * K - 1)
    lower = nu * Pxx / chi2.ppf((1 + ci) / 2, nu)
    upper = nu * Pxx / chi2.ppf((1 - ci) / 2, nu)

    return f, Pxx, (lower, upper)


def find_stable_segments(
    df: pd.DataFrame,
    *,
    freq_col: str = "frequency_THz",
    n: int = 2,
    threshold: float | None = None,
    jump_factor: float = 20,
    trim_left: int = 2,
    trim_right: int = 0,
):
    """Return the `n` longest mode-hop-free segments of `df`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame as returned by ``load_lta_file``.
    freq_col : str, default "frequency_THz"
        Column to detect jumps on.
    n : int, default 2
        Number of longest segments to return.
    threshold : float, optional
        Jump size in `freq_col` units. Auto-detected if omitted, as
        ``jump_factor * median(positive steps)``.
    jump_factor : float, default 20
        Multiplier on the median step size for the automatic threshold.
    trim_left : int, default 2
        Number of points to drop from the start of each segment.
    trim_right : int, default 0
        Number of points to drop from the end of each segment.

    Returns
    -------
    list of pandas.DataFrame
        Longest segment first. Row index and both ``time_ms``/``time_s``
        are reset to start at 0 within each segment.
    """
    diffs = df[freq_col].diff().abs()

    if threshold is None:
        typical = diffs[diffs > 0].median()
        threshold = jump_factor * typical

    segment_id = (diffs > threshold).cumsum()

    segments = []
    for _, grp in df.groupby(segment_id):
        end = len(grp) - trim_right if trim_right > 0 else len(grp)
        grp = grp.iloc[trim_left:end]
        if len(grp) == 0:
            continue
        grp = grp.reset_index(drop=True)
        for time_col in ("time_ms", "time_s"):
            if time_col in grp.columns:
                grp = grp.assign(**{time_col: grp[time_col] - grp[time_col].iloc[0]})
        segments.append(grp)

    segments.sort(key=len, reverse=True)
    return segments[:n]
