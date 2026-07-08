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


def compute_oadev(data, *, rate=None, time_s=None, data_type="freq", taus="all", ci=None, alpha=None):
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
    ci : float, optional
        If given, also compute a chi-squared confidence interval (e.g.
        ``0.6827`` for a 1-sigma equivalent, or ``0.95``) for each `tau`,
        following the two-step method of Riley [1]_: (1) identify the
        dominant power-law noise type via ``allantools.autocorr_noise_id``
        (or fix it with `alpha`), (2) compute the equivalent degrees of
        freedom via ``allantools.edf_greenhall`` [2]_ and derive asymmetric
        bounds via ``allantools.confidence_interval``.
    alpha : int, optional
        Power-law noise exponent to use for every `tau` instead of
        estimating it per-tau (e.g. ``alpha=0`` for white FM, typical for a
        locked laser). Only used when `ci` is given. When left as `None`,
        the noise type is auto-identified per `tau` via
        ``allantools.autocorr_noise_id``, which needs at least 30 points
        after averaging by `tau`; at large `tau` (few points left) this
        falls back to ``alpha=0`` (white FM).

    Returns
    -------
    tau : numpy.ndarray
        Averaging times in seconds.
    dev : numpy.ndarray
        Overlap Allan deviation values.
    dev_err : numpy.ndarray
        Naive error of `dev` as returned by ``allantools`` (``dev / sqrt(n)``).
        This does **not** account for the actual noise type or the
        correlation introduced by overlapping samples and is **not** a
        statistically correct confidence interval — use `ci` for that.
    n : numpy.ndarray
        Number of pairs used at each `tau`.
    ci_bounds : tuple of numpy.ndarray, only if `ci` is given
        ``(lower, upper)`` chi-squared confidence bounds for `dev`.

    Raises
    ------
    ValueError
        If neither `rate` nor `time_s` is given.

    References
    ----------
    .. [1] W. J. Riley, "Handbook of Frequency Stability Analysis",
       NIST Special Publication 1065 (2008), chapter 5.
    .. [2] C. A. Greenhall & W. J. Riley, "Uncertainty of Stability
       Variances Based on Finite Differences", Proc. 35th Annual PTTI
       Systems and Applications Meeting (2003).
    """
    if rate is None:
        if time_s is None:
            raise ValueError("Either 'rate' or 'time_s' must be given.")
        rate = _estimate_rate(time_s)

    data = np.asarray(data)
    tau, dev, dev_err, n = at.oadev(data, rate=rate, data_type=data_type, taus=taus)

    if ci is None:
        return tau, dev, dev_err, n

    N = len(data)
    lower = np.empty_like(dev)
    upper = np.empty_like(dev)
    for i, (t, d) in enumerate(zip(tau, dev)):
        m = max(int(round(t * rate)), 1)
        a = alpha
        if a is None:
            try:
                a = at.autocorr_noise_id(data, m, data_type=data_type)[0]
            except NotImplementedError:
                # too few points left after averaging by m to identify the
                # noise type; assume white FM (alpha=0), the common case
                # for a locked laser
                a = 0
        edf = at.edf_greenhall(alpha=a, d=2, m=m, N=N, overlapping=True, modified=False)
        lower[i], upper[i] = at.confidence_interval(d, edf, ci=ci)

    return tau, dev, dev_err, n, (lower, upper)


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
