"""Pure computation: Allan deviation, PSD and stable-segment extraction (no plotting)."""

from __future__ import annotations

import contextlib
import os
import warnings

import numpy as np
import pandas as pd
import allantools as at
from scipy.signal import welch
from scipy.stats import chi2

# Greenhall confidence intervals need allantools' ``ci`` submodule. Its API has
# moved between releases, so probe (never raise) at import time; ``compute_oadev``
# raises a clear ImportError only when ``ci`` is actually requested.
try:
    _HAS_GREENHALL_CI = all(
        hasattr(at.ci, _name)
        for _name in ("edf_greenhall", "confidence_interval", "autocorr_noise_id")
    )
except AttributeError:
    _HAS_GREENHALL_CI = False


def _estimate_rate(time_s):
    """Estimate a sampling rate in Hz from timestamps in seconds.

    Uses ``1 / median(positive time differences)`` — robust against gaps.
    """
    dt = pd.Series(time_s).diff().dropna()
    dt = dt[dt > 0]
    return 1.0 / np.median(dt)


def _greenhall_ci(data, tau, dev, *, rate, data_type, ci, d=2, overlapping=True):
    """Per-tau Greenhall confidence interval for an (O)ADEV estimate.

    Returns ``(ci_lo, ci_hi, diag)``. ``ci_lo``/``ci_hi`` are **absolute
    deviation values** (same native unit as ``dev``), not half-widths. Points
    for which no interval can be computed are NaN in both bounds and flagged in
    ``diag["skipped"]``.

    ``d=2`` selects ADEV (``d=3`` would be Hadamard). ``modified=False`` because
    ``compute_oadev`` computes ``oadev`` (overlapping), not ``omdev``.

    Parameters
    ----------
    data, tau, dev : array_like
        The series and the per-tau ADEV result from ``at.oadev``.
    rate : float
        Sampling rate in Hz (used to recover the averaging factor
        ``af = round(tau_i * rate)``).
    data_type : str
        ``"freq"`` or ``"phase"``; forwarded to ``autocorr_noise_id``.
    ci : float
        Two-sided coverage, e.g. ``0.6826894921370859`` for 1 sigma.
    d : int, default 2
        Difference order for ``edf_greenhall`` (2 = ADEV).
    overlapping : bool, default True
        Must be True — ``compute_oadev`` uses the overlapping estimator.
    """
    # Phase-0 pairing requirement: compute_oadev uses at.oadev (overlapping),
    # so the Greenhall EDF must be computed with overlapping=True.
    assert overlapping, "Greenhall EDF must use overlapping=True to match at.oadev"

    data = np.asarray(data, dtype=float)
    tau = np.asarray(tau, dtype=float)
    dev = np.asarray(dev, dtype=float)
    N = len(data)
    n = len(tau)

    ci_lo = np.full(n, np.nan)
    ci_hi = np.full(n, np.nan)
    edf_arr = np.full(n, np.nan)
    alpha_used = np.full(n, np.nan)
    alpha_raw = np.full(n, np.nan)
    clamped = np.zeros(n, dtype=bool)
    skipped = np.zeros(n, dtype=bool)

    last_alpha = None
    # allantools' ci functions print to stdout on the short-series case; keep
    # that chatter out of the caller's stdout (notebooks, capsys) — the
    # summarising warnings.warn below still reports what happened.
    with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull):
        for i in range(n):
            af = int(round(tau[i] * rate))
            if af < 1:
                af = 1

            # Guard C (part 1): too few independent estimates -> skip.
            if N - 2 * af < 8:
                skipped[i] = True
                continue

            # Step 1 + Guard A: noise identification on the decimated series.
            try:
                alpha_int, _alpha_f, _d_est, _rho = at.ci.autocorr_noise_id(
                    data, af, data_type=data_type, dmin=0, dmax=2
                )
                alpha_i = int(alpha_int)
                alpha_raw[i] = alpha_i
                last_alpha = alpha_i
            except NotImplementedError:
                # Fallback: reuse the last successfully identified alpha (noise
                # type is usually stable across neighbouring tau); only assume
                # white FM (0) if none exists. Record alpha_raw as NaN to show
                # it was imputed.
                alpha_i = last_alpha if last_alpha is not None else 0
                alpha_raw[i] = np.nan

            # Guard B: clamp alpha into [-2, 2].
            alpha_c = min(2, max(-2, alpha_i))
            if alpha_c != alpha_i:
                clamped[i] = True
            alpha_used[i] = alpha_c

            # Step 5: EDF. edf_greenhall can still raise at the clamp boundary
            # for exotic cases -> treat as a skip rather than propagate.
            try:
                edf = at.ci.edf_greenhall(
                    alpha=alpha_c, d=d, m=af, N=N,
                    overlapping=overlapping, modified=False,
                )
            except (AssertionError, NotImplementedError):
                skipped[i] = True
                continue

            # Guard C (part 2): degenerate EDF -> skip.
            if not np.isfinite(edf) or edf < 1:
                skipped[i] = True
                continue

            edf_arr[i] = edf
            lo, hi = at.ci.confidence_interval(dev=float(dev[i]), edf=edf, ci=ci)
            ci_lo[i] = lo
            ci_hi[i] = hi

    # One summarising warning per call (not one per tau).
    n_clamped = int(np.sum(clamped))
    n_skipped = int(np.sum(skipped))
    if n_clamped or n_skipped:
        warnings.warn(
            f"Greenhall CI: {n_clamped} tau point(s) had the noise-type exponent "
            f"clamped into [-2, 2]; {n_skipped} point(s) were skipped "
            "(no interval computable; ci_lo/ci_hi are NaN there).",
            stacklevel=2,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff_octaves = (
            float(np.log2(np.max(tau) / np.min(tau)))
            if n > 1 and np.min(tau) > 0
            else 0.0
        )

    diag = {
        "edf": edf_arr,
        "alpha": alpha_used,
        "alpha_raw": alpha_raw,
        "clamped": clamped,
        "skipped": skipped,
        "n_eff_octaves": n_eff_octaves,
    }
    return ci_lo, ci_hi, diag


def compute_oadev(
    data, *, rate=None, time_s=None, data_type="freq", taus="all",
    ci=None, ci_diagnostics=False,
):
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
        Passed through to ``allantools.oadev`` (and to the noise-type
        identification when `ci` is requested).
    taus : str or array_like, default "all"
        Averaging times, passed through to ``allantools.oadev``.
    ci : float, optional
        If given (e.g. ``0.6826894921370859`` for 1 sigma), also compute a
        per-tau Greenhall confidence interval and return it as one extra
        element. ``None`` (the default) leaves the return value a 4-tuple,
        unchanged. The interval uses the Greenhall equivalent-degrees-of-freedom
        with the noise type identified per tau via lag-1 autocorrelation
        (``d=2``, ``overlapping=True``). Two guards keep it robust: the
        noise-type exponent is clamped into ``[-2, 2]``, and points with too few
        independent estimates (short decimated series) are skipped with NaN
        bounds. Requires allantools' ``ci`` submodule; a clear ``ImportError`` is
        raised only if it is missing and `ci` is requested.
    ci_diagnostics : bool, default False
        Only meaningful together with `ci`. If True, also return a diagnostics
        dict (see Returns) for the thesis methods section and debugging.

    Returns
    -------
    tau : numpy.ndarray
        Averaging times in seconds.
    dev : numpy.ndarray
        Overlap Allan deviation values.
    dev_err : numpy.ndarray
        Naive error of `dev`, as returned directly by ``allantools.oadev``
        (``dev / sqrt(n)``). It does not account for the actual noise type or
        the correlation introduced by overlapping samples; use `ci` for a
        noise-aware interval.
    n : numpy.ndarray
        Number of pairs used at each `tau`.
    (ci_lo, ci_hi) : tuple of numpy.ndarray, only if `ci` is given
        Absolute lower/upper deviation bounds (same native unit as `dev`), NaN
        where no interval could be computed.
    diag : dict, only if `ci` is given and `ci_diagnostics` is True
        Keys ``"edf"``, ``"alpha"`` (clamped integer exponent used),
        ``"alpha_raw"`` (raw ``autocorr_noise_id`` output, NaN where imputed),
        ``"clamped"`` (bool), ``"skipped"`` (bool), ``"n_eff_octaves"``
        (``log2(tau_max/tau_min)``).

    Raises
    ------
    ValueError
        If neither `rate` nor `time_s` is given.
    ImportError
        If `ci` is requested but allantools' ``ci`` submodule is unavailable.
    """
    if rate is None:
        if time_s is None:
            raise ValueError("Either 'rate' or 'time_s' must be given.")
        rate = _estimate_rate(time_s)

    data = np.asarray(data)
    tau, dev, dev_err, n = at.oadev(data, rate=rate, data_type=data_type, taus=taus)

    if ci is None:
        return tau, dev, dev_err, n

    if not _HAS_GREENHALL_CI:
        raise ImportError(
            "Greenhall confidence intervals require allantools' 'ci' submodule "
            "(edf_greenhall, confidence_interval, autocorr_noise_id), which the "
            "installed allantools does not provide. Upgrade to allantools>=2024.6."
        )

    ci_lo, ci_hi, diag = _greenhall_ci(
        data, tau, dev, rate=rate, data_type=data_type, ci=ci, d=2, overlapping=True,
    )
    if ci_diagnostics:
        return tau, dev, dev_err, n, (ci_lo, ci_hi), diag
    return tau, dev, dev_err, n, (ci_lo, ci_hi)


DEFAULT_ADEV_REGION_BOUNDARIES = (0.25, 2.0)


def summarize_adev_regions(
    tau, dev, dev_err, boundaries=DEFAULT_ADEV_REGION_BOUNDARIES, agg="mean",
    *, ci_bounds=None, error_model=None,
):
    """Aggregate Allan deviation values into τ regions (e.g. short/mid/long term).

    Splits `tau` into ``len(boundaries) + 1`` half-open regions
    ``[0, b_0), [b_0, b_1), ..., [b_n, inf)`` and reduces the `dev` values
    falling into each region to a single value plus an error estimate.

    Legacy error (``error`` key)
    ----------------------------
    The legacy `error` propagates the individual `dev_err` values of a region
    through quadrature, ``sqrt(sum(dev_err_i**2)) / n``. This **understates** the
    true uncertainty: overlapping ADEV points at neighbouring τ are nearly
    perfectly correlated, so their errors do not average down as ``1/sqrt(n)``.
    It is retained unchanged for backward compatibility and is still populated
    even when `ci_bounds` is supplied. For a corrected estimate pass
    `ci_bounds` (see below) with ``error_model="correlated"``.

    The still-valid reason the raw **spread** of `dev` across a region is *not*
    used as an error bar: the Allan deviation typically has a real trend across a
    region (e.g. it falls as ``tau**-0.5`` for white FM), so that spread partly
    reflects the trend, not measurement uncertainty. This propagated/aggregated
    error avoids that conflation, for both ``agg="mean"`` and ``agg="median"``
    (for the median it is an approximation).

    Greenhall error (``error_lo``/``error_hi`` keys, opt-in via `ci_bounds`)
    -----------------------------------------------------------------------
    When `ci_bounds=(ci_lo, ci_hi)` is given, each region dict additionally gets
    asymmetric `error_lo`/`error_hi`, `n_ci`, `dev_min`, `dev_max`. Per point the
    absolute bounds are converted to half-widths ``half_lo = dev - ci_lo`` and
    ``half_hi = ci_hi - dev`` (clipped at 0) *before* aggregating. With
    ``error_model="correlated"`` (the default when `ci_bounds` is given) the
    standard deviation of the region mean is taken as the *mean of the individual
    half-widths* — no ``1/sqrt(n)`` — because overlapping estimators are
    correlated. ``error_model="independent"`` instead divides by
    ``sqrt(n_eff)`` with ``n_eff = log2(tau_max/tau_min)`` (octaves spanned, not
    the point count). NaN bounds (skipped points, clustering at large τ) are
    excluded via ``nanmean``; `n_ci` counts the finite points separately from
    `n`, and if `n_ci == 0` the region's `error_lo`/`error_hi` are NaN (never a
    silent fall-back to the quadrature error). `value` is never recomputed over
    the reduced point set, so the headline number stays comparable.

    Caveat: a Greenhall interval describes a single-τ ADEV estimator. The mean
    over a region is a derived quantity, so its interval is a plausibility band
    rather than a rigorous CI.

    Parameters
    ----------
    tau : array_like
        Averaging times in seconds (e.g. from `compute_oadev`).
    dev : array_like
        Allan deviation values, same length as `tau`.
    dev_err : array_like or None
        Per-tau naive error of `dev` (e.g. `compute_oadev`'s `dev_err`), same
        length as `tau`, used for the legacy quadrature `error`. If None, the
        legacy `error` is NaN (only sensible together with `ci_bounds`).
    boundaries : sequence of float, default (0.25, 2.0)
        τ boundaries in seconds (need not be pre-sorted). ``k`` boundaries
        produce ``k + 1`` regions.
    agg : {"mean", "median"}, default "mean"
        Aggregation statistic applied to `dev` within each region.
    ci_bounds : tuple of array_like, optional
        ``(ci_lo, ci_hi)`` absolute deviation bounds (same unit/length as `dev`,
        e.g. from ``compute_oadev(..., ci=...)``). Enables the additive Greenhall
        keys. NaN entries are treated as skipped points.
    error_model : {"correlated", "independent"}, optional
        Only used when `ci_bounds` is given; defaults to ``"correlated"``.
        Ignored (not validated) otherwise.

    Returns
    -------
    list of dict
        One dict per non-empty region, ascending τ order. Always contains
        ``tau_min``, ``tau_max`` (``inf`` for the last region), ``value``,
        ``error``, ``n``. When `ci_bounds` is given, each dict additionally
        contains ``error_lo``, ``error_hi``, ``n_ci``, ``dev_min``, ``dev_max``.

    Raises
    ------
    ValueError
        If `agg` is not ``"mean"``/``"median"``, if any `boundaries` entry is not
        positive, or if `error_model` (when `ci_bounds` is given) is unknown.

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
    dev_err = None if dev_err is None else np.asarray(dev_err)

    if ci_bounds is not None:
        ci_lo = np.asarray(ci_bounds[0], dtype=float)
        ci_hi = np.asarray(ci_bounds[1], dtype=float)
        # Pitfall 1: convert absolute sigma bounds to half-widths *before*
        # aggregating; clip at 0 to absorb floating-point noise.
        half_lo = np.clip(dev - ci_lo, 0, None)
        half_hi = np.clip(ci_hi - dev, 0, None)
        if error_model is None:
            error_model = "correlated"
        if error_model not in ("correlated", "independent"):
            raise ValueError(
                f"Unknown error_model {error_model!r}; expected 'correlated' or 'independent'"
            )

    boundaries = sorted(boundaries)
    if any(b <= 0 for b in boundaries):
        raise ValueError("boundaries must be positive")

    edges = [0.0, *boundaries, np.inf]
    agg_func = np.mean if agg == "mean" else np.median

    regions = []
    empty_ci_regions = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (tau >= lo) & (tau < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        value = float(agg_func(dev[mask]))
        if dev_err is not None:
            error = float(np.sqrt(np.sum(dev_err[mask] ** 2)) / n)
        else:
            error = float("nan")
        region = {"tau_min": lo, "tau_max": hi, "value": value, "error": error, "n": n}

        if ci_bounds is not None:
            hlo = half_lo[mask]
            hhi = half_hi[mask]
            finite = np.isfinite(hlo) & np.isfinite(hhi)
            n_ci = int(np.sum(finite))
            if n_ci == 0:
                error_lo = float("nan")
                error_hi = float("nan")
                empty_ci_regions += 1
            else:
                # Pitfall 2: overlapping ADEV points are ~fully correlated, so
                # the SD of their mean is the mean of the SDs (no 1/sqrt(n)).
                mean_lo = float(np.nanmean(hlo))
                mean_hi = float(np.nanmean(hhi))
                if error_model == "correlated":
                    error_lo = mean_lo
                    error_hi = mean_hi
                else:  # "independent"
                    tau_region = tau[mask].astype(float)
                    pos = tau_region[tau_region > 0]
                    if pos.size >= 2:
                        n_eff = float(np.log2(np.max(pos) / np.min(pos)))
                    else:
                        n_eff = 1.0
                    n_eff = max(n_eff, 1.0)
                    error_lo = mean_lo / np.sqrt(n_eff)
                    error_hi = mean_hi / np.sqrt(n_eff)
            region["error_lo"] = error_lo
            region["error_hi"] = error_hi
            region["n_ci"] = n_ci
            # Free bonus: actual span of dev across the region (no assumptions).
            region["dev_min"] = float(np.min(dev[mask]))
            region["dev_max"] = float(np.max(dev[mask]))

        regions.append(region)

    if ci_bounds is not None and empty_ci_regions:
        warnings.warn(
            f"Greenhall CI region error: {empty_ci_regions} region(s) had no finite "
            "CI bounds (all points skipped); their error_lo/error_hi are NaN while "
            "value/error stay from the full point set.",
            stacklevel=2,
        )

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
