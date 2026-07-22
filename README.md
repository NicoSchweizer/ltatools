# ltatools

HighFinesse wavemeter `.lta` file analysis: loading, Allan deviation, power spectral density,
timeseries and overview plots.

## Install

```bash
pip install git+https://github.com/NicoSchweizer/ltatools.git
```

For local development (editable install, with `pytest`):

```bash
git clone https://github.com/NicoSchweizer/ltatools.git
cd ltatools
pip install -e ".[dev]"
```

## Quick start

`plot()` is the single entry point for building and (optionally) saving any plot — it accepts
either an already-loaded DataFrame or a raw `.lta` path, and always returns `None` so nothing is
echoed at the end of a notebook cell.

```python
from ltatools import plot

plot("data/2026-07-01_lock_test.lta", kind="overview", save="figures/lock_test_overview")
```

That's a timeseries panel (frequency + power) on top, frequency and power Allan deviation panels
below, saved as `figures/lock_test_overview.png`.

`data` can be a `.lta` path or an already-loaded `pandas.DataFrame` (auto-detected) — a path is
loaded internally via `load_lta_file`. Pass `cleanup=True` to drop invalid rows (same as
`load_lta_file`'s own `cleanup` parameter) when passing a path directly:

```python
plot("data/run.lta", kind="adev", quantity="power", cleanup=True)
```

`cleanup` is ignored if `data` is already a DataFrame — call `load_lta_file(..., cleanup=True)`
yourself beforehand in that case.

## `plot()` — all plot kinds

| `kind`        | Produces                                              | Notable kwargs                          |
|---------------|--------------------------------------------------------|------------------------------------------|
| `"overview"`  | timeseries + frequency/power ADEV (default)             | `freq_unit`, `power_unit`, `errorbars`, `regions`, `ci` |
| `"psd"`       | frequency + power PSD/ASD, side by side                 | `scaling="psd"\|"asd"`, `ci`            |
| `"timeseries"`| frequency-or-wavelength + power over time                | `freq_unit`, `power_unit`, `lines`      |
| `"adev"`      | Allan deviation of one column                            | `quantity="frequency"\|"power"`, `unit`, `regions`, `ci` |
| `"spectrum"`  | PSD/ASD of one column                                    | `quantity="frequency"\|"power"`, `scaling` |

```python
from ltatools import plot, load_lta_file

df = load_lta_file("data/run.lta", cleanup=True)

plot(df, kind="adev", quantity="power", unit="uW")
plot(df, kind="psd", scaling="asd", ci=0.95, save="figures/run_psd")
plot(df, kind="timeseries", freq_unit="MHz")
```

## Old measurement with mode hops

Split into the longest mode-hop-free segments first, then build an overview per segment:

```python
from ltatools import load_lta_file, find_stable_segments, plot

df = load_lta_file("data/2025-11-03_free_running.lta", cleanup=True)
for i, seg in enumerate(find_stable_segments(df, n=3)):
    plot(seg, kind="overview", freq_unit="MHz", save=f"figures/free_running_seg{i}")
```

Or in one call via `lta_overview`:

```python
from ltatools import lta_overview

results = lta_overview(
    "data/2025-11-03_free_running.lta", cleanup=True, segments=True, n_segments=3, freq_unit="MHz"
)
```

## Short/mid/long-term ADEV summary (`regions`)

`plot_adev`/`overview_figure`/`plot(..., kind="adev"|"overview")` can annotate the ADEV curve with
a mean or median per τ region, each with a propagated error — e.g. a "short/mid/long term
linewidth" summary as commonly shown in frequency-stability plots.

```python
from ltatools import load_lta_file, plot

df = load_lta_file("data/run.lta", cleanup=True)
plot(df, kind="adev", quantity="frequency", unit="MHz", regions=True)
```

`regions=True` uses the default boundaries (0.25 s, 2 s); pass a list of τ values in seconds (e.g.
`regions=[0.1, 10]`) for custom boundaries. `region_agg="mean"|"median"` selects the aggregation
(default `"mean"`). The annotated error is the **propagated** error of the region's `dev_err`
values (quadrature sum divided by the point count), not the standard deviation of the ADEV values
in the region — the ADEV typically has a real trend across a region, so its raw spread would
overstate the actual measurement uncertainty. The annotation is shown one unit step finer than the
axis unit (e.g. kHz on an MHz axis, see `ltatools.style.finer_unit`).

## Greenhall confidence intervals (`ci`)

By default the ADEV error bars use the naive `dev / sqrt(n)` error from `allantools`. Pass a
coverage probability as `ci` to get **noise-aware Greenhall confidence intervals** instead —
opt-in and fully backward-compatible (the default remains the legacy error).

```python
from ltatools import load_lta_file, plot

df = load_lta_file("data/run.lta", cleanup=True)
plot(df, kind="adev", quantity="frequency", unit="MHz", ci=0.6826894921370859)  # 1 sigma
plot(df, kind="overview", ci=0.6826894921370859, regions=True)
```

`ci=0.6826894921370859` is the 1σ (68.27 %) two-sided interval. The bars become asymmetric; points
where the interval can't be computed (short decimated series) simply draw no bar. When combined with
`regions`, the region annotation reports the correlated Greenhall error (`+a / -b` when asymmetric)
instead of the legacy propagated error.

Using `compute_oadev` directly, `ci` adds one element to the return tuple (and `ci_diagnostics=True`
a second) — the 4-tuple return is unchanged when `ci` is not set:

```python
from ltatools import compute_oadev

tau, dev, dev_err, n = compute_oadev(df["frequency_THz"], time_s=df["time_s"])            # 4-tuple
tau, dev, dev_err, n, (ci_lo, ci_hi) = compute_oadev(                                     # 5-tuple
    df["frequency_THz"], time_s=df["time_s"], ci=0.6826894921370859
)
```

`ci_lo`/`ci_hi` are absolute deviation values in the same unit as `dev` (not half-widths).

## Error bar styling

`plot_adev`/`overview_figure`/`plot()` support `errorbars=False` (hide them), `capsize` (end caps,
default `0` = none), and `errorbar_color` (default: a darkened version of the marker color).

```python
from ltatools import load_lta_file, plot

df = load_lta_file("data/run.lta", cleanup=True)
plot(df, kind="overview", errorbars=False)
plot(df, kind="adev", capsize=3)
```

## PSD/ASD with a confidence band

```python
from ltatools import load_lta_file, psd_figure

df = load_lta_file("data/run.lta", cleanup=True)
fig, axes = psd_figure(df, scaling="asd", ci=0.95, save="figures/run_psd.png")
```

`psd_figure` is intentionally **not** part of `overview_figure` — call it separately (or via
`plot(df, kind="psd")`) when a spectral view is needed. `compute_psd`/`plot_psd` are available
individually for custom axes, the same way `compute_oadev`/`plot_adev` are.

## Custom axes / composing your own figure

The building blocks (`plot_timeseries`, `plot_adev`, `plot_psd`) each take an `ax=` and a `save=`
and return the axes they drew into, for composing a custom `matplotlib` figure:

```python
import matplotlib.pyplot as plt
from ltatools import load_lta_file, compute_oadev, plot_adev

df = load_lta_file("data/run.lta", cleanup=True)
tau, dev, dev_err, n = compute_oadev(df["frequency_THz"], time_s=df["time_s"])

fig, ax = plt.subplots()
plot_adev(tau, dev, dev_err, unit="kHz", quantity="frequency", ax=ax, save="figures/adev.png")
```

## Notes

- All plotting functions except `plot()` return `fig`/`ax` (or a tuple of axes) and never call
  `plt.show()`; display happens in the caller (Jupyter renders figures automatically). `plot()`
  itself always returns `None`, specifically so it doesn't echo an `Axes` repr at the end of a
  notebook cell.
- `compute_oadev`'s `dev_err` is the naive error returned directly by `allantools.oadev`
  (`dev / sqrt(n)`) — it doesn't account for noise type or overlap correlation, and it is the
  default. For a noise-aware interval pass `ci` to `compute_oadev`/`plot(kind="adev"|"overview")`
  (see "Uncertainty model" below). `compute_psd`'s `ci` confidence band is a separate,
  chi-squared PSD band and is unrelated to the ADEV `ci`.
- `compute_psd` treats frequency and power in Hz and µW respectively; `psd_figure`/
  `plot(kind="spectrum", quantity="frequency")` do the THz→Hz conversion before calling it.
  `compute_psd` itself stays unit-agnostic, same as `compute_oadev`.
- Default text size across all plots is set via `matplotlib.rcParams` at import time
  (`ltatools.style`), slightly larger than matplotlib's defaults for readability.
- `overview_figure`/`plot(kind="overview")` default to `taus="octave"` (powers of two) — on a
  145k-row file this is ~450x faster than `"all"` with a visually indistinguishable ADEV curve on
  the usual log-log plot. `compute_oadev` directly and `plot(kind="adev")` still default to
  `taus="all"` (the exhaustive, slower computation); pass `taus="octave"` explicitly to speed
  either of those up too.

## Uncertainty model

Two error models are available for ADEV:

- **Legacy (default):** `compute_oadev`'s `dev_err = dev / sqrt(n)` from `allantools`, and
  `summarize_adev_regions`' region error `sqrt(sum(dev_err_i^2)) / n`. This is convenient but
  **understates** the uncertainty, because overlapping ADEV estimates at neighbouring τ are nearly
  perfectly correlated, so their errors do not average down as `1/sqrt(n)`.
- **Greenhall confidence intervals (opt-in via `ci`):** a per-τ interval built from the Greenhall
  equivalent degrees of freedom (`edf_greenhall`, `d=2` for ADEV, `overlapping=True`, `modified=False`),
  with the noise-type exponent α identified per τ from the lag-1 autocorrelation of the decimated
  series (`autocorr_noise_id`). α is clamped into `[-2, 2]`; τ points with too few independent
  estimates (short decimated series) are skipped (NaN bounds, no error bar).

  For region summaries, `summarize_adev_regions(..., ci_bounds=(ci_lo, ci_hi))` reports asymmetric
  `error_lo`/`error_hi`. The default `error_model="correlated"` takes the region's uncertainty as
  the *mean of the per-point half-widths* (no `1/sqrt(n)`) — the honest reading of fully correlated
  overlapping estimators. `error_model="independent"` divides by `sqrt(n_eff)` with
  `n_eff = log2(tau_max/tau_min)` (octaves spanned, not the point count) for comparison. `n_ci`
  records how many points had a finite interval, separately from `n`.

  Caveat: a Greenhall interval describes a single-τ estimator; the interval attached to a region
  mean is a plausibility band, not a rigorous CI.

Both models leave `value`/`n` unchanged, so the headline numbers stay comparable. The legacy error
remains the default everywhere; nothing changes unless `ci`/`ci_bounds` is set.

## References

- P. D. Welch, *The Use of Fast Fourier Transform for the Estimation of Power Spectra*, IEEE
  Trans. Audio Electroacoust. **15**(2), 70–73 (1967).
- D. B. Percival & A. T. Walden, *Spectral Analysis for Physical Applications*, Cambridge
  University Press (1993), ch. 6 — source of the equivalent-degrees-of-freedom approximation
  `nu = 36*K^2 / (19*K - 1)` used by `compute_psd`'s confidence band (Hann window, 50% overlap).
- J. S. Bendat & A. G. Piersol, *Random Data: Analysis and Measurement Procedures*, 4th ed.,
  Wiley (2010).
- IEEE Std 1139-2008, *Standard Definitions of Physical Quantities for Fundamental Frequency and
  Time Metrology*.
- D. A. Howe, D. W. Allan & J. A. Barnes, *Properties of Signal Sources and Measurement
  Methods*, Proc. 35th Annual Symposium on Frequency Control (1981).
