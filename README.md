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
| `"overview"`  | timeseries + frequency/power ADEV (default)             | `freq_unit`, `power_unit`, `errorbars`, `regions` |
| `"psd"`       | frequency + power PSD/ASD, side by side                 | `scaling="psd"\|"asd"`, `ci`            |
| `"timeseries"`| frequency-or-wavelength + power over time                | `freq_unit`, `power_unit`, `lines`      |
| `"adev"`      | Allan deviation of one column                            | `quantity="frequency"\|"power"`, `unit`, `regions` |
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
  (`dev / sqrt(n)`) — it doesn't account for noise type or overlap correlation. There is no
  built-in chi-squared confidence interval for ADEV in this package (kept deliberately simple);
  `compute_psd`'s `ci` confidence band is unrelated and unaffected.
- `compute_psd` treats frequency and power in Hz and µW respectively; `psd_figure`/
  `plot(kind="spectrum", quantity="frequency")` do the THz→Hz conversion before calling it.
  `compute_psd` itself stays unit-agnostic, same as `compute_oadev`.
- Default text size across all plots is set via `matplotlib.rcParams` at import time
  (`ltatools.style`), slightly larger than matplotlib's defaults for readability.

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
