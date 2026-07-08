# ltatools

HighFinesse wavemeter `.lta` file analysis: loading, Allan deviation, power spectral density,
timeseries and overview plots.

Successor to the `wavemeter-helper` prototype (see `legacy/prototype.py`, reference only — not
imported from).

## Install

```bash
pip install -e ".[dev]"
```

## Standard workflow

Two lines cover the common case: load a file, get a combined overview figure (timeseries on top,
frequency and power Allan deviation below).

```python
from ltatools import load_lta_file, overview_figure

df = load_lta_file("data/2026-07-01_lock_test.lta", cleanup=True)
fig, axes = overview_figure(df)
```

## Use cases

**New, stable measurement (standard case):**

```python
from ltatools import load_lta_file, overview_figure

df = load_lta_file("data/2026-07-01_lock_test.lta", cleanup=True)
fig, axes = overview_figure(df, freq_unit="MHz", save="figures/lock_test_overview.png")
```

**Old measurement with mode hops:**

```python
from ltatools import load_lta_file, find_stable_segments, overview_figure

df = load_lta_file("data/2025-11-03_free_running.lta", cleanup=True)
for i, seg in enumerate(find_stable_segments(df, n=3)):
    overview_figure(seg, freq_unit="MHz", save=f"figures/free_running_seg{i}.png")
```

Or in one call via `lta_overview`:

```python
from ltatools import lta_overview

results = lta_overview(
    "data/2025-11-03_free_running.lta", cleanup=True, segments=True, n_segments=3, freq_unit="MHz"
)
```

**Frequency ADEV in kHz with a chi-squared confidence interval (assuming white FM):**

```python
from ltatools import load_lta_file, compute_oadev, plot_adev

df = load_lta_file("data/run.lta", cleanup=True)
tau, dev, err, n, (lo, hi) = compute_oadev(
    df["frequency_THz"], time_s=df["time_s"], ci=0.683, alpha=0  # alpha=0: white FM
)
plot_adev(tau, dev, ci_bounds=(lo, hi), unit="kHz", quantity="frequency")
```

The naive `err` (`dev / sqrt(n)`) from `allantools` ignores noise type and overlap correlation and
is not a real confidence interval — pass `ci` to get a proper chi-squared interval instead (see
References below). `overview_figure(df, ci=...)` applies the same interval to both its ADEV panels.

**PSD/ASD with a confidence band, as a separate figure:**

```python
from ltatools import load_lta_file, psd_figure

df = load_lta_file("data/run.lta", cleanup=True)
fig, axes = psd_figure(df, scaling="asd", ci=0.95, save="figures/run_psd.png")
```

`psd_figure` is intentionally **not** part of `overview_figure` — call it separately when a
spectral view is needed. `compute_psd`/`plot_psd` are available individually for custom axes, the
same way `compute_oadev`/`plot_adev` are.

## Notes

- All plotting functions return `fig`/`ax` and never call `plt.show()`; display happens in the
  caller (Jupyter renders figures automatically).
- The prototype's `lta_to_adev`/`lta_to_t_s` wrappers were intentionally not ported — `lta_overview`
  (a thin wrapper around `load_lta_file` + `overview_figure` + `find_stable_segments`) replaces
  both with a single overview-focused entry point. There is no `lta_psd`; use `load_lta_file` +
  `psd_figure` directly (two lines, same pattern as the standard workflow above).
- `compute_psd` treats frequency and power in Hz and µW respectively; `psd_figure` does the THz→Hz
  conversion before calling it. `compute_psd` itself stays unit-agnostic, same as `compute_oadev`.

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
- W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST Special Publication 1065 (2008),
  ch. 5 — chi-squared confidence intervals and noise-type identification for `compute_oadev`.
- C. A. Greenhall & W. J. Riley, *Uncertainty of Stability Variances Based on Finite
  Differences*, Proc. 35th PTTI Systems and Applications Meeting (2003) — the EDF algorithm
  behind `allantools.edf_greenhall`.
- S. R. Stein, *Frequency and Time – Their Measurement and Characterization*, in
  *Precision Frequency Control*, Vol. 2, Academic Press (1985).
- D. A. Howe, D. W. Allan & J. A. Barnes, *Properties of Signal Sources and Measurement
  Methods*, Proc. 35th Annual Symposium on Frequency Control (1981).
