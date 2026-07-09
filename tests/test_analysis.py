import numpy as np
import pandas as pd
import pytest

from ltatools.analysis import compute_oadev, compute_psd, find_stable_segments


def test_compute_oadev_white_freq_noise_slope():
    rng = np.random.default_rng(42)
    n = 4000
    rate = 10.0
    data = rng.normal(0, 1.0, n)

    tau, dev, dev_err, ns = compute_oadev(data, rate=rate, taus="octave")

    k = max(3, len(tau) // 2)
    log_tau = np.log(tau[:k])
    log_dev = np.log(dev[:k])
    slope, _ = np.polyfit(log_tau, log_dev, 1)

    assert slope == pytest.approx(-0.5, abs=0.15)


def test_compute_oadev_rate_from_time_s():
    dt = 0.1
    n = 500
    time_s = np.arange(n) * dt
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1.0, n)

    tau, dev, dev_err, ns = compute_oadev(data, time_s=time_s, taus="octave")

    assert tau[0] == pytest.approx(dt, rel=0.2)


def test_compute_oadev_requires_rate_or_time_s():
    with pytest.raises(ValueError):
        compute_oadev([1.0, 2.0, 3.0])


def test_compute_psd_white_noise_level():
    rng = np.random.default_rng(3)
    fs = 1000.0
    sigma = 2.0
    data = rng.normal(0, sigma, 200_000)

    f, Pxx = compute_psd(data, rate=fs)

    expected = 2 * sigma**2 / fs
    mean_level = np.mean(Pxx[3:-3])
    assert mean_level == pytest.approx(expected, rel=0.2)


def test_compute_psd_sine_peak():
    rng = np.random.default_rng(5)
    fs = 1000.0
    f0 = 50.0
    n = 200_000
    t = np.arange(n) / fs
    data = 3.0 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, 0.1, n)

    f, Pxx = compute_psd(data, rate=fs, nperseg=4096)

    peak_f = f[np.argmax(Pxx)]
    bin_width = f[1] - f[0]
    assert peak_f == pytest.approx(f0, abs=bin_width)


def test_compute_psd_ci_coverage():
    rng = np.random.default_rng(11)
    fs = 1000.0
    sigma = 2.0
    data = rng.normal(0, sigma, 50_000)

    f, Pxx, (lower, upper) = compute_psd(data, rate=fs, ci=0.95, nperseg=1024)

    true_level = 2 * sigma**2 / fs
    coverage = np.mean((lower <= true_level) & (true_level <= upper))
    assert 0.85 <= coverage <= 0.99


def test_compute_psd_jitter_warns_and_stays_plausible():
    rng = np.random.default_rng(9)
    n = 2000
    dt_nominal = 0.01
    dt = dt_nominal + rng.normal(0, 0.05 * dt_nominal, n)
    dt[100] += 0.5 * dt_nominal  # inject one clear outlier beyond tolerance
    time_s = np.cumsum(np.abs(dt))
    data = rng.normal(0, 1.0, n)

    with pytest.warns(UserWarning, match="irregularly spaced"):
        f, Pxx = compute_psd(data, time_s=time_s)

    assert np.all(np.isfinite(Pxx))
    assert len(f) == len(Pxx)


def test_compute_psd_requires_rate_or_time_s():
    with pytest.raises(ValueError):
        compute_psd([1.0, 2.0, 3.0])


def test_find_stable_segments_splits_on_jump():
    n = 100
    jump_at = 50
    rng = np.random.default_rng(7)
    freq = 300.0 + rng.normal(0, 1e-5, n)
    freq[jump_at:] += 5.0
    time_ms = np.arange(n) * 10.0
    df = pd.DataFrame(
        {
            "frequency_THz": freq,
            "time_ms": time_ms,
            "time_s": time_ms * 1e-3,
        }
    )

    segments = find_stable_segments(df, n=2, trim_left=2, trim_right=0)

    assert len(segments) == 2
    lengths = sorted((len(s) for s in segments), reverse=True)
    assert lengths == [jump_at - 2, (n - jump_at) - 2]
    assert len(segments[0]) >= len(segments[1])
    for seg in segments:
        assert seg["time_ms"].iloc[0] == 0
        assert seg["time_s"].iloc[0] == 0
        assert seg["frequency_THz"].max() - seg["frequency_THz"].min() < 1.0
