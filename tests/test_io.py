import pytest
from scipy.constants import c

from ltatools.io import load_lta_file


def test_load_lta_file_normalizes_columns(write_lta):
    path = write_lta(n_points=50)
    df = load_lta_file(path)
    for col in ("time_ms", "time_s", "wavelength_nm", "power_uW", "frequency_THz"):
        assert col in df.columns


def test_frequency_matches_physics(write_lta):
    path = write_lta(n_points=10, wavelength_nm=1064.0)
    df = load_lta_file(path)
    expected_THz = c / (1064.0 * 1e-9) * 1e-12
    assert df["frequency_THz"].mean() == pytest.approx(expected_THz, rel=1e-6)
    assert df["frequency_THz"].mean() == pytest.approx(281.76, abs=0.01)


def test_cleanup_removes_invalid_rows(write_lta):
    path = write_lta(n_points=100, invalid_rows=5)
    df_raw = load_lta_file(path, cleanup=False)
    df_clean = load_lta_file(path, cleanup=True)
    assert len(df_raw) == 100
    assert len(df_clean) == 95
    assert (df_clean["wavelength_nm"] > 0).all()


def test_missing_header_raises(write_lta):
    path = write_lta(n_points=10, include_header=False)
    with pytest.raises(ValueError):
        load_lta_file(path)


def test_unknown_extra_column_survives(tmp_path):
    content = (
        "Preamble line\n"
        "Time  [ms]\tSignal 1  Wavelength, vac.  [nm]\tSignal 1 Power  [µW]\tExtra  Column [x]\n"
        "0,000000\t1064,000000\t500,000000\t1,000000\n"
        "10,000000\t1064,000001\t499,000000\t2,000000\n"
    )
    path = tmp_path / "extra.lta"
    with open(path, "w", encoding="cp1252") as f:
        f.write(content)
    df = load_lta_file(path)
    assert "extra_column_x" in df.columns
