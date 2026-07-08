# Plan: optionale Fehlerbalken, einheitliche Einheiten, kleinere Marker, ADEV-Titel

Baseline vor diesen Änderungen: Commit `bd79c69` (dieser Repo-Historie).

## Befund: Power-Einheiten (Punkt 2)

Kein Bug in `io.py`/`load_lta_file` — der reale `.lta`-Header lautet
`"Signal 1 Power  [µW]"`, `power_uW` enthält also tatsächlich µW-Werte.

Die eigentliche Inkonsistenz liegt in `plotting.py`: `plot_timeseries()`
zeigt die Power-Achse fest in **mW**, während der ADEV-Power-Panel in
`overview_figure()` standardmäßig **µW** zeigt. Wird durch einen
`power_unit`-Parameter in `plot_timeseries()` behoben, der von
`overview_figure()` durchgereicht wird (Default vereinheitlicht auf `"uW"`).

## Checkliste

- [x] Privates GitHub-Repo `ltatools` angelegt, Baseline-Commit gepusht
- [x] `plot_adev()`: `errorbars: bool = True` + `title: str | None = None`
- [x] `plot_timeseries()`: `freq_unit="THz"`, `power_unit="uW"`, `markersize=4`
- [x] `overview_figure()`: `errorbars`/`markersize` durchreichen, ADEV-Panels
      bekommen Titel ("Frequency Allan Deviation" / "Power Allan Deviation")
- [x] Tests in `tests/test_plotting.py` ergänzt (Einheiten, `errorbars=False`,
      Titel, Regressionstest Timeseries-vs-ADEV-Einheit)
- [x] `pytest` grün (36 passed)
- [x] Manueller Smoke-Check (Overview-Figure mit `errorbars=False`,
      `freq_unit="GHz"`, `power_unit="mW"` vs. Default) — per Bild bestätigt
- [x] Commit + Push nach GitHub
