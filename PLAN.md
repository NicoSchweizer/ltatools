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

# Runde 2: ADEV-CI entfernen, save-Parameter, plot()-Wrapper, Docstring-Beispiele

Baseline: Commit `17bbb15` (Ende Runde 1).

- [x] `compute_oadev` (`analysis.py`): `ci`/`alpha` + Chi²-CI-Zweig (Riley-
      Methode über `autocorr_noise_id`/`edf_greenhall`/`confidence_interval`)
      entfernt; gibt nur noch `tau, dev, dev_err, n` zurück (die naive,
      direkt von `allantools.oadev()` gelieferte Fehlerangabe).
      `compute_psd`/`plot_psd`/`psd_figure` und deren eigenes, unabhängiges
      Chi²-Konfidenzband bleiben unangetastet.
- [x] `plot_adev`: `ci_bounds`-Parameter entfernt, `save`-Parameter ergänzt.
- [x] `plot_timeseries` / `plot_psd`: `save`-Parameter ergänzt (speichert
      über `ax.get_figure()`, funktioniert auch mit von außen übergebenem
      `ax`).
- [x] `overview_figure`: `ci`-Parameter entfernt, ADEV-Panels vereinfacht.
- [x] Neue Wrapper-Funktion `plot(data, kind=..., quantity=..., save=...,
      cleanup=...)` in `plotting.py`: nimmt DataFrame ODER `.lta`-Pfad,
      deckt `kind` ∈ {overview, psd, timeseries, adev, spectrum} ab,
      `save` ist ein Name (Endung wird ergänzt), gibt `None` zurück (kein
      `<Axes: ...>`-Repr-Text mehr im Notebook).
- [x] `__init__.py`: `plot` exportiert.
- [x] Docstring-`Examples` ergänzt bei `plot`, `overview_figure`,
      `psd_figure`, `lta_overview`, `plot_adev`, `plot_timeseries`.
- [x] Tests: 2 ADEV-CI-Tests in `test_analysis.py` gelöscht
      (`test_compute_oadev_ci_brackets_dev`,
      `test_compute_oadev_noise_id_auto_widens_relatively_at_large_tau`),
      2 ADEV-CI-Tests in `test_plotting.py` gelöscht
      (`test_plot_adev_ci_bounds_overrides_dev_err`,
      `test_overview_figure_ci_smoke`), `test_psd_figure_ci_band`
      unverändert gelassen; Save-Tests für die drei Einzel-Panel-Funktionen
      und Dispatch-/Save-/Fehler-Tests für `plot()` ergänzt.
      `pytest` grün (51 passed).
- [x] Manueller Smoke-Check aller 5 `kind`s über `plot()` (DataFrame-Input,
      inkl. `save="name"` ohne Endung) — per Bild bestätigt, alle Aufrufe
      liefern `None`.
- [x] Commit + Push nach GitHub

## Nachtrag: Fehlerbalken-Styling (Endcaps + dunklere Farbe)

Baseline: Commit `227a82e` (Ende Runde 2).

- [x] `style.py`: `darken_color(color, factor=0.7)` ergänzt (RGB-Skalierung
      über `matplotlib.colors.to_rgb`).
- [x] `plot_adev`: neue Parameter `capsize=0` (Default weiterhin ohne
      Endcaps — `capsize=3` z.B. fügt sie hinzu) und `errorbar_color=None`
      (Default: dunklere Variante der Punktefarbe via `darken_color`,
      standardmäßig aktiv; explizite Farbe oder z.B. `COLORS["frequency"]`
      übergeben, um wie vorher gleichfarbig zu sein).
- [x] `overview_figure`: `capsize`/`errorbar_color` ergänzt (gleicher
      Default `capsize=0`), an beide ADEV-Panels durchgereicht.
- [x] `plot()`-Wrapper: keine Änderung nötig — `capsize`/`errorbar_color`
      laufen bereits über `**kwargs` durch (sowohl `kind="adev"` direkt zu
      `plot_adev`, als auch `kind="overview"` zu `overview_figure`).
- [x] Tests ergänzt: dunklere Farbe standardmäßig vorhanden (aber keine
      Endcaps), `capsize=3` (o.ä.) schaltet Endcaps zu, `errorbar_color`
      überschreibbar, Durchreichung in `overview_figure` geprüft. `pytest`
      grün (56 passed).
- [x] Manueller Smoke-Check per Bild bestätigt.
- [x] Commit + Push nach GitHub (`ff1b3fa`)

**Rückgängig machen:** `errorbar_color=COLORS["frequency"]`/`COLORS["power"]`
an `plot_adev`/`overview_figure`/`plot()` übergeben, um wieder gleichfarbige
Fehlerbalken wie vorher zu bekommen (Endcaps sind per Default `capsize=0`
ohnehin aus). Zusätzlich komplett per `git revert ff1b3fa` — Baseline
`227a82e` bleibt dafür sauber erhalten.

## Nachtrag: Docstring-Beispiele mit tatsächlichem Output

Baseline: Commit `ff1b3fa`.

- [x] `Examples`-Sektionen in `plot_timeseries`, `plot_adev`,
      `overview_figure`, `psd_figure`, `lta_overview` um die tatsächliche
      REPL-Ausgabe der jeweiligen Aufrufe ergänzt (nicht nur den Aufruf
      selbst) — Reprs gegen echte Läufe mit synthetischen Daten verifiziert.
      `plot()`-Beispiele bleiben ohne Output-Zeile, da die Funktion bewusst
      `None` zurückgibt (dafür wird in einer REPL nichts angezeigt).
- [x] `pytest` grün (56 passed, keine Funktionsänderung).
- [ ] Commit + Push nach GitHub
