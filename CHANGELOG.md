# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); this project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-01

First public release.

### Added
- **Dataset profiler** — auto-detects the time column (with name hints), sampling frequency, numeric
  signals, and missing-value sentinels; works on CSV and Excel.
- **Detectors** — a robust (median/MAD) z-score ensemble over differenced + STL-deseasonalised
  signals, an optional **Matrix Profile** shape scan, and an **Isolation Forest** baseline.
- **Truth layer** (`query.py`) — exact stats verified `== pandas`, plus correlation, missing-data,
  skewness, and trend.
- **Reasoning** — plain-English "why" per anomaly with a cross-signal confidence verdict
  (real event vs isolated glitch).
- **Natural-language interface** — extremes, averages, thresholds, year/month scoping, comparisons,
  trend, correlation and missing-data questions; optional local-LLM rewording and cause suggestions
  (numbers always computed by code).
- **Web app** (`localhost:3020`) — drag-and-drop upload, per-signal charts with anomaly markers,
  editable time-column, deep-shape toggle, and a Q&A box.
- **Evaluation** — verification gate (`verify.py`), synthetic + real benchmarks (NAB, UCR), and a
  PR-AUC detector comparison (imbalance-robust).
- Packaging: `pip install`, console scripts, Dockerfile, CI.

### Notes
- Fully local and private — no cloud, no API keys.
- Scope: analyses the uploaded dataset only; it does **not** forecast future values.
