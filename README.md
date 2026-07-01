# Anomaly Detector

Upload **any** time-series dataset → the system finds the anomalies and lets you **ask questions
about it in plain English**. Runs **fully locally**: exact numbers are computed by code, and a local
LLM only *explains* them — it never invents figures.

## What it does

- **Drop in a CSV or Excel file** — the system auto-detects the time column, frequency and signals.
- **Detects anomalies** with a statistical ensemble — differencing + STL-deseasonalised, robust
  (median/MAD) z-score — for spikes, level shifts and seasonal deviations. An **optional Matrix
  Profile** scan (`--shape`) adds unusual-*pattern* detection; an LSTM autoencoder is included as a
  benchmark comparator.
- **Explains them** — "26 Jan is unusual because signals A, B and C all moved together," with a
  confidence verdict (real event vs isolated glitch).
- **Answers questions** — "what was the highest value?", "what's the average in 2020?",
  "how many times did it go above 100?", "was 2023 higher than 2022?", "is it rising over time?",
  "were there any anomalies?" (deterministic parsing today; free phrasing is future work).
- **Offline & private** — no cloud, no API keys. A local LLM (Ollama, optional) only *rewords*
  answers and suggests possible causes; it never computes a number.

> **Scope:** the system analyses the dataset you upload — it does **not forecast** future values.
> Every figure is computed from data that exists.

## Screenshot

The web app (`localhost:3020`): drag in a CSV, and each signal gets a trace with anomalies marked,
a table explaining *why* each was flagged (and which other sensors moved with it), and a box to ask
questions in plain English.

<!-- Add a screenshot at assets/screenshot.png, then uncomment the line below: -->
<!-- ![Anomaly Detector web UI](assets/screenshot.png) -->

## Quickstart

```bash
git clone https://github.com/bellosahmed/detector_ano.git
cd detector_ano
pip install -r requirements.txt        # one-time

# web UI — drag in a CSV, see the anomalies on a chart
python3.12 -m anomaly_detector.web.app   # → http://localhost:3020

# or the command line
python3.12 scripts/analyze.py samples/sensor.csv
```

### Install as a package (adds the `anomaly-detector` command)

```bash
pip install .                          # or: pip install ".[all]" for deep + shape detectors
anomaly-detector samples/sensor.csv
```

### Run with Docker (no Python setup needed)

```bash
docker build -t anomaly-detector .
docker run -p 3020:3020 anomaly-detector   # → http://localhost:3020
```

Optional extras: `pip install ".[deep]"` (LSTM comparator), `".[shape]"` (Matrix Profile).
The local-LLM rewording is optional too — install [Ollama](https://ollama.com) and it lights up
automatically; without it, answers stay in exact computed form.

The LLM explanation layer is **optional**: with no model installed, answers still work in
"template mode" (every figure shown plainly). Install Ollama + a small model to enable AI phrasing.

## How it works

**Computation = code, explanation = language model.** Deterministic Python computes every figure
(the *truth layer*, verified against pandas); a local LLM only *rewords* the results, so it can't
invent a number. Detection is an ensemble of robust (median/MAD) z-scores on the differenced and
STL-deseasonalised signal, with an optional Matrix Profile scan for unusual *shapes*. Anomalies are
explained across sensors (which other signals moved) with a confidence verdict.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, method choices, and references.

## Repo layout

```
├── src/anomaly_detector/   # the engine (importable library)
│   └── web/                # Flask web app: upload + chart + Q&A
├── tests/                  # automated tests
├── benchmarks/             # evaluation scripts (NAB, UCR, tuning, detector comparison)
├── scripts/                # analyze.py, verify.py, make_samples.py
└── samples/                # tiny example datasets to try it out
```

Handy shortcuts: `make test`, `make verify`, `make web`.

## Development

```bash
pip install -e .          # editable install
python3.12 -m pytest      # run the test suite
python3.12 scripts/verify.py   # verification gate (results == pandas)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project layout and conventions, and
[CHANGELOG.md](CHANGELOG.md) for release notes.

## Licence

MIT — see [LICENSE](LICENSE).
