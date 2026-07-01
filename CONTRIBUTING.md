# Contributing

Thanks for your interest in improving Anomaly Detector.

## Development setup

```bash
git clone https://github.com/bellosahmed/detector_ano.git
cd detector_ano
pip install -e ".[all]"     # editable install + optional deep/shape extras
python3.12 -m pytest        # run the tests (should be all green)
```

Handy `make` targets: `make test`, `make verify`, `make web`.

## Project layout

```
src/anomaly_detector/    the library (the engine)
  ├─ profile.py          infer time column, frequency, signals, sentinels
  ├─ detect.py           detectors + metrics (robust z, Matrix Profile, Isolation Forest, PR-AUC…)
  ├─ query.py            exact stats — the "truth layer" (verified == pandas)
  ├─ reasoning.py        cross-signal explanation of anomalies
  ├─ interface.py        natural-language question → answer
  ├─ llm.py              optional local-LLM rewording (never computes numbers)
  ├─ core.py             analyze() — ties profiling + detection + reasoning together
  └─ web/                Flask app (upload, chart, Q&A)
tests/                   one test module per source module
benchmarks/              evaluation scripts (NAB, UCR, tuning, comparison)
scripts/                 analyze.py, verify.py, make_samples.py
samples/                 tiny example datasets
```

## Core principle (please preserve it)

**Computation = code; explanation = language model.** Every number a user sees must come from
deterministic code (the query engine / detectors), verifiable against pandas. The LLM only *rewords*
results — it must never compute or invent a figure. New features should keep this separation.

## Conventions

- **Test-driven.** Write a failing test first, then the minimal code to pass it. Every source module
  has a matching `tests/test_*.py`.
- **Keep the truth layer exact.** Anything in `query.py` must match a direct pandas computation;
  `scripts/verify.py` enforces this and must stay green (13/13).
- Optional dependencies (`torch`, `stumpy`, `ollama`) are imported lazily and degrade gracefully —
  the core must run without them.
- Match the surrounding style; keep functions small and documented.

## Before opening a PR

```bash
python3.12 -m pytest        # all tests pass
python3.12 scripts/verify.py  # verification gate passes (13/13)
```

Describe what changed and why, and include a test for any new behaviour or bug fix.
