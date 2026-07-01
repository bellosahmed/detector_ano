# Architecture

## The one principle

**Computation = code. Explanation = language model.**
Every number the user sees is produced by deterministic Python (the *truth layer*) and is verifiable
against pandas. The local LLM only rephrases those numbers into friendly sentences — it is
structurally prevented from calculating, so it cannot hallucinate a figure. Keeping the language
model away from arithmetic is the deliberate defence against the well-documented tendency of LLMs to
fabricate numbers ([Ji et al., 2023](#references)).

## End-to-end data flow

```
   Upload CSV / Excel
          │  loader.read_table
          ▼
   1. profile ............ infer time column (name + content), frequency, numeric
      (DatasetProfile)     signals, missing-value sentinels, univariate vs multivariate
          │  (user can override the detected time column)
          ▼
   2. per signal ......... coerce numeric, mask sentinels → NaN, interpolate (follow local trend)
          ▼
   3. detect ............. robust median/MAD z-score on an ensemble of views:
          │                 • raw first-difference          → spikes & level shifts
          │                 • STL-deseasonalised difference  → seasonal-relative anomalies
          │                (optional) Matrix Profile scan    → unusual shapes
          ▼
   ┌───── 4. query (EXACT) ─────┐   ┌───── 5. reasoning ─────┐
   │ max/min/mean/threshold/    │   │ WHY a point is unusual │
   │ correlation/missing/skew/  │   │ + cross-signal verdict │
   │ trend  (== pandas)         │   │ (many signals move →   │
   └────────────┬───────────────┘   │  real event; one → glitch)
                │                    └───────────┬────────────┘
                └───────────┬────────────────────┘
                            ▼
   6. core.analyze() ....... the single API that ties profiling + detection + reasoning
                             together and returns plain, JSON-able results (one source of numbers)
                            ▼
   7. interface + llm ...... natural-language question → intent → core → answer; the LLM
                             OPTIONALLY rewords or suggests a cause (never computes a number)
                            ▼
   Front-ends: web app (upload · chart · Q&A)  ·  CLI (analyze.py / anomaly-detector)
```

## Modules (`src/anomaly_detector/`)

| Module | Responsibility |
| --- | --- |
| `loader.py` | Read a CSV or Excel file into a DataFrame. |
| `profile.py` | Infer the time column, frequency, signals, and sentinels → `DatasetProfile`. Makes "any dataset" work. |
| `detect.py` | Detectors + metrics: robust z-score ensemble, Matrix Profile, Isolation Forest; confusion matrix, precision/recall/F1, PR-AUC. |
| `query.py` | **Exact** statistics — max/min/mean/correlation/threshold/missing/skew/trend. The truth layer (verified `== pandas`). |
| `reasoning.py` | Explains *why* a point is anomalous and gives a cross-signal confidence verdict. |
| `interface.py` | Natural-language question → intent → the core → a trust-labelled answer. |
| `llm.py` | Optional local Ollama layer — rewords answers, suggests causes; never computes numbers. |
| `deep.py` | Optional LSTM-autoencoder detector (a benchmark comparator). |
| `formatting.py` | Shared human-friendly number formatting. |
| `core.py` | `analyze()` — the single entry point everything else calls. |
| `web/` | Flask app. Imports **only** the core API — no analytics logic lives in the web layer. |

## Why this structure

- **Generic-first.** The profiler decouples the pipeline from any dataset's column names — the whole
  point of "upload any dataset". Time-column detection combines content (parseable dates, monotonic)
  with a name hint (`timestamp`/`date`).
- **One source of numbers (`core.analyze`).** Every front-end (web, CLI) calls the same function, so
  you cannot get a different answer from a different door.
- **Optional dependencies degrade gracefully.** `torch`, `stumpy`, and `ollama` are imported lazily;
  the core runs without them (and without any network).

## Method choices & references

The detector is an **ensemble** because no single method covers every anomaly type — a point
confirmed empirically (PR-AUC comparison in [EVALUATION.md](EVALUATION.md)) and in the literature
([Schmidl et al., 2022](#references)):

- **Robust z-score (median/MAD)** rather than mean/std, so a few large anomalies don't inflate the
  spread and hide each other — standard robust statistics.
- **STL** ([Cleveland et al., 1990](#references)) removes trend and seasonality so seasonal-relative
  anomalies stand out; a per-phase mean is used as a fast fallback on very long series.
- **Matrix Profile** ([Yeh et al., 2016](#references)) finds *discords* — subsequences whose shape
  occurs nowhere else — catching contextual anomalies that differencing misses.
- **Isolation Forest** ([Liu, Ting & Zhou, 2008](#references)) is included as an established baseline.
- **LSTM autoencoder** ([Malhotra et al., 2016](#references)) is included as a deep comparator.

Evaluation follows the guidance that **accuracy/ROC-AUC are misleading under heavy class imbalance**;
we report precision/recall/F1, a confusion matrix, and **PR-AUC** ([Davis & Goadrich, 2006](#references)),
and we validate on the corrected UCR archive ([Wu & Keogh, 2021](#references)) and NAB
([Lavin & Ahmad, 2015](#references)), noting the point-adjustment critique ([Kim et al., 2022](#references)).

### References

1. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). *STL: A Seasonal-Trend
   Decomposition Procedure Based on Loess.* Journal of Official Statistics, 6(1), 3–73.
2. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* IEEE ICDM.
3. Yeh, C.-C. M., et al. (2016). *Matrix Profile I: All Pairs Similarity Joins for Time Series.* IEEE ICDM.
4. Malhotra, P., Ramakrishnan, A., et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly
   Detection.* ICML Anomaly Detection Workshop.
5. Davis, J., & Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves.* ICML.
6. Lavin, A., & Ahmad, S. (2015). *Evaluating Real-time Anomaly Detection Algorithms — the Numenta
   Anomaly Benchmark (NAB).* IEEE ICMLA.
7. Wu, R., & Keogh, E. (2021). *Current Time Series Anomaly Detection Benchmarks are Flawed and are
   Creating the Illusion of Progress.* IEEE TKDE. (The UCR Anomaly Archive.)
8. Schmidl, S., Wenig, P., & Papenbrock, T. (2022). *Anomaly Detection in Time Series: A Comprehensive
   Evaluation.* PVLDB, 15(9).
9. Kim, S., Choi, K., Choi, H.-S., Lee, B., & Yoon, S. (2022). *Towards a Rigorous Evaluation of
   Time-series Anomaly Detection.* AAAI.
10. Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing
    Surveys, 55(12).

> Citations are provided for orientation — confirm exact authors/years/venues before quoting.
