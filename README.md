# 🎓 Student Support and Course Feedback Assistant

**AI for Everyone – Project 01.** Educational Streamlit prototype for a teacher reviewing a class after the
second assessment period. **Decision support only** - the support flag (`needs_support = 1` when G3 < 10) is a
classroom modelling target, not a decision about promotion, admission, punishment or a real student.
A teacher must review every suggestion.

**Live demo:** _paste your Streamlit Community Cloud link here_

## App tabs
| Tab | What it shows | Tool |
|---|---|---|
| 1. Class overview and support prediction | 3 charts, test metrics, example record form with validation | Decision Tree (+ baseline) |
| 2. Learning profiles | group sizes and averages in original units, scatter plot | StandardScaler + KMeans (k = 3) |
| 3. Feedback digit | sample/uploaded digit, processed 28×28 image, teacher confirmation | small Keras CNN |
| 4. Course comments | sentiment + topic counts, comment dropdown, metrics vs reference labels | NLTK VADER + keyword rules |
| 5. Teaching summary | **Offline summary (default)**, optional Groq summary, prompt comparison | Groq Free Plan |

## Results (fixed splits; test set evaluated once)
| Model | Accuracy | Macro F1 | Class 1 precision | Class 1 recall |
|---|---|---|---|---|
| Decision tree (max_depth=4) | 0.917 | 0.907 | 0.857 | 0.900 |
| Baseline (most frequent) | 0.667 | 0.400 | 0.000 | 0.000 |

Test examples: 40 × class 0, 20 × class 1.

| KMeans group | Size | Description (training averages) |
|---|---|---|
| 0 | 25 | lower G1 (6.68), lower G2 (5.48), lower study time (1.52), higher past failures (2.2). Learning profile only; not a permanent label. |
| 1 | 144 | lower G1 (9.09), lower G2 (9.09), about average study time (2.03), about average past failures (0.17). Learning profile only; not a permanent label. |
| 2 | 107 | higher G1 (14.27), higher G2 (14.25), about average study time (2.07), lower past failures (0.09). Learning profile only; not a permanent label. |

Group numbers are arbitrary; a group is a learning profile, not a predicted class.

| Other component | Accuracy | Macro F1 |
|---|---|---|
| CNN (3 epochs, 500 test images) | 0.926 | 0.925 |
| VADER vs 30 reference labels | 0.967 | 0.967 |

Charts and metric files are in `reports/`. Acceptance test log: `reports/test_log.csv`.

## Repository layout
```
app.py                  Streamlit interface (5 tabs)
src/                    config, data loading, validation, models, NLP, summary
models/                 supervised.joblib, clusters.joblib, images.keras, metadata
data/                   main.csv, feedback.csv, example_inputs.csv, meta.json,
                        topic_keywords.json, prompt_test_cases.json  (no images.npz)
sample_images/          one demo image per digit class
nltk_data/              offline VADER lexicon (no download at startup)
source_notices/         data attribution - keep with the data
reports/                charts, metrics, test log
tests/test_basic.py     pytest checks
requirements.txt        pinned to the training versions
.streamlit/             config.toml, secrets.toml.example (real secrets never committed)
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Models are trained in the Colab notebook; the app only loads them and never retrains.

## Deploy (free)
1. **GitHub Free:** New repository → *Add file → Upload files* → drag in **all files and folders of this repo**
   (not the ZIP itself) → *Commit changes*. Do not upload keys or `images.npz`.
2. **Streamlit Community Cloud:** *Create app* → this repository, branch `main`, main file **`app.py`**,
   *Advanced settings* → Python **3.13** → *Deploy*.
3. Optional Groq: *App settings → Secrets* → `GROQ_API_KEY = "..."` (Free Plan only). Offline summary stays the default.

GitHub stores the code; Community Cloud runs it (GitHub Pages cannot run Streamlit). If a free limit blocks the
live demo, run locally and submit a screen recording.

## Tests
```bash
pip install pytest && pytest -q
```

## Limits
Small dataset; results show associations, not causes. The CNN was trained on MNIST and does not prove reliability
on real handwriting or scanned forms - never use the digit to assign a mark. VADER can misread negation, sarcasm
and mixed comments; English only (Urdu / Roman Urdu are outside this baseline). Groq free quotas can run out.

## Data sources
UCI Student Performance (mathematics subset), MNIST subset, authored classroom comments - see `source_notices/`.
