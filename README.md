# Guidora Machine Learning

This repository contains the Guidora machine-learning workflows for:

- abandonment-risk training and real-time inference with LightGBM;
- semantic FAQ embeddings and similarity search;
- feature preparation, evaluation, and model reliability tests.

Repository: [DHRezgui/Guidora-ml](https://github.com/DHRezgui/Guidora-ml)

## Requirements

- Python 3.9 or newer
- pip and a virtual environment
- PostgreSQL 15 or newer when using database-backed workflows
- At least 4 GB of RAM for model training and sentence-transformer workloads

## Install

Run these commands from the root of this repository:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run the environment checks:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_ml_env.py
python verify_lightgbm_setup.py
```

## Repository layout

- `predict.py`: one-shot abandonment-risk inference.
- `abandonment_worker.py`: persistent JSON-lines inference worker.
- `training/`: dataset preparation, feature engineering, training, and evaluation.
- `rag/`: semantic FAQ embedding and retrieval scripts.
- `models/`: trained model and feature-name artifacts.
- `data/`: raw and processed datasets; local data should not contain personal information.
- `results/`: evaluation and embedding outputs.
- `tests/`: ML and prediction tests.
- `config.py`: shared paths and training configuration.

## Prepare a training dataset

The preparation script accepts a CSV export or JSON payload. A JSON input can
contain a `features` array. The core feature columns are:

```text
timeOnPage, scrollDepth, clickMisses, hesitations, abandonmentRisk,
helpTriggered, hasError, multiplePages, label
```

Run it from the ML repository root:

```bash
python training/prepare_dataset.py --input data/raw/dataset.json
```

The pipeline writes prepared data to `data/processed/`, including train,
validation, test, and metadata files. It derives `timePerPage`,
`clickMissRate`, `hesitationRate`, `frictionScore`, `highFriction`, and
`multipleIssues`.

## Train and evaluate the model

```bash
python training/train_model.py --input data/raw/dataset.json
```

The training workflow writes the model and evaluation artifacts under `models/`
and `results/`, including:

- `models/lightgbm_model.pkl`;
- `models/feature_names.json`;
- `results/metrics.json`.

Reported metrics include accuracy, precision, recall, AUC, and confusion-matrix
counts. Do not publish datasets, model artifacts derived from private data, or
run logs containing personal information.

## Real-time inference

`predict.py` loads the trained model and accepts either an input file or JSON on
stdin. Example input:

```json
{
  "features": {
    "timeOnPage": 83,
    "scrollDepth": 41.4,
    "clickMisses": 0,
    "hesitations": 2,
    "abandonmentRisk": 0.2,
    "helpTriggered": 0,
    "hasError": 0,
    "multiplePages": 1
  },
  "threshold": 0.5
}
```

Run a file-based prediction:

```bash
python predict.py --input data/sample-features.json
```

Run through stdin:

```bash
echo '{"features":{"timeOnPage":83,"scrollDepth":41.4,"clickMisses":0,"hesitations":2,"abandonmentRisk":0.2,"helpTriggered":0,"hasError":0,"multiplePages":1}}' | python predict.py
```

The result contains an abandonment probability, a binary `willAbandon` decision,
a confidence score, and the threshold used. Use `--model`, `--features`, and
`--threshold` to override the default paths and decision threshold.

For long-lived backend processes, `abandonment_worker.py` loads the model once
and serves newline-delimited JSON requests over stdin/stdout. Logs are written
to stderr so stdout remains machine-readable.

## Semantic FAQ search

Prepare FAQ entries in `data/raw/faq_questions.json`:

```json
[
  {
    "id": "faq-001",
    "question": "How do I reset my password?",
    "answer": "Use the password reset link on the sign-in page.",
    "category": "auth"
  }
]
```

Generate embeddings and search them from the repository root:

```bash
python rag/generate_faq_embeddings.py
python rag/semantic_search.py "How do I reset my password?"
```

Use `--top-k`, `--min-similarity`, and `--embeddings` to customize retrieval.
Generated embeddings are written under `results/`.

## Backend integration

The backend exposes the ML functionality under its versioned API prefix:

- `POST /api/v1/ml/predictions/abandonment`
- `POST /api/v1/ml/predictions/batch`
- `GET /api/v1/ml/predictions/health`

The backend starts the persistent worker in its Docker development setup and
uses this repository's `predict.py` and model artifacts. See the backend
repository for authentication, authorization, rate limits, and request schemas.

## Tests and quality checks

```bash
python -m pytest tests -v
python -m pytest tests/test_predict.py -v
black --check .
flake8 .
mypy predict.py config.py
```

The prediction tests cover model loading, output shape and ranges, missing
feature handling, threshold decisions, and the CLI contract.

## Configuration and data safety

Training paths and LightGBM settings are defined in `config.py`. Database
connections use environment variables such as `DB_HOST`, `DB_PORT`, `DB_NAME`,
`DB_USER`, and `DB_PASSWORD`.

Never commit `.env` files, credentials, private datasets, or generated outputs
containing personal data. Use `.env.example` as a template where available.

## License

Confirm the intended open-source license and add a `LICENSE` file before a
formal public release.
