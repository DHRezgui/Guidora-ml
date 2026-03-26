# LightGBM ML Environment Setup Guide

## Overview

This directory contains the machine learning pipeline for the **Abandonment Risk Prediction** model using **LightGBM**. The model predicts the risk of user abandonment in real-time to trigger proactive contextual help.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)
- PostgreSQL 15+ (for database access)
- 4GB+ RAM recommended for model training

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Setup Script

```bash
python setup_ml_env.py
```

This script will:
- Verify Python version
- Install all dependencies
- Verify LightGBM installation
- Verify sentence-transformers installation
- Create necessary directories (models, data, results, notebooks)
- Test database connectivity

## Project Structure

```
ml/
├── config.py                 # LightGBM configuration & hyperparameters
├── requirements.txt          # Python dependencies
├── setup_ml_env.py          # Environment initialization script
├── README.md                # This file
├── models/                   # Trained model artifacts
│   ├── lightgbm_model.pkl
│   └── feature_names.json
├── data/                     # Data files (excluded from git)
│   ├── raw/
│   └── processed/
├── results/                  # Model evaluation results & metrics
│   ├── metrics.json
│   └── confusion_matrix.png
├── notebooks/                # Jupyter notebooks
│   ├── exploratory/         # Data exploration & EDA
│   │   ├── 01_data_exploration.ipynb
│   │   └── 02_feature_analysis.ipynb
│   └── training/            # Model training notebooks
│       ├── 03_model_training.ipynb
│       └── 04_model_evaluation.ipynb
└── training/                 # Training scripts
    ├── __init__.py
    ├── prepare_dataset.py
    ├── preprocess.py
    ├── train_model.py
    └── evaluate.py
```

## Key Components

### config.py

Contains all configuration for the LightGBM model:
- **Model parameters**: boosting type, learning rate, number of leaves, etc.
- **Regularization settings**: lambda, alpha, subsampling rates
- **Feature engineering options**: scaling, outlier removal, PCA
- **Dataset parameters**: train/test split, stratification
- **Evaluation settings**: threshold, SHAP values computation
- **File paths**: models, data, results directories

### Main Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| lightgbm | 4.6.0 | Gradient Boosting framework |
| scikit-learn | 1.8.0 | ML utilities & metrics |
| pandas | 3.0.1 | Data manipulation |
| numpy | 2.4.3 | Numerical computing |
| sqlalchemy | 2.1.1 | Database ORM |
| psycopg2 | 2.9.9 | PostgreSQL adapter |
| jupyter | 1.1.1 | Notebook environment |
| pytest | 8.3.2 | Testing framework |
| sentence-transformers | 3.3.1 | Semantic embeddings for FAQ questions |

## Semantic FAQ Embeddings (sentence-transformers)

This sprint introduces semantic search capabilities for natural-language FAQ support.

### Install / Verify

```bash
pip install -r requirements.txt
python setup_ml_env.py
```

### Quick embedding smoke test

```bash
python ml/rag/semantic_embedder.py
```

Expected output includes:

- selected model name
- number of embedded questions
- embedding vector dimension

### Generate FAQ embeddings dataset

Prepare FAQ entries in `ml/data/raw/faq_questions.json` with this schema:

```json
[
    {
        "id": "faq-001",
        "question": "Comment réinitialiser mon mot de passe ?",
        "answer": "...",
        "category": "auth"
    }
]
```

Run generation:

```bash
python ml/rag/generate_faq_embeddings.py
```

Generated artifact:

- `ml/results/faq_embeddings.json`

The output contains model metadata, vector dimension, and one embedding vector for each FAQ question.

### Semantic search (cosine similarity)

Use semantic retrieval to find the most relevant FAQ entries for a user query:

```bash
python ml/rag/semantic_search.py "Je n'arrive pas a reinitialiser mon mot de passe"
```

Optional parameters:

- `--top-k` to control the number of returned results
- `--min-similarity` to filter low-relevance matches
- `--embeddings` to use a custom embeddings file

## Quick Start

### 1. Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Verify Installation

```bash
python setup_ml_env.py
```

Expected output shows all checks passing ✅

Quick reproducible smoke test:

```bash
python verify_lightgbm_setup.py
```

Expected output includes `LightGBM smoke test: OK` and an AUC score.

### 3. Start Jupyter Lab

```bash
jupyter lab
```

Then navigate to `notebooks/exploratory/` to explore data.

### 4. Train Model (When Ready)

```bash
python ml/training/train_model.py --input ml/data/raw/dataset.json
```

The training command automatically:

- loads and prepares the dataset
- trains a LightGBM classifier
- evaluates validation/test splits
- saves model and metrics artifacts

Generated artifacts:

- `ml/models/lightgbm_model.pkl`
- `ml/models/feature_names.json`
- `ml/results/metrics.json`

Metrics report includes:

- `accuracy`
- `precision`
- `recall`
- `auc`
- confusion matrix (tn/fp/fn/tp)

## Data Preparation and Feature Engineering

This step transforms backend exported datasets into clean and structured
training matrices while preserving project business logic.

### Input expectations

- Supports CSV export from backend ML endpoints
- Supports JSON export with `{"features": [...]}` payload
- Expects the backend schema columns: `timeOnPage`, `scrollDepth`, `clickMisses`,
  `hesitations`, `abandonmentRisk`, `helpTriggered`, `hasError`, `multiplePages`, `label`

### Run preparation pipeline

```bash
python ml/training/prepare_dataset.py --input ml/data/raw/dataset.json
```

Generated outputs in `ml/data/processed/`:

- `dataset_prepared.csv`
- `X_train.csv`, `X_valid.csv`, `X_test.csv`
- `y_train.csv`, `y_valid.csv`, `y_test.csv`
- `dataset_metadata.json`

### Derived features created

- `timePerPage`
- `clickMissRate`
- `hesitationRate`
- `frictionScore`
- `highFriction`
- `multipleIssues`

## Automated Tests (Model Reliability)

Automated tests for the prediction model are located in `ml/tests/test_predict.py`.

The test suite validates:

- model artifact loading (`lightgbm_model.pkl`, `feature_names.json`)
- prediction output schema and value ranges
- missing-feature imputation behavior
- CLI real-time prediction via stdin JSON
- threshold-dependent binary decision behavior

### Run only prediction tests

```bash
python -m pytest ml/tests/test_predict.py -v
```

### Run all ML tests

```bash
python -m pytest ml/tests -v
```

## Real-time Model Usage (Integration Contract)

The real-time predictor is exposed through the backend API and backed by
the `ml/predict.py` script.

### REST endpoints

- `POST /ml/predictions/abandonment`
- `POST /ml/predictions/batch`
- `GET /ml/predictions/health`

### Single prediction request example

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

### Single prediction response example

```json
{
    "success": true,
    "prediction": {
        "abandonmentRisk": 0.0106,
        "willAbandon": false,
        "confidence": 0.5106,
        "threshold": 0.5
    },
    "timestamp": "2026-03-25T10:30:00.000Z",
    "metadata": {
        "modelVersion": "1.0",
        "executionTimeMs": 145
    }
}
```

### CLI contract (script-level integration)

`ml/predict.py` supports stdin JSON payloads and prints JSON results to stdout:

```bash
echo '{"features":{"timeOnPage":83,"scrollDepth":41.4,"clickMisses":0,"hesitations":2,"abandonmentRisk":0.2,"helpTriggered":0,"hasError":0,"multiplePages":1}}' | python ml/predict.py
```

This interface is used by the backend prediction service and can be reused by
other consumers (workers, SDK bridges, batch jobs).

## Configuration Customization

Edit `config.py` to adjust model hyperparameters:

```python
LIGHTGBM_CONFIG = {
    "learning_rate": 0.05,      # Lower = slower, more accurate
    "num_leaves": 31,           # Complexity of each tree
    "max_depth": 7,             # Max tree depth
    "n_estimators": 100,        # Number of boosting rounds
    "subsample": 0.8,           # Data sampling rate
    "colsample_bytree": 0.8,   # Feature sampling rate
}
```

### Recommended Starting Points

**For Speed** (Quick experimentation):
```python
"learning_rate": 0.1,
"n_estimators": 50,
"num_leaves": 15,
```

**Balanced** (Default):
```python
"learning_rate": 0.05,
"n_estimators": 100,
"num_leaves": 31,
```

**For Accuracy** (Production):
```python
"learning_rate": 0.02,
"n_estimators": 200,
"num_leaves": 63,
```

## Database Connection Setup

For model training to work, you need to configure database access:

1. Ensure PostgreSQL is running
2. Create `.env` file in the `ml/` directory:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trustdev_db
DB_USER=postgres
DB_PASSWORD=your_password
```

3. Verify connection:

```bash
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://user:pwd@localhost:5432/db'); print('✅ Connection OK')"
```

## Troubleshooting

### Issue: "No module named 'lightgbm'"

**Solution**: Reinstall LightGBM

```bash
pip uninstall lightgbm
pip install --no-cache-dir lightgbm==4.6.0
```

### Issue: "psycopg2.OperationalError: connection refused"

**Solution**: Verify PostgreSQL is running and connection string in `.env` is correct

```bash
# Check PostgreSQL service
# Windows
net start postgresql-x64-15

# macOS
brew services start postgresql

# Linux
sudo systemctl start postgresql
```

### Issue: "Out of Memory" during model training

**Solution**: Reduce batch size and sample size in config.py

```python
"bagging_fraction": 0.5,  # Reduce from 0.8
"feature_fraction": 0.5,  # Reduce from 0.8
```

### Issue: Virtual environment not activating

**Solution**: Check Python path and reinstall venv

```bash
python -m venv --clear venv
# Then activate again
```

## Development Tools

### Code Quality

```bash
# Format code
black setup_ml_env.py config.py

# Check types
mypy setup_ml_env.py

# Lint
flake8 setup_ml_env.py --max-line-length=100
```

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Next Steps

After successful installation:

1. ✅ **Verify setup** with `python setup_ml_env.py`
2. 📓 **Explore data** with Jupyter notebooks
3. 🔍 **Analyze features** in `02_feature_analysis.ipynb`
4. 🎓 **Train model** with `03_model_training.ipynb`
5. 📊 **Evaluate performance** with `04_model_evaluation.ipynb`
6. 🚀 **Deploy to production** (when ready)

## Resources

- [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/modules/model_selection.html)
- [Random Forest vs Gradient Boosting](https://www.kaggle.com/c/titanic)
- [SHAP for Model Interpretability](https://shap-lrjball.readthedocs.io/)

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review setup logs in console output
3. Verify database connection with `.env` file
4. Check PostgreSQL service status

---

**Last Updated**: March 2026
**LightGBM Version**: 4.6.0
**Python Version**: 3.9+
