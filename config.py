# LightGBM Configuration for Abandonment Prediction Model

# Model Parameters
LIGHTGBM_CONFIG = {
    # Learning Control
    "boosting_type": "gbdt",  # Gradient Boosting Decision Tree
    "objective": "binary",  # Binary classification (abandon/no abandon)
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 100,
    
    # Regularization
    "reg_lambda": 1.0,  # L2 regularization
    "reg_alpha": 0.0,   # L1 regularization
    "min_child_weight": 1e-3,
    "min_child_samples": 20,
    "subsample": 0.8,  # Bagging fraction
    "colsample_bytree": 0.8,
    "subsample_freq": 1,
    
    # Training
    "verbose": 1,
    "random_state": 42,
    "n_jobs": -1,  # Use all processors
    "force_row_wise": True,
    
    # Early Stopping
    "early_stopping_rounds": 10,
    "eval_metric": "auc",
}

# Feature Engineering Parameters
FEATURE_ENGINEERING = {
    "scale_features": True,
    "handle_missing": True,
    "remove_outliers": False,
    "apply_pca": False,
}

# Backend-aligned input schema (backend/src/ml/constants/dataset-schema.ts)
RAW_FEATURE_COLUMNS = [
    "timeOnPage",
    "scrollDepth",
    "clickMisses",
    "hesitations",
    "abandonmentRisk",
    "helpTriggered",
    "hasError",
    "multiplePages",
    "label",
]

# Training schema includes derived features from the ML preprocessing pipeline
TRAINING_FEATURE_COLUMNS = [
    "timeOnPage",
    "scrollDepth",
    "clickMisses",
    "hesitations",
    "helpTriggered",
    "hasError",
    "multiplePages",
    "timePerPage",
    "clickMissRate",
    "hesitationRate",
    "frictionScore",
    "highFriction",
    "multipleIssues",
]

PREPROCESSING_PARAMS = {
    "min_rows_for_training": 100,
    "missing_strategy": "median",
    "default_pages_single": 1,
    "default_pages_multi": 2,
    "friction_weights": {
        "clickMisses": 0.3,
        "hesitations": 0.25,
        "timeOnPage": 0.2,
        "scrollDepth": 0.25,
    },
}

# Dataset Parameters
DATASET_PARAMS = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "stratify": True,  # Stratified split for imbalanced data
    "random_state": 42,
}

# Evaluation Parameters
EVALUATION_PARAMS = {
    "threshold": 0.5,  # Classification threshold
    "compute_shap": True,  # Compute SHAP values for interpretability
}

# Semantic FAQ embedding configuration
SEMANTIC_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "normalize_embeddings": True,
    "batch_size": 32,
}

FAQ_EMBEDDING_PATHS = {
    "input_questions": "./data/raw/faq_questions.json",
    "output_embeddings": "./results/faq_embeddings.json",
}

SEMANTIC_SEARCH_CONFIG = {
    "top_k": 5,
    "min_similarity": 0.7,
}

# Paths
PATHS = {
    "models": "./models/",
    "data": "./data/",
    "results": "./results/",
    "notebooks": "./notebooks/",
    "training": "./training/",
}
