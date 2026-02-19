# Insurance Claims Fraud Detection

Gradient-boosted ensemble system for identifying fraudulent insurance claims. Combines XGBoost, LightGBM, and CatBoost in a stacking architecture with 200+ engineered features for high-precision fraud classification.

## Overview

Insurance fraud represents billions in annual losses across the industry. This system applies modern gradient boosting methods with extensive feature engineering to flag suspicious claims for investigation. The ensemble approach balances sensitivity (catching fraud) against specificity (avoiding false positives that waste investigator time).

Key capabilities:

- **200+ engineered features** spanning provider behavior, claim characteristics, temporal patterns, and network relationships
- **Stacking ensemble** with XGBoost, LightGBM, and CatBoost as base learners and logistic regression as the meta-learner
- **Automated hyperparameter optimization** via Optuna with cross-validated average precision as the objective
- **Interpretable predictions** through SHAP values and feature importance analysis
- **Class imbalance handling** using SMOTE, ADASYN, and model-native class weighting

## Architecture

```
Raw Claims Data
      |
      v
Data Preprocessing
(imputation, encoding, balancing)
      |
      v
Feature Engineering (200+ features)
 |          |            |           |
 v          v            v           v
Provider  Claim      Temporal    Network
Features  Features   Features    Features
      \      |          |       /
       v     v          v      v
      Combined Feature Matrix
      |          |          |
      v          v          v
  XGBoost   LightGBM   CatBoost
      \         |         /
       v        v        v
     Stacking Meta-Learner
            |
            v
     Fraud Probability
```

## Installation

```bash
git clone https://github.com/yourusername/fraud-detection-ensemble.git
cd fraud-detection-ensemble

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### Generate synthetic data and train

```python
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import FraudDetectionModel
from src.evaluation import ModelEvaluator

# Generate synthetic demo data
preprocessor = DataPreprocessor()
df = preprocessor.generate_synthetic_data(n_samples=50000, fraud_rate=0.05)

# Preprocess
df = preprocessor.handle_missing_values(df)
df = preprocessor.encode_categoricals(df)
X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(df)

# Engineer features
engineer = FeatureEngineer()
X_train = engineer.build_feature_matrix(X_train)
X_test = engineer.build_feature_matrix(X_test)

# Train ensemble
model = FraudDetectionModel()
model.train_stacking_ensemble(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator(model, X_test, y_test)
evaluator.generate_evaluation_report("reports/")
```

### Make predictions on new data

```bash
python src/predict.py --input data/new_claims.csv --output predictions.csv
```

## Feature Groups

### Provider Features
- Average claim amount per provider
- Claim frequency (daily, weekly, monthly)
- Specialty deviation score
- Patient volume relative to peers
- Out-of-network billing ratio

### Claim Features
- Diagnosis-procedure mismatch score
- Claim amount z-score
- Service duration anomaly flag
- Number of procedures per claim
- Unusual billing code combinations

### Temporal Features
- Rolling window claim counts (7d, 30d, 90d)
- Time between consecutive claims
- Weekend and holiday submission flags
- Claim velocity acceleration

### Network Features
- Provider-patient pair frequency
- Referral chain depth
- Shared patient count between providers
- Geographic outlier score

## Model Performance

Evaluated on held-out test set (10,000 claims, 5% fraud rate):

| Model | AUC-ROC | AUC-PR | Precision@90%Recall | F1 |
|-------|---------|--------|--------------------|----|
| XGBoost | 0.961 | 0.724 | 0.68 | 0.72 |
| LightGBM | 0.958 | 0.718 | 0.66 | 0.71 |
| CatBoost | 0.955 | 0.710 | 0.65 | 0.70 |
| **Stacking Ensemble** | **0.971** | **0.762** | **0.74** | **0.76** |

## Configuration

Model hyperparameters and training settings are defined in `configs/model_config.yaml`. Optuna can be used to automatically optimize these parameters.

## Project Structure

```
fraud-detection-ensemble/
├── configs/
│   └── model_config.yaml
├── notebooks/
│   └── exploration.py
├── src/
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## License

MIT License
