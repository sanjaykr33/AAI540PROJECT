# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#/kaggle/input/datasets/stealthtechnologies/employee-attrition-dataset/train.csv
#/kaggle/input/datasets/stealthtechnologies/employee-attrition-dataset/test.csv

import kagglehub

# Download latest version
path = kagglehub.dataset_download("stealthtechnologies/employee-attrition-dataset")

print("Path to dataset files:", path)

#/kaggle/input/datasets/stealthtechnologies/employee-attrition-dataset


import os

print("Path:", path)
print("Contents:", os.listdir(path))

# For more detail:
for item in os.listdir(path):
    full_path = os.path.join(path, item)
    if os.path.isdir(full_path):
        print(f"Folder: {item}/ →", os.listdir(full_path))
    else:
        print(f"File: {item} ({os.path.getsize(full_path):,} bytes)")


#---------------------------------------------------------

# attrition_prediction_pipeline.py
"""
End-to-end ML pipeline for Employee Attrition Prediction
Based on: https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset

Follows the architecture: Data → ETL/Preprocessing → Feature Engineering → Training → Evaluation → Batch Inference → Monitoring
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

import xgboost as xgb
from xgboost import XGBClassifier

import joblib
from datetime import datetime
import json

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

DATA_PATH_TRAIN = "/kaggle/input/datasets/stealthtechnologies/employee-attrition-dataset/train.csv"  # update path
DATA_PATH_TEST  = "/kaggle/input/datasets/stealthtechnologies/employee-attrition-dataset/test.csv"   # optional holdout

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Attrition"
POS_LABEL = "Left"   # or 1 depending on encoding
NEG_LABEL = "Stayed" # or 0

RANDOM_STATE = 42

# ────────────────────────────────────────────────
# 1. Load & Validate Data
# ────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load dataset and perform basic validation"""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Basic checks
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    
    print("\nTarget distribution:")
    print(df[TARGET].value_counts(normalize=True).round(3))
    
    return df


# ────────────────────────────────────────────────
# 2. Preprocessing & Feature Engineering
# ────────────────────────────────────────────────

def get_feature_types(df: pd.DataFrame):
    """Classify features into numeric / categorical / ordinal"""
    id_cols = ["Employee ID"] if "Employee ID" in df.columns else []
    
    drop_cols = id_cols
    
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in drop_cols + [TARGET]]
    
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_features = [c for c in categorical_features if c not in drop_cols + [TARGET]]
    
    # Common ordinal features in this dataset
    ordinal_features = [
        col for col in ["Education Level", "Job Satisfaction", "Work-Life Balance",
                        "Performance Rating", "Company Reputation", "Employee Recognition"]
        if col in df.columns
    ]
    categorical_features = [c for c in categorical_features if c not in ordinal_features]
    
    print(f"Numeric:      {len(numeric_features)}")
    print(f"Categorical:  {len(categorical_features)}")
    print(f"Ordinal:      {len(ordinal_features)}")
    
    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "ordinal": ordinal_features,
        "drop": drop_cols
    }


def build_preprocessor(feature_types: dict):
    """Create scikit-learn preprocessing pipeline"""
    
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_types["numeric"]),
            ("cat", categorical_transformer, feature_types["categorical"]),
            ("ord", ordinal_transformer, feature_types["ordinal"]),
        ],
        remainder="drop"
    )
    
    return preprocessor


# ────────────────────────────────────────────────
# 3. Full Modeling Pipeline
# ────────────────────────────────────────────────

def build_model_pipeline(preprocessor):
    """Complete pipeline: preprocessing + classifier"""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        scale_pos_weight=3,           # adjust based on class imbalance
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        enable_categorical=False      # we one-hot encode beforehand
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    return pipeline


# ────────────────────────────────────────────────
# 4. Training & Evaluation
# ────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, feature_types: dict):
    X = df.drop(columns=[TARGET] + feature_types["drop"])
    y = df[TARGET].map({NEG_LABEL: 0, POS_LABEL: 1})   # encode target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    preprocessor = build_preprocessor(feature_types)
    pipeline = build_model_pipeline(preprocessor)
    
    print("\nTraining XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "auc_roc":   roc_auc_score(y_test, y_prob)
    }
    
    print("\nModel Performance:")
    print(json.dumps({k: round(v, 4) for k,v in metrics.items()}, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[NEG_LABEL, POS_LABEL]))
    
    return pipeline, metrics, X_test, y_test, y_prob


# ────────────────────────────────────────────────
# 5. Save Model & Artifacts
# ────────────────────────────────────────────────

def save_model(pipeline, metrics: dict, prefix="attrition_xgb"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{prefix}_{timestamp}.joblib")
    metrics_path = os.path.join(MODEL_DIR, f"{prefix}_{timestamp}_metrics.json")
    
    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved: {model_path}")
    return model_path


# ────────────────────────────────────────────────
# 6. Batch Prediction Example
# ────────────────────────────────────────────────

def batch_predict(pipeline, new_data: pd.DataFrame, threshold=0.65):
    """Predict attrition risk for new/current employees"""
    probs = pipeline.predict_proba(new_data)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    result = pd.DataFrame({
        "attrition_probability": probs,
        "predicted_attrition": preds,
        "risk_level": np.where(probs >= 0.85, "High",
                              np.where(probs >= threshold, "Medium", "Low"))
    })
    
    return result


# ────────────────────────────────────────────────
# 7. Simple Monitoring Stub (expand with Evidently, Alibi Detect, etc.)
# ────────────────────────────────────────────────

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """Placeholder for data drift detection"""
    # In production → use evidently.ai or custom KS/PSI tests
    print("Drift detection placeholder:")
    print(f"Reference rows: {len(reference_df)} | Current rows: {len(current_df)}")
    # Add real drift metrics here...


# ────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────

if __name__ == "__main__":
    
    # 1. Load
    df = load_data(DATA_PATH_TRAIN)
    
    # 2. Feature classification
    feature_types = get_feature_types(df)
    
    # 3. Train & evaluate
    pipeline, metrics, X_test, y_test, y_prob = train_and_evaluate(df, feature_types)
    
    # 4. Save
    model_path = save_model(pipeline, metrics)
    
    # 5. Example batch prediction (using test split as proxy)
    print("\nExample batch predictions (first 8 rows):")
    sample_preds = batch_predict(pipeline, X_test.iloc[:8])
    print(sample_preds)
    
    # Optional: feature importance
    if hasattr(pipeline.named_steps["classifier"], "feature_importances_"):
        importance = pd.Series(
            pipeline.named_steps["classifier"].feature_importances_,
            index=pipeline.named_steps["preprocessor"].get_feature_names_out()
        ).sort_values(ascending=False).head(12)
        print("\nTop 12 Features:")
        print(importance.round(4))