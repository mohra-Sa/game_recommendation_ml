

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from ml_gamenew import (
    classification_preprocessing,
    inv_sq,                        # needed to unpickle KNN if best weight was inv_sq
)

warnings.filterwarnings("ignore")


def test(csv_path: str, models_dir: str = 'saved_models_cls'):
    print("=" * 60)
    print("  Milestone 2 – Classification Test Script")
    print("=" * 60)

    # ── 1. Load & preprocess test data ─────────────────────────────────────
    print(f"\n[1/3] Loading test data: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"      Shape: {df_raw.shape}")

    print("\n[2/3] Running classification preprocessing pipeline ...")
    df_processed = classification_preprocessing(df_raw)
    print(f"      Shape after engineering: {df_processed.shape}")

    # ── 2. Split features and target ───────────────────────────────────────
    X = (df_processed
         .drop(columns=['GamePopularity'], errors='ignore')
         .select_dtypes(include=[np.number])
         .fillna(0))

    # Grab ground truth if the TA included it (for accuracy reporting)
    y_true = None
    if 'GamePopularity' in df_processed.columns:
        y_true = df_processed['GamePopularity'].reset_index(drop=True)

    # ── 3. Load saved artefacts ────────────────────────────────────────────
    print(f"\n[3/3] Loading models from: {models_dir}/  →  predicting ...")
    p = Path(models_dir)

    # Shared artefacts (RF, SVM, DT)
    with open(p / 'cls_selected_features.pkl', 'rb') as f:
        cls_selected_features = pickle.load(f)
    with open(p / 'cls_scaler.pkl', 'rb') as f:
        cls_scaler = pickle.load(f)

    # KNN-specific artefacts
    with open(p / 'knn_selected_features.pkl', 'rb') as f:
        knn_selected_features = pickle.load(f)
    with open(p / 'knn_scaler.pkl', 'rb') as f:
        knn_scaler = pickle.load(f)
    with open(p / 'knn_best_params.pkl', 'rb') as f:
        knn_params = pickle.load(f)

    # Models
    with open(p / 'rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open(p / 'svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open(p / 'dt_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open(p / 'knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    # ── Prepare the two feature/scale paths ────────────────────────────────

    # Path A: RF / SVM / DT  →  lgbm features + cls_scaler
    for col in cls_selected_features:
        if col not in X.columns:
            X[col] = 0
    X_main = X[cls_selected_features].fillna(0)
    X_main_scaled = cls_scaler.transform(X_main)

    # Path B: KNN  →  knn features (subset of Path A) + knn_scaler
    for col in knn_selected_features:
        if col not in X.columns:
            X[col] = 0
    X_knn = X[knn_selected_features].fillna(0)
    X_knn_scaled = knn_scaler.transform(X_knn)

    # ── Predict with all 4 models ──────────────────────────────────────────
    models = [
        ('Random Forest', rf_model,  X_main_scaled),
        ('SVM',           svm_model, X_main_scaled),
        ('Decision Tree', dt_model,  X_main_scaled),
        ('KNN',           knn_model, X_knn_scaled),
    ]

    all_preds = {}
    print("\n── Results ─────────────────────────────────────────────")
    for name, model, X_in in models:
        preds = model.predict(X_in)
        all_preds[name] = preds
        if y_true is not None:
            acc = accuracy_score(y_true, preds)
            print(f"  {name:<16} |  Accuracy: {acc * 100:.2f}%")
        else:
            print(f"  {name:<16} |  Predictions generated ({len(preds)} rows)")
    print("-" * 55)

    if y_true is not None:
        print("\n── Detailed Report (Random Forest) ─────────────────────")
        print(classification_report(y_true, all_preds['Random Forest']))

    # ── Save predictions ───────────────────────────────────────────────────
    out_df = pd.DataFrame(all_preds)
    out_df.insert(0, 'Row', range(len(out_df)))
    out_path = 'predictions_milestone2.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}  ({len(out_df)} rows)")
    print("=" * 60)

    return out_df



if __name__ == '__main__':

        test_file_path = "data/test_data2.csv"

        test(csv_path=test_file_path)
