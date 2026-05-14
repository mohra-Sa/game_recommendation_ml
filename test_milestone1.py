import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from ml_gamenew import (
    abd_al_karem_features,
    mohra_raneem_features,
    sama_features,
    engineer_interaction_features,
)

warnings.filterwarnings("ignore")


def preprocess(df):
    df = abd_al_karem_features(df)
    df = mohra_raneem_features(df)
    df = sama_features(df)
    df = engineer_interaction_features(df)
    extra_cols = [
        'QueryID', 'ResponseID', 'QueryName', 'ResponseName',
        'PCRecReqsText', 'LinuxRecReqsText', 'MacRecReqsText',
        'PCMinReqsText', 'LinuxMinReqsText', 'MacMinReqsText',
        'SupportedLanguages', 'ReleaseDate',
    ]
    df = df.drop(columns=[c for c in extra_cols if c in df.columns])
    return df


def test(csv_path, models_dir='saved_models'):
    print("=" * 60)
    print("  Milestone 1 - Regression Test Script")
    print("=" * 60)
    print(f"\n[1/4] Loading test data: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"      Shape: {df_raw.shape}")
    print("\n[2/4] Running feature-engineering pipeline ...")
    df_processed = preprocess(df_raw)
    print(f"      Shape after engineering: {df_processed.shape}")
    print(f"\n[3/4] Loading saved models from: {models_dir}/")
    p = Path(models_dir)

    with open(p / 'selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    with open(p / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(p / 'stacking_model.pkl', 'rb') as f:
        stacking_model = pickle.load(f)
    with open(p / 'xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open(p / 'lgb_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)

    print(f"      Loaded {len(selected_features)} selected features.")

    # ── Align features & scale ──────────────────────────────────────────────
    print("\n[4/4] Predicting ...")
    X = df_processed.select_dtypes(include=[np.number])

    # Separate target if the TA's file includes it (for grading)
    y_true_log = None
    if 'RecommendationCount' in X.columns:
        y_true_log = X['RecommendationCount'].copy()
        X = X.drop(columns=['RecommendationCount'])

    # Handle any feature column missing in the test file
    for col in selected_features:
        if col not in X.columns:
            X[col] = 0

    X = X[selected_features].fillna(0)
    X_scaled = scaler.transform(X)  # transform only – NEVER re-fit on test data

    # ── Predict ─────────────────────────────────────────────────────────────
    y_pred_log = stacking_model.predict(X_scaled)
    y_pred_clipped = np.clip(y_pred_log, a_min=None, a_max=20)
    y_pred = np.expm1(y_pred_clipped)

    # ── Evaluate (MSE & R²) ─────────────────────────────────────────────────
    if y_true_log is not None:
        y_true = np.expm1(y_true_log)
        print("\n── Results ─────────────────────────────────────────────")
        for name, mdl in [('Stacking (main)', stacking_model),
                          ('XGBoost', xgb_model),
                          ('LightGBM', lgb_model)]:
            p_log = mdl.predict(X_scaled)
            p_clipped = np.clip(p_log, a_min=None, a_max=20)
            p_raw = np.expm1(p_clipped)
            mse = mean_squared_error(y_true, p_raw)
            r2 = r2_score(y_true_log, p_log)
            print(f"  {name:<18} |  R2: {r2:.4f}  |  MSE: {mse:.2f}")
        print("-" * 55)
    out = 'predictions_milestone1.csv'
    pd.DataFrame({'Predicted_RecommendationCount': y_pred}).to_csv(out, index=False)
    print(f"\nPredictions saved to: {out}  ({len(y_pred)} rows)")
    print("=" * 60)

    return y_pred
test_file_path = 'data/test_data.csv'
models_directory = 'saved_models'
test(csv_path=test_file_path, models_dir=models_directory)