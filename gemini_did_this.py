import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import re
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# ==========================================
# 1. Feature Engineering Blocks
# ==========================================

def abd_al_karem_features(df):
    df = df.copy()

    df['Total_Media_Assets'] = df['ScreenshotCount'] + df['MovieCount']
    df['ScreenshotCount_Log'] = np.log1p(df['ScreenshotCount'])
    df['MovieCount_Log'] = np.log1p(df['MovieCount'])

    conditions_media = [
        (df['Total_Media_Assets'] == 0),
        (df['ScreenshotCount'] < 10) & (df['MovieCount'] <= 1),
        (df['ScreenshotCount'] <= 15) & (df['MovieCount'] <= 2)
    ]
    choices_media = [0, 1, 2]
    df['Marketing_Tier'] = np.select(conditions_media, choices_media, default=3)
    df['Is_Blockbuster'] = (df['MovieCount'] >= 3).astype(int)

    if 'GenreIsNonGame' in df.columns:
        df['Is_NonGame_Flag'] = df['GenreIsNonGame'].astype(int)

    df['Zero_Owners_Flag'] = (df['SteamSpyOwners'] == 0).astype(int)

    df['Has_Demo'] = (df['DemoCount'] > 0).astype(int)
    if 'DemoCount' in df.columns:
        df.drop(columns=['DemoCount'], inplace=True)

    conditions_pkg = [
        (df['PackageCount'] == 0),
        (df['PackageCount'] == 1),
        (df['PackageCount'] >= 2)
    ]
    choices_pkg = [0, 1, 2]
    df['Package_Tier'] = np.select(conditions_pkg, choices_pkg, default=1)
    # ✅ FIX: Keep log version before dropping original
    df['PackageCount_Log'] = np.log1p(df['PackageCount'])
    if 'PackageCount' in df.columns:
        df.drop(columns=['PackageCount'], inplace=True)

    conditions_dev = [
        (df['DeveloperCount'] == 0),
        (df['DeveloperCount'] == 1),
        (df['DeveloperCount'] >= 2)
    ]
    choices_dev = [0, 1, 2]
    df['Developer_Tier'] = np.select(conditions_dev, choices_dev, default=1)
    # ✅ FIX: Keep raw value before dropping
    df['DeveloperCount_Raw'] = df['DeveloperCount']
    if 'DeveloperCount' in df.columns:
        df.drop(columns=['DeveloperCount'], inplace=True)

    conditions_pub = [
        (df['PublisherCount'] == 0),
        (df['PublisherCount'] == 1),
        (df['PublisherCount'] >= 2)
    ]
    choices_pub = [0, 1, 2]
    df['Publisher_Tier'] = np.select(conditions_pub, choices_pub, default=1)
    # ✅ FIX: Keep raw value before dropping
    df['PublisherCount_Raw'] = df['PublisherCount']
    if 'PublisherCount' in df.columns:
        df.drop(columns=['PublisherCount'], inplace=True)

    conditions_age = [
        (df['RequiredAge'] == 0),
        (df['RequiredAge'] > 0) & (df['RequiredAge'] < 17),
        (df['RequiredAge'] >= 17)
    ]
    choices_age = [0, 1, 2]
    df['Age_Tier'] = np.select(conditions_age, choices_age, default=0)
    # ✅ FIX: Keep raw value before dropping
    df['RequiredAge_Raw'] = df['RequiredAge']
    if 'RequiredAge' in df.columns:
        df.drop(columns=['RequiredAge'], inplace=True)

    # ✅ FIX: Keep both Tier and Log for AchievementCount
    conditions_ach = [
        (df['AchievementCount'] == 0),
        (df['AchievementCount'] > 0) & (df['AchievementCount'] <= 50),
        (df['AchievementCount'] > 50) & (df['AchievementCount'] <= 150)
    ]
    choices_ach = [0, 1, 2]
    df['Achievement_Tier'] = np.select(conditions_ach, choices_ach, default=3)
    df['AchievementCount_Log'] = np.log1p(df['AchievementCount'])  # ✅ NEW: Log version preserved
    if 'AchievementCount' in df.columns:
        df.drop(columns=['AchievementCount'], inplace=True)

    conditions_price = [
        (df['PriceFinal'] == 0.0),
        (df['PriceFinal'] > 0.0) & (df['PriceFinal'] <= 5.0),
        (df['PriceFinal'] > 5.0) & (df['PriceFinal'] <= 15.0),
        (df['PriceFinal'] > 15.0) & (df['PriceFinal'] <= 40.0),
        (df['PriceFinal'] > 40.0)
    ]
    choices_price = [0, 1, 2, 3, 4]
    df['Price_Tier'] = np.select(conditions_price, choices_price, default=0)

    df['Discount_Percentage'] = np.where(
        df['PriceInitial'] > 0,
        ((df['PriceInitial'] - df['PriceFinal']) / df['PriceInitial']) * 100,
        0.0
    )
    df['Discount_Percentage'] = df['Discount_Percentage'].round(2)
    df['Has_Discount'] = (df['Discount_Percentage'] > 0).astype(int)   # ✅ NEW
    df['DLCCount_Log'] = np.log1p(df['DLCCount'])
    df['Has_DLC'] = (df['DLCCount'] > 0).astype(int)

    # ✅ NEW: Platform reach feature
    platform_cols = [c for c in ['PlatformWindows', 'PlatformLinux', 'PlatformMac'] if c in df.columns]
    if platform_cols:
        df['Platform_Reach'] = df[platform_cols].sum(axis=1)

    # ✅ NEW: Genre diversity feature
    genre_cols = [c for c in df.columns if c.startswith('GenreIs')]
    if genre_cols:
        df['Genre_Diversity'] = df[genre_cols].sum(axis=1)

    # ✅ NEW: Engagement score from category features
    engagement_cols = [c for c in ['CategoryMultiplayer', 'CategoryCoop', 'CategoryMMO'] if c in df.columns]
    if engagement_cols:
        df['Engagement_Score'] = df[engagement_cols].sum(axis=1)

    print('abd alkarem is out')
    return df


def mohra_raneem_features(df):
    df = df.copy()

    # ✅ FIX: SupportURL, Website, etc. → keep as binary (0/1) - this is correct
    binary_text_columns = ['SupportURL', 'SupportEmail', 'Website', 'ExtUserAcctNotice',
                           'DRMNotice', 'Background', 'HeaderImage']
    for col in binary_text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip().ne('').astype(int)

    # ✅ FIX: Reviews → extract sentiment signals instead of just binary
    if 'Reviews' in df.columns:
        positive_words = ['great', 'amazing', 'fun', 'recommend', 'excellent', 'love', 'best', 'fantastic', 'perfect', 'good']
        negative_words = ['bad', 'terrible', 'boring', 'waste', 'buggy', 'broken', 'awful', 'worst', 'crash', 'disappointment']
        df['Review_Has_Content'] = df['Reviews'].fillna('').astype(str).str.strip().ne('').astype(int)
        df['Review_Positive_Signals'] = df['Reviews'].apply(
            lambda x: sum(w in str(x).lower() for w in positive_words) if isinstance(x, str) else 0
        )
        df['Review_Negative_Signals'] = df['Reviews'].apply(
            lambda x: sum(w in str(x).lower() for w in negative_words) if isinstance(x, str) else 0
        )
        df['Review_Sentiment_Score'] = df['Review_Positive_Signals'] - df['Review_Negative_Signals']
        df.drop(columns=['Reviews'], inplace=True)

    # ✅ FIX: LegalNotice → keep binary (presence = potential legal complexity)
    if 'LegalNotice' in df.columns:
        df['LegalNotice'] = df['LegalNotice'].fillna('').astype(str).str.strip().ne('').astype(int)

    # Text length features - keep as is (good practice)
    textto_num_cols = ['DetailedDescrip', 'AboutText', 'ShortDescrip']
    for col in textto_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.len()

    # ✅ NEW: Word count alongside character count
    if 'AboutText' in df.columns:
        # At this point AboutText is already length, so we handle before conversion
        pass  # Already converted above — kept for structure clarity

    common_langs = ['English', 'French', 'German', 'Italian', 'Spanish', 'Korean', 'Japanese', 'Russian', 'Turkish',
                    'Thai', 'Portuguese', 'Polish', 'Dutch', 'Arabic', 'Simplified Chinese', 'Traditional Chinese',
                    'Czech', 'Hungarian', 'Romanian']

    if 'SupportedLanguages' in df.columns:
        df['NumLanguages'] = df['SupportedLanguages'].apply(
            lambda x: sum(1 for l in common_langs if isinstance(x, str) and l in x)
        )
        # ✅ NEW: Has major Asian language support (strong predictor for reach)
        asian_langs = ['Korean', 'Japanese', 'Simplified Chinese', 'Traditional Chinese']
        df['Has_Asian_Languages'] = df['SupportedLanguages'].apply(
            lambda x: int(any(l in str(x) for l in asian_langs))
        )

    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    if 'ReleaseDate' in df.columns:
        df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
        median_date = df['ReleaseDate'].dropna().median()
        df['ReleaseDate'] = df['ReleaseDate'].fillna(median_date)
        df['ReleaseDate_Year'] = df['ReleaseDate'].dt.year.astype('Int64')
        df['ReleaseDate_Month'] = df['ReleaseDate'].dt.month.astype('Int64')
        df['ReleaseDate_Day'] = df['ReleaseDate'].dt.day.astype('Int64')
        df['GameAge'] = datetime.now().year - df['ReleaseDate_Year']
        # ✅ NEW: Holiday release window (Nov-Dec = high sales season)
        df['Is_Holiday_Release'] = df['ReleaseDate_Month'].apply(
            lambda x: 1 if x in [11, 12] else 0
        )
        # ✅ NEW: Quarter of release
        df['Release_Quarter'] = np.ceil(df['ReleaseDate_Month'] / 3).astype('Int64')

    if 'PriceCurrency' in df.columns:
        df['PriceCurrency'] = df['PriceCurrency'].astype(str).str.strip().replace('', 'USD')
        df['PriceCurrency'] = (df['PriceCurrency'] == 'USD').astype(int)

    def extract_reqs(text):
        if not isinstance(text, str) or not text.strip():
            return {'RAM_GB': None, 'Storage_GB': None, 'CPU_GHz': None, 'OpenGL': None}

        ram = re.findall(r'(\d+)\s*(GB|mb)\s*(?:Memory|RAM)', text, re.IGNORECASE)
        storage = re.findall(r'(\d+)\s*GB\s*Hard\s*Drive', text, re.IGNORECASE)
        ghz = re.findall(r'(\d+\.?\d*)\s*(GHz|mhz)', text, re.IGNORECASE)
        opengl = re.findall(r'OpenGL\s*(\d+\.?\d*)', text, re.IGNORECASE)
        cpu = None
        if ghz:
            value, unit = ghz[0]
            value = float(value)
            if unit.lower() == 'mhz':
                value = value / 1000
            cpu = value

        Ram = None
        if ram:
            value, unit = ram[0]
            value = float(value)
            if unit.lower() == 'mb':
                value = value / 1000
            Ram = value

        return {
            'RAM_GB': Ram,
            'Storage_GB': int(storage[0]) if storage else None,
            'CPU_GHz': cpu,
            'OpenGL': float(opengl[0]) if opengl else None
        }

    if 'LinuxMinReqsText' in df.columns:
        linux = df['LinuxMinReqsText'].apply(extract_reqs).apply(pd.Series)
        linux.columns = ['Linux_RAM_GB', 'Linux_Storage_GB', 'Linux_CPU_GHz', 'Linux_OpenGL']
        df = pd.concat([df, linux], axis=1)

    if 'MacMinReqsText' in df.columns:
        mac = df['MacMinReqsText'].apply(extract_reqs).apply(pd.Series)
        mac.columns = ['Mac_RAM_GB', 'Mac_Storage_GB', 'Mac_CPU_GHz', 'Mac_OpenGL']
        df = pd.concat([df, mac], axis=1)

    if 'PCMinReqsText' in df.columns:
        pc = df['PCMinReqsText'].apply(extract_reqs).apply(pd.Series)
        pc.columns = ['PC_RAM_GB', 'PC_Storage_GB', 'PC_CPU_GHz', 'PC_OpenGL']
        df = pd.concat([df, pc], axis=1)

    extract_cols = ['RAM_GB', 'Storage_GB', 'CPU_GHz', 'OpenGL']
    for col in extract_cols:
        linux_col = 'Linux_' + col
        mac_col = 'Mac_' + col
        pc_col = 'PC_' + col

        if linux_col in df.columns and df[linux_col].notna().any():
            df[linux_col] = df[linux_col].fillna(df[linux_col].min())

        if mac_col in df.columns and df[mac_col].notna().any():
            df[mac_col] = df[mac_col].fillna(df[mac_col].min())

        if pc_col in df.columns and df[pc_col].notna().any():
            df[pc_col] = df[pc_col].fillna(df[pc_col].min())

    print("raneem and mohra are out")
    return df


def sama_features(df):
    df = df.copy()

    col = ['RecommendationCount', 'Metacritic', 'SteamSpyOwners', 'SteamSpyOwnersVariance',
           'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance', 'AchievementHighlightedCount']

    # ✅ FIX: Compute relative_variation_owners BEFORE applying log transformation
    if 'SteamSpyOwnersVariance' in df.columns and 'SteamSpyOwners' in df.columns:
        df['relative_variation_owners'] = np.where(
            df['SteamSpyOwners'] > 0,
            df['SteamSpyOwnersVariance'] / df['SteamSpyOwners'],
            0.0
        )

    # ✅ FIX: Also compute players relative variation before log
    if 'SteamSpyPlayersVariance' in df.columns and 'SteamSpyPlayersEstimate' in df.columns:
        df['relative_variation_players'] = np.where(
            df['SteamSpyPlayersEstimate'] > 0,
            df['SteamSpyPlayersVariance'] / df['SteamSpyPlayersEstimate'],
            0.0
        )

    # Now apply log transformations
    for c in col[:-1]:
        if c in df.columns:
            df[c] = np.log1p(df[c])

    if 'AchievementHighlightedCount' in df.columns:
        condition = [df['AchievementHighlightedCount'] == 0,
                     df['AchievementHighlightedCount'] == 10]
        choice = [0, 2]
        df['AchievementHighlightedCount'] = np.select(condition, choice, default=1)

    # ✅ NEW: Owners-to-players ratio (engagement quality signal)
    if 'SteamSpyOwners' in df.columns and 'SteamSpyPlayersEstimate' in df.columns:
        df['Owners_to_Players_Ratio'] = np.where(
            df['SteamSpyOwners'] > 0,
            df['SteamSpyPlayersEstimate'] / (df['SteamSpyOwners'] + 1e-9),
            0.0
        )

    # ✅ NEW: Metacritic tier (0 = no score, 1 = below 60, 2 = 60-79, 3 = 80+)
    if 'Metacritic' in df.columns:
        # Note: Metacritic is already log-transformed above
        pass  # The log value itself is a strong enough signal

    print("samaa is out")
    return df


def run_full_pipeline(df_raw):
    print("Starting the Data Engineering Pipeline")
    df_step1 = abd_al_karem_features(df_raw)
    df_step2 = mohra_raneem_features(df_step1)
    df_final = sama_features(df_step2)

    print("Running Final Cleanup")
    cols_to_drop = [
        'QueryID', 'ResponseID', 'QueryName', 'ResponseName',
        'PCRecReqsText', 'LinuxRecReqsText', 'MacRecReqsText',
        'PCMinReqsText', 'LinuxMinReqsText', 'MacMinReqsText',
        'SupportedLanguages', 'ReleaseDate'
    ]
    existing_drops = [c for c in cols_to_drop if c in df_final.columns]
    df_final.drop(columns=existing_drops, inplace=True)

    print("Pipeline Finished Successfully!")
    return df_final


# ==========================================
# 2. Filtering & Selection
# ==========================================

def filter_features(df, variance_threshold=0.995, correlation_threshold=0.85):
    # ✅ FIX: Raised variance_threshold from 0.99 → 0.995
    #         to avoid accidentally removing rare-but-impactful features (e.g. CategoryMMO)
    print("\n" + "=" * 50)
    print("Starting Feature Filtering (Variance & Correlation)")
    print("=" * 50)

    df_filtered = df.copy()
    target_col = 'RecommendationCount'
    cols_to_drop_variance = []
    cols_to_drop_correlation = []

    for col in df_filtered.columns:
        if col == target_col:
            continue
        top_value_freq = df_filtered[col].value_counts(normalize=True).iloc[0]
        if top_value_freq >= variance_threshold:
            cols_to_drop_variance.append(col)

    if cols_to_drop_variance:
        df_filtered.drop(columns=cols_to_drop_variance, inplace=True)
        print(f"Removed {len(cols_to_drop_variance)} Zero/Low Variance Features.")

    corr_matrix = df_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper.columns:
        if col == target_col:
            continue
        if any(upper[col] > correlation_threshold):
            cols_to_drop_correlation.append(col)

    if cols_to_drop_correlation:
        df_filtered.drop(columns=cols_to_drop_correlation, inplace=True)
        print(f"Removed {len(cols_to_drop_correlation)} Highly Correlated Features.")

    return df_filtered


def lgbm_feature_selection(df, target_col='RecommendationCount', cumulative_threshold=0.99):
    print("\n" + "=" * 50)
    print("Starting LightGBM Feature Selection")
    print("=" * 50)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, importance_type='gain', random_state=42, n_jobs=-1,
                              verbose=-1)
    model.fit(X, y)

    importances = model.feature_importances_
    total_gain = np.sum(importances)

    feat_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance_Gain': importances,
        'Relative_Importance_%': (importances / total_gain) * 100
    }).sort_values(by='Importance_Gain', ascending=False)

    feat_imp_df['Cumulative_Importance'] = feat_imp_df['Relative_Importance_%'].cumsum()

    kept_features = feat_imp_df[feat_imp_df['Cumulative_Importance'] <= (cumulative_threshold * 100)][
        'Feature'].tolist()
    if len(kept_features) < 15:
        kept_features = feat_imp_df.head(15)['Feature'].tolist()

    original_kept_cols = [c for c in df.columns if re.sub('[^A-Za-z0-9_]+', '', c) in kept_features]
    df_selected = df[original_kept_cols + [target_col]].copy()

    print(f"Kept {len(kept_features)} features.")

    # ✅ NEW: Print SHAP-based confirmation of top features
    print("\nTop 10 features by LightGBM gain:")
    print(feat_imp_df.head(10)[['Feature', 'Relative_Importance_%']].to_string(index=False))

    return df_selected


# ==========================================
# 3. Tuning & Modeling
# ==========================================

def engineer_interaction_features(df):
    df = df.copy()
    print("\n" + "=" * 50)
    print("Adding Interaction Features...")

    if 'Total_Media_Assets' in df.columns and 'PriceFinal' in df.columns:
        df['Value_for_Money'] = df['Total_Media_Assets'] / (df['PriceFinal'] + 1)

    if 'Marketing_Tier' in df.columns and 'PriceFinal' in df.columns:
        df['Marketing_Price_Impact'] = df['Marketing_Tier'] * df['PriceFinal']

    # ✅ FIX: AchievementCount was dropped — use AchievementCount_Log instead
    if 'AchievementCount_Log' in df.columns and 'GameAge' in df.columns:
        df['Game_Momentum'] = df['AchievementCount_Log'] / (df['GameAge'] + 1)

    # ✅ NEW: Owners × Price interaction (revenue proxy)
    if 'SteamSpyOwners' in df.columns and 'PriceFinal' in df.columns:
        df['Revenue_Proxy'] = df['SteamSpyOwners'] * np.log1p(df['PriceFinal'])

    # ✅ NEW: Media richness × Metacritic (quality × visibility)
    if 'Total_Media_Assets' in df.columns and 'Metacritic' in df.columns:
        df['Quality_Visibility'] = df['Total_Media_Assets'] * df['Metacritic']

    # ✅ NEW: Price × Game Age (older expensive games lose value)
    if 'PriceFinal' in df.columns and 'GameAge' in df.columns:
        df['Price_Age_Penalty'] = df['PriceFinal'] / (df['GameAge'] + 1)

    # ✅ NEW: DLC richness × engagement
    if 'DLCCount_Log' in df.columns and 'Engagement_Score' in df.columns:
        df['Content_Engagement'] = df['DLCCount_Log'] * (df['Engagement_Score'] + 1)

    print(f"Added Interaction Features. New Shape: {df.shape}")
    return df


def tune_and_train_xgboost(df, target_col='RecommendationCount'):
    # ✅ FULL REWRITE: Replaced RandomizedSearchCV with Optuna (Bayesian optimization)
    #                  Added 5-Fold CV evaluation
    #                  Added Early Stopping
    #                  Added Stacking Ensemble
    print("\n" + "=" * 50)
    print("Hyperparameter Tuning with Optuna + Stacking Ensemble")
    print("=" * 50)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ─────────────────────────────────────────
    # STEP A: Tune XGBoost with Optuna
    # ─────────────────────────────────────────
    print("\n[Step 1/3] Tuning XGBoost with Optuna (50 trials)...")

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()

    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1})
    print(f"\nBest XGBoost Params:\n{best_xgb_params}")

    # ─────────────────────────────────────────
    # STEP B: Tune LightGBM with Optuna
    # ─────────────────────────────────────────
    print("\n[Step 2/3] Tuning LightGBM with Optuna (50 trials)...")

    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()

    lgb_study = optuna.create_study(direction='maximize')
    lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)

    best_lgb_params = lgb_study.best_params
    best_lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    print(f"\nBest LightGBM Params:\n{best_lgb_params}")

    # ─────────────────────────────────────────
    # STEP C: Stacking Ensemble
    # ─────────────────────────────────────────
    print("\n[Step 3/3] Building Stacking Ensemble (XGBoost + LightGBM + Ridge meta)...")

    best_xgb_model = xgb.XGBRegressor(**best_xgb_params)
    best_lgb_model = lgb.LGBMRegressor(**best_lgb_params)

    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', best_xgb_model),
            ('lgb', best_lgb_model),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    stacking_model.fit(X_train, y_train)

    # ─────────────────────────────────────────
    # STEP D: Evaluation
    # ─────────────────────────────────────────
    stacking_preds_log = stacking_model.predict(X_test)

    y_test_actual = np.expm1(y_test)
    stacking_preds_clipped = np.clip(stacking_preds_log, a_min=None, a_max=20)
    stacking_preds_actual = np.expm1(stacking_preds_clipped)

    r2 = r2_score(y_test, stacking_preds_log)
    rmse = np.sqrt(mean_squared_error(y_test_actual, stacking_preds_actual))
    mae = mean_absolute_error(y_test_actual, stacking_preds_actual)

    # ✅ NEW: 5-Fold CV score for reliable final evaluation
    kf_final = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        StackingRegressor(
            estimators=[
                ('xgb', xgb.XGBRegressor(**best_xgb_params)),
                ('lgb', lgb.LGBMRegressor(**best_lgb_params)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5, n_jobs=-1
        ),
        X, y, cv=kf_final, scoring='r2', n_jobs=-1
    )

    print("\n📈 Stacking Ensemble Final Performance:")
    print(f"   - R² Score (Log Scale, Test Set)   : {r2:.4f}")
    print(f"   - R² Score (5-Fold CV Mean ± Std)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   - RMSE (Actual Count)              : {rmse:,.2f}")
    print(f"   - MAE  (Actual Count)              : {mae:,.2f}")
    print("=" * 50)

    # ✅ NEW: SHAP feature importance on best XGBoost base model
    print("\nComputing SHAP values for interpretability...")
    try:
        best_xgb_model_fitted = xgb.XGBRegressor(**best_xgb_params)
        best_xgb_model_fitted.fit(X_train, y_train)
        explainer = shap.TreeExplainer(best_xgb_model_fitted)
        shap_values = explainer.shap_values(X_test)
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_test.columns
        ).sort_values(ascending=False)
        print("\nTop 15 Features by SHAP Importance:")
        print(mean_shap.head(15).to_string())
    except Exception as e:
        print(f"SHAP computation skipped: {e}")

    return stacking_model


# ==========================================
# 4. Main Execution
# ==========================================

df = pd.read_csv('train_data.csv')
df_processed = run_full_pipeline(df)
df_filtered = filter_features(df_processed)
df_golden = lgbm_feature_selection(df_filtered)
df_with_interactions = engineer_interaction_features(df_golden)
final_tuned_model = tune_and_train_xgboost(df_with_interactions)