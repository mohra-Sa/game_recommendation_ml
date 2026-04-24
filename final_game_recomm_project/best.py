import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import re
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    if 'PackageCount' in df.columns:
        df.drop(columns=['PackageCount'], inplace=True)

    conditions_dev = [
        (df['DeveloperCount'] == 0),
        (df['DeveloperCount'] == 1),
        (df['DeveloperCount'] >= 2)
    ]
    choices_dev = [0, 1, 2]
    df['Developer_Tier'] = np.select(conditions_dev, choices_dev, default=1)
    if 'DeveloperCount' in df.columns:
        df.drop(columns=['DeveloperCount'], inplace=True)

    conditions_pub = [
        (df['PublisherCount'] == 0),
        (df['PublisherCount'] == 1),
        (df['PublisherCount'] >= 2)
    ]
    choices_pub = [0, 1, 2]
    df['Publisher_Tier'] = np.select(conditions_pub, choices_pub, default=1)
    if 'PublisherCount' in df.columns:
        df.drop(columns=['PublisherCount'], inplace=True)

    conditions_age = [
        (df['RequiredAge'] == 0),
        (df['RequiredAge'] > 0) & (df['RequiredAge'] < 17),
        (df['RequiredAge'] >= 17)
    ]
    choices_age = [0, 1, 2]
    df['Age_Tier'] = np.select(conditions_age, choices_age, default=0)
    if 'RequiredAge' in df.columns:
        df.drop(columns=['RequiredAge'], inplace=True)

    conditions_ach = [
        (df['AchievementCount'] == 0),
        (df['AchievementCount'] > 0) & (df['AchievementCount'] <= 50),
        (df['AchievementCount'] > 50) & (df['AchievementCount'] <= 150)
    ]
    choices_ach = [0, 1, 2]
    df['Achievement_Tier'] = np.select(conditions_ach, choices_ach, default=3)
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
    df['DLCCount_Log'] = np.log1p(df['DLCCount'])
    df['Has_DLC'] = (df['DLCCount'] > 0).astype(int)
    print('abd alkarem is out')
    return df


def mohra_raneem_features(df):
    df = df.copy()

    text_columns = ['SupportURL', 'SupportEmail', 'Website', 'Reviews', 'ExtUserAcctNotice', 'DRMNotice', 'LegalNotice',
                    'Background', 'HeaderImage']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip().ne('').astype(int)

    textto_num_cols = ['DetailedDescrip', 'AboutText', 'ShortDescrip']
    for col in textto_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.len()

    common_langs = ['English', 'French', 'German', 'Italian', 'Spanish', 'Korean', 'Japanese', 'Russian', 'Turkish',
                    'Thai', 'Portuguese', 'Polish', 'Dutch', 'Arabic', 'Simplified Chinese', 'Traditional Chinese',
                    'Czech', 'Hungarian', 'Romanian']

    if 'SupportedLanguages' in df.columns:
        df['NumLanguages'] = df['SupportedLanguages'].apply(
            lambda x: sum(1 for l in common_langs if isinstance(x, str) and l in x)
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
        else:
            cpu = None

        Ram = None
        if ram:
            value, unit = ram[0]
            value = float(value)
            if unit.lower() == 'mb':
                value = value / 1000
            Ram = value
        else:
            Ram = None

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

    for c in col[:-1]:
        if c in df.columns:
            df[c] = np.log1p(df[c])

    if 'AchievementHighlightedCount' in df.columns:
        condition = [df['AchievementHighlightedCount'] == 0,
                     df['AchievementHighlightedCount'] == 10]
        choice = [0, 2]
        df['AchievementHighlightedCount'] = np.select(condition, choice, default=1)

    if 'SteamSpyOwnersVariance' in df.columns and 'SteamSpyOwners' in df.columns:
        df['relative_variation_owners'] = np.where(
            df['SteamSpyOwners'] > 0,
            df['SteamSpyOwnersVariance'] / df['SteamSpyOwners'],
            0.0
        )



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

def filter_features(df, variance_threshold=0.99, correlation_threshold=0.85):
    print("\n" + "=" * 50)
    print("Starting Feature Filtering (Variance & Correlation)")
    print("=" * 50)

    df_filtered = df.copy()
    target_col = 'RecommendationCount'
    cols_to_drop_variance = []
    cols_to_drop_correlation = []

    for col in df_filtered.columns:
        if col == target_col: continue
        top_value_freq = df_filtered[col].value_counts(normalize=True).iloc[0]
        if top_value_freq >= variance_threshold:
            cols_to_drop_variance.append(col)

    if cols_to_drop_variance:
        df_filtered.drop(columns=cols_to_drop_variance, inplace=True)
        print(f"Removed {len(cols_to_drop_variance)} Zero/Low Variance Features.")

    corr_matrix = df_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper.columns:
        if col == target_col: continue
        if any(upper[col] > correlation_threshold):
            cols_to_drop_correlation.append(col)

    if cols_to_drop_correlation:
        df_filtered.drop(columns=cols_to_drop_correlation, inplace=True)
        print(f"✂Removed {len(cols_to_drop_correlation)} Highly Correlated Features.")

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
    if len(kept_features) < 15: kept_features = feat_imp_df.head(15)['Feature'].tolist()

    original_kept_cols = [c for c in df.columns if re.sub('[^A-Za-z0-9_]+', '', c) in kept_features]
    df_selected = df[original_kept_cols + [target_col]].copy()

    print(f"Kept {len(kept_features)} features.")
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
    if 'AchievementCount' in df.columns and 'GameAge' in df.columns:
        df['Game_Momentum'] = df['AchievementCount'] / (df['GameAge'] + 1)

    print(f"✅ Added Interaction Features. New Shape: {df.shape}")
    return df


def tune_and_train_xgboost(df, target_col='RecommendationCount'):
    print("\n" + "=" * 50)
    print("Hyperparameter Tuning & Cross-Validation (XGBoost)")
    print("=" * 50)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    param_dist = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'min_child_weight': [5, 15, 30],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    print("Running Randomized Search with 3-Fold CV...")
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=20,
        scoring='r2',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_

    print(f"\n🏆 Best Parameters Found:\n{random_search.best_params_}")

    xgb_preds_log = best_xgb.predict(X_test)

    y_test_actual = np.expm1(y_test)
    xgb_preds_log_clipped = np.clip(xgb_preds_log, a_min=None, a_max=20)
    xgb_preds_actual = np.expm1(xgb_preds_log_clipped)

    r2 = r2_score(y_test, xgb_preds_log)
    rmse = np.sqrt(mean_squared_error(y_test_actual, xgb_preds_actual))
    mae = mean_absolute_error(y_test_actual, xgb_preds_actual)

    print("\n📈 Tuned XGBoost Final Performance (WITH SAMA'S FEATURES RESTORED):")
    print(f"   - R² Score (Log Scale) : {r2:.4f}")
    print(f"   - RMSE (Actual Count)  : {rmse:,.2f}")
    print(f"   - MAE (Actual Count)   : {mae:,.2f}")
    print("=" * 50)

    return best_xgb

df = pd.read_csv('train_data.csv')
df_processed = run_full_pipeline(df)
df_filtered = filter_features(df_processed)
df_golden = lgbm_feature_selection(df_filtered)
df_with_interactions = engineer_interaction_features(df_golden)
final_tuned_model = tune_and_train_xgboost(df_with_interactions)