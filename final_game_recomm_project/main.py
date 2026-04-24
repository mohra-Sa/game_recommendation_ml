import pandas as pd
import numpy as np

df = pd.read_csv('train_data.csv')
def abd_al_karem_features(df):

    df = df.copy()

    # 1. Media Features
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

    # البتاع دا ممكن يكون بيعمل اوفر فيتينج لسا هشوف

    df['Zero_Owners_Flag'] = (df['SteamSpyOwners'] == 0).astype(int)

    # 2. Demo Features
    df['Has_Demo'] = (df['DemoCount'] > 0).astype(int)
    if 'DemoCount' in df.columns:
        df.drop(columns=['DemoCount'], inplace=True)

    # 3. Package Features
    conditions_pkg = [
        (df['PackageCount'] == 0),
        (df['PackageCount'] == 1),
        (df['PackageCount'] >= 2)
    ]
    choices_pkg = [0, 1, 2]
    df['Package_Tier'] = np.select(conditions_pkg, choices_pkg, default=1)
    if 'PackageCount' in df.columns:
        df.drop(columns=['PackageCount'], inplace=True)

    # 4. Developer Features
    conditions_dev = [
        (df['DeveloperCount'] == 0),
        (df['DeveloperCount'] == 1),
        (df['DeveloperCount'] >= 2)
    ]
    choices_dev = [0, 1, 2]
    df['Developer_Tier'] = np.select(conditions_dev, choices_dev, default=1)
    if 'DeveloperCount' in df.columns:
        df.drop(columns=['DeveloperCount'], inplace=True)

    # 5. Publisher Features
    conditions_pub = [
        (df['PublisherCount'] == 0),
        (df['PublisherCount'] == 1),
        (df['PublisherCount'] >= 2)
    ]
    choices_pub = [0, 1, 2]
    df['Publisher_Tier'] = np.select(conditions_pub, choices_pub, default=1)
    if 'PublisherCount' in df.columns:
        df.drop(columns=['PublisherCount'], inplace=True)

    # 6. Age Features
    conditions_age = [
        (df['RequiredAge'] == 0),
        (df['RequiredAge'] > 0) & (df['RequiredAge'] < 17),
        (df['RequiredAge'] >= 17)
    ]
    choices_age = [0, 1, 2]
    df['Age_Tier'] = np.select(conditions_age, choices_age, default=0)
    if 'RequiredAge' in df.columns:
        df.drop(columns=['RequiredAge'], inplace=True)

    # 7. Achievement Features
    conditions_ach = [
        (df['AchievementCount'] == 0),
        (df['AchievementCount'] > 0) & (df['AchievementCount'] <= 50),
        (df['AchievementCount'] > 50) & (df['AchievementCount'] <= 150)
    ]
    choices_ach = [0, 1, 2]
    df['Achievement_Tier'] = np.select(conditions_ach, choices_ach, default=3)
    if 'AchievementCount' in df.columns:
        df.drop(columns=['AchievementCount'], inplace=True)

    # 8. Price & Discount Features
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


import pandas as pd
import numpy as np
import pycountry
from datetime import datetime
import re


def mohra_raneem_features(df):
    df = df.copy()

    # Fill missing values in text columns with empty strings and create binary features indicating the presence of text
    text_columns = ['SupportURL', 'SupportEmail', 'Website', 'Reviews', 'ExtUserAcctNotice', 'DRMNotice', 'LegalNotice',
                    'Background', 'HeaderImage']
    for col in text_columns:
        df[col] = df[col].fillna('').str.strip().ne('').astype(int)

    # Convert text columns to numeric features by calculating their length (number of characters)(this more accurately captures the amount of information provided in the text, which may be more relevant for recommendation than just the presence of text)
    textto_num_cols = ['DetailedDescrip', 'AboutText', 'ShortDescrip']
    for col in textto_num_cols:
        df[col] = df[col].fillna('').str.len()

    # anather way to list common languages using pycountry, but it may not be comprehensive and may miss some languages
    # common_langs = [lang.name for lang in pycountry.languages  if hasattr(lang, 'alpha_2')]
    common_langs = ['English', 'French', 'German', 'Italian', 'Spanish', 'Korean',
                    'Japanese', 'Russian', 'Turkish', 'Thai', 'Portuguese', 'Polish',
                    'Dutch', 'Arabic', 'Simplified Chinese', 'Traditional Chinese',
                    'Czech', 'Hungarian', 'Romanian']

    df['NumLanguages'] = df['SupportedLanguages'].apply(
        lambda x: sum(1 for l in common_langs if isinstance(x, str) and l in x)
    )

    # Convert boolean columns to integers (0 and 1)
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    if bool_cols:  # تأمين بسيط لتجنب خطأ لو مفيش عواميد Boolean
        df[bool_cols] = df[bool_cols].astype(int)

    # Fill missing values in ReleaseDate, convert to datetime, and extract year, month, day, and calculate game age
    df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
    median_date = df['ReleaseDate'].dropna().median()
    df['ReleaseDate'] = df['ReleaseDate'].fillna(median_date)
    df['ReleaseDate_Year'] = df['ReleaseDate'].dt.year.astype('Int64')
    df['ReleaseDate_Month'] = df['ReleaseDate'].dt.month.astype('Int64')
    df['ReleaseDate_Day'] = df['ReleaseDate'].dt.day.astype('Int64')
    df['GameAge'] = datetime.now().year - df['ReleaseDate_Year']
    print(df[['ReleaseDate', 'ReleaseDate_Year', 'ReleaseDate_Month', 'ReleaseDate_Day', 'GameAge']].head())

    # fill misssing values in PriceCurrency and encode it
    print(df['PriceCurrency'].value_counts(dropna=False))
    df['PriceCurrency'] = df['PriceCurrency'].str.strip().replace('', 'USD')
    df['PriceCurrency'] = (df['PriceCurrency'] == 'USD').astype(int)
    print("Missing PriceCurrency:", df['PriceCurrency'].isnull().sum())

    # Function to extract RAM, Storage, CPU, and OpenGL requirements from LinuxMinReqsText and MacMinReqsText
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

    # Linux
    linux = df['LinuxMinReqsText'].apply(extract_reqs).apply(pd.Series)
    linux.columns = ['Linux_RAM_GB', 'Linux_Storage_GB', 'Linux_CPU_GHz', 'Linux_OpenGL']

    # Mac
    mac = df['MacMinReqsText'].apply(extract_reqs).apply(pd.Series)
    mac.columns = ['Mac_RAM_GB', 'Mac_Storage_GB', 'Mac_CPU_GHz', 'Mac_OpenGL']

    # PC
    pc = df['PCMinReqsText'].apply(extract_reqs).apply(pd.Series)
    pc.columns = ['PC_RAM_GB', 'PC_Storage_GB', 'PC_CPU_GHz', 'PC_OpenGL']

    extract_cols = ['RAM_GB', 'Storage_GB', 'CPU_GHz', 'OpenGL']
    for col in extract_cols:
        linux_col = 'Linux_' + col
        mac_col = 'Mac_' + col
        pc_col = 'PC_' + col

        if linux[linux_col].notna().any():
            linux[linux_col] = linux[linux_col].fillna(linux[linux_col].min())

        if mac[mac_col].notna().any():
            mac[mac_col] = mac[mac_col].fillna(mac[mac_col].min())

        if pc[pc_col].notna().any():
            pc[pc_col] = pc[pc_col].fillna(pc[pc_col].min())

    df = pd.concat([df, linux, mac, pc], axis=1)
    print(df[['Linux_RAM_GB', 'Linux_Storage_GB', 'Linux_CPU_GHz', 'Linux_OpenGL',
              'Mac_RAM_GB', 'Mac_Storage_GB', 'Mac_CPU_GHz', 'Mac_OpenGL',
              'PC_RAM_GB', 'PC_Storage_GB', 'PC_CPU_GHz', 'PC_OpenGL']].iloc[8:21].head())

    print("raneem and mohra are out")
    return df





import pandas as pd
import numpy as np

def sama_features(df):
    # أخدنا نسخة لحماية الميموري
    df = df.copy()

    col = ['RecommendationCount','Metacritic','SteamSpyOwners','SteamSpyOwnersVariance',
           'SteamSpyPlayersEstimate','SteamSpyPlayersVariance','AchievementHighlightedCount']

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
    cols_to_drop = ['SteamSpyOwnersVariance','SteamSpyOwners','SteamSpyPlayersEstimate','SteamSpyPlayersVariance']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    print("samaa is out")
    return df


def rana_features(df):
    df['HasWebsite'] = df['Website'].notna().astype(int)
    df = df.drop(columns=['Website'])

    ## fill the nulls with USD
    df['PriceCurrency'] = df['PriceCurrency'].str.strip()
    df['PriceCurrency'] = df['PriceCurrency'].replace('', 'USD')

    df['HasDRM'] = (df['DRMNotice'].str.strip() != '').astype(int)  # i have 70 non-e epty rows only
    df['HasExternalAcct'] = (df['ExtUserAcctNotice'].str.strip() != '').astype(int)
    df = df.drop(columns=['LegalNotice', 'DRMNotice', 'ExtUserAcctNotice'])

    df['HasBackground'] = df['Background'].str.strip().replace('', None).notna().astype(int)
    df = df.drop(columns=['Background'])

    df['HasHeaderImage'] = df['HeaderImage'].str.strip().replace('', None).notna().astype(int)
    df = df.drop(columns=['HeaderImage'])

    return df



import warnings

warnings.filterwarnings('ignore')  # عشان نخفي الـ Warnings بتاعة البانداس وقت الـ Testing


def run_full_pipeline(df_raw):
    print("🚀 Starting the Data Engineering Pipeline...")

    # 1. Run Abd Al Karem's Features
    print("Running Abd Al Karem's block...")
    df_step1 = abd_al_karem_features(df_raw)

    # 2. Run Mohra & Raneem's Features
    print("Running Mohra & Raneem's block...")
    df_step2 = mohra_raneem_features(df_step1)

    # 3. Run Sama's Features
    print("Running Sama's block...")
    df_final = sama_features(df_step2)

    # 4. Final Cleanup (مسح العواميد اللي نسيناها والـ IDs)
    print("Running Final Cleanup...")
    cols_to_drop = [
        'QueryID', 'ResponseID', 'QueryName', 'ResponseName',  # الـ IDs والأسماء
        'PCRecReqsText', 'LinuxRecReqsText', 'MacRecReqsText',  # متطلبات الـ Rec اللي متعملهاش معالجة
        'PCMinReqsText', 'LinuxMinReqsText', 'MacMinReqsText',  # النصوص الأصلية للمتطلبات بعد ما طلعنا منها الرامات
        'SupportedLanguages', 'ReleaseDate'  # نصوص اللغة والتاريخ الأصلية
    ]
    # نمسح بس اللي موجود منهم عشان الكود ميضربش
    existing_drops = [c for c in cols_to_drop if c in df_final.columns]
    df_final.drop(columns=existing_drops, inplace=True)

    print("✅ Pipeline Finished Successfully!")
    print(f"--> Final Data Shape: {df_final.shape}")
    object_cols = df_final.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"⚠️ Warning: There are still Text/Object columns left: {object_cols}")
    else:
        print("🟢 Excellent: Data is 100% Numeric and ready for XGBoost!")

    return df_final



df_processed = run_full_pipeline(df)


























# ==========================================
# Step 1: Correlation & Variance Filter
# ==========================================

def filter_features(df, variance_threshold=0.99, correlation_threshold=0.85):
    """
    هذه الدالة تقوم بتصفية الداتا من:
    1. الأعمدة شبه الثابتة (Low Variance)
    2. الأعمدة المتكررة بدرجة عالية (High Correlation)
    مع استثناء التارجت (RecommendationCount) من الحذف.
    """
    print("\n" + "=" * 50)
    print("🔍 Starting Feature Filtering (Variance & Correlation)")
    print("=" * 50)

    df_filtered = df.copy()
    target_col = 'RecommendationCount'
    cols_to_drop_variance = []
    cols_to_drop_correlation = []

    # ---------------------------------------------------------
    # 1. Variance Filter (الأعمدة الميتة)
    # ---------------------------------------------------------
    for col in df_filtered.columns:
        if col == target_col:
            continue

        # حساب نسبة القيمة الأكثر تكراراً في العمود
        top_value_freq = df_filtered[col].value_counts(normalize=True).iloc[0]

        # لو القيمة دي متكررة بنسبة أكبر من الثريشولد (مثلاً 99%)
        if top_value_freq >= variance_threshold:
            cols_to_drop_variance.append(col)

    # تنفيذ الحذف
    if cols_to_drop_variance:
        df_filtered.drop(columns=cols_to_drop_variance, inplace=True)
        print(f"🧹 Removed {len(cols_to_drop_variance)} Zero/Low Variance Features:")
        for c in cols_to_drop_variance:
            print(f"   - {c}")
    else:
        print("✅ No Low Variance Features found.")

    # ---------------------------------------------------------
    # 2. Correlation Filter (الأعمدة المكررة)
    # ---------------------------------------------------------
    # حساب مصفوفة الكورليشن (بناخد القيمة المطلقة عشان السالب والموجب)
    corr_matrix = df_filtered.corr().abs()

    # تحديد المثلث العلوي من المصفوفة عشان منقرأش القيم مرتين
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # البحث عن الأعمدة اللي الكورليشن بتاعها أعلى من الثريشولد
    for col in upper.columns:
        if col == target_col:
            continue
        # لو أي قيمة في العمود ده أعلى من 0.85 مع عمود تاني
        if any(upper[col] > correlation_threshold):
            cols_to_drop_correlation.append(col)

    # تنفيذ الحذف
    if cols_to_drop_correlation:
        df_filtered.drop(columns=cols_to_drop_correlation, inplace=True)
        print(f"\n✂️ Removed {len(cols_to_drop_correlation)} Highly Correlated Features (> {correlation_threshold}):")
        for c in cols_to_drop_correlation:
            print(f"   - {c}")
    else:
        print("\n✅ No Highly Correlated Features found.")

    # ---------------------------------------------------------
    # النهاية
    # ---------------------------------------------------------
    print("-" * 50)
    total_dropped = len(cols_to_drop_variance) + len(cols_to_drop_correlation)
    print(f"📊 Original Shape: {df.shape}")
    print(f"📊 Filtered Shape: {df_filtered.shape}")
    print(f"🗑️ Total Features Removed: {total_dropped}")
    print("=" * 50)

    return df_filtered


# --- تشغيل الفلتر ---
# Assuming 'df_processed' is the output from your run_full_pipeline
df_filtered = filter_features(df_processed)




























# ==========================================
# Step 2: LightGBM Feature Selector
# ==========================================
import lightgbm as lgb
import re


def lgbm_feature_selection(df, target_col='RecommendationCount', cumulative_threshold=0.99):
    """
    هذه الدالة تستخدم LightGBM لتحديد أهم الميزات بناءً على الـ Gain.
    ستحتفظ بالميزات التي تمثل 99% من قوة التوقع، وتحذف الباقي.
    """
    print("\n" + "=" * 50)
    print("🚀 Starting LightGBM Feature Selection")
    print("=" * 50)

    # 1. فصل الداتا والتارجت
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # أمان برمجي: تنظيف أسماء الأعمدة لأن LightGBM بيرفض أي مسافات أو رموز غريبة
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    print("⚙️ Training Baseline LightGBM Model...")

    # 2. بناء الموديل (استخدام importance_type='gain' هو الأصح علمياً)
    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        importance_type='gain',  # بنقيس الميزة أضافت دقة قد إيه، مش اتكررت كام مرة
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X, y)

    # 3. حساب الأهمية
    importances = model.feature_importances_
    total_gain = np.sum(importances)

    # عمل جدول بالأهمية
    feat_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance_Gain': importances,
        'Relative_Importance_%': (importances / total_gain) * 100
    }).sort_values(by='Importance_Gain', ascending=False)

    # حساب الأهمية التراكمية (Cumulative)
    feat_imp_df['Cumulative_Importance'] = feat_imp_df['Relative_Importance_%'].cumsum()

    # 4. اختيار الميزات (ناخد اللي بيمثلوا 99% من الأهمية ونرمي الباقي)
    kept_features = feat_imp_df[feat_imp_df['Cumulative_Importance'] <= (cumulative_threshold * 100)][
        'Feature'].tolist()

    # تأمين: لو الثريشولد كان قاسي جداً، نضمن على الأقل 15 ميزة
    if len(kept_features) < 15:
        kept_features = feat_imp_df.head(15)['Feature'].tolist()

    dropped_features = [f for f in X.columns if f not in kept_features]

    # 5. الطباعة الواضحة
    print(
        f"\n✅ Kept {len(kept_features)} features (Accounting for {cumulative_threshold * 100}% of total predictive power).")
    print(f"🗑️ Dropped {len(dropped_features)} features (Noise / Low Contribution).")

    print("\n👑 Top 10 Most Important Features:")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(f"   🥇 {row['Feature']:<25} : {row['Relative_Importance_%']:.2f}%")

    if dropped_features:
        print("\n✂️ Sample of Dropped Features (Zero or near-zero importance):")
        for f in dropped_features[:10]:  # بنطبع أول 10 بس عشان الزحمة
            print(f"   - {f}")

    # 6. تجهيز الداتا النهائية
    final_cols = kept_features + [target_col]
    # نرجع نستخدم أسماء الأعمدة الأصلية (قبل تنظيف الـ Regex) لضمان التوافق
    original_kept_cols = [c for c in df.columns if re.sub('[^A-Za-z0-9_]+', '', c) in kept_features]
    df_selected = df[original_kept_cols + [target_col]].copy()

    print("-" * 50)
    print(f"🎯 Final Golden Dataset Shape: {df_selected.shape}")
    print("=" * 50)

    return df_selected, feat_imp_df


# --- تشغيل الاختيار ---
# Assuming 'df_filtered' is the output from the previous Correlation/Variance step
df_golden, feature_importance_table = lgbm_feature_selection(df_filtered, cumulative_threshold=0.99)











































from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb


def train_baseline_models(df, target_col='RecommendationCount'):
    print("\n" + "=" * 50)
    print("⚙️ Training Baseline Models (XGBoost vs LightGBM)")
    print("=" * 50)

    # 1. Split Data into Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Training Set: {X_train.shape[0]} rows | Test Set: {X_test.shape[0]} rows")

    # 3. Initialize Models (Default Parameters for Baseline)
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

    # 4. Train Models
    print("\n⏳ Training XGBoost...")
    xgb_model.fit(X_train, y_train)

    print("⏳ Training LightGBM...")
    lgb_model.fit(X_train, y_train)

    # 5. Predict on Test Set
    xgb_preds_log = xgb_model.predict(X_test)
    lgb_preds_log = lgb_model.predict(X_test)

    # 6. Transform Predictions and Actuals back to normal scale for Error Metrics
    # np.expm1 is the exact inverse of np.log1p used in data processing
    y_test_actual = np.expm1(y_test)

    # Clip predictions to prevent exponential explosion from outliers before expm1
    xgb_preds_log_clipped = np.clip(xgb_preds_log, a_min=None, a_max=20)
    lgb_preds_log_clipped = np.clip(lgb_preds_log, a_min=None, a_max=20)

    xgb_preds_actual = np.expm1(xgb_preds_log_clipped)
    lgb_preds_actual = np.expm1(lgb_preds_log_clipped)

    # 7. Evaluation Function
    def print_metrics(model_name, y_true_log, y_pred_log, y_true_act, y_pred_act):
        r2 = r2_score(y_true_log, y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_true_act, y_pred_act))
        mae = mean_absolute_error(y_true_act, y_pred_act)

        print(f"\n📈 {model_name} Performance:")
        print(f"   - R² Score (Log Scale) : {r2:.4f}")
        print(f"   - RMSE (Actual Count)  : {rmse:,.2f}")
        print(f"   - MAE (Actual Count)   : {mae:,.2f}")

    # Print Results
    print_metrics("XGBoost", y_test, xgb_preds_log, y_test_actual, xgb_preds_actual)
    print_metrics("LightGBM", y_test, lgb_preds_log, y_test_actual, lgb_preds_actual)

    print("\n" + "=" * 50)
    return xgb_model, lgb_model


# --- التشغيل ---
# Assuming 'df_golden' is your final dataframe from the LightGBM selector
xgb_baseline, lgb_baseline = train_baseline_models(df_golden)

# ==========================================
# Step 4: Feature Interactions & Hyperparameter Tuning (XGBoost)
# ==========================================
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np


def engineer_interaction_features(df):
    """إضافة ميزات مركبة قوية لمساعدة الموديل على فهم العلاقات المعقدة"""
    df = df.copy()

    print("\n" + "=" * 50)
    print("🧬 Adding Interaction Features...")

    # 1. القيمة مقابل السعر (كل ما كان السعر قليل والوصف/الصور كتير، دي قيمة أعلى)
    if 'Total_Media_Assets' in df.columns and 'PriceFinal' in df.columns:
        df['Value_for_Money'] = df['Total_Media_Assets'] / (df['PriceFinal'] + 1)

    # 2. تأثير التسويق بالنسبة للسعر
    if 'Marketing_Tier' in df.columns and 'PriceFinal' in df.columns:
        df['Marketing_Price_Impact'] = df['Marketing_Tier'] * df['PriceFinal']

    # 3. زخم اللعبة (الإنجازات بالنسبة لعمر اللعبة)
    if 'AchievementCount' in df.columns and 'GameAge' in df.columns:
        df['Game_Momentum'] = df['AchievementCount'] / (df['GameAge'] + 1)

    print(f"✅ Added Interaction Features. New Shape: {df.shape}")
    print("=" * 50)
    return df


def tune_and_train_xgboost(df, target_col='RecommendationCount'):
    print("\n" + "=" * 50)
    print("🎯 Hyperparameter Tuning & Cross-Validation (XGBoost)")
    print("=" * 50)

    # 1. فصل الداتا (التارجت هنا هو الـ Log1p اللي جاي من دالة سما جاهز)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. تعريف الموديل الأساسي
    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    # 3. تحديد شبكة البحث (Grid) للباراميترز اللي بتحارب الـ Overfitting
    param_dist = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],  # تقليل العمق بيمنع حفظ الداتا
        'min_child_weight': [5, 15, 30],  # إجبار الموديل إنه ميعملش تفرع لعدد قليل من الألعاب
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]  # أخذ عينات من الأعمدة بيجبره يبص على كل الميزات
    }

    # 4. تشغيل الـ RandomizedSearchCV مع 3-Fold Cross Validation
    print("⏳ Running Randomized Search with 3-Fold CV (This may take a few minutes)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=20,  # هيجرب 20 تركيبة مختلفة عشوائياً (عشان ننجز في الوقت)
        scoring='r2',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    # 5. استخراج أفضل موديل
    best_xgb = random_search.best_estimator_
    print(f"\n🏆 Best Parameters Found:\n{random_search.best_params_}")

    # 6. التقييم النهائي على الـ Test Set
    xgb_preds_log = best_xgb.predict(X_test)

    # حساب الخطأ الحقيقي
    y_test_actual = np.expm1(y_test)
    xgb_preds_log_clipped = np.clip(xgb_preds_log, a_min=None, a_max=20)
    xgb_preds_actual = np.expm1(xgb_preds_log_clipped)

    r2 = r2_score(y_test, xgb_preds_log)
    rmse = np.sqrt(mean_squared_error(y_test_actual, xgb_preds_actual))
    mae = mean_absolute_error(y_test_actual, xgb_preds_actual)

    print("\n📈 Tuned XGBoost Final Performance:")
    print(f"   - R² Score (Log Scale) : {r2:.4f}")
    print(f"   - RMSE (Actual Count)  : {rmse:,.2f}")
    print(f"   - MAE (Actual Count)   : {mae:,.2f}")
    print("=" * 50)

    return best_xgb


# --- التشغيل ---
# df_golden هي الداتا اللي طلعت من الـ LightGBM Selector
df_with_interactions = engineer_interaction_features(df_golden)
final_tuned_model = tune_and_train_xgboost(df_with_interactions)