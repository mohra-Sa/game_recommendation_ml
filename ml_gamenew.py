import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import re
import lightgbm as lgb
import xgboost as xgb
import shap
import optuna
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
import pickle
from pathlib import Path


def mohra_raneem_features(df):
    df = df.copy()

    binary_text_columns = ['SupportURL', 'SupportEmail', 'Website', 'ExtUserAcctNotice',
                           'DRMNotice', 'Background', 'HeaderImage','LegalNotice','IsFree','FreeVerAvail','PlatformMac','PlatformLinux','PlatformWindows',
                           'GenreIsMassivelyMultiplayer','GenreIsRacing','GenreIsSports','GenreIsFreeToPlay','GenreIsEarlyAccess','GenreIsSimulation',
                           'GenreIsRPG','GenreIsStrategy','GenreIsCasual','GenreIsAdventure','GenreIsAction','GenreIsIndie','GenreIsNonGame''CategoryMultiplayer', 'CategoryCoop', 'CategoryMMO']
    for col in binary_text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip().ne('').astype(int)




    # Reviews → extract sentiment signals instead of just binary
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



    # Text length features
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
        # has major Asian language support
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
        #  holiday release window (Nov-Dec = high sales season)
        df['Is_Holiday_Release'] = df['ReleaseDate_Month'].apply(
            lambda x: 1 if x in [11, 12] else 0
        )
        #  Quarter of release
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
    # fill QueryName columns nulls with unkown
    df['QueryName'] = df['QueryName'].fillna('unknown')
    df=df.drop(columns=['LinuxMinReqsText','LinuxRecReqsText','MacMinReqsText','MacRecReqsText','PCMinReqsText','PCRecReqsText','QueryName','ResponseName','QueryID','ResponseID','SupportedLanguages', 'ReleaseDate'])
    return df



def abd_al_karem_features(df):
    df = df.copy()

    df['Total_Media_Assets'] =np.log1p(df['ScreenshotCount'] + df['MovieCount'])
    df['ScreenshotCount_Log'] = np.log1p(df['ScreenshotCount'])
    df['MovieCount_Log'] = np.log1p(df['MovieCount'])

    df['Total_Media_Assets_log']=np.log1p(df['Total_Media_Assets'])

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
    df['AchievementCount_Log'] = np.log1p(df['AchievementCount'])
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

    platform_cols = [c for c in ['PlatformWindows', 'PlatformLinux', 'PlatformMac'] if c in df.columns]
    if platform_cols:
        df['Platform_Reach'] = df[platform_cols].sum(axis=1)

    # Genre diversity feature
    genre_cols = [c for c in df.columns if c.startswith('GenreIs')]
    if genre_cols:
        df['Genre_Diversity'] = df[genre_cols].sum(axis=1)

    # Engagement score from category features
    engagement_cols = [c for c in ['CategoryMultiplayer', 'CategoryCoop', 'CategoryMMO'] if c in df.columns]
    if engagement_cols:
        df['Engagement_Score'] = df[engagement_cols].sum(axis=1)



    return df





def corr_heatmap(df, columns):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
    plt.title('Correlation Heatmap', fontsize=10, fontweight='bold', pad=8)
    plt.xticks(rotation=40, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.show()


def sama_features(df):
    df = df.copy()

    col = ['RecommendationCount', 'Metacritic', 'SteamSpyOwners', 'SteamSpyOwnersVariance',
           'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance', 'AchievementHighlightedCount']

    #abd alkarem was here to check the cols if exist
    existing_cols = [c for c in col if c in df.columns]

    print("Columns distribution before handling")
    if len(existing_cols) > 0:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 20))
        axes = axes.flatten()

        for i, c in enumerate(existing_cols):
            sns.histplot(df[c].dropna(), kde=True, ax=axes[i], bins=100, shrink=0.8, color='#2b6777', edgecolor='white')
            upper_limit = df[c].quantile(0.99)
            axes[i].set_xlim(0, upper_limit)
            axes[i].set_title(f'Distribution of {c}', fontsize=9, fontweight='bold')
            axes[i].set_xlabel(c, fontsize=7)
            axes[i].set_ylabel('Frequency', fontsize=7)

        for j in range(len(existing_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=10.0)
        plt.show()

    if 'SteamSpyOwnersVariance' in df.columns and 'SteamSpyOwners' in df.columns:
        df['relative_variation_owners'] = np.where(
            df['SteamSpyOwners'] > 0,
            np.log1p(df['SteamSpyOwnersVariance'] / df['SteamSpyOwners']),
            0.0
        )

    if 'SteamSpyPlayersVariance' in df.columns and 'SteamSpyPlayersEstimate' in df.columns:
        df['relative_variation_players'] = np.where(
            df['SteamSpyPlayersEstimate'] > 0,
            np.log1p(df['SteamSpyPlayersVariance'] / df['SteamSpyPlayersEstimate']),
            0.0
        )

    # engagement quality signal
    if 'SteamSpyOwners' in df.columns and 'SteamSpyPlayersEstimate' in df.columns:
        df['Owners_to_Players_Ratio'] = np.where(
            df['SteamSpyOwners'] > 0,
            np.log1p(df['SteamSpyPlayersEstimate'] / (df['SteamSpyOwners'] + 1e-9)),
            0.0
        )

    #abd alkarem was here to check the cols if exist
    for c in existing_cols:
        if c != 'AchievementHighlightedCount':
            df[c] = np.log1p(df[c])

    if 'AchievementHighlightedCount' in df.columns:
        condition = [df['AchievementHighlightedCount'] == 0,
                     df['AchievementHighlightedCount'] == 10]
        choice = [0, 2]
        df['AchievementHighlightedCount'] = np.select(condition, choice, default=1)

    clean_colms = ['RecommendationCount', 'Metacritic', 'AchievementHighlightedCount',
                   'relative_variation_owners', 'relative_variation_players', 'Owners_to_Players_Ratio']
    #abd alkarem was here to check the cols if exist
    existing_clean_colms = [c for c in clean_colms if c in df.columns]

    if len(existing_clean_colms) > 1:
        corr_heatmap(df, existing_clean_colms)

    return df


def engineer_interaction_features(df):
    df = df.copy()
    print("\n" + "=" * 50)
    print("Adding Interaction Features...")

    if 'Total_Media_Assets' in df.columns and 'PriceFinal' in df.columns:
        df['Value_for_Money'] = df['Total_Media_Assets'] / (df['PriceFinal'] + 1)

    if 'Marketing_Tier' in df.columns and 'PriceFinal' in df.columns:
        df['Marketing_Price_Impact'] = df['Marketing_Tier'] * df['PriceFinal']

    if 'AchievementCount_Log' in df.columns and 'GameAge' in df.columns:
        df['Game_Momentum'] = df['AchievementCount_Log'] / (df['GameAge'] + 1)

    # revenue proxy
    if 'SteamSpyOwners' in df.columns and 'PriceFinal' in df.columns:
        df['Revenue_Proxy'] = df['SteamSpyOwners'] * np.log1p(df['PriceFinal'])

    # quality × visibility
    if 'Total_Media_Assets' in df.columns and 'Metacritic' in df.columns:
        df['Quality_Visibility'] = df['Total_Media_Assets'] * df['Metacritic']

    # older expensive games lose value
    if 'PriceFinal' in df.columns and 'GameAge' in df.columns:
        df['Price_Age_Penalty'] = df['PriceFinal'] / (df['GameAge'] + 1)

    # DLC richness × engagement
    if 'DLCCount_Log' in df.columns and 'Engagement_Score' in df.columns:
        df['Content_Engagement'] = df['DLCCount_Log'] * (df['Engagement_Score'] + 1)

    if 'PriceFinal' in df.columns and 'PriceInitial' in df.columns:
        df['price'] = df['PriceFinal'] + df['PriceInitial']
        # set 'IsFree' and 'FreeVerAvail' based on whether the calculated 'price' is 0
        df['IsFree'] = (df['price'] == 0).astype(int)
        df['FreeVerAvail'] = (df['price'] == 0).astype(int)
        df['price'] = np.log1p(df['price'])

    new_features = ['RecommendationCount', 'Value_for_Money', 'Marketing_Price_Impact',
                    'Game_Momentum', 'Revenue_Proxy', 'Quality_Visibility',
                    'Price_Age_Penalty', 'Content_Engagement']

    for c in new_features[1:]:
        if c in df.columns:
            df[c] = np.log1p(df[c])

    existing_new_features = [c for c in new_features if c in df.columns]

    if len(existing_new_features) > 1:
        corr_heatmap(df, existing_new_features)

    return df



def filter_features(df, target_col='RecommendationCount', variance_threshold=0.995, correlation_threshold=0.85):

    print("\n" + "=" * 50)
    print("Starting Feature Filtering (Variance & Correlation)")
    print("=" * 50)

    df_filtered = df.copy()
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
        print(f"Removed {cols_to_drop_variance} , ")
        print(f"Removed {len(cols_to_drop_variance)} Zero/Low Variance Features.")

    #abd alkarem was here to clc only numircal f
    corr_matrix = df_filtered.select_dtypes(include=[np.number]).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper.columns:
        if col == target_col:
            continue
        if any(upper[col] > correlation_threshold):
            cols_to_drop_correlation.append(col)

    if cols_to_drop_correlation:
        df_filtered.drop(columns=cols_to_drop_correlation, inplace=True)
        print(f"Removed {cols_to_drop_correlation} , ")
        print(f"Removed {len(cols_to_drop_correlation)} Highly Correlated Features.")

    return df_filtered


def lgbm_feature_selection(X, Y, cumulative_threshold=0.99, task='regression'):
    print("\n" + "=" * 50)
    print(f"Starting LightGBM Feature Selection ({task.capitalize()})")
    print("=" * 50)

    # Store original column names
    original_cols = X.columns.tolist()
    X_renamed = X.rename(columns=lambda col: re.sub('[^A-Za-z0-9_]+', '', col))

    #abd alkarem was here to make the code work om both lines
    if task == 'classification':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y_ready = le.fit_transform(Y)
        model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, importance_type='gain', random_state=42,
                                   n_jobs=-1, verbose=-1)
    else:
        Y_ready = Y
        model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, importance_type='gain', random_state=42,
                                  n_jobs=-1, verbose=-1)

    model.fit(X_renamed, Y_ready)

    importances = model.feature_importances_
    total_gain = np.sum(importances)

    feat_imp_df = pd.DataFrame({
        'Feature_Renamed': X_renamed.columns,
        'Importance_Gain': importances,
        'Relative_Importance_%': (importances / total_gain) * 100
    }).sort_values(by='Importance_Gain', ascending=False)

    feat_imp_df['Cumulative_Importance'] = feat_imp_df['Relative_Importance_%'].cumsum()

    kept_features_renamed = feat_imp_df[feat_imp_df['Cumulative_Importance'] <= (cumulative_threshold * 100)][
        'Feature_Renamed'].tolist()
    if len(kept_features_renamed) < 15:
        kept_features_renamed = feat_imp_df.head(15)['Feature_Renamed'].tolist()

    # Create a mapping from renamed names back to original names
    rename_map = {re.sub('[^A-Za-z0-9_]+', '', col): col for col in original_cols}
    kept_features_original = [rename_map[renamed_col] for renamed_col in kept_features_renamed if
                              renamed_col in rename_map]

    print(f"Kept {len(kept_features_original)} features.")
    print("\nTop 10 features by LightGBM gain (Original Names):")
    top_10_renamed = feat_imp_df.head(10)
    top_10_original_names = [rename_map[f] for f in top_10_renamed['Feature_Renamed'].tolist()]
    top_10_display_df = pd.DataFrame({
        'Feature': top_10_original_names,
        'Relative_Importance_%': top_10_renamed['Relative_Importance_%'].tolist()
    })
    print(top_10_display_df.to_string(index=False))

    # feature vs importance graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Relative_Importance_%', y='Feature', data=top_10_display_df, palette='viridis')
    plt.title('Top 10 Features by LightGBM Gain (Original Names)')
    plt.xlabel('Relative Importance (%)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return top_10_original_names


def run_full_pipeline(df_raw):
    df = abd_al_karem_features(df_raw)
    df = mohra_raneem_features(df)
    df = sama_features(df)
    df = engineer_interaction_features(df)
    df = filter_features(df)

    cols_to_drop = [
        'QueryID', 'ResponseID', 'QueryName', 'ResponseName',
        'PCRecReqsText', 'LinuxRecReqsText', 'MacRecReqsText',
        'PCMinReqsText', 'LinuxMinReqsText', 'MacMinReqsText',
        'SupportedLanguages', 'ReleaseDate'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    return df



def inv_sq(distances):
    return 1 / (distances ** 2 + 1e-10)


def classification_preprocessing(df):
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



if __name__ == '__main__':
    df = pd.read_csv('data/train_data.csv')
    df_processed = run_full_pipeline(df)
    X = df_processed.drop(columns=['RecommendationCount']).select_dtypes(include=[np.number]).fillna(0)
    y = df_processed['RecommendationCount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selected_features = lgbm_feature_selection(X_train, y_train)
    print(f"Selected Features: {selected_features}")
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
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
        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=kf,
            scoring='r2',
            n_jobs=-1
        )
        return scores.mean()
    # LightGBM to search best hyperparameters
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

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=kf,
            scoring='r2',
            n_jobs=-1
        )

        return scores.mean()


    # XGBoost

    print("=" * 50)
    print("Tuning XGBoost...")
    print("=" * 50)

    xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)

    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    })

    print("Best XGBoost Params:")
    print(best_xgb_params)

    #  LightGBM

    print("=" * 50)
    print("Tuning LightGBM...")
    print("=" * 50)

    lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)

    best_lgb_params = lgb_study.best_params
    best_lgb_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })

    print("Best LightGBM Params:")
    print(best_lgb_params)

    #  Build Models
    best_xgb_model = xgb.XGBRegressor(**best_xgb_params)
    best_lgb_model = lgb.LGBMRegressor(**best_lgb_params)

    best_xgb_model.fit(X_train_scaled, y_train)
    best_lgb_model.fit(X_train_scaled, y_train)
    xgb_test_r2 = r2_score(y_test, best_xgb_model.predict(X_test_scaled))
    lgb_test_r2 = r2_score(y_test, best_lgb_model.predict(X_test_scaled))
    print("R² of XGBoost model:", xgb_test_r2)
    print("R² of LightGBM model:", lgb_test_r2)

    # Stacking model
    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', best_xgb_model),
            ('lgb', best_lgb_model)
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )

    # Train Model
    print("=" * 50)
    print("Training Stacking Model...")
    print("=" * 50)

    stacking_model.fit(X_train_scaled, y_train)

    #  Prediction
    y_pred_log = stacking_model.predict(X_test_scaled)

    y_test_actual = np.expm1(y_test)
    y_pred_clipped = np.clip(y_pred_log, a_min=None, a_max=20)
    y_pred_actual = np.expm1(y_pred_clipped)

    # Evaluation by MSE and RMSE and apply cross validation
    r2 = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)

    kf_final = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        stacking_model,
        X,
        y,
        cv=kf_final,
        scoring='r2',
        n_jobs=-1
    )

    print("\nFinal Performance:")
    print(f"R² Score (Test): {r2:.4f}")
    print(f"R² CV Mean: {cv_scores.mean():.4f}")
    print(f"R² CV Std : {cv_scores.std():.4f}")
    print(f"RMSE      : {rmse:.2f}")
    print(f"MAE       : {mae:.2f}")


    save_dir = Path('saved_models')
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / 'selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    print(f"✓  selected_features  ({len(selected_features)} features)")
    with open(save_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓  scaler")
    with open(save_dir / 'stacking_model.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)
    print("✓  stacking_model")
    with open(save_dir / 'xgb_model.pkl', 'wb') as f:
        pickle.dump(best_xgb_model, f)
    print("✓  xgb_model")

    with open(save_dir / 'lgb_model.pkl', 'wb') as f:
        pickle.dump(best_lgb_model, f)
    print("✓  lgb_model")
    print(f"\nAll artefacts saved to  ./{save_dir}/")
    print("Files in folder:", [p.name for p in save_dir.iterdir()])
    print("\n" + "=" * 50)
    print("MILESTONE 2 – Classification Pipeline")
    print("=" * 50)



    # ── 1. Load & preprocess classification data ──────────────────────────────
    df_cls = pd.read_csv('data/train2.csv')
    df_cls_processed = classification_preprocessing(df_cls)

    # filter_features knows GamePopularity is the target
    df_cls_processed = filter_features(df_cls_processed)

    # Separate features and target (keep string labels: 'Low'/'Medium'/'High')
    X_cls = (df_cls_processed
             .drop(columns=['GamePopularity'])
             .select_dtypes(include=[np.number])
             .fillna(0))
    y_cls = df_cls_processed['GamePopularity']

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # ── 2. Feature selection (lgbm) + scaling  →  used by RF, SVM, DT ────────
    cls_selected_features = lgbm_feature_selection(X_cls_train, y_cls_train, task='classification')
    print(f"LGBM Selected Features (RF/SVM/DT): {cls_selected_features}")

    X_cls_train_sel = X_cls_train[cls_selected_features]
    X_cls_test_sel = X_cls_test[cls_selected_features]

    cls_scaler = StandardScaler()
    X_cls_train_scaled = cls_scaler.fit_transform(X_cls_train_sel)
    X_cls_test_scaled = cls_scaler.transform(X_cls_test_sel)

    # ── 3. KNN: SelectKBest (top 20) on lgbm-filtered features + own scaler ──
    K_FEATURES = 20
    knn_selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    knn_selector.fit(X_cls_train_sel, y_cls_train)

    knn_selected_features = X_cls_train_sel.columns[knn_selector.get_support()].tolist()
    print(f"KNN Selected Features: {knn_selected_features}")

    X_knn_train = X_cls_train_sel[knn_selected_features]
    X_knn_test = X_cls_test_sel[knn_selected_features]

    knn_scaler = StandardScaler()
    X_knn_train_sc = knn_scaler.fit_transform(X_knn_train)
    X_knn_test_sc = knn_scaler.transform(X_knn_test)

    # ── 4. Train Random Forest (with SMOTEENN resampling)
    print("\n--- Training Random Forest ---")
    smote_enn = SMOTEENN(random_state=42)
    X_rf_res, y_rf_res = smote_enn.fit_resample(X_cls_train_scaled, y_cls_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_rf_res, y_rf_res)
    rf_acc = accuracy_score(y_cls_test, rf_model.predict(X_cls_test_scaled))
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

    # ── 5. Train SVM
    print("\n--- Training SVM ---")
    svm_model = SVC(kernel='rbf', C=1, gamma='scale',
                    class_weight='balanced', random_state=42)
    svm_model.fit(X_cls_train_scaled, y_cls_train)
    svm_acc = accuracy_score(y_cls_test, svm_model.predict(X_cls_test_scaled))
    print(f"SVM Accuracy: {svm_acc * 100:.2f}%")

    # ── 6. Train Decision Tree (GridSearchCV)
    print("\n--- Training Decision Tree (GridSearchCV) ---")
    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    grid_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        dt_params, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_dt.fit(X_cls_train_scaled, y_cls_train)
    best_dt = grid_dt.best_estimator_
    dt_acc = accuracy_score(y_cls_test, best_dt.predict(X_cls_test_scaled))
    print(f"Decision Tree Best Params: {grid_dt.best_params_}")
    print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")

    # ── 7. Train KNN (tune k then weights)
    print("\n--- Training KNN ---")
    k_values = [3, 5, 7, 9, 11]
    k_results = []
    for k in k_values:
        knn_tmp = KNeighborsClassifier(n_neighbors=k, weights='uniform',
                                       metric='minkowski', n_jobs=-1)
        knn_tmp.fit(X_knn_train_sc, y_cls_train)
        acc = accuracy_score(y_cls_test, knn_tmp.predict(X_knn_test_sc))
        k_results.append({'k': k, 'accuracy': acc})
        print(f"  k={k:2d} → Accuracy = {acc:.4f}")

    k_df = pd.DataFrame(k_results)
    best_k = int(k_df.loc[k_df['accuracy'].idxmax(), 'k'])
    print(f"Best k = {best_k}")

    weight_options = ['uniform', 'distance', inv_sq]
    weight_labels = ['uniform', 'distance', 'inv_sq']
    w_results = []
    for w, label in zip(weight_options, weight_labels):
        knn_tmp = KNeighborsClassifier(n_neighbors=best_k, weights=w,
                                       metric='minkowski', n_jobs=-1)
        knn_tmp.fit(X_knn_train_sc, y_cls_train)
        acc = accuracy_score(y_cls_test, knn_tmp.predict(X_knn_test_sc))
        w_results.append({'weights': label, 'accuracy': acc})
        print(f"  weights={label:12s} → Accuracy = {acc:.4f}")

    w_df = pd.DataFrame(w_results)
    best_w_label = w_df.loc[w_df['accuracy'].idxmax(), 'weights']
    best_w_func = weight_options[weight_labels.index(best_w_label)]
    print(f"Best weights = {best_w_label}")

    # Train final KNN with best params
    best_knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w_func,
                                    metric='minkowski', n_jobs=-1)
    best_knn.fit(X_knn_train_sc, y_cls_train)
    knn_acc = accuracy_score(y_cls_test, best_knn.predict(X_knn_test_sc))
    print(f"KNN Final Accuracy: {knn_acc * 100:.2f}%")




    # ── 8. Save all classification artefacts ──────────────────────────────────
    save_dir_cls = Path('saved_models_cls')
    save_dir_cls.mkdir(exist_ok=True)
    with open(save_dir_cls / 'cls_selected_features.pkl', 'wb') as f:
        pickle.dump(cls_selected_features, f)
    print("✓  cls_selected_features")

    with open(save_dir_cls / 'cls_scaler.pkl', 'wb') as f:
        pickle.dump(cls_scaler, f)
    print("✓  cls_scaler")

    with open(save_dir_cls / 'rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓  rf_model")

    with open(save_dir_cls / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    print("✓  svm_model")

    with open(save_dir_cls / 'dt_model.pkl', 'wb') as f:
        pickle.dump(best_dt, f)
    print("✓  dt_model")
    with open(save_dir_cls / 'knn_selector.pkl', 'wb') as f:
        pickle.dump(knn_selector, f)
    print("✓  knn_selector")

    with open(save_dir_cls / 'knn_selected_features.pkl', 'wb') as f:
        pickle.dump(knn_selected_features, f)
    print("✓  knn_selected_features")

    with open(save_dir_cls / 'knn_scaler.pkl', 'wb') as f:
        pickle.dump(knn_scaler, f)
    print("✓  knn_scaler")
    with open(save_dir_cls / 'knn_best_params.pkl', 'wb') as f:
        pickle.dump({'best_k': best_k, 'best_w_label': best_w_label}, f)
    print("✓  knn_best_params")

    with open(save_dir_cls / 'knn_model.pkl', 'wb') as f:
        pickle.dump(best_knn, f)
    print("✓  knn_model")

    print(f"\nAll classification artefacts saved to ./{save_dir_cls}/")
    print("Files:", [p.name for p in save_dir_cls.iterdir()])






