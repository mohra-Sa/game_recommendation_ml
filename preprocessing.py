import pandas as pd
import numpy as np
import pycountry
from datetime import datetime

df=pd.read_csv("D:/3rd year/2nd term/ML/game-recommendation-ml/data/train_data.csv")


# Fill missing values in text columns with empty strings and create binary features indicating the presence of text
text_columns = ['MacMinReqsText','LinuxMinReqsText','PCRecReqsText','PCMinReqsText','SupportURL','SupportEmail','Website','Reviews','ExtUserAcctNotice','DRMNotice','LegalNotice']
for col in text_columns:
    df[col] = df[col].fillna('').str.strip().ne('').astype(int)
# Convert text columns to numeric features by calculating their length (number of characters)(this more accurately captures the amount of information provided in the text, which may be more relevant for recommendation than just the presence of text)
textto_num_cols = ['DetailedDescrip','AboutText','ShortDescrip']
for col in textto_num_cols:
    df[col] = df[col].fillna('').str.len()

  # anather way to list common languages using pycountry, but it may not be comprehensive and may miss some languages 
# common_langs = [lang.name for lang in pycountry.languages  if hasattr(lang, 'alpha_2')]  
common_langs= ['English','French','German','Italian','Spanish','Korean',
               'Japanese','Russian','Turkish','Thai','Portuguese','Polish',
               'Dutch','Arabic','Simplified Chinese','Traditional Chinese',
               'Czech','Hungarian','Romanian']

df['NumLanguages'] = df['SupportedLanguages'].apply(
    lambda x: sum(1 for l in common_langs if isinstance(x, str) and l in x)
)

# for col in ['NumLanguages', text_columns,textto_num_cols]:
#     print(df[col].head(5))

# Convert boolean columns to integers (0 and 1)
bool_cols = df.select_dtypes(include='bool').columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)

    
# Fill missing values in ReleaseDate, convert to datetime, and extract year, month, day, and calculate game age
df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
median_date = df['ReleaseDate'].dropna().median()
df['ReleaseDate'] = df['ReleaseDate'].fillna(median_date)
df['ReleaseDate_Year']  = df['ReleaseDate'].dt.year.astype('Int64')
df['ReleaseDate_Month'] = df['ReleaseDate'].dt.month.astype('Int64')
df['ReleaseDate_Day']   = df['ReleaseDate'].dt.day.astype('Int64')
df['GameAge'] = datetime.now().year - df['ReleaseDate_Year']
print(df[['ReleaseDate','ReleaseDate_Year','ReleaseDate_Month','ReleaseDate_Day','GameAge']].head())


# fill misssing values in PriceCurrency and encode it

print(df['PriceCurrency'].value_counts(dropna=False)) 
df['PriceCurrency'] = df['PriceCurrency'].str.strip().replace('', 'USD')
df['PriceCurrency'] = (df['PriceCurrency'] == 'USD').astype(int)
print(df['PriceCurrency'].isnull().sum()) 
