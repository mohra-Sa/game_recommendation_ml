import pandas as pd
import numpy as np
import pycountry
from datetime import datetime
import re

df=pd.read_csv("D:/3rd year/2nd term/ML/game-recommendation-ml/data/train_data.csv")


# Fill missing values in text columns with empty strings and create binary features indicating the presence of text
text_columns = ['SupportURL','SupportEmail','Website','Reviews','ExtUserAcctNotice','DRMNotice','LegalNotice','Background','HeaderImage']
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



# Function to extract RAM, Storage, CPU, and OpenGL requirements from LinuxMinReqsText and MacMinReqsText
def extract_reqs(text):
    if not isinstance(text, str) or not text.strip():
        return {'RAM_GB': None, 'Storage_GB': None, 'CPU_GHz': None, 'OpenGL': None}

    ram     = re.findall(r'(\d+)\s*(GB|mb)\s*(?:Memory|RAM)', text, re.IGNORECASE)
    storage = re.findall(r'(\d+)\s*GB\s*Hard\s*Drive',   text, re.IGNORECASE)
    ghz = re.findall(r'(\d+\.?\d*)\s*(GHz|mhz)', text, re.IGNORECASE)
    opengl  = re.findall(r'OpenGL\s*(\d+\.?\d*)',         text, re.IGNORECASE)
    cpu=None
    if ghz: 
      value, unit = ghz[0]
      value = float(value)
      
      if unit.lower() == 'mhz':
        value = value / 1000
      cpu = value
    else:
        cpu = None
    
    Ram=None
    if ram: 
      value, unit = ram[0]
      value = float(value)

      if unit.lower() == 'mb':
        value = value / 1000
      Ram = value
    else:
        Ram = None

    return {
        'RAM_GB'    : Ram ,
        'Storage_GB': int(storage[0])  if storage else None,
        'CPU_GHz'   : cpu,
        'OpenGL'    : float(opengl[0]) if opengl  else None
    }

# Linux
linux = df['LinuxMinReqsText'].apply(extract_reqs).apply(pd.Series)
linux.columns = ['Linux_RAM_GB','Linux_Storage_GB','Linux_CPU_GHz','Linux_OpenGL']

# Mac
mac = df['MacMinReqsText'].apply(extract_reqs).apply(pd.Series)
mac.columns = ['Mac_RAM_GB','Mac_Storage_GB','Mac_CPU_GHz','Mac_OpenGL']

# PC
pc = df['PCMinReqsText'].apply(extract_reqs).apply(pd.Series)
pc.columns = ['PC_RAM_GB','PC_Storage_GB','PC_CPU_GHz','PC_OpenGL']

extract_cols = ['RAM_GB','Storage_GB','CPU_GHz','OpenGL']
for col in extract_cols:
    linux_col = 'Linux_' + col
    mac_col   = 'Mac_' + col
    pc_col    = 'PC_' + col

    if linux[linux_col].notna().any():
        linux[linux_col] = linux[linux_col].fillna(linux[linux_col].min())

    if mac[mac_col].notna().any():
        mac[mac_col] = mac[mac_col].fillna(mac[mac_col].min())

    if pc[pc_col].notna().any():
        pc[pc_col] = pc[pc_col].fillna(pc[pc_col].min())



    

df = pd.concat([df, linux, mac, pc], axis=1)
print(df[['Linux_RAM_GB','Linux_Storage_GB','Linux_CPU_GHz','Linux_OpenGL',
           'Mac_RAM_GB','Mac_Storage_GB','Mac_CPU_GHz','Mac_OpenGL',
          'PC_RAM_GB','PC_Storage_GB','PC_CPU_GHz','PC_OpenGL']].iloc[8:21].head())


