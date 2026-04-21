// binary flag for whether these is a website or not
df['HasWebsite'] = df['Website'].notna().astype(int)
df = df.drop(columns=['Website']) 

## fill the nulls with USD
df['PriceCurrency'] = df['PriceCurrency'].str.strip()
df['PriceCurrency'] = df['PriceCurrency'].replace('', 'USD')


df['HasDRM']  = (df['DRMNotice'].str.strip() != '').astype(int)    # i have 70 non-e epty rows only 
df['HasExternalAcct'] = (df['ExtUserAcctNotice'].str.strip() != '').astype(int)
df = df.drop(columns=['LegalNotice', 'DRMNotice', 'ExtUserAcctNotice'])


df['HasBackground'] = df['Background'].str.strip().replace('', None).notna().astype(int)
df = df.drop(columns=['Background'])



df['HasHeaderImage'] = df['HeaderImage'].str.strip().replace('', None).notna().astype(int)
df = df.drop(columns=['HeaderImage'])


// 
