// binary flag for whether these is a website or not
df['HasWebsite'] = df['Website'].notna().astype(int)
df = df.drop(columns=['Website']) 

## fill the nulls with USD
df['PriceCurrency'] = df['PriceCurrency'].str.strip()
df['PriceCurrency'] = df['PriceCurrency'].replace('', 'USD')
