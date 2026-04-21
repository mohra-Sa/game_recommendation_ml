// binary flag for whether these is a website or not
df['HasWebsite'] = df['Website'].notna().astype(int)
df = df.drop(columns=['Website']) 
