import pandas as pd

vix = pd.read_excel('quant_strat_interview/q1_vix.xlsx', header=0, engine='openpyxl')
vix = vix.dropna()         # Drop any empty rows
vix['Date'] = pd.to_datetime(vix['Date']) # Ensure it's a date object

sp = pd.read_excel('quant_strat_interview/q1_sp.xlsx', header=0, engine='openpyxl')
sp = sp.dropna()         # Drop any empty rows
sp['Date'] = pd.to_datetime(sp['Date']) # Ensure it's a date object


df = pd.merge(sp, vix, on='Date', how='inner')
df.set_index('Date', inplace=True)
print(df.head())