import pandas as pd


sites = sorted(list(range(1 ,101)) * 3)
local = ['A', 'B', 'C'] * 100
data = {"sites": sites, "local": local}
data['sites'] = [str(x).zfill(3) for x in data['sites']]

df = pd.DataFrame(data=data)

df['label'] = 'Soil - ' + df['sites'] + df['local']
df['content'] = df['sites'] + df['local']
df = df.drop(['sites', 'local'], axis=1)
df.to_excel(r'C:\Users\fotxo\OneDrive\Desktop\soil_labels.xlsx', index=False)
