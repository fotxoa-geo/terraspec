import pandas as pd
import geopandas as gp
import os
gis_data = os.path.join(r'C:\Users\fotxo\OneDrive - UCLA IT Services\EMIT\terraspec\gis\Observation.shp')

df = gp.read_file(gis_data)
df = df.drop(columns='Type')
df = df.drop(columns='Descriptio')
df['longitude'] = df['geometry'].x.round(4).map("{:.4f}".format)
df['latitude'] = df['geometry'].y.round(4).map("{:.4f}".format)
df = df.drop(columns='geometry')
df = df.drop(columns='Date & Tim')
df = df.drop(columns='EMIT Overp')
df['lat_long_combined'] = df['latitude'] + ', ' + df['longitude']
df = df.drop(columns=['latitude', 'longitude'])
df['Ground'] = ''
df = df.reindex(columns=['Name', 'lat_long_combined', 'Ground', 'EMIT Date'])
df['EMIT Date'] = pd.to_datetime(df['EMIT Date'])
df['EMIT Date'] = df['EMIT Date'].dt.strftime('%m/%d/%Y')
print(df.columns)
print(df.to_latex(index=False))