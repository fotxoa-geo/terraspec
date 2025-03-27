import pandas as pd
import geopandas as gp
import os

slpit_data = r'G:\My Drive\terraspec\slpit\figures\fraction_output.csv'
shift_data = r'G:\My Drive\terraspec\shift\figures\shift_fraction_output.csv'
skip = ['SRA-000-SPRING', 'SRB-047-SPRING', 'SRB-004-FALL', 'SRB-050-FALL', 'SRB-200-FALL']

df_emit = pd.read_csv(slpit_data)
df_emit['cmp'] = 'ss'
df_shift = pd.read_csv(shift_data)
df_shift = df_shift[~df_shift['plot'].isin(skip)]
df_shift['cmp'] = 'shift'

df = pd.concat([df_emit, df_shift], ignore_index=True)

df = df[(df['normalization'] == 'none')].copy()


for i in sorted(list(df.unmix_mode.unique())):
    df_unmix = df[(df['unmix_mode'] == i)].copy()

    for instrument in sorted(list(df_unmix.instrument.unique())):
        df_instrument = df_unmix[(df_unmix['instrument'] == instrument)].copy()

        for campaign in sorted(list(df_instrument.cmp.unique())):
            df_ins = df_instrument[(df_instrument['cmp'] == campaign)].copy()

            mean = df_ins['npv_se'].mean()
            # for lib in sorted(list(df_ins.lib_mode.unique())):
            #     df_lib = df_ins[(df_ins['lib_mode'] == lib)].copy()


            print(f"{i}, {instrument}, {campaign}, , {mean:.2f}")


gis_data = os.path.join(r'C:\Users\spect\PycharmProjects\pythonProject\terraspec\gis\Observation.shp')
df = gp.read_file(gis_data)
print(df)
df = df.drop(columns='Type')

df['longitude'] = df['geometry'].x.round(4).map("{:.4f}".format)
df['latitude'] = df['geometry'].y.round(4).map("{:.4f}".format)
#df = df.drop(columns='geometry')
#df = df.drop(columns='Date & Tim')
#df = df.drop(columns='EMIT DATE')
df['lat_long_combined'] = df['latitude'] + ', ' + df['longitude']

print(df)
df = df.drop(columns=['latitude', 'longitude'])
df['Ground'] = ''
df = df.reindex(columns=['Name', 'lat_long_combined', 'Ground', 'EMIT DATE'])
#df['EMIT DATE'] = pd.to_datetime(df['EMIT DATE'])
#df['EMIT DATE'] = df['EMIT DATE'].dt.strftime('%Y-%m-%d')
print(df.columns)
print(df.to_latex(index=False))

gis_data = os.path.join(r'C:\Users\spect\PycharmProjects\pythonProject\terraspec\gis\shift_transects_centroid.shp')
df = gp.read_file(gis_data)
print(df.columns)

df['longitude'] = df['geometry'].x.round(4).map("{:.4f}".format)
df['latitude'] = df['geometry'].y.round(4).map("{:.4f}".format)
df['lat_long_combined'] = df['latitude'] + ', ' + df['longitude']

print(df)
df = df.drop(columns=['latitude', 'longitude'])
df['Ground'] = ''
df = df.reindex(columns=['plot', 'lat_long_combined', 'Ground', 'EMIT DATE'])
df['EMIT DATE'] = pd.to_datetime(df['EMIT DATE'])
df['EMIT DATE'] = df['EMIT DATE'].dt.strftime('%Y-%m-%d')
print(df.columns)
print(df.to_latex(index=False))



corresponding_flight = { 'DPA-004-FALL' : '20220915t195816',
                         'DPB-003-FALL' : '20220915t195816',
                         'DPB-004-FALL' : '20220915t200714',
                         'DPB-005-FALL' : '20220915t195816',
                         'DPB-020-SPRING' : '20220322t204749',
                         'DPB-027-SPRING' : '20220412t205405',
                         'SRA-000-SPRING' : '20220420t195351',
                         'SRA-007-FALL': '20220914t184300',
                         'SRA-008-FALL': '20220914t184300',
                         'SRA-019-SPRING' : '20220308t190523',
                         'SRA-020-SPRING' : '20220308t205512',
                         'SRA-021-SPRING' : '20220308t204043',
                         'SRA-033-SPRING' : '20220316t210303',
                         'SRA-034-SPRING' : '20220316t210303',
                         'SRA-056-FALL' : '20220914t184300',
                         'SRA-109-SPRING' : '20220511t190344',
                         'SRB-004-FALL' : '20220914t184300',
                         'SRB-010-FALL' : '20220915t203517',
                         'SRB-021-SPRING' : '20220308t205512',
                         'SRB-026-SPRING' : '20220308t191151',
                         'SRB-045-FALL' : '20220915t185652',
                         'SRB-046-FALL' : '20220915t203517',
                         'SRB-047-SPRING' : '20220405t201359',
                         'SRB-050-FALL' : '',
                         'SRB-084-SPRING' : '20220511t212317',
                         'SRB-100-FALL' : '20220915t203517',
                         'SRB-200-FALL' : '',}

for key, value in corresponding_flight.items():
    print(key, value)


