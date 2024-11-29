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