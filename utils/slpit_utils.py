import time

import pandas as pd

class slpit:
    "ceratin utilities for split processing of asd data and arranging data"
    def __init__(self):
        print("")

    @classmethod
    def df_white_ref_table(cls, record):
        df_white_ref = pd.json_normalize(record['white_ref'])
        df_white_ref = df_white_ref.iloc[:, 14:]

        return df_white_ref

    @classmethod
    def df_em_table(cls, record):

        if not record["solo_slpit_toggle"]:
            df_transect_em = pd.json_normalize(record['emit_transect_endmembers'])
            df_transect_em = df_transect_em.iloc[:, 14:]

        else:
            df_transect_solo = pd.json_normalize(record['solo_slpit'])
            df_transect_solo = df_transect_solo.iloc[:, 14:]
            df_transect_solo.sort_values('start_em_range')

            df_em_rows = []
            for _row, row in df_transect_solo.iterrows():
                if not row['bad_em']:
                    bad_em = []
                else:
                    bad_em = pd.json_normalize(row['bad_em'])
                    bad_em = bad_em.iloc[:, 14:]
                    bad_em = bad_em['erroneous_endmembers'].tolist()

                for i in range(row['start_em_range'], row['end_em_range'] + 1):
                    if i in bad_em:
                        em_condition = 'bad'
                    else:
                        em_condition = ''

                    if row['em_classification'] == 'Soil':
                        species = ''
                    else:
                        species = 'UNK-'
                    df_em_rows.append([i, row["line_num"], row['em_classification'], species, '', '', '', em_condition])

                df_transect_em = pd.DataFrame(df_em_rows)
                df_transect_em.columns = ['asd_file_num', "transect_line_num", 'endmembers', "species", 'photo_toggle',
                                          'em_photo', 'notes', 'em_condition']

        return df_transect_em


