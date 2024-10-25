import pandas as pd
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.create_tree import create_directory
from utils.spectra_utils import spectra


class ecosis_table:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.sim_directory = os.path.join(self.base_directory, 'simulation')
        self.slpit_directory = os.path.join(self.base_directory, 'slpit')
        self.shift_directory = os.path.join(self.base_directory, 'shift')

        create_directory(os.path.join(self.base_directory, 'ecosis'))
        self.ecosis_directory = os.path.join(self.base_directory, 'ecosis')

        # set asd wvls
        self.asd_wvls = spectra.load_asd_wavelenghts()

        # em_labels
        self.ems = ['npv', 'pv', 'soil']

        # load species key
        species_key = os.path.join('utils', 'species_santabarbara_ca.csv')
        df_species = pd.read_csv(species_key)
        df_species['label'] = df_species['label'].apply(lambda x: x.split('-')[0].strip())
        self.df_species = df_species



    def reformat_simulation_ecosis(self):

        tables = glob(os.path.join(self.sim_directory, 'output', 'geofilter', '*.csv'))
        dfs = []

        df_species = self.df_species
        mapping = df_species.set_index('key_value')['label'].to_dict()

        # convolved library
        df_emit_lib_global = pd.read_csv(os.path.join(self.sim_directory, 'output', 'convolved',
                                                      'geofilter_convolved.csv'))

        for table in tables:
            dataset = os.path.basename(table).split(".")[0].split("_")[-1]

            if dataset in ['DP', 'SR', 'JORN']:
                df = pd.read_csv(table)
                df = df.sort_values('level_1')

                df.insert(5, 'latin_genus', '')
                df.insert(6, 'latin_species', '')

                # modify vegetation
                df_veg = df.loc[df['level_1'] != 'soil'].copy()
                df_veg['level_2'] = df_veg['level_2'].fillna("UNK-")
                df_veg['level_2'] = df_veg['level_2'].replace("UNKN-", 'UNK-')
                df_veg['level_2'] = df_veg['level_2'].map(mapping)

                # fill in genus and species if known
                df_veg.loc[df_veg['level_2'] != 'Unknown', 'latin_genus'] = df_veg['level_2'].str.split(' ').str[0]
                df_veg.loc[df_veg['level_2'] != 'Unknown', 'latin_species'] = df_veg['level_2'].str.split(' ').str[1]

                # fill in remaining genus and species with unknown
                df_veg.loc[df_veg['level_2'] == 'Unknown', 'latin_genus'] = df_veg['level_1'] + ' ' + df_veg['level_2']
                df_veg.loc[df_veg['level_2'] == 'Unknown', 'latin_species'] = df_veg['level_1'] + ' ' + df_veg['level_2']

                # fill in level 2 with level 1 for unks
                df_veg.loc[df_veg['level_2'] == 'Unknown', 'level_2'] = df_veg['level_1'] + ' ' + df_veg['level_2']
                df_veg.loc[df_veg['level_2'] == 'Unknown', 'level_2'] = df_veg['level_1'] + ' ' + df_veg['level_2']

                # modify soils
                df_soil = df.loc[df['level_1'] == 'soil'].copy()
                df_soil['level_2'] = df_soil['level_1']
                df_dataset_concat = pd.concat([df_veg, df_soil], axis=0, ignore_index=True)

                # check that these spectra are within filtered library
                check = df_dataset_concat['fname'].isin(df_emit_lib_global['fname']).all()

                if check:
                    print(f'all spectra in {dataset} are present in EMIT global lib')
                    dfs.append(df_dataset_concat)
                else:
                    spectra_not_present = df_dataset_concat[~df_dataset_concat['fname'].isin(df_emit_lib_global['fname'])]['fname']
                    print("Following spectra not found in EMIT library:")
                    print(spectra_not_present.tolist())

            else:
                print(f"{dataset} is not being published on ecosis")

        df_final = pd.concat(dfs, axis=0, ignore_index=True)
        df_final.drop(columns=['level_3'], inplace=True)
        df_final = df_final.sort_values('level_1')
        df_final.to_csv(os.path.join(self.ecosis_directory, 'simulation_asd_data.csv'), index=False)


def run_ecosis(base_directory):
    ecosis_table(base_directory=base_directory).reformat_simulation_ecosis()