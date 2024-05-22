import os
import subprocess
import time
from itertools import product
from sys import platform
from datetime import datetime
import numpy as np
from simulation.run_unmix import call_unmix
import pandas as pd
from glob import glob
from utils.spectra_utils import spectra
from utils.create_tree import create_directory

if "win" in platform:
    level_arg = 'level_1'
    n_cores = '8'
else:
    level_arg = 'level_1'
    n_cores = '40'

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

def aviris_data(aviris_csv, shift_plot, df_em, season):

    try:
        df_aviris = pd.read_csv(aviris_csv, low_memory=False)
        df_select_aviris = df_aviris.loc[df_aviris['plot_name'] == shift_plot].copy()
        spectra = df_select_aviris.groupby('wavelength')['reflectance'].mean()
        sample_feld_date = df_em['date'].values[0]
        img_date = df_select_aviris['time'].values[0]
        date1 = datetime.strptime(img_date, "%Y-%m-%d")
        date2 = datetime.strptime(sample_feld_date, "%Y-%m-%d")

        # Calculate the difference between the two dates
        difference = date1 - date2

        del df_aviris
        return [shift_plot, season, img_date, sample_feld_date, difference] + spectra.tolist()

    except:
        print(shift_plot, aviris_csv, 'failed...')

class unmix_runs:
    def __init__(self, base_directory: str, dry_run = False):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        create_directory(os.path.join(self.output_directory, 'outlogs'))
        create_directory(os.path.join(self.output_directory, 'scratch'))

        self.output_directory = os.path.join(base_directory, 'output')
        self.spectral_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')
        self.spectral_em_directory = os.path.join(self.output_directory, 'spectral_transects', 'endmembers')
        self.gis_directory = os.path.join(base_directory, 'gis')

        # create output destination for instrument scenes
        create_directory(os.path.join(self.output_directory, 'spectral_transects', 'airborne'))
        self.airborne_directory = os.path.join(self.output_directory, 'spectral_transects', 'airborne')

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='aviris_ng')

        # load dry run parameter
        self.dry_run = dry_run

        self.scale = '0.5'

        # load convolved EMIT global library
        self.emit_global = os.path.join(self.base_directory, 'output', 'shift_convex_hull__n_dims_4_unmix_library.csv')
        self.spectra_starting_column_local = '11'
        self.spectra_starting_column_global = '8'

        # optimal parameters for unmixing
        self.optimal_parameters_sma = ['--num_endmembers 20', '--n_mc 25', '--normalization brightness']
        self.optimal_parameters_mesma = ['--max_combinations 100', '--n_mc 25', '--normalization brightness']

        self.num_cmb = ['--max_combinations 10', '--max_combinations 20', '--max_combinations 30', '--max_combinations 40',
                        '--max_combinations 50', '--max_combinations 60', '--max_combinations 70', '--max_combinations 80',
                        '--max_combinations 90', '--max_combinations 100' , '--max_combinations 5']

    def unmix_calls(self, mode:str):
        # Start unmixing process
        print("commencing... spectral unmixing")

        exclude = ['.hdr', '.csv', '.ini', '.xml']

        # get plot reflectances
        reflectance_files = sorted(glob(os.path.join(self.spectral_transect_directory, '*')))

        # loop counter
        create_directory(os.path.join(self.output_directory, mode))
        count = 0
        for reflectance_file in reflectance_files:

            if os.path.splitext(reflectance_file)[1] in exclude:
                continue

            # site info
            plot_num = os.path.basename(reflectance_file)

            # get aviris reflectances
            shift_plot = plot_num.split("_")[0]
            season = plot_num.split("_")[1]

            # em transect file
            em_file = os.path.join(self.spectral_em_directory, f"{shift_plot}_{season}_aviris_ng.csv")

            # load 3x3 windows from extracts
            img_date = corresponding_flight[f"{shift_plot}-{season}"]
            aviris_line = glob(os.path.join(self.base_directory, 'gis', 'shift-data-clip', f"*{shift_plot}-{season}_ang{img_date}*"))
            aviris_line = [file for file in aviris_line if not file.endswith(".hdr")]

            if not aviris_line:
                continue
            else:
                aviris_refl = aviris_line[0]

            if os.path.isfile(em_file):
                # run unmix codes
                # get information about run for metadata
                output_dest = os.path.join(self.output_directory, mode)

                if mode == 'mesma':
                    simulation_parameters = self.optimal_parameters_mesma
                else:
                    simulation_parameters = self.optimal_parameters_sma

                updated_parameters = simulation_parameters
                out_param_string = " ".join(updated_parameters)
                out_param_name = f'{shift_plot.replace(" ", "")}-{season}___{out_param_string.replace("--", "").replace("_", "-").replace(" ", "_")}'


                # asd unmixing - slpit with local endmembers
                count = count + 1
                call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_file, em_file=em_file,
                           parameters=updated_parameters, output_dest=os.path.join(output_dest,os.path.join(output_dest, f'asd-local___{out_param_name}')),
                           spectra_starting_column=self.spectra_starting_column_local, scale=self.scale)

                # airborne unmixing with local endmembers
                count = count + 1
                call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=aviris_refl, em_file=em_file,
                           parameters=updated_parameters, output_dest=os.path.join(output_dest,f'aviris-local___{out_param_name}'),
                           spectra_starting_column=self.spectra_starting_column_local, scale=self.scale)

                # asd unmixing with global endmembers
                count = count + 1
                call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_file, em_file=self.emit_global,
                           parameters=updated_parameters, output_dest=os.path.join(output_dest,f'asd-global___{out_param_name}'),
                           spectra_starting_column=self.spectra_starting_column_global, scale=self.scale)

                # airborne unmixing with global endmembers
                count = count + 1
                call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=aviris_refl, em_file=self.emit_global,
                           parameters=updated_parameters, output_dest=os.path.join(output_dest,f'aviris-global___{out_param_name}'),
                           spectra_starting_column=self.spectra_starting_column_global, scale=self.scale)

            else:
                print(em_file + " does not exist!")
                pass
                raise FileNotFoundError

        print('total unmix calls: ', count)

def run_shift_unmix(base_directory, dry_run):
    all_runs = unmix_runs(base_directory, dry_run)
    all_runs.unmix_calls(mode='sma')
    all_runs.unmix_calls(mode='mesma')
