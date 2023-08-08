import os
import subprocess
import time
from itertools import product
from sys import platform
from figures import load_wavelengths
import pandas as pd
from convolve import create_directory
import logging
from glob import glob


level_arg = 'level_1'
n_cores = '40'

def execute_call(cmd_list, dry_run=False):
    if dry_run:
        print(cmd_list)
    else:
        subprocess.call(cmd_list)


def call_unmix(mode: str, reflectance_file: str, em_file: str, dry_run: bool, parameters: list, output_dest: str):
    base_call = f'julia -p {n_cores} ~/EMIT/SpectralUnmixing/unmix.jl {reflectance_file} {em_file} ' \
                f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column 8 --refl_scale 1 ' \
                f'{" ".join(parameters)} '

    # execute_call(['sbatch', '-N', "1", '-c', n_cores,'--mem', "80G", '--wrap', f'{base_call}'], dry_run)
    sbatch_cmd = f"sbatch -N 1 -c {n_cores} --mem 80G --wrap='{base_call}'"
    subprocess.check_output(sbatch_cmd, shell=True, text=True)


class unmix:
    def __init__(self, base_directory: str, normalization=[], num_em =[], mc_runs=[], num_cmb=[], wavelength_file='',
                 dry_run = False):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'spectral-surveying', 'output')
        self.spectral_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')
        self.spectral_em_directory = os.path.join(self.output_directory, 'spectral_transects', 'endmembers')
        self.gis_directory = os.path.join(base_directory, 'spectral-surveying', 'gis')

        # load model parameters
        self.normalization = normalization
        self.num_em = num_em
        self.mc_runs = mc_runs
        self.num_cmb = num_cmb

        # load wavelengths
        self.wvls, self.fwhm = load_wavelengths(wavelength_file)

        # load dry run parameter
        self.dry_run = dry_run

    def sma(self, mode='sma-best'):
        # Start unmixing process
        print("commencing... spectral unmixing")

        # get plot reflectances
        reflectance_files = sorted(glob(os.path.join(self.spectral_transect_directory, '*reflectance-emit_date*[!.hdr]')))

         # loop counter
        count = 0
        meta_rows = []
        for reflectance_file in reflectance_files:
            # site info
            plot_num = '-'.join(os.path.basename(reflectance_file).split("-")[:2])

            # unmix parameters ; every possible combination
            sma_options = product(self.normalization, self.num_em, self.mc_runs)
            all_sma_runs = [[e for e in result if e is not None] for result in sma_options]

            # em transect file
            em_file = os.path.join(self.spectral_em_directory, plot_num + '-emit.csv')

            # get cropped emit reflectance file
            emit_refl_file = glob(os.path.join(self.gis_directory, 'emit-data-clip', 'SPEC-' + plot_num.split("-")[1] + "_*[!.hdr!.aux.xml]"))
            if os.path.isfile(em_file):
                df_em = pd.read_csv(em_file)
                total_number_of_ems = len(df_em.index)

                for simulation_parameters in all_sma_runs:

                    # get information about run for metadata
                    split_run_params = ' '.join(simulation_parameters).replace('--', '').split(" ")

                    if 'num_endmembers' in split_run_params:
                        number_em_sim = split_run_params[split_run_params.index('num_endmembers') + 1]

                    if 'normalization' in split_run_params:
                        brightness = split_run_params[split_run_params.index('normalization') + 1]

                    if 'n_mc' in split_run_params:
                        n_mc = split_run_params[split_run_params.index('n_mc') + 1]

                    # check for number of ems; if number is smaller unmix with global library
                    #if total_number_of_ems < int(number_em_sim):
                    #    em_file = self.em_file_emit

                    # asd unmixing
                    count += 1
                    call_unmix(base_directory=self.output_directory, mode=mode, dry_run=self.dry_run,
                               reflectance_file=reflectance_file, em_file=em_file,
                               parameters=simulation_parameters, output_name='asd-' + plot_num)

                    # # spaceborne unmixing
                    count += 1
                    call_unmix(base_directory=self.output_directory, mode=mode, dry_run=self.dry_run,
                        reflectance_file=emit_refl_file[0], em_file=em_file, parameters=simulation_parameters, output_name='emit-' + plot_num)

                    meta_rows.append(['ground-' + plot_num, em_file, brightness, int(number_em_sim), int(n_mc)])
                    meta_rows.append(['emit-' + plot_num, em_file, brightness, int(number_em_sim), int(n_mc)])

            else:
                print(em_file + " does not exist!")
                raise FileNotFoundError

        df_meta = pd.DataFrame(meta_rows)
        df_meta.columns = ['output_name', 'em_file', 'brightness', 'num_em', 'n_mc']
        df_meta.to_csv(os.path.join(self.output_directory, 'unmix-metadata.csv'), index=False)
        print('total simulations: ', count)
