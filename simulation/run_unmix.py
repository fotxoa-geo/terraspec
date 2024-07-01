import os
import subprocess
from sys import platform
from utils.create_tree import create_directory
from utils.text_guide import execute_call
from utils.envi import get_meta, save_envi
from utils.spectra_utils import spectra
from glob import glob
from itertools import product
from osgeo import gdal
from p_tqdm import p_map
from functools import partial
import pandas as pd
import time
import shutil

level_arg = 'level_1'
n_cores = '40'



def unmix_sim_menu():
    print("You are in unmixing mode for simulations...")
    print("A... Unmix the no atmosphere reflectances")
    print("B... Unmix Hypertrace")
    print("C... Exit")



def create_uncertainty(uncertainty_file: str, wvls):
    output = os.path.join(os.path.dirname(uncertainty_file), 'reflectance_uncertainty.hdr')
    if os.path.isfile(output):
        pass
    else:
        ds_uncertainty = gdal.Open(uncertainty_file, gdal.GA_ReadOnly)
        uncertainty_array = ds_uncertainty.ReadAsArray().transpose((1, 2, 0))
        uncertainty_array = uncertainty_array[:, :, :-2]

        uncertainty_meta = get_meta(lines=uncertainty_array.shape[0], samples=uncertainty_array.shape[1], bands=wvls, wvls=True)
        save_envi(output_file=output, meta=uncertainty_meta, grid=uncertainty_array)


def call_unmix(mode: str, reflectance_file: str, em_file: str, dry_run: bool, parameters: list, output_dest:str, scale:str,
               spectra_starting_column:str, uncertainty_file=None, io_bug=None, scenes=None):
    
    out_dir_path = os.path.dirname(os.path.dirname(output_dest))
    
    create_directory( os.path.join(out_dir_path, 'outlogs'))
    outlog_name = os.path.join(out_dir_path, 'outlogs', f"{mode}-{os.path.basename(output_dest)}.out")  
    

    if scenes == None:
        scrtch_rfl = os.path.join(out_dir_path, 'scratch', f"{mode}-{os.path.basename(output_dest)}")
        scrtch_hdr = os.path.join(out_dir_path, 'scratch', f"{mode}-{os.path.basename(output_dest)}.hdr") 
        scrtch_csv = os.path.join(out_dir_path, 'scratch', f"{mode}-{os.path.basename(output_dest)}.csv")

        shutil.copyfile(reflectance_file, scrtch_rfl)
        shutil.copyfile(reflectance_file + '.hdr' , scrtch_hdr)
        shutil.copyfile(em_file, scrtch_csv)
    
    # the io bug call allows us to completely start all the runs 
    if io_bug:
        report_path = os.path.join(os.path.dirname(os.path.dirname((os.path.dirname(output_dest)))), 'figures',
                                   'computing_performance_report.csv')
        # open report
        df_report = pd.read_csv(report_path)
        df_report = df_report.loc[(df_report['error'] == 1)].copy()
        df_report = df_report.replace('"', '', regex=True)

        if scrtch_rfl in df_report['reflectance_file'].values:
            base_call = f'julia ~/EMIT/SpectralUnmixing/unmix.jl {scrtch_rfl} {scrtch_csv} ' \
                        f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column {spectra_starting_column} --refl_scale {scale} ' \
                        f'{" ".join(parameters)} '

            execute_call(
                ['sbatch', '-N', '1', '--tasks-per-node', "1", '--mem', "50G", '--output', outlog_name, '--job-name', 'emit.unmix' ,'--wrap',
                 f'{base_call}'], dry_run)
        else:
            pass
 
    else:
        if scenes:
            base_call = f'julia ~/EMIT/SpectralUnmixing/unmix.jl {reflectance_file} {em_file} ' \
                        f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column {spectra_starting_column} --refl_scale {scale} ' \
                        f'{" ".join(parameters)} '

        else:
            base_call = f'julia ~/EMIT/SpectralUnmixing/unmix.jl {scrtch_rfl} {scrtch_csv} ' \
                        f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column {spectra_starting_column} --refl_scale {scale} ' \
                        f'{" ".join(parameters)} '

        execute_call(['sbatch', '-N', '1', '--tasks-per-node', '1', '--mem', "50G", '--output', outlog_name, '--job-name', 'emit.unmix', '--wrap',
                      f'{base_call}'], dry_run)




def hypertrace_unmix(base_directory: str, mode: str, reflectance_file: str, em_file: str, dry_run: bool, parameters: list):
    # create results directory
    create_directory((os.path.join(base_directory, "output", 'hypertrace', 'fractions')))
    create_directory((os.path.join(base_directory, "output", 'hypertrace', 'outlogs')))
    create_directory((os.path.join(base_directory, "output", 'hypertrace', 'outlogs', 'unmix')))

    # get metadata from hypertrace outputs
    atmosphere = os.path.abspath(reflectance_file).split('/')[-7].split("__")[0].split("_")[1:]
    atmosphere = "-".join(atmosphere)
    altitude = os.path.abspath(reflectance_file).split('/')[-7].split("__")[1].split("_")[1]
    doy = os.path.abspath(reflectance_file).split('/')[-7].split("__")[2].split("_")[1]
    lat = os.path.abspath(reflectance_file).split('/')[-7].split("__")[3].split("_")[1]
    long = os.path.abspath(reflectance_file).split('/')[-7].split("__")[4].split("_")[1]

    # get geometry information from hypertrace output
    azimuth = os.path.abspath(reflectance_file).split('/')[-6].split("__")[0].split("_")[1]
    sensor_zenith = os.path.abspath(reflectance_file).split('/')[-6].split("__")[1].split("_")[1]
    time = os.path.abspath(reflectance_file).split('/')[-6].split("__")[2].split("_")[1]
    elev = os.path.abspath(reflectance_file).split('/')[-6].split("__")[3].split("_")[1]

    # get atmoshperic conditions from hypertrace output
    aod = os.path.abspath(reflectance_file).split('/')[-3].split("__")[0].split("_")[1]
    h2o = os.path.abspath(reflectance_file).split('/')[-3].split("__")[1].split("_")[1]

    basename = [atmosphere, altitude, doy, long, lat, azimuth, sensor_zenith, time, elev, aod, h2o]
    basename = "_".join(basename)

    # output destination
    output_dest = os.path.join(base_directory, "output", 'hypertrace', 'fractions', mode + "_" + basename.replace(".", "-"))

    # path to reflectance uncertainty file
    uncertainty_file = os.path.join(os.path.dirname(reflectance_file), "reflectance_uncertainty")
    outlog_name = os.path.join(base_directory, "output", 'hypertrace', 'outlogs', 'unmix', f'{mode}_{basename.replace(".", "-")}.out')

    base_call = f'julia ~/EMIT/SpectralUnmixing/unmix.jl {reflectance_file} {em_file} ' \
                f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column 8 ' \
                f'--reflectance_uncertainty_file {uncertainty_file} ' \
                f'{" ".join(parameters)}'

    execute_call(['sbatch', '-N', "1", '--tasks-per-node', '1', '--mem', "50G", '--output', outlog_name, '--wrap', f'{base_call}'], dry_run)


class runs:
    def __init__(self, base_directory, dry_run):

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='emit')

        # load dry run parameter
        self.dry_run = dry_run

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        create_directory(os.path.join(self.output_directory, 'outlogs'))
        create_directory(os.path.join(self.output_directory, 'scratch'))

        # load em libraries output
        self.em_libraries_output = os.path.join(self.output_directory, "endmember_libraries")

        # model parameters
        self.num_cmb = ['--max_combinations 10', '--max_combinations 25', '--max_combinations 50', '--max_combinations 100', '--max_combinations 500', '--max_combinations 1000']
        self.normalization = ['--normalization brightness', '--normalization none', '--normalization 1500']
        self.num_em = ['--num_endmembers 3', '--num_endmembers 5', '--num_endmembers 10', '--num_endmembers 20', '--num_endmembers 30']
        self.mc_runs = ['--n_mc 50', '--n_mc 25', '--n_mc 10', '--n_mc 5', '--n_mc 1']

        # simulation parameters for spatial and hypertrace unmix
        self.optimal_parameters_sma = ['--num_endmembers 30', '--n_mc 25', '--normalization brightness']
        self.optimal_parameters_mesma = ['--max_combinations 100', '--n_mc 25', '--normalization brightness']

        # set scale to 1 reflectance
        self.scale = '1'
        self.spectra_starting_col_julia = '8'

    def geographic_sma(self, mode:str):
        print("commencing geographic spectral unmixing...")

        # Start unmixing process
        reflectance_files = glob(os.path.join(self.output_directory, '*withold--*spectra'))
        create_directory((os.path.join(self.base_directory, "output", 'spatial')))
        
        for reflectance_file in reflectance_files:
            basename = os.path.basename(reflectance_file)

            site = os.path.basename(reflectance_file)

            # output destination
            output_dest = os.path.join(self.base_directory, "output", 'spatial', site + "_" + mode + " ".join(self.optimal_parameters)).replace(
                "--", "_").replace(" ", "_").replace("__", "_")

            # get em file
            em_file = os.path.join(self.em_libraries_output, "_".join(basename.split("__")[0:1]) + '__n_dims_4_unmix_library.csv')

            call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_file, em_file=em_file,
                       parameters=self.optimal_parameters, output_dest=output_dest, scale=self.scale,
                       spectra_starting_column=self.spectra_starting_col_julia)


    def latin_hypercubes(self, mode:str, io_bug):
        # Start unmixing process with all options
        print(f"commencing latin hypercube {mode} spectral unmixing...")

        if mode == 'mesma':
            options = product(self.normalization, self.num_cmb, self.mc_runs)
        else:
            options = product(self.normalization, self.num_em, self.mc_runs)

        all_sma_runs = list(options)

        # create results directory
        create_directory((os.path.join(self.base_directory, "output", mode)))
        
        count = 0
        for simulation_parameters in all_sma_runs:
            dfs = glob(os.path.join(self.em_libraries_output, '*latin_hypercube_*.csv'))

            for df in dfs:
                n_dimensions = os.path.basename(df).split("_")[5]
                reflectance_file = os.path.join(self.output_directory, 'latin_hypercube__n_dims_' + str(n_dimensions) + '_spectra')

                # output name
                output_name = 'latin_hypercube__n_dims_' + str(n_dimensions) + '_spectra'

                # output destination
                output_dest = os.path.join(self.base_directory, "output", mode,
                                           output_name + "_" + mode + " ".join(simulation_parameters)).replace(
                                            "--", "_").replace(" ", "_").replace("__", "_")

                call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_file, em_file=df, parameters=simulation_parameters, output_dest=output_dest, scale=self.scale, spectra_starting_column=self.spectra_starting_col_julia, io_bug=io_bug)
                count += 1

        print(f"total number of {mode} calls: {count}")

    def convex_hulls(self, mode:str, io_bug):
        print(f"commencing convex hull {mode} spectral unmixing...")
        
        if mode == 'mesma':
            options = product(self.normalization, self.num_cmb, self.mc_runs)
        else:
            options = product(self.normalization, self.num_em, self.mc_runs)

        all_sma_runs = list(options) 

        # create results directory
        create_directory((os.path.join(self.base_directory, "output", mode)))
        
        count = 0 
        for simulation_parameters in all_sma_runs:
            dfs = glob(os.path.join(self.em_libraries_output, '*convex_hull_*.csv'))
            for df in dfs:
                n_dimensions = os.path.basename(df).split("_")[5]
                reflectance_file = os.path.join(self.output_directory, 'convex_hull__n_dims_' + str(n_dimensions) + '_spectra')

                output_name = 'convex_hull__n_dims_' + str(n_dimensions) + '_spectra'

                # output destination
                output_dest = os.path.join(self.base_directory, "output", mode, output_name + "_" + mode + " ".join(simulation_parameters)).replace("--", "_").replace(" ", "_").replace("__", "_")
                call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_file, em_file=df, parameters=simulation_parameters, output_dest=output_dest, scale=self.scale, spectra_starting_column=self.spectra_starting_col_julia, io_bug=io_bug)
                count += 1
                
        print(f"total number of {mode} calls: {count}")

    def hypertrace_call(self, mode:str):
        # start hypertrace unmixing
        print(f"commencing hypertrace spectral unmixing...")

        # load em library for hypertrace unmixing
        hyp_em_lib = os.path.join(self.output_directory, "endmember_libraries", 'convex_hull__n_dims_4_unmix_library.csv')

        print("loading hypertrace outputs...")
        estimated_reflectances = glob(os.path.join(self.base_directory, "output", "hypertrace", '**',
                                                   '*estimated-reflectance'), recursive=True)

        uncertainty_files = []
        for reflectance_file in estimated_reflectances:
            uncertainty_file = os.path.join(os.path.dirname(reflectance_file), 'posterior-uncertainty')
            uncertainty_files.append(uncertainty_file)

        p_map(partial(create_uncertainty, wvls=self.wvls), uncertainty_files,
              **{"desc": "\t\t saving new uncertainty files...", "ncols": 150})

        if mode == 'mesma':
            opt_params = self.optimal_parameters_mesma
        else:
            opt_params = self.optimal_parameters_sma
        
        for reflectance_file in estimated_reflectances:
            hypertrace_unmix(base_directory=self.base_directory, mode=mode, dry_run=self.dry_run,
                             reflectance_file=reflectance_file, em_file=hyp_em_lib,
                             parameters=opt_params)

    
def run_unmix_workflow(base_directory, dry_run, io_bug):
    all_runs = runs(base_directory=base_directory, dry_run=dry_run)
    while True:
        unmix_sim_menu()
        user_input = input('\nPlease indicate the desired unmix mode: ').upper()

    #geo = all_runs.geographic_sma(mode='sma-best')
        if user_input == 'A':
            sma_convex = all_runs.convex_hulls(mode='sma', io_bug=io_bug)
            mesma_convex = all_runs.convex_hulls(mode='mesma', io_bug=io_bug)
            lh_sma = all_runs.latin_hypercubes(mode='sma', io_bug=io_bug)
            lh_mesma = all_runs.latin_hypercubes(mode='mesma', io_bug=io_bug)
            sma_convex = all_runs.convex_hulls(mode='sma-best', io_bug=io_bug)
            lh_sma = all_runs.latin_hypercubes(mode='sma-best', io_bug=io_bug)
        elif user_input == 'B':
            all_runs.hypertrace_call(mode='mesma')
            all_runs.hypertrace_call(mode='sma')
        elif user_input == 'C':
            print('returning to main menu...')
            break
