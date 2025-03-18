import os
import time

import pandas as pd
from glob import glob
from p_tqdm import p_umap
from functools import partial
from isofit.core.sunposition import sunpos
from datetime import datetime, timezone, timedelta
from utils.results_utils import load_fraction_files, error_processing, uncertainty_processing, load_data, performance_log, atmosphere_file
from utils.create_tree import create_directory
from utils.results_utils import param_search
from osgeo import gdal
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simulation.latex_tables import run_latex_tables
from functools import partial


def table_sim_menu():
    print("You are in table mode for simulations...")
    print("A... Computing Performance Tables")
    print("B... Error Tables")
    print("C... Latex Tables")
    print("D... Geographic Tables")
    print("E... Exit")


class tables:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")

        # check for figure directory
        create_directory(self.fig_directory)

        # em_labels
        self.ems = ['non-photosynthetic\nvegetation', 'photosynthetic\nvegetation', 'soil']
        self.ems_short = ['npv', 'pv', 'soil']

    def unmix_error_table(self, mode:str):
        fraction_files = load_fraction_files(self.base_directory, mode, '*_fractional_cover')
        fraction_files = [i for i in fraction_files if not ('withold' in i)]
        results = p_umap(partial(error_processing, output_directory=self.output_directory), fraction_files,
                         **{"desc": f"\t\t processing {mode} error tables...", "ncols": 150})

        cols_df = ['scenario', 'normalization', 'num_em', 'cmbs', 'dims', 'mc_runs', 'npv_mae', 'pv_mae', 'soil_mae',
                   'npv_rmse', 'pv_rmse', 'soil_rmse', 'npv_r2', 'pv_r2', 'soil_r2', 'npv_mc_unc', 'pv_mc_unc', 'soil_mc_unc',
                   'npv_stde', 'pv_stde', 'soil_stde', 'npv_mean_unc', 'pv_mean_unc', 'soil_mean_unc']

        df = pd.DataFrame(results, columns=cols_df)
        df.to_csv(os.path.join(self.fig_directory, f'{mode}_unmix_error_report.csv'), index=False)

    def geographic_table(self, mode:str):

        fraction_files = load_fraction_files(self.base_directory, mode, '*_fractional_cover')

        results = p_umap(partial(error_processing, output_directory=self.output_directory), fraction_files,
                         **{"desc": f"\t\t processing {mode} error tables...", "ncols": 150})

        cols_df = ['scenario', 'normalization', 'num_em', 'dimensions', 'continent_sim', 'mc_runs', 'npv_mae', 'pv_mae', 'soil_mae',
                   'npv_rmse', 'pv_rmse', 'soil_rmse', 'npv_r2', 'pv_r2', 'soil_r2', 'npv_mc_unc', 'pv_mc_unc', 'soil_mc_unc',
                   'npv_stde', 'pv_stde', 'soil_stde', 'npv_mean_unc', 'pv_mean_unc', 'soil_mean_unc']

        df = pd.DataFrame(results, columns=cols_df)
        df.to_csv(os.path.join(self.fig_directory, "geographic_error_report.csv"), index=False)

    def performance_table(self):
        outfiles = glob(os.path.join(self.output_directory, 'outlogs', '*.out'))

        dfs = p_umap(performance_log, outfiles,
                        **{"desc": f"\t\t processing performance tables...", "ncols": 150})
        
        df_performance = pd.concat(dfs)
        df_performance.to_csv(os.path.join(self.fig_directory, "computing_performance_report.csv"), index=False)
        
    def atmosphere_table(self):
        fraction_files = glob(os.path.join(self.base_directory, 'output', 'hypertrace', '**', '*fractional_cover'),
                              recursive=True)
        
        results = p_umap(partial(atmosphere_file, base_directory=self.base_directory), fraction_files, **{"desc": f"\t\t processing atmosphere fraction files...", "ncols": 150})
        df_atmos = pd.DataFrame(results)
        
        col_names = ['mode', 'atmosphere', 'altitude', 'doy', 'long', 'lat', 'azimuth', 'sensor_zenith', 'time', 'elev', 'aod', 'h2o', 'solar_zenith',
                        'npv_rmse', 'npv_mae', 'npv_stde', 'npv_mc_unc', 
                        'pv_rmse', 'pv_mae', 'pv_stde', 'pv_mc_unc',
                        'soil_rmse', 'soil_mae', 'soil_stde', 'soil_mc_unc']

        df_atmos.columns = col_names
        df_atmos.to_csv(os.path.join(self.fig_directory, "atmosphere_error_report.csv"), index=False)


def run_build_tables(base_directory):
    run_tables = tables(base_directory=base_directory)
    while True:
        table_sim_menu()

        user_input = input('\nPlease indicate the desired tables: ').upper()

        if user_input == 'A':
            run_tables.performance_table()
        elif user_input == 'B':
            modes = ['mesma', 'sma']
            for i in modes:
                run_tables.unmix_error_table(mode=i)
            
            run_tables.atmosphere_table()
        elif user_input == 'C':
            run_latex_tables(base_directory=base_directory)
        elif user_input == 'D':
            run_tables.geographic_table(mode='spatial')
        elif user_input == 'E':
            print("returning to simulation main menu...")
            break
