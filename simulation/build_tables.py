import os
import pandas as pd
from glob import glob
from p_tqdm import p_umap
from functools import partial
from isofit.core.sunposition import sunpos
from datetime import datetime, timezone, timedelta
from utils.results_utils import load_fraction_files, error_processing, uncertainty_processing, load_data
from utils.create_tree import create_directory
from osgeo import gdal
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simulation.latex_tables import run_latex_tables

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
                   'npv_rmse', 'pv_rmse', 'soil_rmse', 'npv_r2', 'pv_r2', 'soil_r2', 'npv_std', 'pv_std', 'soil_std',
                   'npv_stde', 'pv_stde', 'soil_stde']

        df = pd.DataFrame(results, columns=cols_df)
        df.to_csv(os.path.join(self.fig_directory, mode + "_unmix_error_report.csv"), index=False)

    def unmix_unceratinty_table(self, mode:str):
        uncertainty_files = load_fraction_files(self.base_directory, mode, '*_fractional_cover_uncertainty')
        uncertainty_files = [i for i in uncertainty_files if not ('withold' in i)]
        results = p_umap(partial(uncertainty_processing,output_directory=self.output_directory), uncertainty_files,
                         **{"desc": f"\t\t processing {mode} uncertainty tables...", "ncols": 150})

        cols_df = ['scenario', 'normalization', 'num_em', 'cmbs', 'dims', 'mc_runs', 'npv_uncer', 'pv_uncer', 'soil_uncer',
                   'npv_std', 'pv_std', 'soil_std',  'npv_stde', 'pv_stde', 'soil_stde']

        df = pd.DataFrame(results, columns=cols_df)
        df.to_csv(os.path.join(self.fig_directory,  mode + "_unmix_uncertainty_report.csv"), index=False)

    def atmosphere_table(self):
        fraction_files = glob(os.path.join(self.base_directory, 'output', 'hypertrace', '**', '*fractional_cover'),
                              recursive=True)
        print('processing atmospheric table...')
        # this was the spectra used for hypertrace with the lowest error!!!
        true_file = os.path.join(self.base_directory, 'output', 'convex_hull__n_dims_4_fractions')
        df_rows = []

        for file in fraction_files:
            basename = os.path.basename(file)
            truth_array, estimated_array = load_data(true_file, file)

            # get hypertrace run parameters
            atmosphere = basename.split("_")[1][4:]
            altitude = basename.split("_")[2].replace("-", ".")
            doy = basename.split("_")[3]
            long = basename.split("_")[4].replace("-", ".")
            lat = basename.split("_")[5].replace("-", ".")
            azimuth = basename.split("_")[6].replace("-", ".")
            sensor_zenith = basename.split("_")[7].replace("-", ".")
            time_str = basename.split("_")[8].replace("-", ".")
            elev = basename.split("_")[9].replace("-", ".")
            aod = basename.split("_")[10].replace("-", ".")
            h2o = basename.split("_")[11].replace("-", ".")

            # get solar zenith from run results
            hypertrace_directory = os.path.join(self.base_directory, 'output', 'hypertrace',
                                                'veg-simulation-atmospheres',
                                                'atm_ATM_' + atmosphere.replace("-", "_") + '__alt_' + altitude + '__doy_' + doy + '__lat_' + lat + '__lon_' + long,
                                                'az_' + azimuth + '__zen_' + sensor_zenith + '__time_' + time_str + '__elev_' + elev,
                                                'noise_noise_coeff_sbg_cbe0', 'prior_emit__inversion_inversion',
                                                'aod_' + aod + '__h2o_' + h2o, 'cal_NONE__draw_0__scale_0',
                                                'fwd_lut')

            # get sma_uncertainty
            sma_uncertainty = file + '_uncertainty'
            ds_sma_uncertainty = gdal.Open(sma_uncertainty, gdal.GA_ReadOnly)
            sma_uncertainty_array = ds_sma_uncertainty.ReadAsArray().transpose((1, 2, 0))
            # get solar zenith angle using hypertrace params
            hour = int(datetime.strptime(str(timedelta(hours=float(time_str))), '%H:%M:%S').strftime('%H'))
            minute = int(datetime.strptime(str(timedelta(hours=float(time_str))), '%H:%M:%S').strftime('%M'))
            month = int(datetime.strptime(doy, '%j').strftime('%m'))
            day_month = int(datetime.strptime(doy, '%j').strftime('%d'))

            # create
            utc_time = datetime(2021, month, day_month-1, hour, minute, 0, tzinfo=timezone.utc)
            geometry_resutls = sunpos(utc_time, float(lat), float(long) * -1, float(elev) * 1000) # we use -1 to adjust for quadrant 4 of earth
            sza = geometry_resutls[1]

            # calculate errors
            errors = []

            # open uncertainty files
            for _em, em in enumerate(self.ems):
                x = truth_array[:, :, _em].flatten()
                y = estimated_array[:, :, _em].flatten()
                sma_uncer = sma_uncertainty_array[:, :, _em].flatten()
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                uncertainty = sma_uncer[~np.isnan(y)]

                sma_un = np.mean(uncertainty)
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                errors.append([rmse, mae, sma_un])

            error_flat = [item for sublist in errors for item in sublist]
            row = [str(atmosphere), float(altitude), float(doy), float(long), float(lat), float(azimuth),
                   float(sensor_zenith), float(time_str), float(elev), float(aod), float(h2o),
                   float(sza)] + error_flat
            df_rows.append(row)

        col_names = ['atmosphere', 'altitude', 'doy', 'long', 'lat', 'azimuth', 'sensor_zenith',
                     'time', 'elev', 'aod', 'h2o', "solar_zenith",
                     'npv_rmse', 'npv_mae', 'npv_sma-uncertainty',
                     'pv_rmse', 'pv_mae', 'pv_sma-uncertainty',
                     'soil_rmse', 'soil_mae', 'soil_sma-uncertainty']

        df = pd.DataFrame(df_rows, columns=col_names)
        df.to_csv(os.path.join(self.fig_directory, "atmosphere_error_report.csv"), index=False)
        print('\t done')

    def geographic_table(self, mode:str):
        fraction_files = load_fraction_files(self.base_directory, mode, '*withold*_fractional_cover')
        results = p_umap(partial(error_processing, output_directory=self.output_directory), fraction_files,
                         **{
                             "desc": f"\t\t processing {mode} error tables...",
                             "ncols": 150})

        cols_df = ['scenario', 'normalization', 'num_em', 'cmbs', 'dims', 'mc_runs', 'npv_mae', 'pv_mae', 'soil_mae',
                   'npv_rmse', 'pv_rmse', 'soil_rmse', 'npv_r2', 'pv_r2', 'soil_r2', 'npv_std', 'pv_std', 'soil_std',
                   'npv_stde', 'pv_stde', 'soil_stde']

        df = pd.DataFrame(results, columns=cols_df)
        df.to_csv(os.path.join(self.fig_directory, "geographic_error_report.csv"), index=False)


def run_build_tables(base_directory):
    run_tables = tables(base_directory=base_directory)
    run_tables.unmix_error_table(mode='sma-best')
    run_tables.unmix_unceratinty_table(mode='sma-best')
    run_tables.unmix_error_table(mode='mesma')
    run_tables.unmix_unceratinty_table(mode='mesma')
    #run_tables.atmosphere_table()
    run_tables.geographic_table(mode='spatial')

    # print latex tables
    run_latex_tables(base_directory=base_directory)
