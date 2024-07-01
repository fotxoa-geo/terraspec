import time
from glob import glob
import os

import pandas as pd
from osgeo import gdal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import sem
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from utils.envi import envi_to_array
from datetime import datetime, timezone, timedelta
from isofit.core.sunposition import sunpos


def load_fraction_files(base_directory: str, mode: str, search_kw: str):
    files = glob(os.path.join(base_directory, "output", mode, search_kw), recursive=True)
    return files


def load_data(x, y):
    ds_truth = gdal.Open(x, gdal.GA_ReadOnly)
    ds_fractions = gdal.Open(y, gdal.GA_ReadOnly)
    truth_array = ds_truth.ReadAsArray().transpose((1, 2, 0))
    estimated_array = ds_fractions.ReadAsArray().transpose((1, 2, 0))
    return truth_array, estimated_array


def error_metrics(truth_array, estimated_array, mc_unc_array, mc_runs):
    r2 = []
    rmse = []
    mae = []
    mc_unc = []
    std_error = []
    mc_avg = []

    for em in range(truth_array.shape[2]):
        x = truth_array[:, :, em].flatten()
        y = estimated_array[:, :, em].flatten()

        if mc_runs > 1:
            sstd = mc_unc_array[: ,:, em].flatten()
            sum_square_sstd = np.sum(np.square(sstd))
            unc = np.sqrt(sum_square_sstd)/sstd.shape[0]
            mc_unc.append(unc)
            mc_avg.append(np.mean(sstd))

        else:
            mc_unc.append(-9999)
            mc_avg.append(-9999)

        rmse.append(mean_squared_error(x, y, squared=False))
        r2.append(r2_calculations(x, y))
        mae.append(mean_absolute_error(x, y))
        std_error.append(sem(a=np.abs(x-y), ddof=1, nan_policy='omit'))

    return mae + rmse + r2 + mc_unc + std_error + mc_avg


def uncertainty_metrics(uncertainty_array):
    uncertainty = []
    stds = []
    std_error = []

    for em in range(uncertainty_array.shape[2] -1):
        uncer = np.mean(uncertainty_array[:, :, em])
        std = np.std(uncertainty_array[:, :, em])
        uncertainty.append(uncer)
        stds.append(std)
        std_error.append(np.std(np.abs(uncertainty_array[:, :, em]), ddof=1) / np.sqrt(np.size(uncertainty_array[:, :, em])))

    return uncertainty + stds + std_error


def param_search(vars, key):
    params = vars.split("_")
    start_idx = params.index(key)
    end_idx = start_idx + 1

    return params[start_idx], params[end_idx]


def error_processing(file, output_directory):
    basename = os.path.basename(file)

    
    if any("withold" in s for s in basename.split("_")):
        truth_base = 'geographic'
    else:
        truth_base = basename.split("_")[0]

    dims = os.path.basename(file).split("_")[4]

    if truth_base == 'convex':
        truth_file = os.path.join(output_directory, 'convex_hull__n_dims_' + dims + '_fractions')

    elif truth_base == 'latin':
        truth_file = os.path.join(output_directory, 'latin_hypercube__n_dims_' + dims + '_fractions')

    elif truth_base == 'geographic':
        file_path = basename.split("_")[:7]
        truth_base = file_path[2]
        del file_path[file_path.index("spectra")]
        file_path[1] += "--"
        file_path = "_".join(file_path).replace("--_", "--").replace("_n", "__n")
        truth_file = os.path.join(output_directory, file_path + '_fractions')
    else:
        raise FileNotFoundError(basename + "not found.")

    truth_array, estimated_array = load_data(truth_file, file)
    
    # get normalization parameters
    if 'normalization' in basename:
        norm_opts = param_search(basename, 'normalization')
        normalization = norm_opts[1]
    else:
        normalization = np.nan

    # endmember parameters
    if 'num_endmembers' in basename:
        norm_opts = param_search(basename, 'endmembers')
        num_em = norm_opts[1]
    else:
        num_em = np.nan

    # endmember parameters
    if 'n_mc' in basename:
        norm_opts = param_search(basename, 'mc')
        mc_runs = int(norm_opts[1])
    else:
        mc_runs = np.nan

    # endmember parameters
    if 'combinations' in basename:
        norm_opts = param_search(basename, 'combinations')
        combs = norm_opts[1]
    else:
        combs = np.nan
    
    if mc_runs > 1:
        unc_file = f"{file}_uncertainty"
        mc_unc_array = envi_to_array(unc_file)
    else:
        mc_unc_array = False


    error = error_metrics(truth_array, estimated_array, mc_unc_array, mc_runs)
    
    return [truth_base, normalization, num_em, combs, dims, mc_runs] + error
    del truth_array, estimated_array


def uncertainty_processing(file, output_directory):
    basename = os.path.basename(file)

    ds_uncertainty = gdal.Open(file, gdal.GA_ReadOnly)
    uncertainty_array = ds_uncertainty.ReadAsArray().transpose((1, 2, 0))
    uncertainty = uncertainty_metrics(uncertainty_array)

    scenario = basename.split("_")[0]
    dims = os.path.basename(file).split("_")[4]

    # get normalization parameters
    if 'normalization' in basename:
        norm_opts = param_search(basename, 'normalization')
        normalization = norm_opts[1]
    else:
        normalization = np.nan

    # endmember parameters
    if 'num_endmembers' in basename:
        norm_opts = param_search(basename, 'endmembers')
        num_em = norm_opts[1]
    else:
        num_em = np.nan

    # endmember parameters
    if 'n_mc' in basename:
        norm_opts = param_search(basename, 'mc')
        mc_runs = norm_opts[1]
    else:
        mc_runs = np.nan

    # endmember parameters
    if 'combinations' in basename:
        norm_opts = param_search(basename, 'combinations')
        combs = norm_opts[1]
    else:
        combs = np.nan

    return [scenario, normalization, num_em, combs, dims, mc_runs] + uncertainty
    del uncertainty_array


def performance_log(out_file:str):

    with open(out_file) as f:
        lines = f.readlines()

    total_seconds = []
    error_flag = 0
    worker_counter = 0
    total_lines = 0
    
    try:
        for line in lines:
            # pattern for time
            time_match = re.search(r"seconds:\s+([\d.]+)", line.strip())
            if time_match:
                pixel_s = float(time_match.group(1))
                total_seconds.append(pixel_s)
                worker_counter += 1

            argument_match = re.search(r'Arguments:\s*\(([^)]+)\)', line.strip())
        
            if argument_match:
                argument_string = argument_match.group(1)
                arguments = dict(re.findall(r'(\w+)\s*=\s*([^,)]+)', argument_string))
        
            error_match = re.search(r'GDALError \(CE_Failure, code 10\):', line.strip())
        
            if error_match:
                error_flag = 1

            host_name_match = re.search(r'Unmixing was processed on: (.+)' , line.strip())
            if host_name_match:
                cpu_host = str(host_name_match.group(1))

            total_line_match = re.search(r'Running from lines: (\d+) - (\d+)', line.strip())
            if total_line_match:
                total_lines = str(total_line_match.group(2))
    
        if worker_counter !=  int(total_lines) - 1:
            error_flag = 1
    
        df = pd.DataFrame([arguments], columns=arguments.keys())
        df['spectra_per_s'] = worker_counter/(np.sum(total_seconds))
        df['total_time'] = np.sum(total_seconds)
        df['error'] = error_flag
        df['worker_count'] = worker_counter
        df['node'] = cpu_host
        
        return df
    except:
        print(out_file, "failed!")

def r2_calculations(x_vals, y_vals):
    X = np.array(x_vals)
    y = np.array(y_vals)
    X = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X)
    r2 = r2_score(y_vals, y_pred)

    return np.round(r2,2)




def atmosphere_file(fraction_file, base_directory):

    # this was the spectra used for hypertrace with the lowest error!!!
    true_file = os.path.join(base_directory, 'output', 'convex_hull__n_dims_4_fractions')
    df_rows = []

    basename = os.path.basename(fraction_file)
    truth_array, estimated_array = load_data(true_file, fraction_file)

    # get mode
    mode = basename.split("_")[0]

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
    #hypertrace_directory = os.path.join(self.base_directory, 'output', 'hypertrace', 'veg-simulation-atmospheres',
    #                                            'atm_ATM_' + atmosphere.replace("-", "_") + '__alt_' + altitude + '__doy_' + doy + '__lat_' + lat + '__lon_' + long,
    #                                            'az_' + azimuth + '__zen_' + sensor_zenith + '__time_' + time_str + '__elev_' + elev,
    #                                            'noise_noise_coeff_sbg_cbe0', 'prior_emit__inversion_inversion',
    #                                            'aod_' + aod + '__h2o_' + h2o, 'cal_NONE__draw_0__scale_0',
    #                                            'fwd_lut')

    # get uncertainty
    uncertainty_file = f'{fraction_file}_uncertainty'
    ds_uncertainty = gdal.Open(uncertainty_file, gdal.GA_ReadOnly)
    mc_uncertainty_array = ds_uncertainty.ReadAsArray().transpose((1, 2, 0))
    
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
    for _em, em in enumerate(['npv', 'pv', 'soil']):
        x = truth_array[:, :, _em].flatten()
        y = estimated_array[:, :, _em].flatten()
        #x = x[~np.isnan(y)]
        #y = y[~np.isnan(y)]
        #uncertainty = mc_uncertainty_array[~np.isnan(y)]
        
        # calcualte errors
        rmse = mean_squared_error(x, y, squared=False)
        mae = mean_absolute_error(x, y)
        
        # calculate uncertainty between f and f-hat
        std_error = sem(a=np.abs(x-y), ddof=1, nan_policy='omit')

        # calculate monte carlo uncertainty
        mc_unc = np.mean(mc_uncertainty_array[: ,:, _em]/np.sqrt(25)) 
        
        # append results to rows
        errors.append([rmse, mae, std_error, mc_unc])

    error_flat = [item for sublist in errors for item in sublist]
    row = [mode, str(atmosphere), float(altitude), float(doy), float(long), float(lat), float(azimuth),
                   float(sensor_zenith), float(time_str), float(elev), float(aod), float(h2o),
                   float(sza)] + error_flat
    
    return row



