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

def load_fraction_files(base_directory: str, mode: str, search_kw: str):
    files = glob(os.path.join(base_directory, "output", mode, search_kw), recursive=True)
    return files


def load_data(x, y):
    ds_truth = gdal.Open(x, gdal.GA_ReadOnly)
    ds_fractions = gdal.Open(y, gdal.GA_ReadOnly)
    truth_array = ds_truth.ReadAsArray().transpose((1, 2, 0))
    estimated_array = ds_fractions.ReadAsArray().transpose((1, 2, 0))
    return truth_array, estimated_array


def error_metrics(truth_array, estimated_array):
    r2 = []
    rmse = []
    mae = []
    stds = []
    std_error = []
    for em in range(truth_array.shape[2]):
        x = truth_array[:, :, em].flatten()
        y = estimated_array[:, :, em].flatten()
        stds.append(np.std(np.abs(x-y)))
        rmse.append(mean_squared_error(x, y, squared=False))
        r2.append(r2_score(x, y))
        mae.append(mean_absolute_error(x, y))
        std_error.append(sem(a=np.abs(x-y), ddof=1, nan_policy='omit'))

    return mae + rmse + r2 + stds + std_error


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
    error = error_metrics(truth_array, estimated_array)

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

    start_times = []
    end_times = []
    elapsed_times = []

    for line in lines:
        # pattern for time
        time_match = re.search(r"Elapsed Time: ([\d.e-]+), Start Time: ([\d.e-]+), End Time: ([\d.e-]+)", line.strip())
        if time_match:
            elapsed_time = float(time_match.group(1))
            start_time = float(time_match.group(2))
            end_time = float(time_match.group(3))
            elapsed_times.append(elapsed_time)
            start_times.append(start_time)
            end_times.append(end_time)

        argument_match = re.search(r'Arguments:\s*\(([^)]+)\)', line.strip())
        if argument_match:
            argument_string = argument_match.group(1)
            arguments = dict(re.findall(r'(\w+)\s*=\s*([^,)]+)', argument_string))

    start_times = [start - 1.0e9 for start in start_times]
    end_times = [end - 1.0e9 for end in end_times]

    try:
        df = pd.DataFrame([arguments], columns=arguments.keys())
        df['elapsed_time'] = np.max(np.array(end_times)) - np.min(np.array(start_times))
        df['worker_time'] = np.mean(np.array(elapsed_times))

        return df
    except:
        print(out_file, "had an error...")

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