import time
from datetime import datetime
import numpy as np
import os
from glob import glob
from utils.create_tree import create_directory
from utils.envi import envi_to_array, save_envi, get_meta
from spectral.io import envi
from functools import partial
from p_tqdm import p_map
from scipy import stats

envi_typemap = {
    'uint8': 1,
    'int16': 2,
    'int32': 3,
    'float32': 4,
    'float64': 5,
    'complex64': 6,
    'complex128': 9,
    'uint16': 12,
    'uint32': 13,
    'int64': 14,
    'uint64': 15
}


def z_shift(time_series_values, time_points, perturbation_date):
    time_series_values[time_series_values == -9999. ] = np.nan
    non_nan_mask = ~np.isnan(time_series_values)
    filtered_series = time_series_values[non_nan_mask]

    dates = [datetime.strptime(d, '%Y%m%dT%H%M%S') for d in time_points]
    filtered_dates = np.array(dates)[non_nan_mask]

    pre_fire_mask = filtered_dates < perturbation_date
    post_fire_mask = filtered_dates >= perturbation_date

    pre_perturbation_signal = filtered_series[pre_fire_mask]
    post_perturbation_signal = filtered_series[post_fire_mask]

    z_score_pre = (pre_perturbation_signal - np.mean(pre_perturbation_signal))/np.std(pre_perturbation_signal)
    z_score_post = (post_perturbation_signal - np.mean(post_perturbation_signal))/np.std(post_perturbation_signal)

    z_score_trajectory = (post_perturbation_signal - np.mean(pre_perturbation_signal))/np.std(pre_perturbation_signal)
    z_post_average = np.mean(z_score_trajectory)
    z_shift = z_post_average - 0

    return z_shift

def welch_test(time_series_values, time_points, perturbation_date):
    time_series_values[time_series_values == -9999.] = np.nan
    non_nan_mask = ~np.isnan(time_series_values)
    filtered_series = time_series_values[non_nan_mask]

    dates = [datetime.strptime(d, '%Y%m%dT%H%M%S') for d in time_points]
    filtered_dates = np.array(dates)[non_nan_mask]

    pre_fire_mask = filtered_dates < perturbation_date
    post_fire_mask = filtered_dates >= perturbation_date

    pre_perturbation_signal = filtered_series[pre_fire_mask]
    post_perturbation_signal = filtered_series[post_fire_mask]

    t_stat, p_val = stats.ttest_ind(pre_perturbation_signal, post_perturbation_signal, equal_var=False)
    alpha = 0.05
    if p_val < alpha:
        return p_val
    else:
        return -9999.

def sam(time_series_values, time_points, perturbation_date, row, col):
    time_series_values[time_series_values == -9999.] = np.nan
    non_nan_mask = ~np.isnan(time_series_values)
    filtered_series = time_series_values[non_nan_mask]

    dates = [datetime.strptime(d, '%Y%m%dT%H%M%S') for d in time_points]
    filtered_dates = np.array(dates)[non_nan_mask]

    pre_fire_mask = filtered_dates < perturbation_date
    post_fire_mask = filtered_dates >= perturbation_date

    pre_perturbation_signal = filtered_series[pre_fire_mask]
    post_perturbation_signal = filtered_series[post_fire_mask]

    filtered_timepoints = np.array(time_points)[non_nan_mask]

    pre_fire_timepoints = filtered_timepoints[pre_fire_mask]
    max_value_of_pre_signal = np.argmax(pre_perturbation_signal)
    date_of_max_pre_signal = pre_fire_timepoints[max_value_of_pre_signal]

    pre_fire_rfl = envi_to_array(os.path.join('terraspec_output', 'fire', 'output', 'EXT', f'sedgwick_boundary_approx_RFL_{date_of_max_pre_signal}_EXT'))
    pre_fire_max_rfl = pre_fire_rfl[row, col, :]
    pre_fire_max_rfl[pre_fire_max_rfl == -9999.] = np.nan

    post_fire_timepoints = filtered_timepoints[post_fire_mask]
    max_value_of_post_signal = np.argmax(post_perturbation_signal)
    date_of_max_post_signal = post_fire_timepoints[max_value_of_post_signal]

    post_fire_rfl = envi_to_array(os.path.join('terraspec_output', 'fire', 'output', 'EXT',
                                              f'sedgwick_boundary_approx_RFL_{date_of_max_post_signal}_EXT'))
    post_fire_max_rfl = post_fire_rfl[row, col, :]
    post_fire_max_rfl[post_fire_max_rfl == -9999.] = np.nan

    t = np.array(post_fire_max_rfl)
    r = np.array(pre_fire_max_rfl)
    mask = ~np.isnan(t) & ~np.isnan(r)

    t_valid = t[mask]
    r_valid = r[mask]

    if t_valid.size == 0:
        return -9999.

    dot_product = np.dot(t_valid, r_valid)

    norm_t = np.linalg.norm(t_valid)
    norm_r = np.linalg.norm(r_valid)

    if norm_t == 0 or norm_r == 0:
        return -9999.

    cos_alpha = np.clip(dot_product / (norm_t * norm_r), -1.0, 1.0)

    return np.degrees(np.arccos(cos_alpha))

def row_parallel_processing(row, _row, time_points, perturbation_date):
    score_row = np.ones((row.shape[0], 3)) * -9999.

    for _col, col in enumerate(row):
        score_row[_col, 0] = z_shift(time_series_values=col, time_points=time_points, perturbation_date=perturbation_date)
        score_row[_col, 1] = welch_test(time_series_values=col, time_points=time_points,
                                     perturbation_date=perturbation_date)
        score_row[_col, 2] = sam(time_series_values=col, time_points=time_points,
                                        perturbation_date=perturbation_date, row=_row, col=_col)
    return score_row

class time_series:

    def __init__(self, base_directory: str, sensor: str, aoi: str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')

        # create output directories
        create_directory(os.path.join(self.output_directory, 'time_series'))
        self.time_series_directory = os.path.join(self.output_directory, 'time_series')

        # input data directories
        self.aoi_extraction = os.path.join(base_directory, 'gis', f'{sensor}-data', 'aoi', f'{os.path.basename(aoi.split('.')[0])}', 'EXT')

        # instrument to indicate wavelengths in output folder
        self.instrument = sensor
        self.aoi = aoi


    def build_time_series(self):
        mask_files = sorted(list(glob(os.path.join(self.aoi_extraction, '*_MASK*_EXT'))))
        meta = envi.read_envi_header(f'{mask_files[0]}.hdr')

        for _em, em in enumerate(['npv', 'pv', 'soil']):

            rows = 0
            cols = 0

            mask_files_update = []

            for _i, i in enumerate(mask_files):
                try:
                    mask_array = envi_to_array(i)
                    rows = max(rows, mask_array.shape[0])
                    cols = max(cols, mask_array.shape[1])
                    mask_files_update.append(i)

                except:
                    print(f'{i} could not be read!')
                    continue

            print(f"The largest dimensions are: {rows} x {cols}")
            fraction_grid = np.ones((rows, cols, len(mask_files_update))) * -9999.

            band_names = []
            for _i, i in enumerate(mask_files_update):
                overpass = i.split('_')[-2]
                band_names.append(overpass)
                fractional_cover = glob(os.path.join(self.output_directory, 'emc2', f'*_{overpass}*_fractional_cover'))[0]

                current_array = envi_to_array(fractional_cover)[:, :, _em]
                mask_array = envi_to_array(i)

                current_array[mask_array[:, :, -1] == 1] = -9999.
                r, c = current_array.shape
                fraction_grid[:r, :c, _i] = current_array[:, :]

            metadata = {'lines': fraction_grid.shape[0],
                        'samples': fraction_grid.shape[1],
                        'bands': fraction_grid.shape[2],
                        'interleave': 'BIL',
                        'header offset': 0,
                        'file type': 'ENVI Standard',
                        'data type': envi_typemap[str(fraction_grid.dtype)],
                        'byte order': 0,
                        'map info': meta['map info'],
                        'coordinate system string': meta['coordinate system string'],
                        'band names': band_names,
                        'data ignore value': -9999.}

            output_raster = os.path.join(self.time_series_directory, f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_time_series.hdr')
            save_envi(output_raster, metadata, fraction_grid)
            print(f'Saved... {output_raster}')

    def build_scores(self):

        for _em, em in enumerate(['npv', 'pv', 'soil']):
            fractional_cover = os.path.join(self.time_series_directory,
                                         f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_time_series')

            frac_array = envi_to_array(fractional_cover)
            print(f"The largest dimensions are: {frac_array.shape[0]} x {frac_array.shape[1]}")
            meta = envi.read_envi_header(f'{fractional_cover}.hdr')

            scores_grid = np.ones((frac_array.shape[0], frac_array.shape[1], 4)) * -9999.
            fire_date = datetime(2024, 7, 4)  # Lakefire date

            time_points = meta['band names']
            func = partial(row_parallel_processing, time_points=time_points, perturbation_date=fire_date)
            indices = range(frac_array.shape[0])

            results = p_map(func, [frac_array[_row, :, :] for _row in range(frac_array.shape[0])], indices,
                            **{"desc": f"\t\t calculating perturbation statistics...", "ncols": 150})

            for _row, row in enumerate(results):
                scores_grid[_row, :, 0:3] = row

            scores_grid[:,:, -1] = scores_grid[:,:, 2] / scores_grid[:,:, 0] # sam/z-score ratio

            metadata = {'lines': scores_grid.shape[0],
                        'samples': scores_grid.shape[1],
                        'bands': 4,
                        'interleave': 'BIL',
                        'header offset': 0,
                        'file type': 'ENVI Standard',
                        'data type': envi_typemap[str(scores_grid.dtype)],
                        'byte order': 0,
                        'map info': meta['map info'],
                        'coordinate system string': meta['coordinate system string'],
                        'band names': ['z_score_shift', 'welch-t_test', 'sam', 'sam_to_z_score_ratio'],
                        'data ignore value': -9999.}

            output_raster = os.path.join(self.time_series_directory,
                                         f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_scores.hdr')

            save_envi(output_raster, metadata, scores_grid)
            print(f'Saved... {output_raster}')


def run_build_workflow(base_directory, sensor, aoi):
    ts = time_series(base_directory=base_directory, sensor=sensor, aoi=aoi)
    ts.build_time_series()
    ts.build_scores()