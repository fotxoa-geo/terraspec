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
from osgeo import gdal
from utils.spectra_utils import spectra
import isofit.core.common as isc

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

def resample_grid(input_file, output_base, warp_kwargs):
    # Open the 30m source
    options = gdal.WarpOptions(**warp_kwargs)

    src_ds = gdal.Open(input_file)
    if src_ds is None:
        print(f"Error: Could not open {input_file}")
        return

    output_file = os.path.join(output_base, os.path.basename(input_file))
    # Execute the operation
    gdal.Warp(output_file, src_ds, options=options)

    # Close resources
    src_ds = None
    print(f"Success! Saved to: {output_file}")


def resample_spectrally(input_file, output_base):
    try:
        spectral_array = envi_to_array(input_file)
        meta = envi.read_envi_header(f'{input_file}.hdr')

        enmap_wvls, enmap_fwhm = spectra.load_wavelengths(sensor='EnMAP')
        emit_wvls, emit_fwhm = spectra.load_wavelengths(sensor='emit')

        def resample_pixel_spectrum(pixel_vector):
            return isc.resample_spectrum(
                x=pixel_vector,
                wl=emit_wvls,
                wl2=enmap_wvls,
                fwhm2=enmap_fwhm,
                fill=False
            )
        spectral_rs_grid = np.apply_along_axis(resample_pixel_spectrum, axis=2, arr=spectral_array)

        metadata = {'lines': spectral_rs_grid.shape[0],
                    'samples': spectral_rs_grid.shape[1],
                    'bands': spectral_rs_grid.shape[2],
                    'wavelength': enmap_wvls,
                    'interleave': 'BIL',
                    'header offset': 0,
                    'file type': 'ENVI Standard',
                    'data type': envi_typemap[str(spectral_rs_grid.dtype)],
                    'byte order': 0,
                    'map info': meta['map info'],
                    'coordinate system string': meta['coordinate system string'],
                    'band names': enmap_wvls,
                    'data ignore value': -9999.,
                    'wavelength units': 'nm'}

        output_raster = os.path.join(output_base, f'{os.path.basename(input_file)}.hdr')
        save_envi(output_raster, metadata, spectral_rs_grid)
    except Exception as e:
        # Catch any failure (IOError, ValueError, AttributeError, etc.)
        # inside this process, print the log, and allow p_map to continue.
        print(f"\nCRITICAL ERROR processing file [{os.path.basename(input_file)}]: {str(e)}")

    finally:
        # Explicit clean up step to release file handles and avoid memory leaks
        spectral_array = None
        spectral_rs_grid = None


def filter_mask_by_derivative(series, initial_mask):
    """
    Refines a mask to keep only the points where the
    vegetation fraction is actively increasing to the next point.
    """
    # Get the indices where the mask is currently True
    indices = np.where(initial_mask)[0]

    if len(indices) < 2:
        return np.zeros_like(initial_mask, dtype=bool)

    valid_indices = []

    # Verify every sequential point pair in the mask
    for i in range(len(indices) - 1):
        idx_current = indices[i]
        idx_next = indices[i + 1]

        # Calculate the derivative (difference) between consecutive mask points
        derivative = series[idx_next] - series[idx_current]

        # If the change is positive, both points are part of an increasing phase
        if derivative > 0:
            valid_indices.append(idx_current)
            valid_indices.append(idx_next)

    # Remove duplicates and create a new refined boolean mask
    valid_indices = np.unique(valid_indices)
    refined_mask = np.zeros_like(initial_mask, dtype=bool)
    refined_mask[valid_indices] = True

    return refined_mask

def z_shift(time_series_values, time_points, perturbation_date):
    time_series_values[time_series_values == -9999. ] = np.nan
    non_nan_mask = ~np.isnan(time_series_values)
    filtered_series = time_series_values[non_nan_mask]

    dates = [datetime.strptime(d.split("_")[0], '%Y%m%dt%H%M%S') for d in time_points]
    filtered_dates = np.array(dates)[non_nan_mask]

    pre_fire_mask = filtered_dates < perturbation_date
    post_fire_mask = filtered_dates >= perturbation_date

    refined_pre_mask = filter_mask_by_derivative(filtered_series, pre_fire_mask)
    refined_post_mask = filter_mask_by_derivative(filtered_series, post_fire_mask)

    pre_perturbation_signal = filtered_series[refined_pre_mask]
    post_perturbation_signal = filtered_series[refined_post_mask]

    z_score_pre = (pre_perturbation_signal - np.mean(pre_perturbation_signal))/np.std(pre_perturbation_signal)
    z_score_post = (post_perturbation_signal - np.mean(post_perturbation_signal))/np.std(post_perturbation_signal)

    z_score_trajectory = (post_perturbation_signal - np.mean(pre_perturbation_signal))/np.std(pre_perturbation_signal)
    z_post_average = np.mean(z_score_trajectory)

    return z_post_average

def welch_test(time_series_values, time_points, perturbation_date):
    time_series_values[time_series_values == -9999.] = np.nan
    non_nan_mask = ~np.isnan(time_series_values)
    filtered_series = time_series_values[non_nan_mask]

    dates = [datetime.strptime(d.split("_")[0], '%Y%m%dt%H%M%S') for d in time_points]
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

    dates = [datetime.strptime(d.split("_")[0], '%Y%m%dt%H%M%S') for d in time_points]
    filtered_dates = np.array(dates)[non_nan_mask]

    pre_fire_mask = filtered_dates < perturbation_date
    post_fire_mask = filtered_dates >= perturbation_date

    pre_perturbation_signal = filtered_series[pre_fire_mask]
    post_perturbation_signal = filtered_series[post_fire_mask]

    filtered_timepoints = np.array(time_points)[non_nan_mask]

    pre_fire_timepoints = filtered_timepoints[pre_fire_mask]
    max_value_of_pre_signal = np.argmax(pre_perturbation_signal)
    date_of_max_pre_signal = pre_fire_timepoints[max_value_of_pre_signal]

    sensor_of_max_value = date_of_max_pre_signal.split("_")[1]

    wvls, fwhm = spectra.load_wavelengths(sensor='EnMAP')
    good_enmap_bands = spectra.get_good_bands_mask(wvls, wavelength_pairs=[[350, 400], [900, 1000], [1310, 1490], [1725, 2050],
                                                                     [2450, 2500]])
    wvls[~good_enmap_bands] = np.nan

    if sensor_of_max_value == "emit":
        pre_fire_rfl = envi_to_array(os.path.join('terraspec_output', 'fire', 'output', 'EMIT_to_ENMAP_RFL', f'sedgwick_boundary_approx_RFL_{date_of_max_pre_signal.split("_")[0].upper()}_EXT'))
    else:
        pre_fire_rfl = sorted(list(glob(os.path.join('terraspec_output', 'fire', 'output', 'EnMAP_to_Spatial_EMIT_RFL',
                                                  f'*_{date_of_max_pre_signal.split("_")[0].upper()}Z_*'))))
        pre_fire_rfl = envi_to_array(pre_fire_rfl[0])

    pre_fire_max_rfl = pre_fire_rfl[row, col, :]

    pre_fire_max_rfl[pre_fire_max_rfl == -9999.] = np.nan
    pre_fire_max_rfl[(pre_fire_max_rfl < 0.0) | (pre_fire_max_rfl > 1.0)] = np.nan
    pre_fire_max_rfl[~good_enmap_bands] = np.nan

    post_fire_timepoints = filtered_timepoints[post_fire_mask]
    max_value_of_post_signal = np.argmax(post_perturbation_signal)
    date_of_max_post_signal = post_fire_timepoints[max_value_of_post_signal]

    sensor_of_max_value = date_of_max_post_signal.split("_")[1]

    if sensor_of_max_value == "emit":
        post_fire_rfl = envi_to_array(os.path.join('terraspec_output', 'fire', 'output', 'EMIT_to_ENMAP_RFL',
                                                  f'sedgwick_boundary_approx_RFL_{date_of_max_post_signal.split("_")[0].upper()}_EXT'))
    else:
        post_fire_rfl = sorted(list(glob(os.path.join('terraspec_output', 'fire', 'output', 'EnMAP_to_Spatial_EMIT_RFL',
                                                     f'*_{date_of_max_post_signal.split("_")[0].upper()}Z_*'))))
        post_fire_rfl = envi_to_array(post_fire_rfl[0])

    post_fire_max_rfl = post_fire_rfl[row, col, :]
    post_fire_max_rfl[(post_fire_max_rfl < 0.0) | (post_fire_max_rfl > 1.0)] = np.nan
    post_fire_max_rfl[~good_enmap_bands] = np.nan

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
        self.gis_directory = os.path.join(base_directory, 'gis')
        self.aoi_extraction = os.path.join(base_directory, 'gis', f'{sensor}-data', 'aoi', f'{os.path.basename(aoi.split('.')[0])}', 'EXT')
        self.aoi_extraction_EnMAP = os.path.join(base_directory, 'gis', f'enmap-data', 'aoi',
                                           f'{os.path.basename(aoi.split('.')[0])}', 'emc2')

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



    def build_merged_time_series(self):
        time_series = sorted(glob(os.path.join(self.time_series_directory, '*emit_time_series')))

        enmap_fractions = sorted(glob(os.path.join(self.output_directory, 'EnMAP_to_EMIT_emc2', '*_fractional_cover')))
        enmap_timestamps = sorted(list(set([os.path.basename(item).split('_')[5].lower() for item in enmap_fractions])))

        for _i, i in enumerate(time_series):
            em = os.path.basename(i).split("_")[-4]
            meta = envi.read_envi_header(f'{i}.hdr')

            # Capture the COMPLETE timestamp for EMIT from the header
            emit_timestamps = sorted(list(set([item.strip().lower() for item in meta['band names']])))

            # Calculate common dates by strictly looking at the day string [:8]
            emit_days = set([t[:8] for t in emit_timestamps])
            enmap_days = set([t[:8] for t in enmap_timestamps])
            common_dates = list(emit_days & enmap_days)
            print(f'The common calendar dates between observations are: {common_dates}')

            # Merge full timestamps instead of just the dates to preserve multiple overpasses
            merged_timestamps = sorted(list(set(emit_timestamps + enmap_timestamps)))
            total_dates = len(merged_timestamps)

            current_array = envi_to_array(i)[:, :, :]
            fraction_merged_grid = np.ones((current_array.shape[0], current_array.shape[1], total_dates)) * -9999.

            # 1. Fill EMIT layers based on complete timestamp matching
            for emit_idx, timestamp in enumerate(emit_timestamps):
                merged_idx = merged_timestamps.index(timestamp)
                fraction_merged_grid[:, :, merged_idx] = current_array[:, :, emit_idx]

            # 2. Process EnMAP files
            for enmap_file in enmap_fractions:
                # Extract full timestamp and the specific 8-digit date
                enmap_timestamp = os.path.basename(enmap_file).split('_')[5].lower()
                enmap_date = enmap_timestamp[:8]

                # Rule 1: If this EnMAP calendar date already has an EMIT asset on it,
                # skip loading EnMAP to preserve EMIT priority for that day.
                if enmap_date in emit_days:
                    print(f"Skipping EnMAP file {os.path.basename(enmap_file)}: Overlapping date prioritized for EMIT.")
                    continue

                # Find where this specific EnMAP timestamp fits into the master timeline
                merged_idx = merged_timestamps.index(enmap_timestamp)

                # Open the resampled 60m EnMAP file
                enmap_ds = envi_to_array(enmap_file)[:, :, _i]
                valid_data_mask = (enmap_ds != -9999.)

                # Merge data array
                fraction_merged_grid[:, :, merged_idx] = np.where(
                    valid_data_mask,
                    enmap_ds,
                    fraction_merged_grid[:, :, merged_idx]
                )
                enmap_ds = None  # Close handle

            # --- TAG TIMESTAMPS WITH SENSOR IDENTIFIERS ---
            tagged_merged_dates = []
            for timestamp in merged_timestamps:
                # Check if the specific timestamp came from EMIT
                if timestamp in emit_timestamps:
                    tagged_merged_dates.append(f"{timestamp}_emit")
                else:
                    tagged_merged_dates.append(f"{timestamp[:-1]}_enmap") # drop z to match emit timestamps

            metadata = {'lines': fraction_merged_grid.shape[0],
                        'samples': fraction_merged_grid.shape[1],
                        'bands': fraction_merged_grid.shape[2],
                        'interleave': 'BIL',
                        'header offset': 0,
                        'file type': 'ENVI Standard',
                        'data type': envi_typemap[str(fraction_merged_grid.dtype)],
                        'byte order': 0,
                        'map info': meta['map info'],
                        'coordinate system string': meta['coordinate system string'],
                        'band names': tagged_merged_dates,  # Safely outputs YYYYMMDDtHHMMSS_sensor
                        'data ignore value': -9999.}

            output_raster = os.path.join(self.time_series_directory,
                                         f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_merged_time_series.hdr')
            save_envi(output_raster, metadata, fraction_merged_grid)
            print(f'Saved... {output_raster}')

    def resample_EnMAP_to_EMIT(self):
        fraction_files = sorted(list(glob(os.path.join(self.aoi_extraction_EnMAP, '*_fractional_cover'))))
        emit_fractions_files = sorted(list(glob(os.path.join(self.gis_directory, 'emit-data', 'aoi', os.path.basename(self.aoi).split('.')[0], 'emc2', '*fractional_cover'))))

        template_ds = gdal.Open(f'{emit_fractions_files[0]}')
        geo_transform = template_ds.GetGeoTransform()
        projection = template_ds.GetProjection()
        cols = template_ds.RasterXSize
        rows = template_ds.RasterYSize

        minx = geo_transform[0]
        maxy = geo_transform[3]
        maxx = minx + geo_transform[1] * cols
        miny = maxy + geo_transform[5] * rows
        template_extent = [minx, miny, maxx, maxy]

        x_res = geo_transform[1]
        y_res = abs(geo_transform[5])

        # Bundle all settings into WarpOptions
        warp_settings = {
            "format": 'ENVI',
            "outputBounds": template_extent,  # structured as standard list [minx, miny, maxx, maxy]
            "xRes": x_res,
            "yRes": y_res,
            "dstSRS": projection,  # Projection string/WKT is a standard string
            "resampleAlg": gdal.GRIORA_Average
        }

        create_directory(os.path.join(self.output_directory, 'EnMAP_to_EMIT_emc2'))
        output_base = os.path.join(self.output_directory, 'EnMAP_to_EMIT_emc2')
        func = partial(resample_grid, output_base=output_base, warp_kwargs=warp_settings)

        p_map(func, fraction_files, **{"desc": f"\t\t Resampling EnMAP to EMIT grid...", "ncols": 150})


    def spatial_resample_EnMAP_to_EMIT(self):
        enmap_rfls = sorted(list(glob(os.path.join(self.aoi_extraction, '*_EXT'))))
        emit_rfls = sorted(list(glob(
            os.path.join(self.gis_directory, 'emit-data', 'aoi', os.path.basename(self.aoi).split('.')[0], 'EXT',
                         '*_EXT'))))

        template_ds = gdal.Open(f'{emit_rfls[0]}')
        geo_transform = template_ds.GetGeoTransform()
        projection = template_ds.GetProjection()
        cols = template_ds.RasterXSize
        rows = template_ds.RasterYSize

        minx = geo_transform[0]
        maxy = geo_transform[3]
        maxx = minx + geo_transform[1] * cols
        miny = maxy + geo_transform[5] * rows
        template_extent = [minx, miny, maxx, maxy]

        x_res = geo_transform[1]
        y_res = abs(geo_transform[5])

        # Bundle all settings into WarpOptions
        warp_settings = {
            "format": 'ENVI',
            "outputBounds": template_extent,  # structured as standard list [minx, miny, maxx, maxy]
            "xRes": x_res,
            "yRes": y_res,
            "dstSRS": projection,  # Projection string/WKT is a standard string
            "resampleAlg": gdal.GRIORA_Average
        }

        create_directory(os.path.join(self.output_directory, 'EnMAP_to_Spatial_EMIT_RFL'))
        output_base = os.path.join(self.output_directory, 'EnMAP_to_Spatial_EMIT_RFL')
        func = partial(resample_grid, output_base=output_base, warp_kwargs=warp_settings)

        p_map(func, enmap_rfls, **{"desc": f"\t\t Resampling EnMAP Rfl to EMIT grid...", "ncols": 150})

    def spectral_resample_emit_to_EnMAP(self):

        emit_rfls = sorted(glob(os.path.join(self.aoi_extraction, '*_RFL_*EXT')))
        create_directory(os.path.join(self.output_directory, 'EMIT_to_ENMAP_RFL'))
        output_base = os.path.join(self.output_directory, 'EMIT_to_ENMAP_RFL')
        func = partial(resample_spectrally, output_base=output_base)

        p_map(func, emit_rfls, **{"desc": f"\t\t Resampling EMIT to EnMAP spectrally...", "ncols": 150})

    def build_scores(self):

        for _em, em in enumerate(['npv', 'pv', 'soil']):
            fractional_cover = os.path.join(self.time_series_directory,
                                         f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_merged_time_series')

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
    #ts.resample_EnMAP_to_EMIT()
    ts.build_time_series()
    #ts.build_merged_time_series()
    #ts.spatial_resample_EnMAP_to_EMIT()
    #ts.spectral_resample_emit_to_EnMAP()
    ts.build_scores()