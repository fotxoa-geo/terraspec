import numpy as np
import isofit.core.common as isc
import os
import itertools
from sklearn.decomposition import PCA
import pandas as pd
import time
from p_tqdm import p_map
from functools import partial
from utils.envi import get_meta, save_envi
from utils import asdreader
from utils import sedreader
from glob import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from utils.slpit_download import load_pickle, save_pickle
import struct
import tetracorder.tetracorder as tc
import spectral.io.envi as envi
from emit_utils.file_checks import envi_header
from scipy.interpolate import interp1d

def get_dd_coords(coord):
    dd_mm = float(str(coord).split(".")[0][-2:] + "." + str(coord).split(".")[1])/60
    dd_dd = float(str(coord).split(".")[0][:-2])
    dd = dd_dd + dd_mm
    return dd

# bad wavelength regions
bad_wv_regions = [[0, 440], [1310, 1490], [1770, 2050], [2440, 2880]]


def load_white_ref_correction():
    white_ref_array = np.loadtxt(os.path.join('utils', 'splib07a_Spectralon99WhiteRef_LSPHERE_ASDFRa_AREF.txt'), skiprows=1)

    return white_ref_array


def gps_asd(latitude_ddmm, longitude_ddmm, file):
    try:
        dd_lat = get_dd_coords(latitude_ddmm)
        dd_long = get_dd_coords(longitude_ddmm) * -1  # used to correct longitude

    except:
        gdf = gpd.read_file('gis/Observation.shp')
        gdf['longitude'] = gdf['geometry'].x
        gdf['latitude'] = gdf['geometry'].y
        plot_name = os.path.basename(os.path.dirname(file)).replace('Spectral', 'SPEC')
        df = gdf.drop(columns='geometry')

        long = df.loc[(df['Name'] == plot_name), 'longitude'].iloc[0]
        lat = df.loc[(df['Name'] == plot_name), 'latitude'].iloc[0]

        dd_lat = lat
        dd_long = long

    return dd_lat, dd_long

class spectra:
    "spectra class allows for different calls for instrument and asd wavelengths"
    def __init__(self):
        print("")

    @classmethod
    def load_wavelengths(cls, sensor: str):
        wavelength_file = os.path.join('utils', 'wavelengths', sensor + '_wavelengths.txt')
        wl = np.loadtxt(wavelength_file, usecols=1)
        fwhm = np.loadtxt(wavelength_file, usecols=2)
        if np.all(wl < 100):
            wl *= 1000
            fwhm *= 1000
        return wl, fwhm

    @classmethod
    def get_good_bands_mask(cls, wavelengths, wavelength_pairs:None):
        wavelengths = np.array(wavelengths)
        if wavelength_pairs is None:
            wavelength_pairs = bad_wv_regions
        good_bands = np.ones(len(wavelengths)).astype(bool)

        for wvp in wavelength_pairs:
            wvl_diff = wavelengths - wvp[0]
            wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
            lower_index = np.nanargmin(wvl_diff)

            wvl_diff = wvp[1] - wavelengths
            wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
            upper_index = np.nanargmin(wvl_diff)
            good_bands[lower_index:upper_index + 1] = False
        return good_bands

    @classmethod
    def convolve(cls, df_row, wvl, fwhm, asd_wvl, spectra_starting_col):
        # Convolve spectra
        refl_convolve = isc.resample_spectrum(x=df_row[1].values[spectra_starting_col:], wl=asd_wvl, wl2=wvl, fwhm2=fwhm,
                                              fill=False)
        return refl_convolve

    @classmethod
    def convolve_asdfile(cls, asd_file, wvl, fwhm):
        # Load asd data

        # get file type
        file_type = os.path.splitext(asd_file)[1]

        if file_type == '.asd':
            data = asdreader.reader(asd_file)
            ins_wl = data.wavelengths
            refl = np.round(data.reflectance, 4)

        elif file_type == '.sed':
            data = sedreader.reader(asd_file)
            ins_wl = data.wavelengths
            refl = np.round(data.reflectance, 4)

        # Convolve spectra
        refl_convolve = isc.resample_spectrum(x=refl, wl=ins_wl, wl2=wvl, fwhm2=fwhm, fill=False)

        return refl_convolve

    @classmethod
    def load_asd_wavelenghts(cls):
        wavelengths_asd = np.linspace(350, 2500, 2151).tolist()

        return wavelengths_asd

    @classmethod
    def load_global_library(cls, output_directory):
        df = pd.read_csv(os.path.join(output_directory, 'convolved', 'geofilter_convolved.csv'))
        return df

    @classmethod
    def latin_hypercubes(cls, points, get_quadrants_index=False):
        ndims = points.shape[1]
        existing_quadrants = list(itertools.product([-1, 1], repeat=ndims))
        quadrants_dict = dict(zip(existing_quadrants, range(len(existing_quadrants))))
        sign_points = np.sign(points)
        quadrants_idx = np.apply_along_axis(lambda x: quadrants_dict[(tuple(x.astype(int)))], 1, sign_points)

        if get_quadrants_index:
            return quadrants_idx
        points_split_into_quadrants = []

        for i in set(quadrants_idx):
            points_split_into_quadrants.append(points[quadrants_idx == i])

        return points_split_into_quadrants

    @classmethod
    def pca_analysis(cls, df, spectra_starting_col:int):

        # target values
        df_select = df.loc[(df['level_1'] == 'soil')].copy()
        metadata = df_select.iloc[:, :spectra_starting_col].reset_index(drop=True)

        # Separating out the features
        x = df_select.iloc[:, spectra_starting_col:].values

        # # PCA analysis for chosen em
        pca = PCA(n_components=x.shape[1])
        df_pca = pd.DataFrame(pca.fit_transform(x))
        df_pca = pd.concat([metadata, df_pca], axis=1)

        return df_pca

    @classmethod
    def increment_synthetic_reflectance(cls, data, em, em_fraction, seed, wvls, spectra_start):
        np.random.seed(seed)

        # calculate the fractions
        if em == 'soil':
            soil_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            npv_frac = np.random.uniform(0, remaining_fraction)  # Generate a random number between 0 and the target_sum
            pv_frac = remaining_fraction - npv_frac  # Calculate the second number to ensure the sum matches the target_sum

        if em == 'npv':
            npv_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            pv_frac = np.random.uniform(0, remaining_fraction)  # Generate a random number between 0 and the target_sum
            soil_frac = remaining_fraction - pv_frac  # Calculate the second number to ensure the sum matches the target_sum

        if em == 'pv':
            pv_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            soil_frac = np.random.uniform(0, remaining_fraction)  # Generate a random number between 0 and the target_sum
            npv_frac = remaining_fraction - soil_frac  # Calculate the second number to ensure the sum matches the target_sum

        fractions = [npv_frac, pv_frac, soil_frac]

        # crate grid to store reflectance
        col_spectra = np.zeros((data.shape[0], len(wvls)))

        # create grid to store index
        col_index = np.zeros((data.shape[0], 3)).astype(int)

        for _row, row in enumerate(data):
            col_index[_row, :] = list(map(int, [row[0][0], row[1][0], row[2][0]]))
            col_spectra[_row, :] = (row[0][spectra_start:].astype(dtype=float) * npv_frac) + \
                                      (row[1][spectra_start:].astype(dtype=float) * pv_frac) + \
                                      (row[2][spectra_start:].astype(dtype=float) * soil_frac)

        return fractions, col_index, col_spectra


    @classmethod
    def synthetic_reflectance(cls, data):
        row_spectra = []
        row_fractions = []
        row_index = [data[2][0, 0], data[2][1, 0], data[2][2, 0]]

        for seed in data[1]:
            np.random.seed(seed)
            fractions = np.random.dirichlet(np.ones(3))

            spectra = np.array(data[0][0]).astype(dtype=float) * fractions[0] + \
                      np.array(data[0][1]).astype(dtype=float) * fractions[1] + \
                      np.array(data[0][2]).astype(dtype=float) * fractions[2]
            row_spectra.append(spectra)
            row_fractions.append(fractions)

        return row_spectra, row_fractions, row_index

    @classmethod
    def create_spectral_bundles(cls, df, level, spectral_bundles):
        ts = time.time()
        # define seed for random sampling of spectral bundles
        np.random.seed(13)

        df = df
        df = df.reset_index(drop=True)
        df.insert(0, 'index', df.index)
        class_names = sorted(list(df[level].unique()))
        class_lists = []

        for em in class_names:
            df_select = df.loc[df[level] == em].copy()
            df_select = df_select.values.tolist()
            class_lists.append(df_select)

        output_pickle = 'spectraL_bundles'
        if os.path.isfile(os.path.join('objects', output_pickle + '.pickle')):
            all_combinations = load_pickle(output_pickle)
        else:
            all_combinations = list(itertools.product(*class_lists))
            save_pickle(all_combinations, output_pickle)

        if len(all_combinations) < spectral_bundles:
            index = np.random.choice(len(all_combinations), replace=False, size=len(all_combinations))
        else:
            index = np.random.choice(len(all_combinations), replace=False, size=spectral_bundles)

        picked_spectra = [all_combinations[i] for i in index]

        # spectra array - combinations x # of classes (each col is an em) x wavelengths
        spec_array = np.array(picked_spectra)

        return spec_array

    @classmethod
    def generate_em_fractions(cls, em, em_fraction, seed):
        np.random.seed(seed)
        # calculate the fractions
        if em == 'soil':
            soil_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            npv_frac = np.random.uniform(0, remaining_fraction)  # Generate a random number between 0 and the target_sum
            pv_frac = remaining_fraction - npv_frac  # Calculate the second number to ensure the sum matches the target_sum

        if em == 'npv':
            npv_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            pv_frac = np.random.uniform(0, remaining_fraction)  # Generate a random number between 0 and the target_sum
            soil_frac = remaining_fraction - pv_frac  # Calculate the second number to ensure the sum matches the target_sum

        if em == 'pv':
            pv_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            soil_frac = np.random.uniform(0,
                                          remaining_fraction)  # Generate a random number between 0 and the target_sum
            npv_frac = remaining_fraction - soil_frac  # Calculate the second number to ensure the sum matches the target_sum

        return npv_frac, pv_frac, soil_frac

    @classmethod
    def row_reflectance(cls, col_size, columns, wavelengths, spectra_start, em, spectral_bundle, row_index):
        mixed_spectra = np.zeros((1, columns, len(wavelengths)))
        index = np.zeros((1, columns, 3))
        fractions = np.zeros((1, columns, 3))

        for _col, col in enumerate(range(0, columns)):
            increment_frac = np.round(col * col_size, 2)
            npv_frac, pv_frac, soil_frac = spectra.generate_em_fractions(em=em, em_fraction=increment_frac, seed= row_index + _col)

            mixed_spectra[0, _col, :] = (spectral_bundle[0][spectra_start:].astype(dtype=float) * npv_frac) + \
                                        (spectral_bundle[1][spectra_start:].astype(dtype=float) * pv_frac) + \
                                        (spectral_bundle[2][spectra_start:].astype(dtype=float) * soil_frac)
            fractions[0, _col, :] = [npv_frac, pv_frac, soil_frac]
            index[0, _col, :] = list(map(int, [spectral_bundle[0][0], spectral_bundle[1][0], spectral_bundle[2][0]]))

        return mixed_spectra, fractions, index

    @classmethod
    def increment_reflectance(cls, class_names: list, simulation_table: str, level: str, spectral_bundles:int,
                              increment_size:float, output_directory: str, wvls, name: str, spectra_starting_col:int,
                              endmember:str, simulation_library_array):

        cols = int(1 / increment_size) + 1
        fraction_grid = np.zeros((spectral_bundles, cols, len(class_names)))
        spectra_grid = np.zeros((spectral_bundles, cols, len(wvls)))
        index_grid = np.zeros((spectral_bundles, cols, len(class_names)))
        em_grid = np.zeros((spectral_bundles, cols, len(wvls)))

        spec_array = spectra.create_spectral_bundles(df=simulation_table, level=level, spectral_bundles=spectral_bundles)

        results = p_map(partial(spectra.row_reflectance, increment_size, cols, wvls,spectra_starting_col, endmember),
                        [bundle for bundle in spec_array], [_index for _index,index in enumerate(spectra_grid)],
                        **{"desc": "\t\t processing reflectance...", "ncols": 150})
        em_indices = {'npv': 0, 'pv': 1, 'soil': 2}

        # populate the results
        for _row, row in enumerate(results):
            spectra_grid[_row, :, :] = row[0]
            fraction_grid[_row, :, :] = row[1]
            index_grid[_row, :, :] = row[2]
            picked_em = int(row[2][0][0][em_indices[endmember]])
            #em_spectra = simulation_library_array[picked_em, :, :] * row[1][0][:,[em_indices[endmember]]]
            em_grid[_row, :, :] = simulation_library_array[picked_em, :, :] * row[1][0][:,[em_indices[endmember]]]

        # save the datasets
        refl_meta = get_meta(lines=spectral_bundles, samples=cols, bands=wvls, wvls=True)
        index_meta = get_meta(lines=spectral_bundles, samples=cols, bands=class_names, wvls=False)
        fraction_meta = get_meta(lines=spectral_bundles, samples=cols, bands=class_names, wvls=False)
        em_meta = get_meta(lines=spectral_bundles, samples=cols, bands=wvls, wvls=True)

        # save index, spectra, fraction grid
        output_files = [os.path.join(output_directory, f'{name}_index.hdr'),
                        os.path.join(output_directory, f'{name}_spectra.hdr'),
                        os.path.join(output_directory, f'{name}_fractions.hdr'),
                        os.path.join(output_directory, f'{name}_em_spectra.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta, em_meta]
        grids = [index_grid, spectra_grid, fraction_grid, em_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})
        del index_grid, spectra_grid, fraction_grid


    @classmethod
    def cont_removal(cls, wavelengths, reflectance, feature):
        left_inds = np.where(np.logical_and(wavelengths >= feature[0], wavelengths <= feature[1]))[0]
        left_x = wavelengths[int(left_inds.mean())]
        left_y = reflectance[left_inds].mean()

        right_inds = np.where(np.logical_and(wavelengths >= feature[2], wavelengths <= feature[3]))[0]
        right_x = wavelengths[int(right_inds.mean())]
        right_y = reflectance[right_inds].mean()

        feature_inds = np.logical_and(wavelengths >= feature[0], wavelengths <= feature[3])

        continuum = interp1d([left_x, right_x], [left_y, right_y],
                             bounds_error=False, fill_value='extrapolate')(wavelengths)
        depths = reflectance[feature_inds] / continuum[feature_inds]
        return depths, wavelengths[feature_inds]


    @classmethod
    def nearest_index_to_wavelength(cls, wavelengths, target_wavelength):
        wvl_nearest_index = (np.abs(wavelengths - target_wavelength)).argmin()

        return wvl_nearest_index

    @classmethod
    def mineral_group_retrival(cls, mineral_index, spectra_observed):
        decoded_expert = tc.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                          log_file=None, log_level='INFO')

        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                      'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

        spectral_reference_library_files = SPECTRAL_REFERENCE_LIBRARY

        group_wvl_center = {'group.2um': 2.24, 'group.1um': 0.79}

        # array to be returned with following positions: group number, rb, rc, rbo, rco
        return_array = np.ones((5)) * -9999.

        # mineral matrix
        if mineral_index != 0:
            df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
            record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral_index), 'Record'].iloc[0]
            filename = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Filename'].iloc[0]
            group = filename.split('.depth.gz')[0].replace('/', '\\').split(os.sep)[0]
            group_num = float(group.split('.')[1][0])

            # this will loop through both libraries
            for key, item in spectral_reference_library_files.items():
                library = envi.open(envi_header(item), item)
                library_reflectance = library.spectra.copy()
                library_records = [int(q) for q in library.metadata['record']]

                if record not in library_records:
                    continue

                hdr = envi.read_envi_header(envi_header(item))
                wavelengths = np.array([float(q) for q in hdr['wavelength']])

                for cont_feat in decoded_expert[filename.split('.depth.gz')[0].replace('/', '\\')]['features']:
                    # get rc and rb from tetracorder library
                    refl_cont, wl = spectra.cont_removal(wavelengths, library_reflectance[library_records.index(record), :], cont_feat['continuum'])
                    rc_nearest_index = spectra.nearest_index_to_wavelength(wavelengths=wl, target_wavelength=group_wvl_center[group])
                    rc = refl_cont[rc_nearest_index]
                    rb_nearest_index = spectra.nearest_index_to_wavelength(wavelengths=wavelengths, target_wavelength=group_wvl_center[group])
                    rb = library_reflectance[library_records.index(record), :][rb_nearest_index]

                    # get rco and rbo - observed spectra
                    refl_cont_o, wl_o = spectra.cont_removal(wavelengths, spectra_observed, cont_feat['continuum'])
                    rco_nearest_index = spectra.nearest_index_to_wavelength(wavelengths=wl_o, target_wavelength=group_wvl_center[group])
                    rco = refl_cont_o[rco_nearest_index]
                    rbo_nearest_index = spectra.nearest_index_to_wavelength(wavelengths=wavelengths, target_wavelength=group_wvl_center[group])
                    rbo = spectra_observed[rbo_nearest_index]

                    return_array[:] = [group_num, rb, rc, rbo, rco]

        else:
            pass

        return return_array

    @classmethod
    def mineral_group_row(cls, mineral_index_row, spectra_row):

        row_return_array = np.ones((mineral_index_row.shape[0], 10)) * -9999.

        for _col, col in enumerate(mineral_index_row):
            for _band, band in enumerate(col):
                if _band in [0, 2]:
                    pass
                else:
                    col_spectra = spectra_row[_col,:]
                    mineral_retrival = spectra.mineral_group_retrival(mineral_index=band, spectra_observed=col_spectra)

                    group_num = mineral_retrival[0]

                    if group_num == 1:
                        row_return_array[_col, :5] = mineral_retrival
                    else:
                        row_return_array[_col, 5:] = mineral_retrival

        return row_return_array

    @classmethod
    def mineral_components(cls, index_array, spectra_array, output_file):

        # cont grid - corresponds to Rc and Rc-observed - for both group 1 and group 2
        output_grid = np.zeros((index_array.shape[0], index_array.shape[1], 10))

        results = p_map(spectra.mineral_group_row, [index_array[_row, :, :] for _row, row in enumerate(index_array)],
                        [spectra_array[_row, :, :] for _row, row in enumerate(spectra_array)],
                        **{"desc": "\t\t processing continuum reflectance...", "ncols": 150})

        for _row, row in enumerate(results):
            output_grid[_row, :, :] = row

        # save spectra
        meta = get_meta(lines=index_array.shape[0], samples= index_array.shape[1], bands=[i for i in range(10)], wvls=False)
        meta['data ignore value'] = -9999
        save_envi(output_file=output_file, meta=meta, grid=output_grid)

    @classmethod
    def generate_reflectance(cls, class_names: list, simulation_table: str, level: str, spectral_bundles:int, cols:int,
                             output_directory: str, wvls, name: str, spectra_starting_col:int):

        ts = time.time()
        # define seed for random sampling of spectral bundles
        np.random.seed(13)

        df = simulation_table
        df = df.reset_index(drop=True)
        df.insert(0, 'index', df.index)
        class_lists = []

        for em in class_names:
            df_select = df.loc[df[level] == em].copy()
            df_select = df_select.values.tolist()
            class_lists.append(df_select)

        all_combinations = list(itertools.product(*class_lists))

        if len(all_combinations) < spectral_bundles:
            index = np.random.choice(len(all_combinations), replace=False, size=len(all_combinations))
        else:
            index = np.random.choice(len(all_combinations), replace=False, size=spectral_bundles)

        spectra_all = [all_combinations[i] for i in index]
        fraction_grid = np.zeros((len(index), cols, len(class_names)))
        spectra_grid = np.zeros((len(index), cols, len(wvls)))
        index_grid = np.zeros((len(index), cols, len(class_names)))

        # spectra array - combinations x # of classes (each col is an em) x wavelengths
        spec_array = np.array(spectra_all)

        # Process row in parallel
        seeds = list(range(0, len(index) * cols))
        seeds_array = np.asarray(seeds)
        seeds_array = seeds_array.reshape(len(index), cols)

        # parallel spectra processes ; # we are using +1 since we added an index identifier
        process_spectra = p_map(spectra.synthetic_reflectance,
                                [(spec_array[_row, :, spectra_starting_col + 1:], seeds_array[_row, :], spec_array[_row, :, :4]) for _row, row
                                 in enumerate(spectra_grid)], **{"desc": "\t\t generating fractions...", "ncols": 150})

        # Populate results row by row
        for _row, row in enumerate(process_spectra):
            for _col, (refl, frac) in enumerate(zip(row[0], row[1])):
                spectra_grid[_row, _col, :] = refl
                fraction_grid[_row, _col, :] = frac
                index_grid[_row, _col, :] = np.array(row[2])

        # save the datasets
        refl_meta = get_meta(lines=len(index), samples=cols, bands=wvls, wvls=True)
        index_meta = get_meta(lines=len(index), samples=cols, bands=class_names, wvls=False)
        fraction_meta = get_meta(lines=len(index), samples=cols, bands=class_names, wvls=False)

        # save index, spectra, fraction grid
        output_files = [os.path.join(output_directory, name + '_index.hdr'),
                        os.path.join(output_directory, name + '_spectra.hdr'),
                        os.path.join(output_directory, name + '_fractions.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta]
        grids = [index_grid, spectra_grid, fraction_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})
        del index_grid, spectra_grid, fraction_grid

    @classmethod
    def simulate_reflectance(cls, df_sim, df_unmix, dimensions, sim_libraries_output, mode, level, spectral_bundles, cols,
                             output_directory, wvls, spectra_starting_col:int):
        """
        @param df_sim: Simulation csv format
        @param df_unmix: Unmixing library
        @param dimensions: dimensions used in convex hull or PCA
        @param sim_libraries_output: output to save csv for simulation
        @param mode: latin hypercube, convex hull, or geographic
        @param level: column having spectral em classification
        @param spectral_bundles: spectral bundles with
        @param cols: number of columns to use in output
        @param output_directory: directory to save outputs
        @param wvls: instrument wavelengths
        @param spectra_starting_col: spectral starting column of dataframe
        @return: none
        """
    
        # check for duplicates from dataframes using actual wavelengths
        df_sim_array = df_sim.iloc[:, spectra_starting_col:].to_numpy()
        df_unmix_array = df_unmix.iloc[:, spectra_starting_col:].to_numpy()

        # check for duplicates again
        dup_check = list((df_unmix_array[None, :] == df_sim_array[:, None]).all(-1).any(0))
        if dup_check.count(True) > 0:
            raise Exception(
                "The simulation found duplicates in both the simulation and unmixing library at: " + str(dimensions))

        df_sim.to_csv(
            os.path.join(sim_libraries_output, mode + '__n_dims_' + str(dimensions) + '_simulation_library.csv'),
            index=False)

        # # create the reflectance file
        spectra.generate_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim, level=level,
                         spectral_bundles=spectral_bundles, cols=cols, output_directory=output_directory,
                         wvls=wvls, name=mode + '__n_dims_' + str(dimensions), spectra_starting_col=spectra_starting_col)

    @classmethod
    def get_reflectance_endmember(cls, df_row, plot_directory:str, team_name_key:str):
        file_num = df_row[0]
        em_classification = df_row[1]
        species = df_row[2]
        plot_name = os.path.basename(plot_directory)
        file_name = os.path.join(plot_directory, team_name_key + "_" + f"{file_num:05d}.asd")
        asd = asdreader.reader(file_name)
        asd_refl = asd.reflectance
        asd_gps = asd.get_gps()
        latitude_ddmm, longitude_ddmm, elevation, utc_time = asd_gps[0], asd_gps[1], asd_gps[2], asd_gps[3]
        utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])

        dd_lat, dd_long = gps_asd(latitude_ddmm=latitude_ddmm, longitude_ddmm=longitude_ddmm, file=file_name)

        return [plot_name, file_name, em_classification, species, dd_long, dd_lat, elevation, utc_time] + list(asd_refl)

    @classmethod
    def get_reflectance_transect(cls, file, plot_directory:str, team_name_key:str):
        white_ref_correction = load_white_ref_correction()
        plot_name = os.path.basename(plot_directory)

        # get file type
        file_type = os.path.splitext(file)[1]

        if file_type == '.asd':
            asd = asdreader.reader(file)
            refl = asd.reflectance * white_ref_correction
            asd_gps = asd.get_gps()
            latitude_ddmm, longitude_ddmm, elevation, utc_time = asd_gps[0], asd_gps[1], asd_gps[2], asd_gps[3]

            if int(utc_time[0]) + int(utc_time[1]) + int(utc_time[2]) == 0:
                file_time = asd.get_save_time()
                utc_time = str(file_time[2]) + ":" + str(file_time[1]) + ":" + str(file_time[0])
            else:
                utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])

            file_num = int(os.path.basename(file).split(".")[0].split("_")[-1])

            dd_lat, dd_long = gps_asd(latitude_ddmm=latitude_ddmm, longitude_ddmm=longitude_ddmm, file=file)

        elif file_type == '.sed':
            sed = sedreader.reader(file)
            refl = sed.reflectance * white_ref_correction
            dd_long, dd_lat, utc_time,elevation = sed.gps
            file_num = int(os.path.basename(file).split(".")[0].split("_")[-1])

        return [plot_name, file, file_num, dd_long, dd_lat, elevation, utc_time] + list(refl)

    @classmethod
    def get_asd_binary(cls, data):
        # unpack the binary asd file
        asdformat = '<3s 157s 18s b b b b l b l f f b b b b b H 128s 56s L hh H H f f f f h b 4b H H H b L HHHH f f f 5b'

        file_version, comment, save_time, parent_version, format_version, itime, dc_corrected, dc_time, \
            data_type, ref_time, ch1_wave, wave1_step, data_format, old_dc_count, old_ref_count, old_sample_count, \
            application, channels, app_data, gps_data, intergration_time, fo, dcc, calibration, instrument_num, \
            ymin, ymax, xmin, xmax, ip_numbits, xmode, flags1, flags2, flags3, flags4, dc_count, ref_count, \
            sample_count, instrument, cal_bulb_id, swir1_gain, swir2_gain, swir1_offset, swir2_offset, \
            splice1_wavelength, splice2_wavelength, smart_detector_type, \
            spare1, spare2, spare3, spare4, spare5 = struct.unpack_from(asdformat, data)

        return save_time, gps_data, file_version, format_version


    @classmethod
    def get_shift_transect(cls, file, season, plot_directory:str):
        line_num = os.path.split(os.path.split(os.path.split(file)[0])[0])[1]
        plot_name = os.path.split(os.path.split(os.path.split(os.path.split(file)[0])[0])[0])[1]

        file_extension = os.path.splitext(file)[1]

        if file_extension == '.asd':
            asd = asdreader.reader(file)
            try:
                asd_refl = asd.reflectance
                asd_gps = asd.get_gps()
                latitude_ddmm, longitude_ddmm, elevation, utc_time = asd_gps[0], asd_gps[1], asd_gps[2], asd_gps[3]

                if int(utc_time[0]) + int(utc_time[1]) + int(utc_time[2]) == 0:
                    file_time = asd.get_save_time()
                    utc_time = str(file_time[2]) + ":" + str(file_time[1]) + ":" + str(file_time[0])

                else:
                    utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])

                file_num = int(os.path.basename(file).split(".")[0].split("_")[-1])

                try:
                    dd_lat = get_dd_coords(latitude_ddmm)
                    dd_long = get_dd_coords(longitude_ddmm) * -1  # used to correct longitude

                except:
                    # get long lat from shift plots csv
                    df_coords = pd.read_csv(os.path.join('gis', 'shift_plot_coordinates.csv'))
                    long = df_coords.loc[(df_coords['Plot Name'] == plot_name) & (df_coords['Season'] == season.upper()), 'longitude'].iloc[0]
                    lat = df_coords.loc[(df_coords['Plot Name'] == plot_name) & (df_coords['Season'] == season.upper()), 'latitude'].iloc[0]

                    dd_lat = lat
                    dd_long = long

                return [f'{plot_name}-{season}', file, line_num, file_num, dd_long, dd_lat, elevation, utc_time] + list(asd_refl)

            except Exception as e:
                print("An error occurred:", e)


        else:
            # read data on the old ASD files
            data = open(file, "rb").read()
            file_num = int(os.path.basename(file).split(".")[1])

            meta_data_asd = spectra.get_asd_binary(data)

            # get gps data
            gps_binary = struct.unpack('=5d 2b 2b b b l 2b 2b b b', meta_data_asd[1])

            latitude_ddmm, longitude_ddmm, elevation, utc_time = gps_binary[2], gps_binary[3], gps_binary[4], (
                gps_binary[10], gps_binary[9], gps_binary[8])
            utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])
            dd_lat = get_dd_coords(latitude_ddmm)
            dd_long = get_dd_coords(longitude_ddmm) * -1  # used to correct longitude

            # asd reflectance
            spectrum = data[484:]
            asd_refl = np.array(list(struct.iter_unpack('<f', spectrum)), dtype=float).flatten()
            asd_refl[:651] *= asd_refl[651] / asd_refl[650]

            return [plot_name + '-' + season, file, line_num, file_num, dd_long, dd_lat, elevation, utc_time] + list(asd_refl)

    @classmethod
    def first_derivative(cls, df_row, spectral_starting_col, wvls):
        spectral_sample = np.array(df_row[1].values[spectral_starting_col:])

        first_derivative = []
        for _i, i in enumerate(wvls):
            # last position is a duplicate of previous ?
            if _i == len(wvls) - 1:
                first_derivative.append(first_derivative[-1])
            else:
                ds = spectral_sample[_i + 1] - spectral_sample[_i]
                dx = wvls[_i + 1] - wvls[_i]
                first_derivative.append(ds/dx)

        return first_derivative

    @classmethod
    def get_all_ems(cls,output_directory: str, instrument: str):
        #spectral_endmembers = glob(os.path.join(output_directory, 'spectral_endmembers', '*' + instrument + ".csv"))
        emit_transect_endmembers = glob(os.path.join(output_directory, 'spectral_transects', 'endmembers-raw', '*' + instrument + ".csv"))
        emit_transect_endmembers = [item for item in emit_transect_endmembers if "Thermal" not in item]
        all_ems = emit_transect_endmembers

        return all_ems

    @classmethod
    def df_to_shapefile(cls,df, base_directory: str, out_name):
        df_shp = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        df_shp.to_file(os.path.join(base_directory, "gis", out_name + '.shp'), driver='ESRI Shapefile')

    @classmethod
    def save_df_em(cls, df, output, instrument):
        df = df.sort_values('level_1')
        df.to_csv(output.replace(" ", "") + '-' + instrument + '.csv', index=False)

    @classmethod
    def df_to_envi(cls, df, spectral_starting_column:int, wvls, output_raster):

        df_array = df.iloc[:, spectral_starting_column:].to_numpy()
        spectra_grid = np.zeros((df_array.shape[0], 1, len(wvls)))

        # fill spectral data
        for _row, row in enumerate(df_array):
            spectra_grid[_row, 0, :] = row

        # save the spectra
        print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls,
                                wvls=True)
        save_envi(output_raster, meta_spectra, spectra_grid)

    @classmethod
    def plot_asd_file(cls, asd_file, out_directory):
        # Load asd data
        data = asdreader.reader(asd_file)
        asd_wl = data.wavelengths

        try:
            outfname = os.path.join(out_directory, os.path.basename(asd_file) + '.png')
            if os.path.isfile(outfname):
                pass

            else:
                asd_refl = data.reflectance

                plt.plot(asd_wl, asd_refl, label=os.path.basename(asd_file))
                plt.legend()
                plt.ylabel("Reflectance (%)")
                plt.xlabel("Wavelenghts (nm)")
                plt.ylim([0, 1.1])

                plt.savefig(outfname, bbox_inches='tight')
                plt.clf()
                plt.close()

        except:
            print(asd_file, out_directory)

    @classmethod
    def plot_sed_file(cls, sed_file, out_directory):
        # load sed data
        data = sedreader.reader(sed_file)
        sed_wvl = data.wavelengths


        try:
            outfname = os.path.join(out_directory, os.path.basename(sed_file) + '.png')
            if os.path.isfile(outfname):
                pass

            else:
                sed_refl = data.reflectance

                plt.plot(sed_wvl, sed_refl, label=os.path.basename(sed_file))
                plt.legend()
                plt.ylabel("Reflectance (%)")
                plt.xlabel("Wavelenghts (nm)")
                plt.ylim([0, 1.1])

                plt.savefig(outfname, bbox_inches='tight')
                plt.clf()
                plt.close()

        except:
            print(sed_file, out_directory)
    
    @classmethod
    def vector_normalize_spectrum(cls, array):
        norm = np.linalg.norm(array)
        
        if norm == 0:
            return array
        
        return array / norm



