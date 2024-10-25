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
import ast

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

mineral_groupings = mineral_groups = {'Calcite': 1,
                  'Chlorite': 1,
                  'Dolomite': 1,
                  'Goethite': 0,
                  'Gypsum': 1,
                  'Hematite': 0,
                  'Illite+Muscovite': 2,
                  'Kaolinite': 2,
                  'Montmorillonite': 2,
                  'Vermiculite': 2}

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
    def create_spectral_bundles(cls, df, level, spectral_bundles, spectral_bundle_project):
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

        output_pickle = f'spectral_bundles_{spectral_bundle_project}'
        if os.path.isfile(os.path.join('objects', f"output_pickle_{spectral_bundle_project}.pickle")):
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
        vegetation_spectra = np.zeros((1, columns, len(wavelengths)))
        soil_spectra = np.zeros((1, columns, len(wavelengths)))

        for _col, col in enumerate(range(0, columns)):
            increment_frac = np.round(col * col_size, 2)
            npv_frac, pv_frac, soil_frac = spectra.generate_em_fractions(em=em, em_fraction=increment_frac, seed= row_index + _col)

            mixed_spectra[0, _col, :] = (spectral_bundle[0][spectra_start:].astype(dtype=float) * npv_frac) + \
                                        (spectral_bundle[1][spectra_start:].astype(dtype=float) * pv_frac) + \
                                        (spectral_bundle[2][spectra_start:].astype(dtype=float) * soil_frac)

            vegetation_spectra[0, _col, :] = (spectral_bundle[0][spectra_start:].astype(dtype=float) * npv_frac) + \
                                             (spectral_bundle[1][spectra_start:].astype(dtype=float) * pv_frac)

            soil_spectra[0, _col, :] = spectral_bundle[2][spectra_start:].astype(dtype=float) * soil_frac

            fractions[0, _col, :] = [npv_frac, pv_frac, soil_frac]
            index[0, _col, :] = list(map(int, [spectral_bundle[0][0], spectral_bundle[1][0], spectral_bundle[2][0]]))

        return mixed_spectra, fractions, index, vegetation_spectra, soil_spectra

    @classmethod
    def increment_reflectance(cls, class_names: list, simulation_table, level: str, spectral_bundles:int,
                              increment_size:float, output_directory: str, wvls, name: str, spectra_starting_col:int,
                              endmember:str, simulation_library_array, spectral_bundle_project):

        spec_array = spectra.create_spectral_bundles(df=simulation_table, level=level,
                                                     spectral_bundles=spectral_bundles,
                                                     spectral_bundle_project=spectral_bundle_project)

        cols = int(1 / increment_size) + 1
        fraction_grid = np.zeros((len(spec_array), cols, len(class_names)))
        spectra_grid = np.zeros((len(spec_array), cols, len(wvls)))
        index_grid = np.zeros((len(spec_array), cols, len(class_names)))
        veg_grid = np.zeros((len(spec_array), cols, len(wvls)))
        soil_grid = np.zeros((len(spec_array), cols, len(wvls)))

        results = p_map(partial(spectra.row_reflectance, increment_size, cols, wvls,spectra_starting_col, endmember),
                        [bundle for bundle in spec_array], [_index for _index,index in enumerate(spectra_grid)],
                        **{"desc": "\t\t processing reflectance...", "ncols": 150})

        # populate the results
        for _row, row in enumerate(results):
            spectra_grid[_row, :, :] = row[0]
            fraction_grid[_row, :, :] = row[1]
            index_grid[_row, :, :] = row[2]
            veg_grid[_row, :, :] = row[3]
            soil_grid[_row, :, :] = row[4]

        # save the datasets
        refl_meta = get_meta(lines=spectra_grid.shape[0], samples=cols, bands=wvls, wvls=True)
        index_meta = get_meta(lines=index_grid.shape[0], samples=cols, bands=class_names, wvls=False)
        fraction_meta = get_meta(lines=fraction_grid.shape[0], samples=cols, bands=class_names, wvls=False)
        veg_meta = get_meta(lines=veg_grid.shape[0], samples=cols, bands=wvls, wvls=True)
        soil_meta = get_meta(lines=soil_grid.shape[0], samples=cols, bands=wvls, wvls=True)

        # save index, spectra, fraction grid
        output_files = [os.path.join(output_directory, f'{name}_index.hdr'),
                        os.path.join(output_directory, f'{name}_spectra.hdr'),
                        os.path.join(output_directory, f'{name}_fractions.hdr'),
                        os.path.join(output_directory, f'{name}_vegetation.hdr'),
                        os.path.join(output_directory, f'{name}_soils.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta, veg_meta, soil_meta]
        grids = [index_grid, spectra_grid, fraction_grid, veg_grid, soil_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})
        del index_grid, spectra_grid, fraction_grid


    @classmethod
    def cont_removal(cls, wavelengths, reflectance, library_reflectance, expert_file_selection, veg_rfl=None, veg_correction=False,
                     constraints=None):

        #thresholds from expert file system
        ct_thresholds = {'CTHRESH1': 0.01, 'CTHRESH2': 0.02, 'CTHRESH4': 0.04, 'CTHRESH5': 0.05}

        # this holds the multiple values of bd if multiple features are passed by the expert file
        features_array = np.ones((len(expert_file_selection))) * -9999
        wavelength_array = np.ones((len(expert_file_selection))) * -9999
        integrals_array = np.ones((len(expert_file_selection))) * -9999
        fit_array = np.ones((len(expert_file_selection))) * -9999

        rc_array = np.ones((len(expert_file_selection))) * -9999
        rb_array = np.ones((len(expert_file_selection))) * -9999

        # subtract vegetation from spectra leaving only soil spectra
        if veg_correction:
            # extract vegetation spectra from reflectance; leaves only the soil component
            reflectance = reflectance - veg_rfl
        else:
            reflectance = reflectance

        # loop through features
        for _cont_feat, cont_feat in enumerate(expert_file_selection):
            feature = cont_feat['continuum']

            # calculate left indices
            left_inds = np.where(np.logical_and(wavelengths >= feature[0], wavelengths <= feature[1]))[0]
            left_x = wavelengths[int(left_inds.mean())]
            left_y_obs = reflectance[left_inds].mean()
            left_y_lib = library_reflectance[left_inds].mean()

            # calculate right indices
            right_inds = np.where(np.logical_and(wavelengths >= feature[2], wavelengths <= feature[3]))[0]
            if right_inds.size == 0:
                right_inds = spectra.nearest_index_to_wavelength(wavelengths=wavelengths,
                                                                 target_wavelength=(feature[2] + feature[3])/2) # this takes the mean of right bounds
            else:
                pass

            right_x = wavelengths[int(right_inds.mean())]
            right_y_obs = reflectance[right_inds].mean()
            right_y_lib = library_reflectance[right_inds].mean()

            # calculate features
            feature_inds = np.logical_and(wavelengths >= feature[0], wavelengths <= feature[3])

            # calculate continuum
            continuum_obs = interp1d([left_x, right_x], [left_y_obs, right_y_obs], bounds_error=False,
                                 fill_value='extrapolate')(wavelengths)

            continuum_lib = interp1d([left_x, right_x], [left_y_lib, right_y_lib], bounds_error=False,
                                     fill_value='extrapolate')(wavelengths)

            # calculate fit
            oc = reflectance/continuum_obs
            lc = library_reflectance/continuum_lib
            f = np.corrcoef(oc[feature_inds], lc[feature_inds])[0, 1]

            # implement fit constraint
            #if None != constraints:
            #    depth_fit_constraint = float(constraints['DEPTH-FIT'][0])
            #
            #    if f < depth_fit_constraint:
            #        continue
            #else:
            #    pass

            fit_array[_cont_feat] = f

            # implement threshold from tetracorder detections is found!
            try:
                feature_threshold = ct_thresholds[cont_feat['ct'][0][1:-1]]
                if np.any(reflectance[feature_inds] < feature_threshold):
                    continue
            except:
                pass

            # implement rct/lct threshold ratio
            try:
                right_cont_ratio = cont_feat['rct/lct>'][0]
                rct = continuum_obs[feature_inds][-1]
                lct = continuum_obs[feature_inds][0]

                if not rct/lct > right_cont_ratio:
                    continue
            except:
                pass

            # implement lct/rct threshold ratio
            try:
                left_cont_ratio = cont_feat['lct/rct>'][0]
                rct = continuum_obs[feature_inds][-1]
                lct = continuum_obs[feature_inds][0]

                if not lct / rct > left_cont_ratio:
                    continue
            except:
                pass

            # calculate band depth; get max index of band depth ignore calculations with negative reflectance!
            if np.any(reflectance[feature_inds] < 0):
                continue

            depth = 1 - np.array(reflectance[feature_inds]/continuum_obs[feature_inds])
            depth_max_index = depth.argmax()

            # calculate integral of library reference
            h_x = lc[feature_inds]/lc[feature_inds] - lc[feature_inds]
            integral = np.trapz(h_x, wavelengths[feature_inds])

            # append max band bepth
            bd = depth[depth_max_index]
            features_array[_cont_feat] = bd

            # append wvl center
            bd_wl_max_center = wavelengths[feature_inds][depth_max_index]
            wavelength_array[_cont_feat] = bd_wl_max_center

            # append rb and rc
            rc_array[_cont_feat] = continuum_obs[feature_inds][depth_max_index]
            rb_array[_cont_feat] = reflectance[feature_inds][depth_max_index]

            # append integral area
            integrals_array[_cont_feat] = integral

        # correct data for -9999.
        integrals_array[integrals_array == -9999] = np.nan
        features_array[features_array == -9999] = np.nan
        fit_array[fit_array == -9999] = np.nan

        # this will return depths that are weighted
        relative_area = integrals_array/np.nansum(integrals_array)
        bd_w = np.nansum(relative_area * features_array * fit_array)

        rb_return = -9999
        rc_return = -9999
        wvl_return = -9999

        return bd_w, rb_return, rc_return, wvl_return
        
        
    @classmethod
    def nearest_index_to_wavelength(cls, wavelengths, target_wavelength):
        wvl_nearest_index = (np.abs(wavelengths - target_wavelength)).argmin()

        return wvl_nearest_index

    @classmethod
    def group_wvl_center(cls):
        group_wvl_center = {
            'group.2um': 2.24,
            'group.1um': 0.79}

        return group_wvl_center

    @classmethod
    def mineral_group_retrival(cls, mineral_index, spectra_observed, veg_rfl=None, plot=False, veg_correction=False,
                               veg_fraction=None):
        decoded_expert = tc.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                          log_file=None, log_level='INFO')

        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                      'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

        # array to be returned with following positions: group number, rb, rc, rbo, rco, aggregated group num
        return_array = np.ones((10)) * -9999.
        soil_fraction = 1 - veg_fraction

        if soil_fraction < .15: # vegetation fraction check; ignoring very low values; no sense in wasting computing resources here
            pass
        else:

            # mineral matrix
            if mineral_index not in [0, 1, 13, 15, 20, 21, 22, 25, 28, 29, 37, 38, 40, 41, 49, 56, 57, 60, 82, 83, 94,
                                     96, 97, 98, 99, 100, 105, 106, 135, 136, 182, 144, 148, 152, 194, 196, 228, 234,
                                     238, 270, 271]: # this excludes minerals not used for simulation!

                df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
                df_mineral_matrix = df_mineral_matrix.fillna(-9999)
                record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral_index), 'Record'].iloc[0]
                filename = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Filename'].iloc[0]
                group_num = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Group'].iloc[0]
                group = f'group.{group_num}um'
                ref_library = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Library'].iloc[0]

                # row index pertains specifically to df; not value from Tetracorder!
                row_index = df_mineral_matrix[df_mineral_matrix['Record'] == record].index[0]
                mineral_row = df_mineral_matrix.iloc[row_index, 7:]
                mineral_row = mineral_row.apply(pd.to_numeric, errors='coerce')

                # load library
                item = SPECTRAL_REFERENCE_LIBRARY[ref_library]
                library = envi.open(envi_header(item), item)
                library_reflectance = library.spectra.copy()
                library_records = [int(q) for q in library.metadata['record']]

                hdr = envi.read_envi_header(envi_header(item))
                wavelengths = np.array([float(q) for q in hdr['wavelength']])
                normalized_group_name = os.path.normpath(filename.split('.depth.gz')[0]) # need this to be compatible for windows; not sure if needed for linux.

                try:
                    constraints = decoded_expert[normalized_group_name]['constituent_constraints']
                except:
                    constraints = None
                    print(decoded_expert[normalized_group_name]['longname'], 'has no constraints!')

                if veg_correction:
                    # get rco and rbo - observed spectra
                    bdo, rbo, rco, wl_o = spectra.cont_removal(wavelengths, spectra_observed, library_reflectance[library_records.index(record), :],
                                                               decoded_expert[normalized_group_name]['features'],
                                                               veg_correction=veg_correction, veg_rfl=veg_rfl,
                                                               constraints=constraints)

                else:
                    # get rco and rbo - observed spectra
                    bdo, rbo, rco, wl_o = spectra.cont_removal(wavelengths, spectra_observed, library_reflectance[library_records.index(record), :],
                                                               decoded_expert[normalized_group_name]['features'],
                                                               veg_correction=veg_correction, constraints=constraints)

                return_array[:] = [group_num, mineral_index, -9999, -9999, -9999, -9999, bdo, rbo, rco, wl_o]

            else:
                pass

        return return_array

    @classmethod
    def mineral_group_row(cls, mineral_index_row, spectra_row, fraction_row=None, vegetation_row=None,
                          veg_correction=False, group=None):

        group_band_index = {'g1': 1, 'g2': 3}
        row_return_array = np.ones((mineral_index_row.shape[0], 10)) * -9999.

        for _col, col in enumerate(mineral_index_row):
            mineral_index = mineral_index_row[_col, group_band_index[group]]
            veg_fraction = fraction_row[_col, 0] + fraction_row[_col, 1]
            col_spectra = spectra_row[_col, :]

            if veg_correction:
                veg_rfl = vegetation_row[_col, :]
                mineral_retrival = spectra.mineral_group_retrival(mineral_index=mineral_index,
                                                                  spectra_observed=col_spectra,
                                                                  veg_correction=veg_correction,
                                                                  veg_rfl=veg_rfl, veg_fraction=veg_fraction)

            else:
                mineral_retrival = spectra.mineral_group_retrival(mineral_index=mineral_index,
                                                                  spectra_observed=col_spectra,
                                                                  veg_correction=veg_correction,
                                                                  veg_fraction=veg_fraction)

            row_return_array[_col, :] = mineral_retrival

        return row_return_array

    @classmethod
    def mineral_components(cls, index_array, spectra_array, output_file, group, fractions_array=None,
                           vegetation_array=None, veg_correction=False):

        # cont grid - corresponds to Rc and Rc-observed
        output_grid = np.zeros((index_array.shape[0], index_array.shape[1], 10))

        if veg_correction:
            results = p_map(partial(spectra.mineral_group_row, veg_correction=veg_correction, group=group),
                            [index_array[_row, :, :] for _row, row in enumerate(index_array)],
                            [spectra_array[_row, :, :] for _row, row in enumerate(spectra_array)],
                            [fractions_array[_row, :, :] for _row, row in enumerate(fractions_array)],
                            [vegetation_array[_row, :, :] for _row, row in enumerate(vegetation_array)],

                            **{"desc": "\t\t processing continuum reflectance (veg correction enabled) ...",
                                "ncols": 150})

        else:
            results = p_map(partial(spectra.mineral_group_row, veg_correction=veg_correction,  group=group),
                            [index_array[_row, :, :] for _row, row in enumerate(index_array)],
                            [spectra_array[_row, :, :] for _row, row in enumerate(spectra_array)],
                            [fractions_array[_row, :, :] for _row, row in enumerate(fractions_array)],
                            **{"desc": "\t\t processing continuum reflectance...", "ncols": 150})

        for _row, row in enumerate(results):
            output_grid[_row, :, :] = row

        # save spectra
        meta = get_meta(lines=index_array.shape[0], samples=index_array.shape[1], bands=[i for i in range(10)], wvls=False)
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



