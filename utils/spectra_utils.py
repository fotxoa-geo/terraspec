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

        try:
            gdf = gpd.read_file('gis/Observation.json')
            gdf['longitude'] = gdf['geometry'].x
            gdf['latitude'] = gdf['geometry'].y
            plot_name = os.path.basename(os.path.dirname(file)).replace('Spectral', 'SPEC')
            df = gdf.drop(columns='geometry')

            long = df.loc[(df['Name'] == plot_name), 'longitude'].iloc[0]
            lat = df.loc[(df['Name'] == plot_name), 'latitude'].iloc[0]

            dd_lat = lat
            dd_long = long

        except:
            dd_lat = -9999.
            dd_long = -9999.

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
        wavelength_file = os.path.join('utils', 'wavelengths', f'{sensor}_wavelengths.txt')
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
    def load_global_library(cls, output_directory, sensor, spectra_starting_col=None, geo_filter=True):

        if geo_filter:
            df = pd.read_csv(os.path.join(output_directory, 'convolved', f'geofilter_sensor_{sensor}_convolved.csv'))

        else:
            df = pd.read_csv(os.path.join(output_directory, 'convolved', f'all_data_sensor_{sensor}_convolved.csv'))

            # drop spectra with nan rows
            spectra_cols = df.columns[spectra_starting_col:]
            df = df.dropna(subset=spectra_cols)

        return df

    @classmethod
    def load_global_library_metadata(cls):
        df = pd.read_csv(os.path.join('simulation', 'emit_global_lib_all.csv'))
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
    def create_spectral_bundles(cls, df, level, spectral_bundles, spectral_bundle_project, new_simulation_bundles):
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
        pickle_path = os.path.join('objects', f"output_pickle_{spectral_bundle_project}.pickle")

        if new_simulation_bundles or not os.path.isfile(pickle_path):
            all_combinations = list(itertools.product(*class_lists))
            save_pickle(all_combinations, output_pickle)
        else:
            all_combinations = load_pickle(output_pickle)

        print(f'total spectral bundles available for simulation: {len(all_combinations)}')

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
        soil_spectra = np.zeros((1, columns, len(wavelengths)))
        gv_spectra = np.zeros((1, columns, len(wavelengths)))
        npv_spectra = np.zeros((1, columns, len(wavelengths)))

        for _col, col in enumerate(range(0, columns)):
            increment_frac = np.round(col * col_size, 2)
            npv_frac, pv_frac, soil_frac = spectra.generate_em_fractions(em=em, em_fraction=increment_frac,
                                                                         seed=row_index+_col)

            mixed_spectra[0, _col, :] = (spectral_bundle[0][spectra_start:].astype(dtype=float) * npv_frac) + \
                                        (spectral_bundle[1][spectra_start:].astype(dtype=float) * pv_frac) + \
                                        (spectral_bundle[2][spectra_start:].astype(dtype=float) * soil_frac)

            soil_spectra[0, _col, :] = spectral_bundle[2][spectra_start:].astype(dtype=float) * soil_frac

            fractions[0, _col, :] = [npv_frac, pv_frac, soil_frac]
            index[0, _col, :] = list(map(int, [spectral_bundle[0][0], spectral_bundle[1][0], spectral_bundle[2][0]]))

            gv_spectra[0, _col, :] = spectral_bundle[1][spectra_start:].astype(dtype=float)
            npv_spectra[0, _col, :] = spectral_bundle[0][spectra_start:].astype(dtype=float)

        return mixed_spectra, fractions, index, soil_spectra, gv_spectra, npv_spectra

    @classmethod
    def increment_reflectance(cls, class_names: list, simulation_table, level: str, spectral_bundles:int,
                              increment_size:float, output_directory: str, wvls, name: str, spectra_starting_col:int,
                              endmember:str, spectral_bundle_project, new_simulation_bundles):

        spec_array = spectra.create_spectral_bundles(df=simulation_table, level=level,
                                                     spectral_bundles=spectral_bundles,
                                                     spectral_bundle_project=spectral_bundle_project,
                                                     new_simulation_bundles=new_simulation_bundles)

        cols = int(1 / increment_size) + 1
        fraction_grid = np.zeros((len(spec_array), cols, len(class_names)))
        spectra_grid = np.zeros((len(spec_array), cols, len(wvls)))
        index_grid = np.zeros((len(spec_array), cols, len(class_names)))
        soil_grid = np.zeros((len(spec_array), cols, len(wvls)))
        npv_grid = np.zeros((len(spec_array), cols, len(wvls)))
        gv_grid = np.zeros((len(spec_array), cols, len(wvls)))
        np.random.seed(13)
        random_grid = np.random.rand(len(spec_array), cols)

        results = p_map(partial(spectra.row_reflectance, increment_size, cols, wvls, spectra_starting_col, endmember),
                        [bundle for bundle in spec_array], [_index for _index,index in enumerate(spectra_grid)],
                        **{"desc": "\t\t processing reflectance...", "ncols": 150})

        # populate the results
        for _row, row in enumerate(results):
            spectra_grid[_row, :, :] = row[0]
            fraction_grid[_row, :, :] = row[1]
            index_grid[_row, :, :] = row[2]
            soil_grid[_row, :, :] = row[3]
            gv_grid[_row, :, :] = row[4]
            npv_grid[_row, :, :] = row[5]

        # save the datasets
        refl_meta = get_meta(lines=spectra_grid.shape[0], samples=cols, bands=wvls, wvls=True)
        index_meta = get_meta(lines=index_grid.shape[0], samples=cols, bands=class_names, wvls=False)
        fraction_meta = get_meta(lines=fraction_grid.shape[0], samples=cols, bands=class_names, wvls=False)
        soil_meta = get_meta(lines=soil_grid.shape[0], samples=cols, bands=wvls, wvls=True)
        gv_meta = get_meta(lines=gv_grid.shape[0], samples=cols, bands=wvls, wvls=True)
        npv_meta = get_meta(lines=npv_grid.shape[0], samples=cols, bands=wvls, wvls=True)

        # save index, spectra, fraction grid
        output_files = [os.path.join(output_directory, f'{name}_index.hdr'),
                        os.path.join(output_directory, f'{name}_spectra.hdr'),
                        os.path.join(output_directory, f'{name}_fractions.hdr'),
                        os.path.join(output_directory, f'{name}_soils.hdr'),
                        os.path.join(output_directory, f'{name}_gv.hdr'),
                        os.path.join(output_directory, f'{name}_npv.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta, soil_meta, gv_meta, npv_meta]
        grids = [index_grid, spectra_grid, fraction_grid, soil_grid, gv_grid, npv_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})
        del index_grid, spectra_grid, fraction_grid



    @classmethod
    def cont_removal(cls, wavelengths, reflectance, library_reflectance, expert_file_selection,
                     constraints=None, npv_fraction=None, gv_fraction=None, pnpv=None,
                     pgv=None, soil_fraction=None, psoil=None, plot=None, output_directory=None, plot_info=None):

        #thresholds from expert file system
        ct_thresholds = {'CTHRESH1': 0.01, 'CTHRESH2': 0.02, 'CTHRESH4': 0.04, 'CTHRESH5': 0.05}

        # this holds the multiple values of bd if multiple features are passed by the expert file
        integrals_array = np.ones((len(expert_file_selection))) * -9999
        bd_array = np.ones((len(expert_file_selection))) * -9999
        bd_prime_array = np.ones((len(expert_file_selection))) * -9999
        bd_library_array = np.ones((len(expert_file_selection))) * -9999

        valid_wavelengths = ~np.isnan(reflectance)

        # loop through features
        for _cont_feat, cont_feat in enumerate(expert_file_selection):

            feature = cont_feat['continuum']
            print(feature)

            if soil_fraction == 0:
                continue

            left_inds = np.where(np.logical_and.reduce((wavelengths >= feature[0], wavelengths <= feature[1], valid_wavelengths)))[0]
            right_inds = np.where(np.logical_and.reduce((wavelengths >= feature[2], wavelengths <= feature[3], valid_wavelengths)))[0]

            # calculate features start/stop
            feature_inds = np.logical_and(wavelengths >= wavelengths[left_inds][0], wavelengths <= wavelengths[right_inds][-1])

            # x boundaries - used for all calculations
            x1, x2 = wavelengths[feature_inds][0], wavelengths[feature_inds][-1] #λi, λj

            # calculate continuum for tetracorder library
            m_l = (library_reflectance[feature_inds][-1] - library_reflectance[feature_inds][0]) / (x2 - x1)
            b_l = library_reflectance[feature_inds][0] - m_l * x1
            lc = m_l * wavelengths + b_l
            bd_l = 1 - np.array(library_reflectance[feature_inds] / lc[feature_inds])
            bd_max_l = bd_l.argmax()
            bd_library_array[_cont_feat] = bd_l[bd_max_l]

            # calculate integral of tetracorder library reference
            h_x = lc[feature_inds] / lc[feature_inds] - lc[feature_inds]
            integral = np.trapz(h_x, wavelengths[feature_inds])
            integrals_array[_cont_feat] = integral

            # calculate vegetation correction
            Ci = x1/(x2-x1)
            Ri = reflectance[feature_inds][0]
            Rj = reflectance[feature_inds][-1]

            fnpv = npv_fraction
            fgv = gv_fraction
            pnpv_j = pnpv[feature_inds][-1]
            pnpv_i = pnpv[feature_inds][0]
            pgv_j = pgv[feature_inds][-1]
            pgv_i = pgv[feature_inds][0]

            # this is our soil spectrum
            psoil = (1/soil_fraction) * (reflectance - (fnpv*pnpv + fgv*pgv))

            # these are the continuum for Bd
            m = (Rj - Ri)/(x2-x1)
            b = Ri - m*x1
            rc = m * wavelengths + b
            bd = 1 - np.array(reflectance[feature_inds]/rc[feature_inds])
            bd_max = np.nanargmax(bd)
            bd_array[_cont_feat] = bd[bd_max]

            # these are the continuum for Bd'
            b_prime = (1 / soil_fraction) * ((1 + Ci) * (Ri - fnpv * pnpv_i - fgv * pgv_i) - Ci * (Rj - fnpv * pnpv_j - fgv * pgv_j))
            m_prime = (Ri - fnpv * pnpv_i - fgv * pgv_i) / (x1 * soil_fraction) - (b_prime / x1)
            rc_prime = m_prime * wavelengths + b_prime
            bd_ρsoil = 1 - np.array(psoil[feature_inds]/rc_prime[feature_inds])
            bd_max_ρsoil = np.nanargmax(bd_ρsoil)
            bd_prime_array[_cont_feat] = bd_ρsoil[bd_max_ρsoil]

            if plot:
                slopes = [m, m_prime]
                intercepts = [b, b_prime]
                rfls = [reflectance, psoil]
                rcs = [rc, rc_prime,]
                labels = ["Bd", "Bd'",]
                spectrum_labels = ["ρ", "ρ$_s$"]
                rc_labels = ["ρ$_c$", "ρ$_c'$"]
                colors = ['red', 'blue']
                bd_plot = 1 - np.array(reflectance[feature_inds] / rc[feature_inds])
                bd_prime_plot = 1 - np.array(psoil[feature_inds] / rc_prime[feature_inds])
                bd_max = np.nanargmax(bd_plot)
                bd_max_prime = np.nanargmax(bd_prime_plot)

                bd_maxes = [bd_max, bd_max_prime]


                plt.plot(wavelengths, reflectance, label="ρ", color='red')
                #plt.plot(wavelengths, psoil, label="ρ$_{soil}$", color='blue')
                plt.legend()
                plt.ylabel('Reflectance')
                plt.xlabel('Wvls')
                plt.savefig(os.path.join(output_directory, f'spectra_complete-cont_feat{_cont_feat}.png'))
                plt.clf()
                plt.close()

                plt.plot(wavelengths, reflectance, label="ρ", color='red')
                plt.plot(wavelengths, psoil, label="ρ$_{soil}$", color='blue')
                plt.legend()
                plt.ylabel('Reflectance')
                plt.xlabel('Wvls')
                plt.axvspan(x1, x2, color='blue', alpha=0.1)
                plt.savefig(os.path.join(output_directory, f'spectra_complete_shaded-cont_feat{_cont_feat}.png'))
                plt.clf()
                plt.close()

                for _i, i in enumerate(slopes):
                    #plt.title(labels[_i])
                    plt.xlim(wavelengths[feature_inds][0] - 0.05, wavelengths[feature_inds][-1] + 0.05)
                    plt.ylim(0, 0.20)
                    plt.plot(wavelengths[feature_inds], rfls[_i][feature_inds], label=spectrum_labels[_i], color=colors[_i])
                    plt.plot(wavelengths[feature_inds], rcs[_i][feature_inds], label=f'Continnum: R$_c$ = {slopes[_i]:.2f}λ$_o$ + {intercepts[_i]:.2f}', color='green')


                    # plot max depth
                    plt.vlines(x=wavelengths[feature_inds][bd_maxes[_i]], ymin=rfls[_i][feature_inds][bd_maxes[_i]], ymax=rcs[_i][feature_inds][bd_maxes[_i]],
                               color='purple', linestyle='--', linewidth=2, label=f"Wvl: {wavelengths[feature_inds][bd_maxes[_i]]:.3f}")

                    # arrows for lambda start and end
                    plt.annotate(f"λ$_i$ = {wavelengths[feature_inds][0]:.2f}",
                                xy=(x1, rfls[_i][feature_inds][0]),  # Arrow points *to* this location
                                xytext=(x1, 0.12),  # Text is placed *at* this location
                                arrowprops=dict(arrowstyle="->", color='blue'),
                                fontsize=12,
                                color='black')

                    plt.annotate(f"λ$_j$ = {wavelengths[feature_inds][-1]:.2f}",
                                 xy=(x2, rfls[_i][feature_inds][-1]),  # Arrow points *to* this location
                                 xytext=(x2, 0.12),  # Text is placed *at* this location
                                 arrowprops=dict(arrowstyle="->", color='blue'),
                                 fontsize=12,
                                 color='black')

                    # arrow for observered spectra
                    plt.annotate(f"{rc_labels[_i]} = {rcs[_i][feature_inds][bd_maxes[_i]]:.2f}",
                                 xy=(wavelengths[feature_inds][bd_maxes[_i]], rcs[_i][feature_inds][bd_maxes[_i]]),  # Arrow points *to* this location
                                 xytext=(wavelengths[feature_inds][bd_maxes[_i]], 0.14),  # Text is placed *at* this location
                                 arrowprops=dict(arrowstyle="->", color='blue'),
                                 fontsize=12,
                                 color='black')

                    # arrow for observered spectra
                    plt.annotate(f"{spectrum_labels[_i]} = {rfls[_i][feature_inds][bd_maxes[_i]]:.2f}",
                                 xy=(wavelengths[feature_inds][bd_maxes[_i]], rfls[_i][feature_inds][bd_maxes[_i]]),
                                 # Arrow points *to* this location
                                 xytext=(wavelengths[feature_inds][bd_maxes[_i]], 0.05),  # Text is placed *at* this location
                                 arrowprops=dict(arrowstyle="->", color='blue'),
                                 fontsize=12,
                                 color='black')

                    plt.legend()
                    plt.ylabel('Reflectance')
                    plt.xlabel('Wvls')
                    plt.savefig(os.path.join(output_directory, f'{plot_info}-cont_feat{_cont_feat}-{labels[_i]}.png'))
                    plt.clf()
                    plt.close()

                # plot continuums
                plt.figure(figsize=(8, 5))
                plt.plot(wavelengths[feature_inds], np.ones(len(wavelengths))[feature_inds], label="Continuum", color='blue')
                plt.plot(wavelengths[feature_inds], np.array(reflectance[feature_inds] / rc[feature_inds]),label="Unknown Compound", color='red')
                #plt.plot(wavelengths[feature_inds], np.array(psoil[feature_inds]/rc_prime[feature_inds]), label="Unknown Mineral Spectra", color='red')
                plt.plot(wavelengths[feature_inds], np.array(library_reflectance[feature_inds] / lc[feature_inds]), label="Reference Compound", color='green')

                plt.vlines(x=wavelengths[feature_inds][bd_max], ymin=np.array(reflectance[feature_inds]/rc[feature_inds])[bd_max],
                           ymax=np.ones(len(wavelengths))[feature_inds][bd_max],
                           color='red', linestyle='--', linewidth=2,
                           label=f"Band Depth\n(e.g., Absorption Stength)")

                # plt.vlines(x=wavelengths[feature_inds][bd_max_prime],
                #            ymin=np.array(psoil[feature_inds] / rc_prime[feature_inds])[bd_max_prime],
                #            ymax=np.ones(len(wavelengths))[feature_inds][bd_max_prime],
                #            color='blue', linestyle='--', linewidth=2,
                #            label=f"Band Depth (e.g., Absorption Stength)")

                plt.legend(loc="lower right", fontsize=8)

                ticks_um = plt.xticks()[0]

                # Set labels in nm
                plt.xticks(ticks=ticks_um, labels=[f"{int(t * 1000)}" for t in ticks_um])

                plt.ylabel('Reflectance')
                plt.xlabel('Wavelength (nm)')
                plt.savefig(os.path.join(output_directory, f'{plot_info}-cont_feat{_cont_feat}-continnum.png'))
                plt.clf()
                plt.close()

        # correct data for -9999.
        integrals_array[integrals_array == -9999] = np.nan

        # array to return all band depths
        bd_return_array = np.ones(4) * -9999

        # calculate weighted band depths
        for _i, i in enumerate([bd_library_array, bd_array, bd_prime_array]):
            i[i == -9999] = np.nan
            relative_area = integrals_array/np.nansum(integrals_array)
            band_depth_w = np.nansum(relative_area * i)

            if band_depth_w <= 1.:

                bd_return_array[_i] = band_depth_w
            else:
                pass

        return bd_return_array
        

    @classmethod
    def tetracorder_id(cls, spectral_libraries, spectrum,  decoded_expert_system):

        def cont_removal_for_id(wvls, spectra, ref_spectra, features):
            # this holds the multiple values of bd if multiple features are passed by the expert file
            integrals_array = np.ones((len(features))) * -9999
            bd_array = np.ones((len(features))) * -9999
            fit_array = np.ones((len(features))) * 9999

            # loop through features
            for _cont_feat, cont_feat in enumerate(features):
                feature = cont_feat['continuum']

                # calculate left indices
                left_inds = np.where(np.logical_and(wavelengths >= feature[0], wavelengths <= feature[1]))[0]
                left_x = wavelengths[int(left_inds.mean())]
                left_y_lib = library_reflectance[left_inds].mean()
                left_y_obs = spectrum[left_inds].mean()

                # calculate right indices
                right_inds = np.where(np.logical_and(wavelengths >= feature[2], wavelengths <= feature[3]))[0]
                if right_inds.size == 0:
                    right_inds = spectra.nearest_index_to_wavelength(wavelengths=wavelengths,
                                                                     target_wavelength=(feature[2] + feature[
                                                                         3]) / 2)  # this takes the mean of right bounds
                else:
                    pass

                right_x = wavelengths[int(right_inds.mean())]
                right_y_lib = library_reflectance[right_inds].mean()
                right_y_obs = spectrum[right_inds].mean()

                # get features
                feature_inds = np.logical_and(wavelengths >= feature[0], wavelengths <= feature[3])

                # calculate continuum for library
                continuum_lib = interp1d([left_x, right_x], [left_y_lib, right_y_lib], bounds_error=False,
                                         fill_value='extrapolate')(wavelengths)

                continuum_obs = interp1d([left_x, right_x], [left_y_obs, right_y_obs], bounds_error=False,
                                         fill_value='extrapolate')(wavelengths)

                # calculate mineral continuum from tetracorder library
                lc = library_reflectance / continuum_lib
                oc = spectra / continuum_obs

                # calculate fit of observed and mineral reference from Clark et al 2003
                b_top = np.sum(oc*lc) - (np.sum(oc) * np.sum(lc))/len(wavelengths[feature_inds])
                b_bottom = np.sum(lc**2) - (np.sum(lc)**2)/len(wavelengths[feature_inds])
                b = b_top/b_bottom
                a = (np.sum(oc) - b * np.sum(lc))/len(wavelengths[feature_inds])

                b_prime_bottom = np.sum(oc**2) - (np.sum(oc)**2)/len(wavelengths[feature_inds])
                b_prime = b_top/b_prime_bottom

                F = (b*b_prime)**(1/2)

                fit_array[_cont_feat] = F

                # calculate integral of tetracorder library reference
                #h_x = lc[feature_inds] / lc[feature_inds] - lc[feature_inds]
                #integral = np.trapz(h_x, wavelengths[feature_inds])
                #integrals_array[_cont_feat] = integral

                # x boundaries - used for all calculations
                x1, x2 = left_x, right_x

            return fit_array

        for i in decoded_expert_system:
            select_expert_system = decoded_expert_system[i]
            select_library = spectral_libraries[select_expert_system['spectral_library']]
            select_record = select_expert_system['record']
            longname = select_expert_system['longname']
            features = select_expert_system['features']

            library = envi.open(envi_header(select_library), select_library)
            library_reflectance = library.spectra.copy()
            library_records = [int(q) for q in library.metadata['record']]

            hdr = envi.read_envi_header(envi_header(select_library))
            wavelengths = np.array([float(q) for q in hdr['wavelength']])

            mineral_reflectance = library_reflectance[library_records.index(select_record), :]
            fit_array = cont_removal_for_id(wvls=wavelengths, spectra=spectrum, ref_spectra=mineral_reflectance, features=features)


    @classmethod
    def nearest_index_to_wavelength(cls, wavelengths, target_wavelength):
        wvl_nearest_index = (np.abs(wavelengths - target_wavelength)).argmin()

        return wvl_nearest_index

    @classmethod
    def get_mineral_information(cls, mineral_index):
        # expert system
        decoded_expert = tc.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                 log_file=None, log_level='INFO')

        # libraries from tetracorder
        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                      'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

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
        print(item)
        library = envi.open(envi_header(item), item)
        print(library)
        library_reflectance = library.spectra.copy()
        library_records = [int(q) for q in library.metadata['record']]

        hdr = envi.read_envi_header(envi_header(item))
        wavelengths = np.array([float(q) for q in hdr['wavelength']])
        normalized_group_name = os.path.normpath(
            filename.split('.depth.gz')[0])  # need this to be compatible for windows; not sure if needed for linux.

        # This is the vegetation correction
        expert_file_selection = decoded_expert[normalized_group_name]['features']
        print(expert_file_selection)
        valid_wavelenghts = ~np.isnan(library_reflectance[library_records.index(record), :] )

        for _cont_feat, cont_feat in enumerate(expert_file_selection):
            feature = cont_feat['continuum']
            left_inds = np.where(np.logical_and.reduce((wavelengths >= feature[0], wavelengths <= feature[1], valid_wavelenghts)))[0]
            right_inds = np.where(np.logical_and.reduce((wavelengths >= feature[2], wavelengths <= feature[3], valid_wavelenghts)))[0]

            # calculate features start/stop
            feature_inds = np.logical_and(wavelengths >= wavelengths[left_inds][0],
                                          wavelengths <= wavelengths[right_inds][-1])

            # x boundaries - used for all calculations
            x1, x2 = wavelengths[feature_inds][0], wavelengths[feature_inds][-1]  # λi, λj


            print(x1, x2)

    @classmethod
    def mineral_group_retrival(cls, mineral_index, spectra_observed, npv_fraction=None, gv_fraction=None, pnpv=None,
                               pgv=None, soil_fraction=None, psoil=None, exclude_minerals=True, plot=None, output_directory=None, plot_info=None):

        # expert system
        decoded_expert = tc.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                          log_file=None, log_level='INFO')

        # libraries from tetracorder
        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                      'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

        # array to be returned with following positions: group number, mineral index, bdw
        return_array = np.ones((6)) * -9999.

        if exclude_minerals:
            minerals_to_exclude = [0, 1, 13, 15, 22, 25, 28, 29, 37, 38, 40, 41, 49, 51, 56, 57, 60, 64, 82, 83, 94,
                                    96, 97, 98, 99, 100, 105, 106, 135, 136, 182, 144, 148, 152, 194, 196, 217, 221, 226,
                                    228, 234, 238, 270, 271, 292] # this excludes minerals not used for simulation!
        else:
            minerals_to_exclude = [0, 60, 96, 97, 98, 99, 100] # no detection and vegetation

        # mineral matrix
        if mineral_index not in minerals_to_exclude:

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
                #print(decoded_expert[normalized_group_name]['longname'], 'has no constraints!')

            # This is the vegetation correction
            bdw = spectra.cont_removal(wavelengths, spectra_observed, library_reflectance[library_records.index(record), :],
                                                       decoded_expert[normalized_group_name]['features'],
                                                       constraints=constraints, soil_fraction=soil_fraction,
                                                       npv_fraction=npv_fraction, gv_fraction=gv_fraction,
                                                       pnpv=pnpv, pgv=pgv, psoil=psoil, plot=plot, output_directory=output_directory, plot_info=plot_info)

            # recalculate tetracorder for new fit
            #psoil_observed = (1/soil_fraction) * (spectra_observed - (npv_fraction*pnpv + gv_fraction*pgv))
            #spectra.tetracorder_id(spectral_libraries=SPECTRAL_REFERENCE_LIBRARY, spectrum=psoil_observed,
            #                       decoded_expert_system=decoded_expert)

            return_array[:] = np.concatenate(([group_num, mineral_index], bdw))

        return return_array

    @classmethod
    def mineral_group_row(cls, mineral_index_row=None, spectra_row=None, fraction_row=None, gv_row=None, npv_row=None, group=None):

        group_band_index = {'g1': 1, 'g2': 3}
        row_return_array = np.ones((mineral_index_row.shape[0], 6)) * -9999.

        for _col, col in enumerate(mineral_index_row):
            mineral_index = mineral_index_row[_col, group_band_index[group]]
            npv_fraction = fraction_row[_col, 0]
            gv_fraction = fraction_row[_col, 1]
            soil_fraction = fraction_row[_col, 2]
            col_spectra = spectra_row[_col, :]
            pnpv = npv_row[_col, :]
            pgv = gv_row[_col, :]

            mineral_retrival = spectra.mineral_group_retrival(mineral_index=mineral_index, spectra_observed=col_spectra,
                                                              npv_fraction=npv_fraction, gv_fraction=gv_fraction, soil_fraction=soil_fraction,
                                                              pnpv=pnpv, pgv=pgv, psoil=None)

            row_return_array[_col, :] = mineral_retrival

        return row_return_array

    @classmethod
    def mineral_components(cls, index_array, spectra_array, output_file, group, fractions_array=None,
                           npv_array=None, gv_array=None):

        # continnum grid
        output_grid = np.zeros((index_array.shape[0], index_array.shape[1], 6))
        func = partial(spectra.mineral_group_row, group=group)
        results = p_map(func,
                        [index_array[_row, :, :] for _row in range(index_array.shape[0])],
                        [spectra_array[_row, :, :] for _row in range(spectra_array.shape[0])],
                        [fractions_array[_row, :, :] for _row in range(fractions_array.shape[0])],
                        [gv_array[_row, :, :] for _row in range(gv_array.shape[0])],
                        [npv_array[_row, :, :] for _row in range(npv_array.shape[0])],
                        **{"desc": "\t\t processing continuum reflectance calculations ...", "ncols": 150})

        for _row, row in enumerate(results):
            output_grid[_row, :, :] = row

        # save spectra
        meta = get_meta(lines=index_array.shape[0], samples=index_array.shape[1], bands=[i for i in range(6)], wvls=False)
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

        print()

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
                                [(spec_array[_row, :, spectra_starting_col + 1:], seeds_array[_row, :],
                                  spec_array[_row, :, :4]) for _row, row in enumerate(spectra_grid)],
                                **{"desc": "\t\t generating fractions...", "ncols": 150})

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
        output_files = [os.path.join(output_directory, f'{name}_index.hdr'),
                        os.path.join(output_directory, f'{name}_spectra.hdr'),
                        os.path.join(output_directory, f'{name}_fractions.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta]
        grids = [index_grid, spectra_grid, fraction_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})

        del index_grid, spectra_grid, fraction_grid

    @classmethod
    def simulate_reflectance(cls, df_sim, df_unmix, dimensions, sim_libraries_output, name, level, spectral_bundles, cols,
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

        # is this secondary check needed?
        df_sim.to_csv(os.path.join(sim_libraries_output, f'{name}_simulation_library.csv'), index=False)

        print(name)
        # # create the reflectance file
        spectra.generate_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim, level=level,
                         spectral_bundles=spectral_bundles, cols=cols, output_directory=output_directory,
                         wvls=wvls, name=name, spectra_starting_col=spectra_starting_col)

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

            try:
                asd = asdreader.reader(file)
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
                plt.ylim([0, 1])

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
                plt.ylim([0, 110])

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

    @classmethod
    def tetracorder_aggregation(cls, txt_files_tetracorder):
        reference_ids = []

        for i in sorted(txt_files_tetracorder):

            emit_group = os.path.basename(i).split('.')[0]

            # skip read me file
            if emit_group in ['AAA']:
                continue

            emit_group = emit_group.split('-')[0]

            with open(i) as f:
                lines = f.readlines()

            # filter lines
            data_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

            if data_lines:
                for line in data_lines:
                    no_comment_line = line.split('#')[0].strip().rstrip()
                    name = no_comment_line.split(" ")[0]
                    cleaned_line = no_comment_line.rstrip()
                    mineral_ref = int(cleaned_line[-4:])
                    library_used = cleaned_line[-17:-10].rstrip()
                    reference_ids.append([name, mineral_ref, emit_group, library_used, os.path.basename(i)])
            else:
                continue

        df_mineral = pd.DataFrame(reference_ids)
        df_mineral.columns = ['Filename', 'Record', 'emit_group', 'Library', 'Base_group']
        df_mineral['Group'] = df_mineral['Filename'].str.split('/').str[0].str.split('.').str[1].str.split('um').str[0].astype(int)

        return df_mineral

    @classmethod
    def get_mineral_reclassification(cls, path_to_tetracorder_minerals):
        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))

        txt_files_tetracorder_sim = glob(os.path.join(path_to_tetracorder_minerals, '*.txt'))
        df_minerals_sim = spectra.tetracorder_aggregation(txt_files_tetracorder=txt_files_tetracorder_sim)

        df_minerals_sim = df_minerals_sim.merge(df_mineral_matrix[['Record', 'Index', 'Library']], on=['Record', 'Library'], how='left')
        df_minerals_sim = df_minerals_sim.dropna()
        df_minerals_sim['Index'] = df_minerals_sim['Index'].astype(int)

        sim_dictionary = df_minerals_sim.set_index('Index')['emit_group'].to_dict()
        sim_dictionary.update({0: "no detection", -9999: "No Data", 96: "vegetation", 97: "vegetation", 98: "vegetation"})

        return sim_dictionary, df_minerals_sim




