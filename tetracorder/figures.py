import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd
import numpy as np
from utils.envi import envi_to_array, load_band_names
from utils.create_tree import create_directory
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import geopandas as gp
from glob import glob
from utils.results_utils import r2_calculations, band_depth_group_aggregate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from isofit.core.sunposition import sunpos
from datetime import datetime, timezone
from tetracorder.aggregator import unique_file_fractions
import tetracorder.tetracorder as tetracorder
from scipy.interpolate import interp1d
import spectral.io.envi as envi
from emit_utils.file_checks import envi_header
import logging
from utils.spectra_utils import spectra
from pypdf import PdfMerger
import matplotlib.image as mpimg
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from p_tqdm import p_map
from matplotlib.ticker import MultipleLocator
from utils.slpit_download import load_pickle, save_pickle
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
import tetracorder.tetracorder as tcs
from functools import partial

mineral_groupings = {'group.1um/copper_precipitate_greenslime': 'Copper',
                     'group.1um/fe2+_chlor+muscphy': 'Fe Oxides',
                     'group.1um/fe2+_goeth+musc': 'Fe Oxides',
                     'group.1um/fe2+fe3+_chlor+goeth.propylzone': 'Fe Oxides',
                     'group.1um/fe2+generic_br33a_bioqtzmonz_epidote': 'Fe Oxides',
                     'group.1um/fe2+generic_carbonate_siderite1': 'Fe Oxides',
                     'group.1um/fe2+generic_nrw.cummingtonite': 'Fe Oxides',
                     'group.1um/fe2+generic_nrw.hs-actinolite': 'Fe Oxides',
                     'group.1um/fe2+generic_vbroad_br20': 'Fe Oxides',
                     'group.1um/fe3+_goethite+qtz.medgr.gds240': 'Fe Oxides',
                     'group.1um/fe3+_goethite.thincoat': 'Fe Oxides',
                     'group.1um/fe3+_hematite.nano.BR34b2': 'Fe Oxides',
                     'group.1um/fe3+_hematite.nano.BR34b2b': 'Fe Oxides',
                     'group.1um/fe3+copper-hydroxide_pitchlimonite': 'Fe Oxides',
                     'group.1um/fe3+mn_desert.varnish1': 'Fe Oxides',
                     'group.2um/calcite+0.2Na-mont': 'Carbonate',
                     'group.2um/calcite+0.5Ca-mont': 'Carbonate',
                     'group.2um/calcite.25+dolom.25+Na-mont.5': 'Carbonate',
                     'group.2um/carbonate_aragonite': 'Carbonate',
                     'group.2um/carbonate_calcite': 'Carbonate',
                     'group.2um/carbonate_calcite+0.2Ca-mont': 'Carbonate',
                     'group.2um/carbonate_calcite+0.3muscovite': 'Carbonate',
                     'group.2um/carbonate_calcite0.7+kaol0.3': 'Carbonate',
                     'group.2um/carbonate_dolo+.5ca-mont': 'Carbonate',
                     'group.2um/carbonate_dolomite': 'Carbonate',
                     'group.2um/chlorite-skarn': 'Silicates',
                     'group.2um/kaolin+musc.intimat': 'Clay',
                     'group.2um/kaolin.5+muscov.medAl': 'Clay',
                     'group.2um/micagrp_lepidolite': 'Clay',
                     'group.2um/micagrp_muscovite-low-Al': 'Clay',
                     'group.2um/micagrp_muscovite-med-Al': 'Clay',
                     'group.2um/micagrp_vermiculite_WS682': 'Clay',
                     'group.2um/organic_drygrass+.17Na-mont': 'Organics',
                     'group.2um/organic_vegetation-dry-grass-golden': 'Organics',
                     'group.2um/sioh_chalcedony': 'Silicates',
                     'group.2um/sioh_hydrated_basaltic_glass': 'Silicates',
                     'group.2um/smectite_montmorillonite_ca_swelling': 'Clay',
                     'group.2um/smectite_montmorillonite_na_highswelling': 'Clay',
                     'group.2um/smectite_nontronite_swelling': 'Clay',
                     'N.D.': 'N.D.'}

mineral_groups = {'Calcite': 'Carbonates',
                  'Chlorite': 'Chlorite',
                  'Dolomite': 'Carbonates',
                  'Goethite-Nano': 'Fe Oxides',
                  'Goethite-Fine': 'Fe Oxides',
                  'Goethite-Med' : 'Fe Oxides',
                  'Goethite-Large': 'Fe Oxides',
                  'Gypsum-Fine': 'Carbonates',
                  'Gypsum-Coarse': 'Carbonates',
                  'Hematite-Nano': 'Fe Oxides',
                  'Hematite-Fine': 'Fe Oxides',
                  'Hematite-Med': 'Fe Oxides',
                  'Hematite-Large': 'Fe Oxides',
                  'Illite+Muscovite': 'Clays',
                  'Kaolinite': 'Clays',
                  'Montmorillonite': 'Clays',
                  'Vermiculite': 'Clays',
                  'Quartz+Feldspar': 'Quartz+Feldspar'}


minerals_to_exclude = [0, 1, 13, 15, 20, 21, 22, 25, 28, 29, 37, 38, 40, 41, 49, 56, 57, 60, 82, 83, 94,
                       96, 97, 98, 99, 100, 105, 106, 135, 136, 182, 144, 148, 152, 194, 196, 228, 234,
                       238, 270, 271]


def tetracorder_library(mineral_index, fig_directory, group):

    if int(mineral_index) not in [0, -9999]:
        decoded_expert = tcs.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                              log_file=None, log_level='INFO')

        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                      'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
        df_mineral_matrix = df_mineral_matrix.fillna(-9999)
        record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral_index), 'Record'].iloc[0]
        filename = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Filename'].iloc[0]
        ref_library = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Library'].iloc[0]
        name = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Name'].iloc[0]

        # load library
        item = SPECTRAL_REFERENCE_LIBRARY[ref_library]
        library = envi.open(envi_header(item), item)
        library_reflectance = library.spectra.copy()
        library_records = [int(q) for q in library.metadata['record']]

        mineral_reflectance = library_reflectance[library_records.index(record), :]

        normalized_group_name = os.path.normpath(filename.split('.depth.gz')[0])
        expert_file_selection = decoded_expert[normalized_group_name]['features']
        hdr = envi.read_envi_header(envi_header(item))

        wavelengths = np.array([float(q) for q in hdr['wavelength']])

        plt.plot(wavelengths, mineral_reflectance)

        # plot the features
        for _cont_feat, cont_feat in enumerate(expert_file_selection):
            feature = cont_feat['continuum']

            left_inds = np.where(np.logical_and(wavelengths >= feature[0], wavelengths <= feature[1]))[0]
            left_x = wavelengths[int(left_inds.mean())]
            left_y_obs = mineral_reflectance[left_inds].mean()

            right_inds = np.where(np.logical_and(wavelengths >= feature[2], wavelengths <= feature[3]))[0]

            if right_inds.size == 0:
                right_inds = spectra.nearest_index_to_wavelength(wavelengths=wavelengths, target_wavelength=(feature[2] + feature[3]) / 2)  # this takes the mean of right bounds
            else:
                pass

            right_x = wavelengths[int(right_inds.mean())]
            right_y_obs = mineral_reflectance[right_inds].mean()

            feature_inds = np.logical_and(wavelengths >= feature[0], wavelengths <= feature[3])
            continuum_obs = interp1d([left_x, right_x], [left_y_obs, right_y_obs], bounds_error=False,
                                     fill_value='extrapolate')(wavelengths)

            h_x = continuum_obs - mineral_reflectance

            # Fill the areas where g(x) > f(x) and f(x) > g(x)
            abs_integral = np.trapz(h_x[feature_inds], wavelengths[feature_inds])

            plt.plot(wavelengths[feature_inds], continuum_obs[feature_inds], label=f'Area: {abs_integral:.3f}; Wvl: {left_x:.2f},{right_x:.2f}')
            plt.fill_between(wavelengths, continuum_obs, mineral_reflectance, where=feature_inds, color='green', alpha=0.3)

        plt.ylim(0,1)
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(fig_directory, f'{int(mineral_index)}.png'))
        plt.clf()
        plt.close()


def simplify_legend(handles, labels):
    unique_labels = {}
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels[label] = handles[i]

    return unique_labels


def cont_rem(wavelengths, reflectance, feature):
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


def bin_sums(x, y, nans:bool):
    mae = []
    x_vals = []
    percent_false_pos = []
    percent_false_neg = []

    for col in range(x.shape[1]):
        fraction = x[0, col]
        vals = y[:, col]
        if nans:
            vals[vals == 0] = np.nan

        mae_calc = np.mean(vals[~np.isnan(vals)])
        x_vals.append(fraction)

        mae.append(mae_calc)

    return x_vals, mae, percent_false_neg, percent_false_pos

def atmosphere_meta(atmosphere):
    basename = os.path.basename(atmosphere)
    aod = basename.split('_')[-4].replace('-', '.')
    h2o = basename.split('_')[-3].replace('-', '.')
    doy = basename.split('_')[1]

    time_dh = basename.split('_')[-6].replace('-', '.')

    hours = int(float(time_dh))
    minutes = (float(time_dh) * 60) % 60
    seconds = (float(time_dh) * 3600) % 60
    hms = "%d%02d%02d" % (hours, minutes, seconds)
    # defaults from hypertrace runs
    latitude = 34.15
    longitude = -118.14
    elevation_m = 10
    acquisition_datetime_utc = datetime.strptime('2023' + doy + hms, "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    geometry_results_emit = sunpos(acquisition_datetime_utc, latitude, longitude, elevation_m)

    return aod, h2o, np.round(geometry_results_emit[1], 2)


def standardize_cont_feature(pure_signal, mixed_singal):
    min_pure = np.min(pure_signal)
    max_pure = np.max(pure_signal)

    min_mixed = np.min(mixed_singal)
    max_mixed = np.max(mixed_singal)

    pure_normalized = (pure_signal - min_pure) / (max_pure - min_pure)
    pure_scaled = pure_normalized * (max_mixed - min_mixed) + min_mixed

    return pure_scaled


class tetracorder_figures:
    def __init__(self, base_directory: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs, legend_text):

        self.base_directory = base_directory
        self.simulation_output_directory = os.path.join(base_directory, 'simulation', 'output')
        self.slpit_output_directory = os.path.join(base_directory, 'slpit', 'output')
        self.output_directory = os.path.join(base_directory, 'tetracorder', 'output')
        self.aug_directory = os.path.join(self.output_directory, 'augmented')
        self.sim_spectra_directory = os.path.join(self.output_directory, 'simulated_spectra')
        self.output_fractions = os.path.join(self.output_directory, 'fractions')
        self.sa_outputs = os.path.join(self.output_directory, 'spectral_abundance')
        self.veg_correction_dir = os.path.join(self.output_directory, 'veg-correction')

        self.slpit_output = os.path.join(base_directory, 'slpit', 'output')
        self.fig_directory = os.path.join(base_directory, 'tetracorder', 'figures')

        #self.bands = load_band_names(
         #   os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_jabun_abs_abundance'))

        create_directory(os.path.join(self.fig_directory, 'plot_minerals'))
        self.major_axis_fontsize = major_axis_fontsize
        self.minor_axis_fontsize = minor_axis_fontsize
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.fig_height = fig_height
        self.fig_width = fig_width
        self.linewidth = linewidth
        self.sig_figs = sig_figs
        self.legend_text = legend_text

        self.wvls, self.fwhms = spectra.load_wavelengths(sensor='emit')

    def get_mineral_reclassification(self, group):
        group_1 = {2: "Iron Oxide",
                    4: "Iron Oxide",
                    8: "Iron Oxide",
                    17: "Chlorite",
                    18: "Iron Oxide",
                    19: "Chlorite",
                    28: "Iron Oxide",
                    30: "Iron Oxide",
                    31: "Iron Oxide",
                    32: "Iron Oxide",
                    43: "Iron Oxide",
                    45: "Iron Oxide",
                    47: "Iron Oxide",
                   0: "No Detection",
                   -9999: "No Data"}

        group_2 = { 125: "Clays",
                    127: "Clays",
                    129: "Clays",
                    130: "Clays",
                    131: "Clays",
                    134: "Clays",
                    139: "Clays",
                    140: "Clays",
                    141: "Clays",
                    143: "Clays",
                    150: "Clays",
                    151: "Clays",
                    166: "Clays",
                    167: "Clays",
                    168: "Clays",
                    184: "Carbonates",
                    213: "Carbonates",
                    215: "Carbonates",
                    216: "Carbonates",
                    218: "Carbonates",
                    219: "Carbonates",
                    225: "Carbonates",
                    227: "Carbonates",
                    229: "Carbonates",
                    231: "Carbonates",
                    232: "Carbonates",
                    0: "No Detection",
                   -9999: "No Data",
                    96: "Vegetation",
                    97: "Vegetation",
                    98: "Vegetation"}

        if group == 'g1':
            return group_1
        else:
            return group_2

    def error_abundance_corrected(self, spectral_abundance_array, pure_soil_array, fractions, index):

        mineral_grid_positions = {'Calcite': 0,
                                  'Dolomite': 1,
                                  'Gypsum-Fine': 2,
                                  'Gypsum-Coarse': 3,
                                  'Chlorite': 0,
                                  'Goethite-Nano': 0,
                                  'Goethite-Fine': 1,
                                  'Goethite-Med': 2,
                                  'Goethite-Large': 3,
                                  'Hematite-Nano': 4,
                                  'Hematite-Fine': 5,
                                  'Hematite-Med': 6,
                                  'Hematite-Large': 7,
                                  'Illite+Muscovite': 0,
                                  'Kaolinite': 1,
                                  'Montmorillonite': 2,
                                  'Vermiculite': 3,
                                  'Quartz+Feldspar': 0}

        # mineral group grid
        oxide_grid_sim = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 8))
        carbonate_grid_sim = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 4))
        clay_grid_sim = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 4))
        quartz_grid = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 1))
        chlorite_grid = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 1))

        for _mineral, mineral in enumerate(self.bands):
            mineral_group = mineral_groups[mineral]
            mineral_grid_position = mineral_grid_positions[mineral]

            if mineral_group == 'Fe Oxides':
                oxide_grid_sim[:, :, mineral_grid_position] = spectral_abundance_array[:,:, _mineral]
            elif mineral_group == 'Carbonates':
                carbonate_grid_sim[:, :, mineral_grid_position] = spectral_abundance_array[:, :, _mineral]
            elif mineral_group == 'Chlorite':
                chlorite_grid[:, :, mineral_grid_position] = spectral_abundance_array[:, :, _mineral]
            elif mineral_group == 'Quartz+Feldspar':
                quartz_grid[:, :, mineral_grid_position] = spectral_abundance_array[:, :, _mineral]
            else:
                clay_grid_sim[:, :, mineral_grid_position] = spectral_abundance_array[:, :, _mineral]

        # merge mineral group grids for simulated spectra
        mineral_grid_sim = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 5))

        for _grid, grid in enumerate([oxide_grid_sim, carbonate_grid_sim, clay_grid_sim, quartz_grid, chlorite_grid]):
            for _row, row in enumerate(grid):
                for _col, col in enumerate(row):

                    group = grid[_row, _col, :]
                    all_zeros = np.all(group == 0)

                    if all_zeros:
                        mineral_grid_sim[_row, _col, _grid] = 0
                    else:
                        mineral_grid_sim[_row, _col, _grid] = np.mean(grid[_row, _col, :][grid[_row, _col, :] != 0])

        # spectral abundance, third dimension is the mineral groups
        error_grid = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], 5))

        # populate error grid
        for _row, row in enumerate(fractions):
            for _col, col in enumerate(row):
                soil_fractions = fractions[_row, _col, 2]
                soil_index = index[_row, 0, 2]
                soil_spectral_abundance = pure_soil_array[int(soil_index), :]

                # calculate the pure abudance
                oxide_pure = []
                carbonate_pure = []
                clay_pure = []
                chlorite_pure = []
                quartz_pure = []

                for _mineral, mineral in enumerate(self.bands):
                    mineral_group = mineral_groups[mineral]
                    pure_sa = soil_spectral_abundance[_mineral]

                    if mineral_group == 'Fe Oxides':
                        oxide_pure.append(pure_sa)
                    elif mineral_group == 'Carbonates':
                        carbonate_pure.append(pure_sa)
                    elif mineral_group == 'Chlorite':
                        chlorite_pure.append(pure_sa)
                    elif mineral_group == 'Quartz+Feldspar':
                        quartz_pure.append(pure_sa)
                    else:
                        clay_pure.append(pure_sa)

                pure_abundace_by_group = []
                for _pure, pure in enumerate([oxide_pure, carbonate_pure, clay_pure, quartz_pure, chlorite_pure]):
                    tmp_array = np.array(pure)
                    all_zeros = np.all(tmp_array == 0)

                    if all_zeros:
                        pure_abundace_by_group.append(0)
                    else:
                        pure_abundace_by_group.append(np.mean(tmp_array[tmp_array != 0]))

                pure_sa_grouped = np.array(pure_abundace_by_group)

                sa_c = mineral_grid_sim[_row, _col, :] #/ np.round(soil_fractions, 2)
                error = np.absolute(sa_c - pure_sa_grouped)
                error_grid[_row, _col, :] = error

        return error_grid


    def mineral_validation(self, x_axis:str):

        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.json")))
        df['Team'] = df['Name'].str.split('-').str[0].str.strip()
        df = df[df['Team'] != 'THERM']
        df = df.sort_values('Name')
        rows_estimated = []
        rows_truth = []
        rows_soil_fractions = []

        if x_axis == 'contact':
            file_kw = '-emit_ems_'
        else:
            file_kw = '_transect_'

        for index, row in df.iterrows():
            plot = row['Name']
            emit_filetime = row['EMIT DATE']
            abundance_emit = glob(os.path.join(self.sa_outputs,
                                               f'*{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_jabun_rel_abundance'))
            grain_size_emit = glob(os.path.join(self.sa_outputs,
                                               f'*{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_grain'))

            abundance_contact_probe = os.path.join(self.sa_outputs,
                                                   f'{plot.replace(" ", "").replace("SPEC", "Spectral") + file_kw}augmented_jabun_rel_abundance')
            grain_size_slpit = os.path.join(self.sa_outputs,
                                                   f'{plot.replace(" ", "").replace("SPEC", "Spectral") + file_kw}augmented_grain')

            fractional_cover = os.path.join(self.slpit_output, 'sma', f'asd-local___{plot.replace(" ", "")}___num-endmembers_20_n-mc_25_normalization_brightness_fractional_cover')

            # load arrays
            truth_array = envi_to_array(abundance_contact_probe)[0, 0, :]
            estimated_array = envi_to_array(abundance_emit[0])

            grain_emit_array = envi_to_array(grain_size_emit[0])[0, 0, -1] * 0.08043
            grain_slpit_array = envi_to_array(grain_size_slpit)[0, 0, -1] * 0.08043

            # load fractional cover
            plot_fractional_cover = np.average(envi_to_array(fractional_cover)[:, :, 2])
            plot_soils = [plot] + [plot_fractional_cover]

            oxides_emit = []
            clays_emit = []
            carbonates_emit = []
            chlorite_emit = []
            q_f_emit = []

            oxides_slpit = []
            clays_slpit = []
            carbonates_slpit = []
            chlorite_slpit = []
            q_f_slpit = []

            for _mineral, mineral in enumerate(self.bands):
                truth_abun = truth_array[_mineral]
                mineral_group = mineral_groups[mineral]

                if mineral_group == 'Fe Oxides':
                    oxides_emit.append(np.mean(estimated_array[0, 0, _mineral]))
                    oxides_slpit.append(truth_abun)
                elif mineral_group == 'Carbonates':
                    carbonates_emit.append(np.mean(estimated_array[0, 0, _mineral]))
                    carbonates_slpit.append(truth_abun)
                elif mineral_group == 'Chlorite':
                    chlorite_emit.append(np.mean(estimated_array[0, 0, _mineral]))
                    chlorite_slpit.append(truth_abun)
                elif mineral_group == 'Quartz+Feldspar':
                    q_f_emit.append(np.mean(estimated_array[0, 0, _mineral]))
                    q_f_slpit.append(truth_abun)
                else:
                    clays_emit.append(np.mean(estimated_array[0, 0, _mineral]))
                    clays_slpit.append(truth_abun)

            row_emit = []
            row_slpit = []

            # clean emit rows
            for i in [oxides_emit, clays_emit, carbonates_emit, chlorite_emit, q_f_emit]:
                tmp_array = np.array([i])
                if np.all(tmp_array == 0):
                    row_emit.append(0)
                else:
                    tmp_array = np.where(tmp_array == 0, np.nan, tmp_array)
                    row_emit.append(np.nanmean(tmp_array))

            # clean slpit rows
            for i in [oxides_slpit, clays_slpit, carbonates_slpit, chlorite_slpit, q_f_slpit]:
                tmp_array = np.array([i])
                if np.all(tmp_array == 0):
                    row_slpit.append(0)
                else:
                    tmp_array = np.where(tmp_array == 0, np.nan, tmp_array)
                    row_slpit.append(np.nanmean(tmp_array))

            # append plot abundances to master rows
            rows_estimated.append([plot] + row_emit + [grain_emit_array])
            rows_truth.append([plot] + row_slpit + [grain_slpit_array])
            rows_soil_fractions.append(plot_soils)

        # master dfs
        df_est = pd.DataFrame(rows_estimated)
        df_est.columns = ['Plot'] + ['Iron Oxides_emit', 'Carbonates_emit', 'Clays_emit', 'Chlorite_emit', 'Quartz+Feldspar_emit', 'Grain Size_emit']
        df_est = df_est.fillna(0)

        df_truth = pd.DataFrame(rows_truth)
        df_truth.columns = ['Plot'] + ['Iron Oxides_slpit', 'Carbonates_slpit', 'Clays_slpit', 'Chlorite_slpit', 'Quartz+Feldspar_slpit', 'Grain Size_slpit']
        df_truth = df_truth.fillna(0)

        df_merge = pd.merge(df_est, df_truth, on='Plot')

        df_soil = pd.DataFrame(rows_soil_fractions)

        df_soil.columns = ['Plot', 'soil_frac']
        df_merge = pd.merge(df_merge, df_soil, on='Plot')
        df_merge = df_merge[df_merge != -9999].dropna()
        df_merge.to_csv(os.path.join(self.fig_directory, f'slpit-emit_{x_axis}_estimated_abundance.csv'), index=False)

        # create figure
        fig = plt.figure(figsize=(12, 8))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.75, hspace=0, figure=fig)
        minor_tick_spacing = 0.1
        major_tick_spacing = 0.2
        counter = 0

        plot_titles = {0: 'Iron Oxides', 1: 'Carbonates', 2: 'Clays', 3: 'Chlorite', 4: 'Quartz+Feldspar', 5: 'Grain Size'}
        plot_lims = {0: (0, 0.03), 1: (0, 0.2), 2: (0, 1), 3:  (0, 1), 4: (0,1), 5: (0, 30)}
        plot_ticks = {0: (0.005, 0.01), 1: (0.05, 0.1), 2: (minor_tick_spacing, major_tick_spacing),
                      3: (minor_tick_spacing, major_tick_spacing), 4: (minor_tick_spacing, major_tick_spacing), 5: (2.5, 5)}

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(plot_titles[counter])
                ax.set_xlabel(f'SLPIT')
                #ax.grid('on', linestyle='--')

                ax.xaxis.set_minor_locator(ticker.MultipleLocator(plot_ticks[counter][0]))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(plot_ticks[counter][1]))
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                #ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                ax.yaxis.set_minor_locator(ticker.MultipleLocator(plot_ticks[counter][0]))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(plot_ticks[counter][1]))
                #ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                ax.set_xlim(plot_lims[counter][0], plot_lims[counter][1])
                ax.set_ylim(plot_lims[counter][0], plot_lims[counter][1])

                if col == 0:
                    ax.set_ylabel('EMIT Spectral\nAbundance')

                # if col != 0:
                #     ax.set_yticklabels([])

                x = df_merge[f'{plot_titles[counter]}_slpit'].values
                y = df_merge[f'{plot_titles[counter]}_emit'].values
                frac = df_merge[f'soil_frac'].values

                df_no_detect = pd.DataFrame({'x': x, 'y': y, 'soil_frac': frac})
                df_no_detect = df_no_detect[(df_no_detect['x'] != 0) | (df_no_detect['y'] != 0)]

                x = df_no_detect['x'].values
                y = df_no_detect['y'].values

                if len (x) != 0 and len(y) != 0:
                    p = ax.scatter(x, y, c=df_no_detect['soil_frac'], cmap='viridis', s=8)
                    rmse = mean_squared_error(x, y, squared=False)
                    mae = mean_absolute_error(x, y)
                    r2 = r2_calculations(x, y)

                    # plot 1 to 1 line
                    one_line = np.linspace(0, plot_lims[counter][1], 101)
                    ax.plot(one_line, one_line, color='red')

                    txtstr = '\n'.join((
                        r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                        r'R$^2$: %.2f' % (r2,),
                        r'n = ' + str(len(x))))

                else:
                    txtstr = 'no points to plot!'

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=6,
                        verticalalignment='top', bbox=props)
                ax.set_aspect('equal', adjustable='box')
                counter += 1

                if col == 3:
                    color_bar_label = 'SLPIT\nSoil Fraction (%)'
                else:
                    color_bar_label = ''
                cbar = fig.colorbar(p, ax=ax, orientation='vertical', label=color_bar_label, fraction=0.05)
                cbar.ax.yaxis.set_tick_params(labelsize=6)

        plt.savefig(os.path.join(self.fig_directory, f'mineral_validation_{x_axis}.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    def mineral_threshold(self):
        # import csvs with abundance estimates
        df_contact = pd.read_csv(os.path.join(self.fig_directory, f'slpit-emit_contact_estimated_abundance.csv'))
        df_contact['mode'] = 'contact'

        df_transect = pd.read_csv(os.path.join(self.fig_directory, f'slpit-emit_transect_estimated_abundance.csv'))
        df_transect['mode'] = 'transect'

        # # create figure
        fig = plt.figure(figsize=(12, 6))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.4, figure=fig)
        minor_tick_spacing = 0.1
        major_tick_spacing = 0.2

        plot_titles = {
            0: 'Iron Oxides',
            1: 'Carbonates',
            2: 'Clays',
            3: 'Chlorite',
            4: 'Quartz+Feldspar',
            5: 'Grain Size'}
        plot_lims = {
            0: (0, 0.1),
            1: (0, 0.35),
            2: (0, 0.75),
            3: (0, 0.35),
            4: (0, 0.75),
            5: (0, 50)}
        plot_ticks = {
            0: (0.005, 0.01),
            1: (0.05, 0.1),
            2: (minor_tick_spacing, major_tick_spacing),
            3: (minor_tick_spacing, major_tick_spacing),
            4: (minor_tick_spacing, major_tick_spacing),
            5: (2.5, 5)}
        counter = 0

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(plot_titles[counter])
                if row == 1:
                    ax.set_xlabel(f'Soil Fractions\n (SLPIT)')
                ax.grid('on', linestyle='--')
                #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
                #ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                #ax.yaxis.set_minor_locator(ticker.MultipleLocator(plot_ticks[counter][0]))
                #ax.yaxis.set_major_locator(ticker.MultipleLocator(plot_ticks[counter][1]))
                #ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))
                ax.set_xlim(0, 1)
                ax.set_ylim(plot_lims[counter][0], plot_lims[counter][1])

                if col == 0:
                    ax.set_ylabel('Absolute Abundance Error\n (SLPIT- EMIT)')

                # if col != 0:
                #     ax.set_yticklabels([])

                for df in [df_contact, df_transect]:
                    x = df[f'{plot_titles[counter]}_slpit'].values
                    y = df[f'{plot_titles[counter]}_emit'].values

                    frac = df[f'soil_frac'].values

                    df_no_detect = pd.DataFrame({'x': x, 'y': y, 'soil_frac': frac})
                    df_no_detect = df_no_detect[(df_no_detect['x'] != 0) | (df_no_detect['y'] != 0)]

                    df_no_detect['error'] = df_no_detect['x'] - df_no_detect['y']
                    df_no_detect['error'] = df_no_detect['error'].abs()
                    df_no_detect = df_no_detect.sort_values('soil_frac')

                    mode = list(df['mode'].unique())[0]

                    if mode == 'contact':
                        marker='s'
                        label = 'Contact Probe'
                    else:
                        marker='^'
                        label = 'Bare Fiber'

                    ax.scatter(df_no_detect['soil_frac'], df_no_detect['error'], edgecolors='black', marker=marker, s=8, label=label)
                # ax.set_aspect('equal', adjustable='box')
                counter += 1

                if col == 2:
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.savefig(os.path.join(self.fig_directory, f'mineral_threshold.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    def tetracorder_libraries(self):

        # plot summary - merged
        merger = PdfMerger()

        # TODO: Get these from....direct input?  Configuration file?
        MINERAL_FRACTION_FILES = ['calcite.group2.txt',
                                  'chlorite.group2.txt',
                                  'dolomite.group2.txt',
                                  'goethite-all-for-reference.group1.txt',
                                  'gypsum.group2.txt',
                                  'hematite-all-for-reference.group1.txt',
                                  'illite+muscovite.group2.txt',
                                  'kaolinite.group2.txt',
                                  'montmorillonite.group2.txt',
                                  'vermiculite.group2.txt',
                                  ]

        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                       'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

        decoded_expert = tetracorder.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                          log_file=None, log_level='INFO')

        mff = [os.path.join('utils', 'tetracorder', 'minerals', x) for x in MINERAL_FRACTION_FILES]
        mineral_fractions = tetracorder.read_mineral_fractions(mff)
        unique_file_names, fractions, scaling, library_names, records, reference_band_depths = unique_file_fractions(
            mineral_fractions, decoded_expert)

        df_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
        spectral_reference_library_files = SPECTRAL_REFERENCE_LIBRARY
        libraries = {}

        transect_data = pd.read_csv(os.path.join(self.slpit_output, 'all-transect-emit.csv'))
        transect_data['Team'] = transect_data['plot_name'].str.split('-').str[0].str.strip()
        transect_data = transect_data[transect_data['Team'] != 'Thermal']

        emit_detections = []
        slpit_detections = []

        for plot in sorted(list(transect_data.plot_name.unique())):
            slpit_ems_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "") +
                                                  '*emit_ems_augmented_min'))

            slpit_ems_spectra = glob(os.path.join(self.aug_directory, '*' + plot.replace(" ", "") +
                                                  '*emit_ems_augmented'))

            emit_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
                                                        '*pixels_augmented_min'))

            emit_window_spectra = glob(os.path.join(self.aug_directory, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
                                                        '*pixels_augmented'))

            mineral_records = []
            mineral_records_emit = []

            if int(envi_to_array(slpit_ems_records[0])[0, 0, 1]) != 0:
                g1_em_records = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 1]), 'Record'].iloc[0]
                mineral_records.append(g1_em_records)
            else:
                slpit_detections.append('none')

            if int(envi_to_array(slpit_ems_records[0])[0, 0, 3]) != 0:
                g2_em_records = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 3]), 'Record'].iloc[0]
                mineral_records.append(g2_em_records)
            else:
                slpit_detections.append('none')

            if int(envi_to_array(emit_records[0])[0, 0, 1]) != 0:
                g1_em_records = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(emit_records[0])[0, 0, 1]), 'Record'].iloc[0]
                mineral_records_emit.append(g1_em_records)
            else:
                emit_detections.append('none')

            if int(envi_to_array(emit_records[0])[0, 0, 3]) != 0:
                g2_em_records = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(emit_records[0])[0, 0, 3]), 'Record'].iloc[0]
                mineral_records_emit.append(g2_em_records)
            else:
                emit_detections.append('none')

            plot_spectra = envi_to_array(slpit_ems_spectra[0])[0, 0, :]
            emit_spectra = envi_to_array(emit_window_spectra[0])[0, 0, :]

            # set up the figure
            fig = plt.figure(figsize=(15, 8))
            gs = gridspec.GridSpec(2, 4, figure=fig)
            map = fig.add_subplot(gs[0, 0])
            ls = fig.add_subplot(gs[0, 1])
            g1_s = fig.add_subplot(gs[0, 2])
            g2_s = fig.add_subplot(gs[0, 3])
            fs = fig.add_subplot(gs[1, :2])
            g1_e = fig.add_subplot(gs[1, 2])
            g2_e = fig.add_subplot(gs[1, 3])


            g1_s.set_title('Continuum Removed Group 1 - SLPIT', fontsize=10)
            g2_s.set_title('Continuum Removed Group 2 - SLPIT', fontsize=10)

            g1_e.set_title('Continuum Removed Group 1 - EMIT', fontsize=10)
            g2_e.set_title('Continuum Removed Group 2 - EMIT', fontsize=10)

            # plot picture
            fig.suptitle(plot, size=16)
            ls.set_title('Landscape\nPicture')
            pic_path = os.path.join(self.slpit_output, 'plot_pictures', 'spectral_transects', plot + '.jpg')
            img = mpimg.imread(pic_path)
            ls.imshow(img)
            ls.axis('off')

            # plot spectra
            emit_wvls, fwhm = spectra.load_wavelengths(sensor='emit')
            good_emit_bands = spectra.get_good_bands_mask(emit_wvls, wavelength_pairs=None)
            emit_wvls[~good_emit_bands] = np.nan
            emit_wvls = emit_wvls/1000

            fs.set_title('Full Spectrum')
            fs.plot(emit_wvls, plot_spectra, label='SLPIT', c='blue')
            fs.plot(emit_wvls, emit_spectra, label='EMIT', c='orange')
            fs.set_ylim(0,1)
            fs.set_ylabel('Reflectance')
            fs.set_xlabel('Wavelengths (µm)')
            fs.legend()

            # plot map
            map.set_title('Plot Map')
            df_transect = transect_data.loc[transect_data['plot_name'] == plot].copy()
            df_transect = df_transect[df_transect.longitude != 'unk']
            m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=45,
                        llcrnrlon=-125, urcrnrlon=-100, ax=map, epsg=4326)
            m.arcgisimage(service='World_Imagery', xpixels=1000, ypixels=1000, dpi=300, verbose=True)
            map.scatter(np.mean(df_transect.longitude), np.mean(df_transect.latitude), color='red', s=12)

            for key, item in spectral_reference_library_files.items():
                library = envi.open(envi_header(item), item)
                library_reflectance = library.spectra.copy()
                library_records = [int(q) for q in library.metadata['record']]

                hdr = envi.read_envi_header(envi_header(item))
                wavelengths = np.array([float(q) for q in hdr['wavelength']])

                if ';;;' in key:
                    key = key.replace(';;;', ',')
                    logging.debug(f'found comma replacement, now: {key}')

                libraries[key] = {'reflectance': library_reflectance,
                                  'library_records': library_records, 'wavelengths': wavelengths}

                df_rows = []
                for _f, (frac, filename, library_name, record) in enumerate(zip(fractions, unique_file_names, library_names.tolist(), records.tolist())):
                    df_rows.append([_f, frac, filename, library_name, record])

                df_lib = pd.DataFrame(df_rows)
                df_lib.columns = ['_frac_index', 'fractions', 'filename', 'library_names', 'records']

                plotted_slipit_library_reference = []

                # plot data
                for _record, slpit_record in enumerate(mineral_records):
                    if slpit_record not in list(df_lib.records.unique()):
                        continue
                    library_name = df_lib.loc[df_lib['records'] == slpit_record, 'library_names'].iloc[0]

                    if library_name == key:
                        filename = df_lib.loc[df_lib['records'] == slpit_record, 'filename'].iloc[0]

                        file_label = filename.split('.depth.gz')[0].replace('/', '\\').split(os.sep)[1]
                        group = filename.split('.depth.gz')[0].replace('/', '\\').split(os.sep)[0]

                        # plot the data
                        for cont_feat in decoded_expert[filename.split('.depth.gz')[0].replace('/', '\\')]['features']:
                            if group == 'group.1um':
                                cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(slpit_record), :],
                                                    cont_feat['continuum'])
                                split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                                emit_cont, ewvls = cont_rem(wavelengths, emit_spectra, cont_feat['continuum'])

                                g1_s.plot(wl, standardize_cont_feature(pure_signal=cont, mixed_singal=split_cont), label=f'{file_label}', c='black', linestyle='dotted')
                                g1_s.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                #g1_s.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

                            if group == 'group.2um':
                                split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                                cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(slpit_record), :], cont_feat['continuum'])
                                emit_cont, ewvls = cont_rem(wavelengths, emit_spectra, cont_feat['continuum'])

                                g2_s.plot(wl, standardize_cont_feature(pure_signal=cont, mixed_singal=split_cont), label=f'{file_label}', c='black', linestyle='dotted')
                                g2_s.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                #g2_s.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

                # plot EMIT data
                for _record, emit_record in enumerate(mineral_records_emit):
                    if emit_record not in list(df_lib.records.unique()):
                        continue
                    library_name = df_lib.loc[df_lib['records'] == emit_record, 'library_names'].iloc[0]

                    if library_name == key:
                        filename = df_lib.loc[df_lib['records'] == emit_record, 'filename'].iloc[0]

                        file_label = filename.split('.depth.gz')[0].replace('/', '\\').split(os.sep)[1]
                        group = filename.split('.depth.gz')[0].replace('/', '\\').split(os.sep)[0]

                        for cont_feat in decoded_expert[filename.split('.depth.gz')[0].replace('/', '\\')]['features']:

                            if group == 'group.1um':
                                cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(emit_record), :], cont_feat['continuum'])
                                split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                                emit_cont, ewvls = cont_rem(wavelengths, emit_spectra, cont_feat['continuum'])

                                g1_e.plot(wl, standardize_cont_feature(pure_signal=cont, mixed_singal=emit_cont), label=f'{file_label}', c='black', linestyle='dotted')
                                #g1_e.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                g1_e.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

                            if group == 'group.2um':
                                cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(emit_record), :], cont_feat['continuum'])
                                split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                                emit_cont, ewvls = cont_rem(wavelengths, emit_spectra, cont_feat['continuum'])

                                g2_e.plot(wl, standardize_cont_feature(pure_signal=cont, mixed_singal=emit_cont), label=f'{file_label}', c='black', linestyle='dotted')
                                #g2_e.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                g2_e.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

            for ax in [g1_s, g2_s, g1_e, g2_e]:
                handles, labels = ax.get_legend_handles_labels()
                unique_labels = simplify_legend(handles, labels)
                ax.legend(unique_labels.values(), unique_labels.keys(), prop={'size': 6})

            plt.savefig(os.path.join(self.fig_directory, 'plot_minerals', plot + '.png'), format="png", dpi=300,
                        bbox_inches="tight")
            plt.savefig(os.path.join(self.fig_directory, 'plot_minerals', plot + '.pdf'), format="pdf", dpi=300,
                        bbox_inches="tight")
            plt.clf()
            plt.close()
            merger.append(os.path.join(self.fig_directory, 'plot_minerals', plot + '.pdf'))

        # write pdf
        merger.write(os.path.join(self.fig_directory, 'plot_minerals', 'plot_summary.pdf'))
        merger.close()

    def confusion_matrix(self, threshold=None, frac_cover=False):
        df_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
        transect_data = pd.read_csv(os.path.join(self.slpit_output, 'all-transect-emit.csv'))
        transect_data['Team'] = transect_data['plot_name'].str.split('-').str[0].str.strip()
        transect_data = transect_data[transect_data['Team'] != 'Thermal']

        emit_detections = []
        slpit_detections = []

        for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):

            slpit_ems_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "") +
                                                  '*emit_ems_augmented_min'))

            emit_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
                                             '*pixels_augmented_min'))

            fractional_cover = os.path.join(self.slpit_output, 'sma', f'asd-local___{plot.replace(" ", "").replace("Spectral", "SPEC")}___num-endmembers_20_n-mc_25_normalization_brightness_fractional_cover')

            # if frac_cover:
            #     soil_frac_cover = np.mean(envi_to_array(fractional_cover)[:,:,2])
            #     if soil_frac_cover > threshold:
            #         continue

            # slpit data - g1
            if int(envi_to_array(slpit_ems_records[0])[0, 0, 1]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 1]), 'Filename'].iloc[0]
                slpit_detections.append(mineral_groupings[filename])
            else:
                slpit_detections.append('N.D.')

            # slpit data - g2
            if int(envi_to_array(slpit_ems_records[0])[0, 0, 3]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 3]), 'Filename'].iloc[0]
                slpit_detections.append(mineral_groupings[filename])
            else:
                slpit_detections.append('N.D.')

            # emit data - g1
            if int(envi_to_array(emit_records[0])[0, 0, 1]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(emit_records[0])[0, 0, 1]), 'Filename'].iloc[0]
                emit_detections.append(mineral_groupings[filename])
            else:
                emit_detections.append('N.D.')

            if int(envi_to_array(emit_records[0])[0, 0, 3]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(emit_records[0])[0, 0, 3]), 'Filename'].iloc[0]
                emit_detections.append(mineral_groupings[filename])

            else:
                emit_detections.append('N.D.')

        plot_list = sorted(list(transect_data.plot_name.unique()), reverse=True)

        # save df
        df = pd.DataFrame({'emit' : emit_detections,
                           'slpit': slpit_detections,
                           'plot' : list(np.repeat(plot_list, 2))} ,
                          columns=['emit', 'slpit', 'plot'])

        df.to_csv(os.path.join(self.fig_directory, 'mineral_report.csv'), index=False)

        # write confusion matrix
        label_encoder = LabelEncoder()
        label_encoder.fit(slpit_detections + emit_detections)

        actual_values_encoded = label_encoder.transform(slpit_detections)
        predicted_values_encoded = label_encoder.transform(emit_detections)

        # Get the class labels
        class_labels = label_encoder.classes_
        cm = metrics.confusion_matrix(predicted_values_encoded, actual_values_encoded)

        plt.figure(figsize=(12, 6))
        plt.grid(True)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
        plt.colorbar()

        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=90)
        plt.yticks(tick_marks, class_labels)

        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white', size=12)

        plt.ylabel('EMIT')
        plt.xlabel('SLPIT')
        plt.tight_layout()

        plt.savefig(os.path.join(self.fig_directory, 'mineral_confusion_matrix.png'), format="png", dpi=300,
                        bbox_inches="tight")

    def confusion_matrices(self):
        group_dict = {'g1': 1, 'g2': 3}

        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))

        for group in ['g1', 'g2']:
            fractions = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_fractions'))

            unique_x_val = np.unique(fractions[0, :, 2]) * 100

            # this is soil from tetracorder output - this is the absolute truth
            bd_tetra = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 20, group_dict[group]]

            # this is sim spectra from tetracorder output w/out corrections
            bd_tetra_sim = envi_to_array(os.path.join(self.sa_outputs,
                                                      f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, group_dict[group]]

            truth_array = np.zeros((fractions.shape[0], fractions.shape[1]))
            truth_array[:] = bd_tetra[:, np.newaxis]
            truth_array[:, 0] = 0

            # make figure
            fig, ax = plt.subplots(1,1, figsize=(20, 20))

            # Flatten the 2D arrays for comparison
            true_flat = truth_array.flatten().astype(int).flatten()
            predicted_flat = bd_tetra_sim.flatten().astype(int).flatten()

            # Get unique values from both true and predicted arrays
            true_labels = np.unique(true_flat)
            predicted_labels = np.unique(predicted_flat)

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(true_flat, predicted_flat, labels=true_labels)

            # get label keys for minerals
            filtered_minerals_df = df_mineral_matrix[df_mineral_matrix['Index'].isin(true_labels)]
            mineral_mapping = dict(zip(filtered_minerals_df['Index'], filtered_minerals_df['Name']))
            present_minerals_result = [mineral_mapping[value] for value in true_labels if value in mineral_mapping]
            present_minerals_result.insert(0, 'No Data')

            # Plot the confusion matrix on the specific axis
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', ax=ax,
                        annot_kws={"size": 14}, cbar=False, xticklabels=present_minerals_result,
                        yticklabels=present_minerals_result)

            # Set axis labels and title
            ax.set_xlabel(f'Predicted')
            ax.set_ylabel(f'True')
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            # Add red outlines around the diagonal elements
            for k in range(min(conf_matrix.shape)):  # Assuming it's square or nearly square
                rect = patches.Rectangle((k, k), 1, 1, fill=False, edgecolor='red', lw=2)
                ax.add_patch(rect)

            # Adjust layout to prevent overlapping
            #plt.tight_layout()
            plt.subplots_adjust(top=0.96, right=0.90)
            plt.savefig(os.path.join(self.fig_directory, f"{group}_confusion_matrix.png"),  bbox_inches='tight')
            plt.clf()
            plt.close()

            # aggregated confusion matrix
            mineral_class = self.get_mineral_reclassification(group=group)

            truth_category_array = np.full(truth_array.shape, 'Other', dtype=object)
            for value, category in mineral_class.items():
                truth_category_array[truth_array == value] = category

            simulated_category_array = np.full(bd_tetra_sim.shape, 'Other', dtype=object)
            for value, category in mineral_class.items():
                simulated_category_array[bd_tetra_sim == value] = category

            # Flatten arrays to use in confusion_matrix
            a_flat = truth_category_array.flatten()
            b_flat = simulated_category_array.flatten()

            # Generate confusion matrix
            labels = sorted(set(a_flat) | set(b_flat))  # Ensures all labels appear

            cm = confusion_matrix(y_pred=b_flat, y_true=a_flat, labels=labels)

            # Convert to DataFrame for display
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f"{group}_confusion_matrix_aggregated.png"), bbox_inches='tight')
            plt.clf()
            plt.close()


    def confusion_matrix_detailed(self):
        group_dict = {'g1': 1, 'g2': 3}
        bd_group_dict = {'g1': 0, 'g2': 2}

        for group in ['g1', 'g2']:
            fractions = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_fractions'))[:,:, 2]

            # this is soil from tetracorder output - this is the absolute truth
            bd_tetra = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 20, group_dict[group]]
            bd_tetra_bd = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 20, bd_group_dict[group]]

            # this is sim spectra from tetracorder output w/out corrections
            bd_tetra_sim = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, group_dict[group]]
            bd_tetra_sim_bd = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, bd_group_dict[group]]

            truth_array = np.zeros((fractions.shape[0], fractions.shape[1]))
            truth_array[:] = bd_tetra[:, np.newaxis]
            truth_array[:, 0] = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, 0, group_dict[group]]

            #truth_array[:, 0] = 0

            truth_bd_array = np.zeros((fractions.shape[0], fractions.shape[1]))
            truth_bd_array[:] = bd_tetra_bd[:, np.newaxis]
            #truth_bd_array[:, 0] = 0

            error = np.absolute(truth_bd_array - bd_tetra_sim_bd)

            # aggregated confusion matrix
            mineral_class = self.get_mineral_reclassification(group=group)
            truth_category_array = np.full(truth_array.shape, 'Other', dtype=object)

            # this is our aggregated ararys
            for value, category in mineral_class.items():
                truth_category_array[truth_array == value] = category

            simulated_category_array = np.full(bd_tetra_sim.shape, 'Other', dtype=object)
            for value, category in mineral_class.items():
                simulated_category_array[bd_tetra_sim == value] = category

            # Flatten arrays to use in confusion_matrix
            a_flat = truth_category_array.flatten()
            b_flat = simulated_category_array.flatten()
            fractions_flat = fractions.flatten()
            error_flat = error.flatten()

            # Generate confusion matrix
            labels = sorted(set(a_flat) | set(b_flat))  # Ensures all labels appear

            # Precompute maximum y value across all bins
            bins = np.arange(0, 1.05, 0.05)  # Bins: 0, 0.05, ..., 1.0
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # create figure
            fig = plt.figure(constrained_layout=True, figsize=(12, 12))
            ncols = len(labels)
            nrows = len(labels)
            gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.05, figure=fig)

            # create figures
            for _row, truth_label in enumerate(labels):
                for _col, predicted_label in enumerate(labels):
                    ax = fig.add_subplot(gs[_row, _col])

                    # Tick formatting
                    if _row == nrows - 1:
                        ax.set_xlabel(f"X-Axis: Soil Fraction\nPred: {predicted_label}", fontsize=8)
                    else:
                        ax.set_xticklabels([])

                    if _col == 0:
                        ax.set_ylabel(f"Truth: {truth_label}", fontsize=8)
                    else:
                        pass
                        #ax.set_yticklabels([])

                    truth_mask = a_flat == truth_label
                    predicted_mask = b_flat == predicted_label
                    combined_mask = truth_mask & predicted_mask

                    #ax.scatter(fractions_flat[combined_mask], error_flat[combined_mask], s=5, color='black')

                    data = fractions_flat[combined_mask]
                    n = len(data)

                    ax.text(0.15, 0.95,  f'n = {n}', ha='center', va='top', fontsize=8)
                    ax.set_xlim(0, 1)
                    #ax.set_ylim(0, 1)

                    if n > 0:
                        #counts, edges = np.histogram(data, bins=bins)
                        #scaled_counts = counts / counts.max() if counts.max() > 0 else counts
                        ax.hist(data, bins=bins, color='black', alpha=0.7, edgecolor='white')
                        #ax.plot(bin_centers, scaled_counts, color='black', linewidth=1.5, marker='o',markersize=3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f"{group}_confusion_matrix_aggregated_detailed.png"), bbox_inches='tight')
            plt.clf()
            plt.close()

            # create figure
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * len(labels), 4),constrained_layout=True)

            for ax, label in zip(axes, labels):
                if label in ['Other', 'No Detection']:
                    continue

                mask = (a_flat == label) & (b_flat == label)
                fractions_in_mask = fractions_flat[mask]
                error_in_mask = error_flat[mask]
                n = len(fractions_in_mask)

                box_data = []
                unique_fractions = np.sort(np.unique(fractions_in_mask))
                mae = []

                for fraction_bin in unique_fractions:
                    fraction_bin_mask = fractions_in_mask == fraction_bin
                    error_bin = error_in_mask[fraction_bin_mask]
                    mae_bin = np.nanmean(np.abs(error_bin))

                    if error_bin.size > 0:
                        box_data.append(error_bin)
                        mae.append(mae_bin)
                    else:
                        box_data.append([np.nan])  # Keep alignment even if empty (unlikely)
                        mae.append([np.nan])

                ax.boxplot(
                    box_data,
                    positions=unique_fractions,
                    widths=0.025,
                    showfliers=True,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightgray', color='black'),
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor='black')
                )

                ax.plot(unique_fractions, mae, color='blue', linewidth=1.5, label='MAE')

                # ✅ Labels and limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Fraction')
                ax.set_ylabel('BD Absolute Error')
                ax.set_title(f'Truth = Pred = {label}\nn = {n}')

                # ✅ Format x-ticks with two decimals
                ax.set_xticks(unique_fractions)
                ax.set_xticklabels([f'{x:.2f}' for x in unique_fractions], rotation=90)

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f"{group}_confusion_matrix_aggregated_diagonal.png"), bbox_inches='tight')
            plt.clf()
            plt.close()



    def veg_correction_fig(self):
        group_dict = {'g1': 0, 'g2': 2}
        bd_group_dict = {
            'g1': 1,
            'g2': 3}

        for group in ['g1', 'g2']:
            fractions = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_fractions'))[:, :, 2]

            # this is simulated spectra corrections
            bds = envi_to_array(os.path.join(self.veg_correction_dir, f'tetracorder_{group}_simulated_spectra'))
            bds[bds == -9999.] = np.nan

            bd_library = bds[:, :, 2]
            bd = bds[:, :, 3]
            bd_prime = bds[:, :, 4]
            bd_prime_prime = bds[:, :, 5]

            # this is for unmixed simulated spectra - Bd '
            bd_emc2 = envi_to_array(os.path.join(self.veg_correction_dir, f'tetracorder_{group}_emc2_spectra'))
            bd_emc2[bd_emc2 == -9999.] = np.nan
            bd_emc2_prime = bd_emc2[:, :, 4]

            # this is for unmixed simulated spectra - Bd''
            bd_emc2_prime_prime = bd_emc2[:, :, 5]

            # load correct mineral IDs @ 100 % soil
            true_mineral_id = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 20, bd_group_dict[group]]
            true_mineral_ids = np.zeros((fractions.shape[0], fractions.shape[1]))
            true_mineral_ids[:] = true_mineral_id[:, np.newaxis]
            true_mineral_ids[:, 0] = 0

            # load simulated spectra mineral IDs
            sim_mineral_detections = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, bd_group_dict[group]]
            bd_from_tetracorder = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, group_dict[group]]

            true_bd = bd[:, 20] # bd_from_tetracorder[:, 20]
            truth_array = np.zeros((fractions.shape[0], fractions.shape[1]))
            truth_array[:] = true_bd[:, np.newaxis]
            truth_array[:, 0] = 0

            # aggregated ids
            mineral_class = self.get_mineral_reclassification(group=group)
            truth_category_array = np.full(fractions.shape, 'Other', dtype=object)

            for value, category in mineral_class.items():
                truth_category_array[true_mineral_ids == value] = category

            simulated_category_array = np.full(fractions.shape, 'Other', dtype=object)
            for value, category in mineral_class.items():
                simulated_category_array[sim_mineral_detections == value] = category

            # flatten arrays
            a_flat = truth_category_array.flatten()
            b_flat = simulated_category_array.flatten()
            fractions_flat = fractions.flatten()
            bd_flat = bd.flatten()
            bd_prime_flat = bd_prime.flatten()
            bd_prime_prime_flat = bd_prime_prime.flatten()
            bd_emc2_prime_flat = bd_emc2_prime.flatten()
            bd_emc2_prime_prime_flat = bd_emc2_prime_prime.flatten()
            bd_from_tetracorder_flat = bd_from_tetracorder.flatten()
            truth_array_flat = truth_array.flatten()

            # Generate confusion matrix
            labels = sorted(set(a_flat) | set(b_flat))  # Ensures all labels appear

            # create figure
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 * len(labels), 4), constrained_layout=True)

            for ax, label in zip(axes, labels):
                if label in ['Other', 'No Detection']:
                    continue

                mask = (a_flat == label) & (b_flat == label)
                fractions_in_mask = fractions_flat[mask]

                bd_in_mask = bd_flat[mask]
                bd_prime_in_mask = bd_prime_flat[mask]
                bd_prime_prime_in_mask = bd_prime_prime_flat[mask]
                bd_emc2_prime_in_mask = bd_emc2_prime_flat[mask]
                bd_emc2_prime_prime_in_mask = bd_emc2_prime_prime_flat[mask]
                bd_from_tetracorder_in_mask = bd_from_tetracorder_flat[mask]
                truth_in_mask = truth_array_flat[mask]

                n = len(fractions_in_mask)

                box_data = []
                unique_fractions = np.sort(np.unique(fractions_in_mask))
                mae_bd = []
                mae_bd_prime = []
                mae_bd_prime_prime = []
                mae_bd_emc2_prime = []
                mae_bd_emc2_prime_prime = []
                mae_bd_from_tetracorder = []

                for fraction_bin in unique_fractions:
                    fraction_bin_mask = fractions_in_mask == fraction_bin
                    error_bd_bin = truth_in_mask[fraction_bin_mask] - bd_in_mask[fraction_bin_mask]
                    error_bd_prime_bin = truth_in_mask[fraction_bin_mask] - bd_prime_in_mask[fraction_bin_mask]
                    error_bd_prime_prime = truth_in_mask[fraction_bin_mask] - bd_prime_prime_in_mask[fraction_bin_mask]
                    error_bd_emc2_prime = truth_in_mask[fraction_bin_mask] - bd_emc2_prime_in_mask[fraction_bin_mask]
                    error_bd_emc2_prime_prime = truth_in_mask[fraction_bin_mask] - bd_emc2_prime_prime_in_mask[fraction_bin_mask]
                    error_bd_from_tetracorder = truth_in_mask[fraction_bin_mask] - bd_from_tetracorder_in_mask[fraction_bin_mask]

                    mae_bd_bin = np.nanmean(np.abs(error_bd_bin))
                    mae_bd_prime_bin = np.nanmean(np.abs(error_bd_prime_bin))
                    mae_bd_prime_prime_bin = np.nanmean(np.abs(error_bd_prime_prime))
                    mae_bd_emc2_prime_bin = np.nanmean(np.abs(error_bd_emc2_prime))
                    mae_bd_emc2_prime_prime_bin = np.nanmean(np.abs(error_bd_emc2_prime_prime))
                    mae_bd_from_tetracorder_bin = np.nanmean(np.abs(error_bd_from_tetracorder))

                    mae_bd.append(mae_bd_bin)
                    mae_bd_prime.append(mae_bd_prime_bin)
                    mae_bd_prime_prime.append(mae_bd_prime_prime_bin)
                    mae_bd_emc2_prime.append(mae_bd_emc2_prime_bin)
                    mae_bd_emc2_prime_prime.append(mae_bd_emc2_prime_prime_bin)
                    mae_bd_from_tetracorder.append(mae_bd_from_tetracorder_bin)

                    # if error_bin.size > 0:
                    #     box_data.append(error_bin)
                    #     #mae.append(mae_bin)
                    # else:
                    #     box_data.append([np.nan])  # Keep alignment even if empty (unlikely)
                    #     #mae.append([np.nan])

                # ax.boxplot(
                #     box_data,
                #     positions=unique_fractions,
                #     widths=0.025,
                #     showfliers=True,
                #     patch_artist=True,
                #     boxprops=dict(facecolor='lightgray', color='black'),
                #     medianprops=dict(color='red'),
                #     whiskerprops=dict(color='black'),
                #     capprops=dict(color='black'),
                #     flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor='black')
                # )

                ax.plot(unique_fractions, mae_bd, color='red', linewidth=1.5, label='Bd')
                ax.plot(unique_fractions, mae_bd_prime, color='green', linewidth=1.5, label="Bd'")
                ax.plot(unique_fractions, mae_bd_prime_prime, color='green', linewidth=3.5, label="Bd''", linestyle='dashed')
                ax.plot(unique_fractions, mae_bd_emc2_prime, color='blue', linewidth=1.5, label="E(MC)$^2$ - Bd'")
                ax.plot(unique_fractions, mae_bd_emc2_prime_prime, color='orange', linewidth=1.5, label="E(MC)$^2$ - BD''")
                ax.plot(unique_fractions, mae_bd_from_tetracorder, color='blue', linewidth=1.5, label="Tetracorder", linestyle='dashed')

                # ✅ Labels and limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 0.2)
                ax.set_xlabel('Soil Fraction')
                ax.set_ylabel('Mean Absolute Error')
                ax.set_title(f'Truth = Pred = {label}\nn = {n}')
                ax.legend()

                # ✅ Format x-ticks with two decimals
                ax.set_xticks(unique_fractions)
                ax.set_xticklabels([f'{x:.2f}' for x in unique_fractions], rotation=90)

           # plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f"{group}_mae_band_depths.png"),
                        bbox_inches='tight')
            plt.clf()
            plt.close()

            # # create lists to store the means
            # mean_y_tetra = []
            # mean_y_tetra_sim = []
            # mean_y_hat = []
            # mean_y_sma = []
            #
            # # create lists to store standard deviations
            # std_y_tetra = []
            # std_y_tetra_sim = []
            # std_y_hat = []
            # std_y_sma = []
            #
            # for _col in range(y.shape[1]):
            #     mean_y_tetra.append(np.nanmean(bd_tetra[:, _col] - bd_tetra[:, _col]))
            #     std_y_tetra.append(np.nanstd(bd_tetra[:, _col] - bd_tetra[:, _col]))
            #
            #     mean_y_tetra_sim.append(np.nanmean(bd_tetra_sim[:, _col] - bd_tetra[:, _col]))
            #     std_y_tetra_sim.append(np.nanstd(bd_tetra_sim[:, _col] - bd_tetra[:, _col]))
            #
            #     mean_y_hat.append(np.nanmean(bd_tetra[:, _col] - bd_hat[:, _col, 6]))
            #     std_y_hat.append(np.nanstd(bd_tetra[:, _col] - bd_hat[:, _col, 6]))
            #
            #     mean_y_sma.append(np.nanmean(bd_tetra[:, _col] - y_sma[:, _col]))
            #     std_y_sma.append(np.nanstd(bd_tetra[:, _col] - y_sma[:, _col]))
            #
            # #ax.plot(unique_x_val, np.absolute(mean_y_tetra),
            # #        label='Tetracorder$_{Truth}$', linestyle='solid', color='red')
            #
            # ax.errorbar(unique_x_val, np.absolute(mean_y_tetra_sim), yerr=std_y_tetra_sim, fmt='o',
            #             label='Mixed - No Correction', linestyle='solid', color='purple', capsize=8, ecolor='purple'
            #             , linewidth=self.linewidth, markersize=20)
            #
            # ax.errorbar(unique_x_val, np.absolute(mean_y_hat), yerr=std_y_hat, fmt='o',
            #             label='Mixed w/ known fraction', linestyle='solid', color='green', capsize=6, ecolor='green'
            #             , linewidth=self.linewidth, markersize = 20)
            #
            # ax.errorbar(unique_x_val, np.absolute(mean_y_sma), yerr=std_y_sma, fmt='o',
            #             label='Mixed w/ derived fraction', linestyle='solid', color='blue', capsize=4, ecolor='blue',
            #             linewidth=self.linewidth, markersize = 20)
            #
            # ax.set_xlabel('% Soil Cover', fontsize=self.axis_label_fontsize)
            # if group == 'g1':
            #     title = 'Iron Oxides'
            # else:
            #     title = 'Clays/Carbonates'
            # ax.set_title(title, fontsize=self.title_fontsize)
            # ax.set_aspect('auto')
            # ax.tick_params(axis='both', labelsize=self.legend_text)
            #
            # # major ticks every 10 units
            # major_ticks = range(0, 101, 10)
            # ax.set_xticks(major_ticks)
            #
            # # Minor ticks every 5 units
            # minor_ticks = range(0, 101, 5)
            # ax.set_xticks(minor_ticks, minor=True)
            #
            # # set tick labels for x-axis
            # ax.set_xticklabels(major_ticks)
            #
            # # major ticks every 10 units - y-axis
            # major_ticks = np.arange(-0.25, 0.25, 0.05)
            # ax.set_yticks(major_ticks)
            #
            # # Minor ticks every 5 units
            # minor_ticks = np.arange(-0.25, 0.25, 0.01)
            # ax.set_yticks(minor_ticks, minor=True)
            #
            # ax.set_yticklabels(major_ticks)
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #
            # ax.legend(loc='upper right', fontsize=self.legend_text)
            # ax.set_ylim(0, .25)
            #
            # ax.set_ylabel('Mean Absolute Error', fontsize=self.axis_label_fontsize)
            #
            # plt.savefig(os.path.join(self.fig_directory, f'tetracorder_{group}_band-depths.png'), format="png", dpi=300,
            #                 bbox_inches="tight")
            #
            # plt.clf()
            # plt.close()
            # col_guide = {0: np.absolute(bd_tetra[:, :] - bd_tetra_sim[:, :]),
            #              1: np.absolute(bd_tetra[:, :] - bd_hat[:, :, 6]),
            #              2: np.absolute(bd_tetra[:, :] - y_sma[:, :])}
            #
            # col_y_label = {0: 'Tetracorder$_{mixed}$',
            #                1: 'Tetracorder$_{vegetation corrected}$',
            #                2: 'Tetracorder$_{sma vegetation corrected}$'}
            #
            # fig = plt.figure(figsize=(12,12))
            # ncols = 3
            # nrows = 1
            # gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.05, width_ratios=[1] * ncols,
            #                        height_ratios=[1] * nrows)
            #
            # for row in range(nrows):
            #     for col in range(ncols):
            #         ax = fig.add_subplot(gs[row, col])
            #         scatter = ax.scatter(bd_tetra[:, :], col_guide[col], c=fractions[:,:,2], cmap='viridis')
            #         ax.set_ylim(0, 1)
            #         ax.set_xlim(0, 1)
            #
            #         ax.set_aspect(1. / ax.get_data_ratio())
            #
            #         ax.set_title(col_y_label[col])
            #         if col == 0:
            #             ax.set_ylabel(f'Absolute Error')
            #
            #         if col != 0:
            #             ax.set_yticklabels([])
            #
            #         ax.set_xlabel('Band Depth')
            #
            #         # add color bar to 3rd plot
            #         if col == 2:
            #             cax = fig.add_subplot(gs[2])  # Independent axis for the color bar
            #             cbar = fig.colorbar(scatter, cax=cax)
            #             cbar.set_label('% Soil Cover')  # Optional label for the colorbar
            #
            # plt.savefig(os.path.join(self.fig_directory, f'tetracorder_{group}_band-depths_scatterplot.png'), format="png", dpi=300,
            #             bbox_inches="tight")
            #
            # plt.clf()
            # plt.close()


    def mineral_ref_figure(self):

        try:
            spectrum = load_pickle('soil_test_tc')
        except:
            spectrum = envi_to_array(os.path.join(self.sim_spectra_directory, 'tetracorder_g1_simulation_spectra'))[0, 17, :]
            save_pickle(spectrum, 'soil_test_tc')

        vegetation_spectra = envi_to_array(os.path.join(self.sim_spectra_directory, 'tetracorder_g1_simulation_vegetation'))[0, 17, :]
        fractions = envi_to_array(os.path.join(self.sim_spectra_directory, 'tetracorder_g1_simulation_fractions'))[0, 17, :]
        gv_spectra = envi_to_array(os.path.join(self.sim_spectra_directory, 'tetracorder_g1_simulation_gv'))[0, 17, :]
        npv_spectra = envi_to_array(os.path.join(self.sim_spectra_directory, 'tetracorder_g1_simulation_npv'))[0, 17, :]
        psoil = envi_to_array(os.path.join(self.sim_spectra_directory, 'tetracorder_g1_simulation_soils'))[0, 20, :]

        data = spectra.mineral_group_retrival(mineral_index=47, spectra_observed=spectrum, npv_fraction=fractions[0], gv_fraction=fractions[1],
                                              pnpv=npv_spectra, pgv=gv_spectra, soil_fraction=fractions[2], psoil=psoil)

        print(data)
    def mineral_sim_library_reference(self):
        sim_library = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_augmented_min'))[:, 0, :]
        group_dict = {'g1': 1, 'g2': 3}
        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))

        for group in ['g1', 'g2']:

            create_directory(os.path.join(self.fig_directory, f'{group}_tetracorder_library'))
            fig_directory_tetracorder = os.path.join(self.fig_directory, f'{group}_tetracorder_library')

            mineral_indices = sim_library[:, group_dict[group]]
            unique_values, counts = np.unique(mineral_indices, return_counts=True)

            p_map(partial(tetracorder_library, fig_directory=fig_directory_tetracorder, group=group),
                  unique_values, **{"desc": "\t\t processing tetracorder library...", "ncols": 150})

            fig, ax = plt.subplots(figsize=(25, 25))

            # get label keys for minerals
            filtered_minerals_df = df_mineral_matrix[df_mineral_matrix['Index'].isin(unique_values)]
            filtered_minerals_df['name_index'] = filtered_minerals_df['Index'].astype(str) + ' - ' + filtered_minerals_df['Name']
            mineral_mapping = dict(zip(filtered_minerals_df['Index'], filtered_minerals_df['name_index']))
            present_minerals_result = [mineral_mapping[value] for value in unique_values if value in mineral_mapping]
            present_minerals_result.insert(0, 'No Data')

            indices = np.arange(len(unique_values))

            bars = ax.bar(indices, counts, tick_label=present_minerals_result,align='center')

            # Set the custom string labels for the x-axis
            ax.set_xticklabels(present_minerals_result, rotation=90, fontsize=12)

            # Add counts on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
                        ha='center', va='bottom', fontsize=20)  # Adjust fontsize as needed

            ax.set_xlabel('Unique Values',  fontsize=20)
            ax.set_ylabel('Counts',  fontsize=20)
            ax.set_title('Tetracorder minerals identified in simulation library',  fontsize=20)

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f'sim_library_minerals_{group}.png'))
            plt.clf()
            plt.close()

    def mineral_sim_spectra_reference(self):

        group_dict = {'g1': 1, 'g2': 3}
        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))

        for group in ['g1', 'g2']:
            sim_library = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, 20, group_dict[group]]

            create_directory(os.path.join(self.fig_directory, f'{group}_tetracorder_simulated_spectra'))
            fig_directory_tetracorder = os.path.join(self.fig_directory, f'{group}_tetracorder_simulated_spectra')

            mineral_indices = sim_library
            unique_values, counts = np.unique(mineral_indices, return_counts=True)

            print(unique_values, counts)
            p_map(partial(tetracorder_library, fig_directory=fig_directory_tetracorder, group=group),
                  unique_values, **{"desc": "\t\t processing tetracorder library...", "ncols": 150})

            fig, ax = plt.subplots(figsize=(25, 25))

            # get label keys for minerals
            filtered_minerals_df = df_mineral_matrix[df_mineral_matrix['Index'].isin(unique_values)]
            filtered_minerals_df['name_index'] = filtered_minerals_df['Index'].astype(str) + ' - ' + filtered_minerals_df['Name']
            mineral_mapping = dict(zip(filtered_minerals_df['Index'], filtered_minerals_df['name_index']))
            present_minerals_result = [mineral_mapping[value] for value in unique_values if value in mineral_mapping]
            #present_minerals_result.insert(0, 'No Data')
            #present_minerals_result.insert(-9999, 'No Data')

            indices = np.arange(len(unique_values))

            bars = ax.bar(indices, counts, tick_label=present_minerals_result, align='center')

            # Set the custom string labels for the x-axis
            ax.set_xticklabels(present_minerals_result, rotation=90, fontsize=12)

            # Add counts on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
                        ha='center', va='bottom', fontsize=10)  # Adjust fontsize as needed

            ax.set_xlabel('Unique Values')
            ax.set_ylabel('Counts')
            ax.set_title('Bar Graph of Unique Values and Their Counts')

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f'sim_spectra_minerals_{group}.png'))
            plt.clf()
            plt.close()

    def veg_correction_by_mineral(self):
        group_dict = {'g1': 0, 'g2': 2}

        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
        df_mineral_matrix = df_mineral_matrix.fillna(-9999)

        for group in ['g1', 'g2']:
            fractions = envi_to_array(os.path.join(self.sim_spectra_directory,
                                                   f'tetracorder_{group}_simulation_fractions'))

            create_directory(os.path.join(self.fig_directory, f'{group}_veg_correction_mineral'))
            fig_directory_tetracorder = os.path.join(self.fig_directory, f'{group}_veg_correction_mineral')

            unique_x_val = np.unique(fractions[0, :, 2]) * 100

            # this is soil from tetracorder output
            bd_tetra = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, :21, group_dict[group]]
            bd_tetra[bd_tetra == 0] = np.nan

            bd_tetra_minerals = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:,:21, group_dict[group] + 1]

            bad_minerals = np.array(minerals_to_exclude)
            bd_tetra_minerals[np.isin(bd_tetra_minerals, bad_minerals)] = np.nan
            bd_tetra_minerals[np.isnan(bd_tetra)] = np.nan

            unique_minerals = np.unique(bd_tetra_minerals)
            unique_minerals = unique_minerals[~np.isnan(unique_minerals)]

            # this is the simulated spectra w/ corrections
            bd_hat = envi_to_array(os.path.join(self.veg_correction_dir, f'tetracorder_{group}_vegetation_correction'))
            bd_hat[bd_hat == -9999.] = np.nan

            # this is sim spectra from tetracorder output w/out corrections
            bd_tetra_sim = envi_to_array(os.path.join(self.sa_outputs,
                                                      f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21,
                           group_dict[group]]
            bd_tetra_sim[bd_tetra_sim == 0] = np.nan

            bd_tetra_sim_minerals = envi_to_array(os.path.join(self.sa_outputs,
                                                               f'tetracorder_{group}_simulation_spectra_augmented_min'))[
                                    :, :21, group_dict[group] + 1]
            bd_tetra_sim_minerals[np.isin(bd_tetra_sim_minerals, bad_minerals)] = np.nan
            bd_tetra_sim_minerals[np.isnan(bd_tetra_sim)] = np.nan

            # this is the sma vegetation reconstructed signal!
            bd_sma = envi_to_array(
                os.path.join(self.veg_correction_dir, f'tetracorder_{group}_vegetation_correction_sma'))
            bd_sma[bd_sma == -9999.] = np.nan

            for mineral in unique_minerals:
                mineral_mask = (bd_tetra_minerals == mineral)

                tetra_soil = bd_tetra.copy()
                tetra_soil[mineral_mask] = np.nan

                tetra_sim = bd_tetra_sim.copy()
                tetra_sim[mineral_mask] = np.nan

                tetra_veg_correction = bd_hat.copy()
                tetra_veg_correction[mineral_mask] = np.nan

                tetra_sma = bd_sma.copy()
                tetra_sma[mineral_mask] = np.nan

                # create lists to store the means
                mean_y_tetra = []
                mean_y_tetra_sim = []
                mean_y_hat = []
                mean_y_sma = []

                # create lists to store standard deviations
                std_y_tetra = []
                std_y_tetra_sim = []
                std_y_hat = []
                std_y_sma = []

                for _col in range(tetra_soil.shape[1]):
                    mean_y_tetra.append(np.nanmean(tetra_soil[:, _col] - tetra_soil[:, _col]))
                    std_y_tetra.append(np.nanstd(tetra_soil[:, _col] - tetra_soil[:, _col]))

                    mean_y_tetra_sim.append(np.nanmean(tetra_soil[:, _col] - tetra_sim[:, _col]))
                    std_y_tetra_sim.append(np.nanstd(tetra_soil[:, _col] - tetra_sim[:, _col]))

                    mean_y_hat.append(np.nanmean(tetra_soil[:, _col] - tetra_veg_correction[:, _col, 6]))
                    std_y_hat.append(np.nanstd(tetra_soil[:, _col] - tetra_veg_correction[:, _col, 6]))

                    mean_y_sma.append(np.nanmean(tetra_soil[:, _col] - tetra_sma[:, _col, 6]))
                    std_y_sma.append(np.nanstd(tetra_soil[:, _col] - tetra_sma[:, _col, 6]))

                record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral), 'Record'].iloc[0]
                name = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Name'].iloc[0]

                # make figure
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.plot(unique_x_val, np.absolute(mean_y_tetra),
                        label='Tetracorder$_{soil}$', linestyle='solid', color='red')

                ax.errorbar(unique_x_val, np.absolute(mean_y_tetra_sim), yerr=std_y_tetra_sim, fmt='o',
                            label='Tetracorder$_{mixed}$', linestyle='solid', color='purple', capsize=8,
                            ecolor='purple')

                ax.errorbar(unique_x_val, np.absolute(mean_y_hat), yerr=std_y_hat, fmt='o',
                            label='Tetracorder$_{vegetation corrected}$', linestyle='solid', color='green', capsize=6,
                            ecolor='green')

                ax.errorbar(unique_x_val, np.absolute(mean_y_sma), yerr=std_y_sma, fmt='o',
                            label='Tetracorder$_{sma vegetation corrected}$', linestyle='solid', color='blue',
                            capsize=4, ecolor='blue')

                ax.set_xlabel('% Soil Cover')
                ax.set_ylabel('Bd - Band Depth')
                ax.legend(loc='upper right')
                ax.set_title(name)

                ax.set_aspect('auto')

                # major ticks every 10 units
                major_ticks = range(0, 101, 10)
                ax.set_xticks(major_ticks)

                # Minor ticks every 5 units
                minor_ticks = range(0, 101, 5)
                ax.set_xticks(minor_ticks, minor=True)

                # set tick labels for x-axis
                ax.set_xticklabels(major_ticks)

                # major ticks every 10 units - y-axis
                major_ticks = np.arange(0, 0.25, 0.05)
                ax.set_yticks(major_ticks)

                # Minor ticks every 5 units
                minor_ticks = np.arange(0, 0.25, 0.01)
                ax.set_yticks(minor_ticks, minor=True)

                ax.set_yticklabels(major_ticks)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.tight_layout()
                plt.savefig(os.path.join(fig_directory_tetracorder, f'{int(mineral)}-bd-veg-corrections.png'))
                plt.clf()
                plt.close()


    def slpit_bd(self):
        def signal_reconstruct(complete_fractions_array, user_em, df_unmix, spectra_start):
            unmix_library_array = df_unmix.iloc[:, spectra_start:].to_numpy()

            df_unmix['level_1'] = df_unmix['level_1'].str.lower()
            min_em_index = np.min(df_unmix[df_unmix['level_1'] == user_em].index)
            max_em_index = np.max(df_unmix[df_unmix['level_1'] == user_em].index)

            unmix_library_array = unmix_library_array[min_em_index:max_em_index + 1, :]
            complete_fractions_array = complete_fractions_array[0, 0, min_em_index:max_em_index + 1]
            spectra_grid = np.zeros((len(self.wvls)))

            if np.sum(complete_fractions_array) == 0:
                return spectra_grid
            else:
                spectra_grid = np.average(unmix_library_array, weights=complete_fractions_array, axis=0)
                return spectra_grid

        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.json")))
        df = df.sort_values('Name')

        df_rows = []

        for index, row in df.iterrows():
            plot = row['Name']
            print(plot)
            plot_num = int(plot.split('-')[1])
            team = plot.split('-')[0]

            if team.strip() in ['THERM']:
                continue

            if plot_num > 60:
                continue

            emit_filetime = row['EMIT DATE']

            for em_lib_type in ['local', 'global']:
                if em_lib_type in ['local']:
                    df_unmix = pd.read_csv(os.path.join(self.slpit_output_directory, 'spectral_transects', 'endmembers', f'{plot.replace("SPEC", "Spectral").replace(" ", "")}-emit.csv'))
                    spectra_start = 10
                else:
                    # construct rho of ems
                    df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv'))
                    spectra_start = 7

                # load reflectances
                emit_rfl = envi_to_array(os.path.join(self.aug_directory, f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented'))[0,0, :]
                slpit_rfl = envi_to_array(os.path.join(self.aug_directory, f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented'))[0,0, :]
                contact_rfl = envi_to_array(os.path.join(self.aug_directory, f'{plot.replace(" ", "").replace("SPEC", "Spectral")}-emit_ems_augmented'))[0,0, :]

                # rebuild vegetation signals
                emit_fractions_complete = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                                      f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_{em_lib_type}_complete_fractions'))
                slpit_fractions_complete = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                                      f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented_{em_lib_type}_complete_fractions'))

                emit_pgv = signal_reconstruct(emit_fractions_complete, user_em='pv', df_unmix=df_unmix, spectra_start=spectra_start)
                slipt_pgv = signal_reconstruct(slpit_fractions_complete, user_em='pv', df_unmix=df_unmix, spectra_start=spectra_start)
                emit_pnpv = signal_reconstruct(emit_fractions_complete, user_em='npv', df_unmix=df_unmix, spectra_start=spectra_start)
                slipt_pnpv = signal_reconstruct(slpit_fractions_complete, user_em='npv', df_unmix=df_unmix, spectra_start=spectra_start)

                # load transect fractions
                emit_fractions = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                                     f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_{em_lib_type}_fractional_cover'))[0,0,:]
                slpit_fractions = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                                      f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented_{em_lib_type}_fractional_cover'))[0,0,:]
                # load the mineral indices
                emit_mineral_indexs = envi_to_array(os.path.join(self.output_directory, 'spectral_abundance',
                                                                      f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_min'))[0,0,:]
                slpit_mineral_indexs = envi_to_array(os.path.join(self.output_directory, 'spectral_abundance',
                                                                      f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented_min'))[0,0,:]
                contact_probe_indices = envi_to_array(os.path.join(self.output_directory, 'spectral_abundance',
                                                                      f'{plot.replace(" ", "").replace("SPEC", "Spectral")}-emit_ems_augmented_min'))[0,0,:]

                for _group, group in enumerate(['g1', 'g2']):
                    index_dict = {0: 1, 1: 3}
                    slpit_mineral_index = slpit_mineral_indexs[index_dict[_group]]
                    emit_mineral_index = emit_mineral_indexs[index_dict[_group]]
                    contact_probe_index = contact_probe_indices[index_dict[_group]]

                    index_src = ['Contact', 'EMIT', 'SLPIT']
                    for _index, index in enumerate([contact_probe_index, emit_mineral_index, slpit_mineral_index]):
                        # contact probe
                        contact_probe_bd = spectra.mineral_group_retrival(mineral_index=index, spectra_observed=contact_rfl,
                                                                          npv_fraction=slpit_fractions[0], gv_fraction=slpit_fractions[1],
                                                                          pnpv=slipt_pnpv, pgv=slipt_pgv, soil_fraction=slpit_fractions[2], psoil=None)

                        # emit data
                        emit_band_bd = spectra.mineral_group_retrival(mineral_index=index, spectra_observed=emit_rfl,
                                                                      npv_fraction=slpit_fractions[0], gv_fraction=slpit_fractions[1],
                                                                      pnpv=emit_pnpv, pgv=emit_pgv, soil_fraction=slpit_fractions[2], psoil=None)

                        # slpit data
                        slpit_bd = spectra.mineral_group_retrival(mineral_index=index, spectra_observed=slpit_rfl,
                                                                  npv_fraction=slpit_fractions[0], gv_fraction=slpit_fractions[1], pnpv=slipt_pnpv,
                                                                  pgv=slipt_pgv, soil_fraction=slpit_fractions[2], psoil=None)

                        # append results
                        df_row = [plot, group, em_lib_type, index_src[_index],  np.int64(index)] + list(contact_probe_bd[2:]) + list(emit_band_bd[2:]) + list(slpit_bd[2:]) + list(emit_fractions) + list(slpit_fractions)
                        df_rows.append(df_row)

        # these are labels for the csv
        data_src_label = ['Contact', "EMIT", "SLPIT"]
        bd_labels = ['Lc', 'Bd', "Bd'", "Bd''"]
        combined_labels = [f"{src}_{bd}" for src in data_src_label for bd in bd_labels]
        ems = ['npv', 'pv', 'soil', 'shade']
        frac_src = ["EMIT", "SLPIT"]
        combined_frac_labels = [f"{em}_{src}" for src in frac_src for em in ems]

        df_results = pd.DataFrame(df_rows)
        df_results.columns = ['plot', 'group', 'em_library', 'index_src', 'index'] + combined_labels + combined_frac_labels
        df_results.to_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'), index=False)

    def slpit_figure(self):
        df_results = pd.read_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'))
        df_results = df_results.replace(-9999, np.nan).dropna()

        col_map = {
            0: 'Iron Oxides',
            1: 'Clays/Carbonates'}

        for em_lib in ['local', 'global']:
            for data_type in ['SLPIT', 'EMIT']:

                # # create figure
                fig = plt.figure(figsize=(self.fig_width, self.fig_height))
                ncols = 2
                nrows = 2
                gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.25, width_ratios=[1] * ncols,
                                       height_ratios=[1] * nrows)

                # loop through figure columns
                for row in range(nrows):
                    for col in range(ncols):
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_ylim(0, 0.3)
                        ax.set_xlim(0, 0.3)
                        ax.set_aspect('auto')

                        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                        if col == 0:
                            df_select = df_results[(df_results['group'] == 'g1') & (df_results['index_src'] == 'Contact')].copy()
                            if row == 0:
                                ax.set_title(col_map[col], fontsize=self.title_fontsize)

                        if col == 1:
                            df_select = df_results[(df_results['group'] == 'g2') & (df_results['index_src'] == 'Contact')].copy()
                            if row == 0:
                                ax.set_title(col_map[col], fontsize=self.title_fontsize)

                        # filter out low plots of soil fraction
                        df_select_val = df_select[(df_select['em_library'] == em_lib) & (df_select['soil_SLPIT'] >= 0.60)].copy()

                        if row == 0:
                            x = df_select_val["Contact_Bd"]
                            y = df_select_val[f"{data_type}_Bd"]
                            ax.set_ylabel(f"{data_type}_Bd", fontsize=self.axis_label_fontsize)
                            ax.set_xlabel("Contact_Bd", fontsize=self.axis_label_fontsize)

                        else:
                            x = df_select_val["Contact_Bd"]
                            y = df_select_val[f"{data_type}_Bd'"]
                            ax.set_ylabel(f"{data_type}_Bd'", fontsize=self.axis_label_fontsize)
                            ax.set_xlabel("Contact_Bd", fontsize=self.axis_label_fontsize)

                        if col != 0:
                            ax.set_yticklabels([])

                        if row == 0:
                            ax.set_xticklabels([])

                        # plot fractional cover values
                        m, b = np.polyfit(x.values, y.values, 1)
                        one_line = np.linspace(0, 1, 101)

                        # plot 1 to 1 line
                        ax.plot(one_line, one_line, color='black')
                        ax.plot(one_line, m * one_line + b, color='red')
                        scatter = ax.scatter(x, y, marker='^', edgecolor='black', label='SLPIT point', zorder=10, s=150,
                                   c=df_select_val['soil_SLPIT'].values, cmap='viridis', vmin=min(df_select_val['soil_SLPIT'].values),
                                             vmax=max(df_select_val['soil_SLPIT'].values))
                        ax.tick_params(axis='both', labelsize=self.legend_text)

                        # Add error metrics
                        rmse = mean_squared_error(x, y)
                        mae = mean_absolute_error(x, y)

                        r2 = r2_calculations(x, y)

                        txtstr = '\n'.join((
                            r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                            r'R$^2$: %.2f' % (r2[0],),
                            r'n = ' + str(len(x)),
                        ))

                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                        ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                                verticalalignment='top', bbox=props)
                        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
                        cbar.set_label('Soil Fraction (%)', fontsize=self.axis_label_fontsize)
                        cbar.ax.tick_params(labelsize=self.legend_text)

                plt.savefig(os.path.join(self.fig_directory, f'field_data_regressions_{em_lib}_{data_type}.png'), format="png", dpi=400,
                        bbox_inches="tight")
                plt.clf()
                plt.close()

    def fraction_soil_vs_bd(self):
        bd_band = {'g1': 0, 'g2': 2}
        minerals = {'g1': 1, 'g2': 3}

        create_directory(os.path.join(self.fig_directory, 'fraction_vs_bd'))

        for group in ['g1', 'g2']:
            create_directory(os.path.join(self.fig_directory, 'fraction_vs_bd', group))

            soil_fractions = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_fractions'))[:,:, 2]
            soil_index = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_index'))[:,:, 2]
            sim_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, 0:21, :]
            sim_soils_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 0:21, :]

            minerals_detected = np.unique(sim_soils_sa_arrary[:, :, minerals[group]])

            for mineral in minerals_detected:
                if int(mineral) in [0]:
                    pass
                else:
                    plt.figure(figsize=(12, 12))

                    mask = ((sim_soils_sa_arrary[:, :, minerals[group]] == mineral) & (sim_sa_arrary[:, :, minerals[group]] == mineral))

                    soil_used = np.unique(soil_index[mask])

                    for soil in soil_used:
                        group_sa = sim_sa_arrary[:, :, bd_band[group]]
                        group_soil = sim_soils_sa_arrary[:, :, bd_band[group]]

                        mask = ((sim_soils_sa_arrary[:, :, minerals[group]] == mineral) & (soil_index == soil))

                        plt.scatter(soil_fractions[mask], group_sa[mask], s=5, alpha=0.6, label="Simulated Spectra")
                        plt.scatter(soil_fractions[mask] + 0.02, group_soil[mask], s=5, alpha=0.6, label="Soil")
                        plt.xlabel('Fractions')
                        plt.ylabel('Band Depth')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(os.path.join(self.fig_directory, 'fraction_vs_bd', group, f'{int(mineral)}_{int(soil)}_fraction_vs_band_depth_{group}.png'), dpi=300)
                        plt.clf()
                        plt.close()


    def fraction_threshold(self):
        df_results = pd.read_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'))
        df_results = df_results.replace(-9999, np.nan).dropna()

        # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 2
        nrows = 1
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.25, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'Iron Oxides',
            1: 'Clays/Carbonates'}

        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_aspect('auto')

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if col == 0:
                    df_select = df_results[
                        (df_results['group'] == 'g1') & (df_results['index_src'] == 'Contact')].copy()

                    ax.set_title(col_map[col], fontsize=self.title_fontsize)

                if col == 1:
                    df_select = df_results[
                        (df_results['group'] == 'g2') & (df_results['index_src'] == 'Contact')].copy()

                    ax.set_title(col_map[col], fontsize=self.title_fontsize)

                # filter out low plots of soil fraction
                for em_lib in ['global', 'local']:
                    values = np.arange(0, 1.00, 0.05)
                    mae_slpit_bd, mae_slpit_bd_prime, mae_emit_bd, mae_emit_bd_prime = [], [], [], []
                    r2_slpit_bd, r2_slpit_bd_prime, r2_emit_bd, r2_emit_bd_prime = [], [], [], []

                    for val in values:
                        df_select_val = df_select[(df_select['em_library'] == em_lib) & (df_select['soil_SLPIT'] >= val)].copy()

                        if df_select_val.empty:
                            mae_slpit_bd.append(np.nan)
                            r2_slpit_bd.append(np.nan)

                            mae_slpit_bd_prime.append(np.nan)
                            r2_slpit_bd_prime.append(np.nan)

                            mae_emit_bd.append(np.nan)
                            r2_emit_bd.append(np.nan)

                            mae_emit_bd_prime.append(np.nan)
                            r2_emit_bd_prime.append(np.nan)
                            continue


                        x = df_select_val["Contact_Bd"]

                        mae_slpit_bd.append(mean_absolute_error(x, df_select_val["SLPIT_Bd"]))
                        r2_slpit_bd.append(r2_calculations(x, df_select_val["SLPIT_Bd"])[0])

                        mae_slpit_bd_prime.append(mean_absolute_error(x, df_select_val["SLPIT_Bd'"]))
                        r2_slpit_bd_prime.append(r2_calculations(x, df_select_val["SLPIT_Bd'"])[0])

                        mae_emit_bd.append(mean_absolute_error(x, df_select_val["EMIT_Bd"]))
                        r2_emit_bd.append(r2_calculations(x, df_select_val["EMIT_Bd"])[0])

                        mae_emit_bd_prime.append(mean_absolute_error(x, df_select_val["EMIT_Bd'"]))
                        r2_emit_bd_prime.append(r2_calculations(x, df_select_val["EMIT_Bd'"])[0])

                    # plot error
                    # ax.plot(values, mae_slpit_bd, label="SLPIT_Bd", color='red')
                    # ax.plot(values, mae_slpit_bd_prime, label="SLPIT_Bd'", color='green')
                    # ax.plot(values, mae_emit_bd, label="EMIT_Bd", color='blue')
                    # ax.plot(values, mae_emit_bd_prime, label="EMIT_Bd'", color='orange')

                    # plot r2
                    if em_lib == 'local':
                        linestyle = 'solid'
                        lw = 2
                    else:
                        linestyle = '--'
                        lw=1

                    #ax.plot(values, r2_slpit_bd, label=f"SLPIT_Bd (R²) - {em_lib}", color='red', linestyle=linestyle, linewidth=lw)
                    #ax.plot(values, r2_slpit_bd_prime, label=f"SLPIT_Bd' (R²) - {em_lib}", color='green',linestyle=linestyle, linewidth=lw)
                    #ax.plot(values, r2_emit_bd, label=f"EMIT_Bd (R²) - {em_lib}", color='blue', linestyle=linestyle, linewidth=lw)
                    #ax.plot(values, r2_emit_bd_prime, label=f"EMIT_Bd' (R²) - {em_lib}", color='orange', linestyle=linestyle, linewidth=lw)

                    ax.plot(values, mae_slpit_bd, label=f"SLPIT_Bd (R²) - {em_lib}", color='red', linestyle=linestyle,
                            linewidth=lw)
                    ax.plot(values, mae_slpit_bd_prime, label=f"SLPIT_Bd' (R²) - {em_lib}", color='green',
                            linestyle=linestyle, linewidth=lw)
                    ax.plot(values, mae_emit_bd, label=f"EMIT_Bd (R²) - {em_lib}", color='blue', linestyle=linestyle,
                            linewidth=lw)
                    ax.plot(values, mae_emit_bd_prime, label=f"EMIT_Bd' (R²) - {em_lib}", color='orange',
                            linestyle=linestyle, linewidth=lw)

                # Combine and deduplicate legend labels
                handles_1, labels_1 = ax.get_legend_handles_labels()

                # Use dict to remove duplicates
                unique = dict(zip(labels_1, handles_1))

                if col == 1:
                   ax.legend(unique.values(), unique.keys(),loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=False)

                ax.set_ylabel("MAE")
                #ax.set_ylabel("R²")
                ax.set_xlabel('Fraction of soil')

        plt.savefig(os.path.join(self.fig_directory, f'field_thresholds.png'), format="png", dpi=400, bbox_inches="tight")
        plt.clf()
        plt.close()



def run_figure_workflow(base_directory):
    ems = ['soil']
    major_axis_fontsize = 22
    minor_axis_fontsize = 20
    title_fontsize = 30
    axis_label_fontsize = 22
    fig_height = 11
    fig_width = 17
    linewidth = 3
    sig_figs = 2
    legend_text = 20

    tc = tetracorder_figures(base_directory=base_directory, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs, legend_text=legend_text)

    #tc.mineral_ref_figure()
    #tc.fraction_soil_vs_bd()

    #tc.mineral_sim_library_reference()
    #tc.mineral_sim_spectra_reference()
    #tc.confusion_matrices()
    tc.confusion_matrix_detailed()

    #tc.slpit_bd()
    #tc.slpit_figure()
    #tc.fraction_threshold()
    #tc.veg_correction_fig()
    #tc.veg_correction_by_mineral()

    #tc.tetracorder_libraries()
    #tc.mineral_validation(x_axis='contact')
    #tc.mineral_validation(x_axis='transect')
    #tc.mineral_threshold()
