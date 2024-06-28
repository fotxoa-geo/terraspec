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
from utils.results_utils import r2_calculations
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
import sns
from mpl_toolkits.basemap import Basemap


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
    def __init__(self, base_directory: str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'tetracorder', 'output')
        self.aug_directory = os.path.join(self.output_directory, 'augmented')
        self.sim_spectra_directory = os.path.join(self.output_directory, 'simulated_spectra')
        self.output_fractions = os.path.join(self.output_directory, 'fractions')
        self.sa_outputs = os.path.join(self.output_directory, 'spectral_abundance')

        self.slpit_output = os.path.join(base_directory, 'slpit', 'output')
        self.fig_directory = os.path.join(base_directory, 'tetracorder', 'figures')

        self.bands = load_band_names(
            os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_jabun_abs_abundance'))

        create_directory(os.path.join(self.fig_directory, 'plot_minerals'))

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

    def simulation_fig(self, xaxis: str):
        # load simulation data - truncate the sa files from augmentation
        sim_index_array = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{xaxis}_index'))
        sim_fractions_array = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{xaxis}_fractions'))
        sim_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{xaxis}_spectra_simulation_augmented_jabun_rel_abundance'))[:, 0:21, :]
        sim_sa_arrary[sim_sa_arrary == -9999] = 0
        soil_sa_sim_pure = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_jabun_rel_abundance'))[:, 0, :]
        soil_sa_sim_pure[soil_sa_sim_pure == -9999] = 0

        error_grid = self.error_abundance_corrected(
            spectral_abundance_array=sim_sa_arrary,
            pure_soil_array=soil_sa_sim_pure,
            fractions=sim_fractions_array, index=sim_index_array) #, unmix_fractions=unmix_fractions_array) #, unmix_abundance=unmix_abundance)

        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ncols = 3
        nrows = 2

        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0, hspace=0, figure=fig)
        minor_tick_spacing = 0.1
        major_tick_spacing = 0.25
        counter = 0

        plot_titles = {
            0: 'Iron Oxides',
            1: 'Carbonates',
            2: 'Clays',
            3: 'Quartz',
            4: 'Chlorite'}

        for row in range(nrows):
            for col in range(ncols):
                if counter == 5:
                    continue
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(plot_titles[counter])
                ax.set_xlabel(f'{xaxis}')
                ax.grid('on', linestyle='--')
                #ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
                #ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
                #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                #ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                if col == 0:
                    ax.set_ylabel('MAE')

                # if col != 0:
                #     ax.set_yticklabels([])

                abs_error = error_grid[:, :, counter]

                if xaxis == 'npv':
                    fractions = sim_fractions_array[:, :, 0]

                if xaxis == 'pv':
                    fractions = sim_fractions_array[:, :, 1]

                if xaxis == 'soil':
                    fractions = sim_fractions_array[:, :, 2]

                # this is no-nans
                x_vals, mae, percent_false_neg, percent_false_pos = bin_sums(x=fractions, y=abs_error, nans=False)
                l1 = ax.plot(x_vals, mae, label='No Atmosphere')

                # this is nan
                # x_vals, mae, percent_false_neg, percent_false_pos = bin_sums(x=fractions, y=abs_error, nans=True)
                # l2 = ax.plot(x_vals, mae, label='No Atmosphere (Nans Muted)')

                # this is the sma line
                # x_vals, mae, percent_false_neg, percent_false_pos = bin_sums(x=fractions, y=abs_unmix_error,
                #                                                              false_pos=mineral_false_positive,
                #                                                              false_neg=mineral_false_negative,
                #                                                              nans=True)
                #l3 = ax.plot(x_vals, mae, label='SMA Approach')
                #ax.set_ylim(0.0, 0.25)
                #ax.set_xlim(0.0, 1.05)

                lns = l1

                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc=0, prop={'size': 6})
                ax.set_aspect(1. / ax.get_data_ratio())

                counter += 1
        plt.savefig(os.path.join(self.fig_directory, 'tetracorder_mae_' + xaxis + '.png'), dpi=300, bbox_inches='tight')

    def mineral_validation(self, x_axis:str):
        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
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


    def veg_correction(self):

        cols = 20 + 1 # total number of cols used; code does not get rid of augemnted cols
        em_sa_array = envi_to_array(os.path.join(self.sa_outputs, 'tetracorder_soil_em_spectra_simulation_augmented_jabun_rel_abundance'))
        em_sa_array = em_sa_array[:, :cols, :]
        em_tetracorder_index = envi_to_array(os.path.join(self.sa_outputs, 'tetracorder_soil_em_spectra_simulation_augmented_min'))
        em_tetracorder_index = em_tetracorder_index[:,:cols, :]

        mixed_sa_array = envi_to_array(os.path.join(self.sa_outputs, 'tetracorder_soil_spectra_simulation_augmented_jabun_rel_abundance'))
        mixed_sa_array = mixed_sa_array[:, :cols, :]
        mixed_tetracorder_index = envi_to_array(os.path.join(self.sa_outputs, 'tetracorder_soil_spectra_simulation_augmented_min'))
        mixed_tetracorder_index = mixed_tetracorder_index[:,:cols, :]




def run_figure_workflow(base_directory):
    ems = ['soil']
    tc = tetracorder_figures(base_directory=base_directory)
    tc.veg_correction()
    #tc.confusion_matrix()
    tc.tetracorder_libraries()
    #tc.mineral_validation(x_axis='contact')
    #tc.mineral_validation(x_axis='transect')
    #tc.mineral_threshold()
    for em in ems:
        tc.simulation_fig(xaxis=em)
