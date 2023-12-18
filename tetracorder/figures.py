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


mineral_groupings = {'group.1um/copper_precipitate_greenslime': 'copper',
                     'group.1um/fe2+_chlor+muscphy': 'chlorite',
                     'group.1um/fe2+_goeth+musc' : 'goethite',
                     'group.1um/fe2+fe3+_chlor+goeth.propylzone': 'goethite',
                     'group.1um/fe2+generic_br33a_bioqtzmonz_epidote': 'goethite',
                     'group.1um/fe2+generic_carbonate_siderite1': 'iron oxide',
                     'group.1um/fe2+generic_nrw.cummingtonite': 'iron oxide',
                     'group.1um/fe2+generic_nrw.hs-actinolite': 'iron oxide',
                     'group.1um/fe2+generic_vbroad_br20': 'iron oxide',
                     'group.1um/fe3+_goethite+qtz.medgr.gds240': 'goethite',
                     'group.1um/fe3+_goethite.thincoat': 'goethite',
                     'group.1um/fe3+_hematite.nano.BR34b2': 'hematite',
                     'group.1um/fe3+_hematite.nano.BR34b2b': 'hematite',
                     'group.1um/fe3+copper-hydroxide_pitchlimonite': 'iron oxide',
                     'group.1um/fe3+mn_desert.varnish1': 'iron oxide',
                     'group.2um/calcite+0.2Na-mont': 'calcite',
                     'group.2um/calcite+0.5Ca-mont': 'calcite',
                     'group.2um/calcite.25+dolom.25+Na-mont.5': 'calcite',
                     'group.2um/carbonate_aragonite': 'carbonate',
                     'group.2um/carbonate_calcite': 'calcite',
                     'group.2um/carbonate_calcite+0.2Ca-mont': 'calcite',
                     'group.2um/carbonate_calcite+0.3muscovite': 'calcite',
                     'group.2um/carbonate_calcite0.7+kaol0.3': 'calcite',
                     'group.2um/carbonate_dolo+.5ca-mont': 'dolomite',
                     'group.2um/carbonate_dolomite': 'dolomite',
                     'group.2um/chlorite-skarn': 'chlorite',
                     'group.2um/kaolin+musc.intimat': 'kaolonite',
                     'group.2um/kaolin.5+muscov.medAl': 'kaolonite',
                     'group.2um/micagrp_lepidolite': 'mica',
                     'group.2um/micagrp_muscovite-low-Al': 'muscovite',
                     'group.2um/micagrp_muscovite-med-Al': 'muscovite',
                     'group.2um/micagrp_vermiculite_WS682': 'vermiculite',
                     'group.2um/organic_drygrass+.17Na-mont': 'npv',
                     'group.2um/organic_vegetation-dry-grass-golden': 'pv',
                     'group.2um/sioh_chalcedony': 'silicates',
                     'group.2um/sioh_hydrated_basaltic_glass': 'silicates',
                     'group.2um/smectite_montmorillonite_ca_swelling': 'montmorillonite',
                     'group.2um/smectite_montmorillonite_na_highswelling': 'montmorillonite',
                     'group.2um/smectite_nontronite_swelling': 'montmorillonite',
                     'none': 'none'}

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


def bin_sums(x, y, false_pos, false_neg):
    mae = []
    x_vals = []
    percent_false_pos = []
    percent_false_neg = []

    for col in range(x.shape[1]):
        fraction = x[0, col]
        vals = y[:, col]
        mae_calc = np.mean(vals)
        x_vals.append(fraction)

        mineral_false_neg = false_neg[:, col]
        mineral_false_pos = false_pos[:, col]

        mae.append(mae_calc)
        percent_false_neg.append(np.sum(mineral_false_neg != 0) / mineral_false_neg.shape[0])
        percent_false_pos.append(np.sum(mineral_false_pos != 0) / mineral_false_pos.shape[0])

    print(mae)
    print(x_vals)
    return x_vals, mae, percent_false_neg, percent_false_pos


def error_abundance_corrected(spectral_abundance_array, pure_soil_array, fractions, index):
    # correct spectral abundance, third dimension is the minerals
    error_grid = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], np.shape(spectral_abundance_array)[2]))
    false_positive_grid = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], np.shape(spectral_abundance_array)[2]))
    false_negative_grid = np.zeros((np.shape(fractions)[0], np.shape(fractions)[1], np.shape(spectral_abundance_array)[2]))

    for _row, row in enumerate(fractions):
        for _col, col in enumerate(row):
            soil_fractions = fractions[_row, _col, 2]

            soil_index = index[_row, 0, 2]

            # correct the abundances
            if int(np.round(soil_fractions, 2)) == 0:
                error_grid[_row, _col, :] = np.absolute(spectral_abundance_array[_row, _col, :] - pure_soil_array[int(soil_index), :])
            else:
                sa_c = spectral_abundance_array[_row, _col, :] / np.round(soil_fractions, 2)
                error = np.absolute(sa_c - pure_soil_array[int(soil_index), :])
                error_grid[_row, _col, :] = error

            # # fill out the detection grid
            # for _mineral, mineral in enumerate(pure_soil_array[int(soil_index), :]):
            #     if pure_soil_array[int(soil_index), _mineral] == 0 and spectral_abundance_array[row, _col, _mineral] != 0:
            #         false_positive_grid[_row, _col, _mineral] = 1
            #     elif pure_soil_array[int(soil_index), _mineral] != 0 and spectral_abundance_array[
            #         _row, _col, _mineral] == 0:
            #         false_negative_grid[_row, _col, _mineral] = 1

    return error_grid, false_positive_grid, false_negative_grid


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


class tetracorder_figures:
    def __init__(self, base_directory: str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'tetracorder', 'output')
        self.aug_directory = os.path.join(base_directory, 'tetracorder', 'output', 'augmented')
        self.slpit_output = os.path.join(base_directory, 'slpit', 'output')
        self.sa_outputs = os.path.join(base_directory, 'tetracorder', 'output', 'spectral_abundance')
        self.fig_directory = os.path.join(base_directory, 'tetracorder', 'figures')

        self.bands = load_band_names(
            os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_abun_mineral'))

        create_directory(os.path.join(self.fig_directory, 'plot_minerals'))

    def simulation_fig(self, xaxis: str):

        # load simulation data - truncate the sa files from augmentation; unmixing is ignored here!
        sim_index_array = envi_to_array(os.path.join(self.output_directory, f'tetracorder_{xaxis}_index'))
        sim_fractions_array = envi_to_array(os.path.join(self.output_directory, f'tetracorder_{xaxis}_fractions'))

        sim_sa_arrary = envi_to_array(
            os.path.join(self.sa_outputs, f'tetracorder_{xaxis}_spectra_simulation_augmented_abun_mineral'))[:, 0:21, :]
        soil_sa_sim_pure = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_abun_mineral'))[:, 0, :]

        error_grid, false_positive_grid, false_negative_grid = error_abundance_corrected(
            spectral_abundance_array=sim_sa_arrary,
            pure_soil_array=soil_sa_sim_pure,
            fractions=sim_fractions_array, index=sim_index_array)

        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        ncols = 5
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0, hspace=0, figure=fig)
        minor_tick_spacing = 0.1
        major_tick_spacing = 0.25
        counter = 0

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(self.bands[counter])
                ax.set_xlabel(f'{xaxis}')
                ax.grid('on', linestyle='--')
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                if col == 0:
                    ax.set_ylabel('MAE')

                if col != 0:
                    ax.set_yticklabels([])

                abs_error = error_grid[:, :, counter]
                mineral_false_positive = false_positive_grid[:, :, counter]
                mineral_false_negative = false_negative_grid[:, :, counter]

                if xaxis == 'npv':
                    fractions = sim_fractions_array[:, :, 0]

                if xaxis == 'pv':
                    fractions = sim_fractions_array[:, :, 1]

                if xaxis == 'soil':
                    fractions = sim_fractions_array[:, :, 2]

                x_vals, mae, percent_false_neg, percent_false_pos = bin_sums(x=fractions, y=abs_error,
                                                                             false_pos=mineral_false_positive,
                                                                             false_neg=mineral_false_negative)
                l1 = ax.plot(x_vals, mae, label='Baseline')
                ax.set_ylim(0.0, 0.25)
                ax.set_xlim(0.0, 1.05)

                lns = l1

                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc=0, prop={'size': 6})
                ax.set_aspect(1. / ax.get_data_ratio())

                counter += 1

        plt.savefig(os.path.join(self.fig_directory, 'tetracorder_mae_' + xaxis + '.png'), dpi=300, bbox_inches='tight')

    def mineral_validation(self):
        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
        df = df.sort_values('Name')
        rows_estimated = []
        rows_truth = []
        rows_soil_fractions = []

        for index, row in df.iterrows():
            plot = row['Name']
            emit_filetime = row['EMIT Date']
            abundance_emit = glob(os.path.join(self.sa_outputs,
                                               f'*{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_abun_mineral'))
            abundance_contact_probe = os.path.join(self.sa_outputs,
                                                   f'{plot.replace(" ", "").replace("SPEC", "Spectral")}-emit_ems_augmented_abun_mineral')
            fractional_cover = os.path.join(self.slpit_output, 'sma-best',
                                            f'asd-local___{plot.replace(" ", "")}___num-endmembers_20_n-mc_25_normalization_brightness_fractional_cover')

            # load arrays
            truth_array = envi_to_array(abundance_contact_probe)[0, 0, :]
            # truth_array[truth_array == 0] = np.nan
            estimated_array = envi_to_array(abundance_emit[0])
            # estimated_array[estimated_array == 0] = np.nan

            # load fractional cover
            plot_fractional_cover = np.average(envi_to_array(fractional_cover)[:, :, 2])

            plot_truth_abun = [plot]
            plot_estimated_abun = [plot]
            plot_soils = [plot, plot_fractional_cover]

            for _mineral, mineral in enumerate(self.bands):
                truth_abun = truth_array[_mineral]
                plot_truth_abun.append(truth_abun)
                estimated_abun = np.mean(estimated_array[0, 0, _mineral]) / plot_fractional_cover
                plot_estimated_abun.append(estimated_abun)

            # append plot abundances to master rows
            rows_estimated.append(plot_estimated_abun)
            rows_truth.append(plot_truth_abun)
            rows_soil_fractions.append(plot_soils)

        # master dfs
        df_est = pd.DataFrame(rows_estimated)
        df_est.columns = ['Plot'] + self.bands
        df_est = df_est.fillna(0)
        df_est.to_csv(os.path.join(self.sa_outputs, 'slpit-emit_estimated_abundance.csv'), index=False)

        df_truth = pd.DataFrame(rows_truth)
        df_truth.columns = ['Plot'] + self.bands
        df_truth = df_truth.fillna(0)
        df_truth.to_csv(os.path.join(self.sa_outputs, 'slpit-contact_estimated_abundance.csv'), index=False)

        df_soil = pd.DataFrame(rows_soil_fractions)
        df_soil.columns = ['Plot', 'soil_frac']
        df_soil.to_csv(os.path.join(self.sa_outputs, 'slpit_soil-fractions.csv'), index=False)

        # create figure
        fig = plt.figure(figsize=(12, 6))
        ncols = 5
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.10, figure=fig)
        minor_tick_spacing = 0.02
        major_tick_spacing = 0.05
        counter = 0

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(self.bands[counter])
                ax.set_xlabel(f'SLPIT (Contact Probe)')
                ax.grid('on', linestyle='--')
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))
                ax.set_xlim(-0.05, 0.25)
                ax.set_ylim(-0.05, 0.25)

                if col == 0:
                    ax.set_ylabel('EMIT Spectral Abundance')

                if col != 0:
                    ax.set_yticklabels([])

                p = ax.scatter(df_truth[self.bands[counter]], df_est[self.bands[counter]], c=df_soil['soil_frac'],
                               cmap='viridis', s=8)
                rmse = mean_squared_error(df_truth[self.bands[counter]], df_est[self.bands[counter]], squared=False)
                mae = mean_absolute_error(df_truth[self.bands[counter]], df_est[self.bands[counter]])
                r2 = r2_calculations(df_truth[self.bands[counter]], df_est[self.bands[counter]])

                # plot 1 to 1 line
                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red')

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(df_truth[self.bands[counter]]))))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=6,
                        verticalalignment='top', bbox=props)
                ax.set_aspect('equal', adjustable='box')
                counter += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(p, cax=cbar_ax, orientation='vertical', label='SLPIT Soil Fraction (%)')

        plt.savefig(os.path.join(self.fig_directory, 'slpit-contact-probe_emit.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    def mineral_error_soil(self):
        # import csvs with abundance estimates
        df_emit = pd.read_csv(os.path.join(self.sa_outputs, 'slpit-emit_estimated_abundance.csv'))
        df_emit = df_emit.fillna(0)
        df_contact = pd.read_csv(os.path.join(self.sa_outputs, 'slpit-contact_estimated_abundance.csv'))
        df_contact = df_contact.fillna(0)

        # these are the x-axis
        df_soil = pd.read_csv(os.path.join(self.sa_outputs, 'slpit_soil-fractions.csv'))

        # # create figure
        fig = plt.figure(figsize=(12, 6))
        ncols = 5
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.5, figure=fig)
        minor_tick_spacing = 0.02
        major_tick_spacing = 0.05
        counter = 0

        # dfs for master error
        dfs = []

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(self.bands[counter])
                ax.set_xlabel(f'Soil Fractions\n (SLPIT)')
                ax.grid('on', linestyle='--')
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))

                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{2}f'))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 0.2)

                if col == 0:
                    ax.set_ylabel('Absolute Abundance Error\n (Contact Probe - EMIT)')

                if col != 0:
                    ax.set_yticklabels([])

                error = np.absolute(df_contact[self.bands[counter]].values - df_emit[self.bands[counter]].values)
                soil_frac = df_soil['soil_frac'].values

                df_mineral = pd.DataFrame({'soil_frac': soil_frac, 'error': error})
                df_mineral = df_mineral.sort_values('soil_frac')
                ax.scatter(df_mineral['soil_frac'], df_mineral['error'])
                # ax.set_aspect('equal', adjustable='box')
                counter += 1

        plt.savefig(os.path.join(self.fig_directory, 'abundance_error_by_soil_cover.png'), dpi=300, bbox_inches='tight')
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

        emit_detections = []
        slpit_detections = []

        for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):
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

            emit_wvls, fwhm = spectra.load_wavelengths(sensor='emit')
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
            fs.set_title('Full Spectrum')
            fs.plot(emit_wvls, plot_spectra, label='SLPIT', c='blue')
            fs.plot(emit_wvls, emit_spectra, label='EMIT', c='orange')
            fs.set_ylim(0,1)
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

                                g1_s.plot(wl, cont, label=f'{file_label}', c='black', linestyle='dotted')
                                g1_s.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                g1_s.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

                            if group == 'group.2um':
                                split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                                cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(slpit_record), :], cont_feat['continuum'])
                                emit_cont, ewvls = cont_rem(wavelengths, emit_spectra, cont_feat['continuum'])

                                g2_s.plot(wl, cont, label=f'{file_label}', c='black', linestyle='dotted')
                                g2_s.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                g2_s.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

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

                                g1_e.plot(wl, cont, label=f'{file_label}', c='black', linestyle='dotted')
                                g1_e.plot(wvls, split_cont, label=f'SLPIT', c='blue')
                                g1_e.plot(ewvls, emit_cont, label=f'EMIT', c='orange')

                            if group == 'group.2um':
                                cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(emit_record), :], cont_feat['continuum'])
                                split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                                emit_cont, ewvls = cont_rem(wavelengths, emit_spectra, cont_feat['continuum'])

                                g2_e.plot(wl, cont, label=f'{file_label}', c='black', linestyle='dotted')
                                g2_e.plot(wvls, split_cont, label=f'SLPIT', c='blue')
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

        emit_detections = []
        slpit_detections = []

        for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):
            slpit_ems_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "") +
                                                  '*emit_ems_augmented_min'))

            emit_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
                                             '*pixels_augmented_min'))

            fractional_cover = os.path.join(self.slpit_output, 'sma-best', f'asd-local___{plot.replace(" ", "").replace("Spectral", "SPEC")}___num-endmembers_20_n-mc_25_normalization_brightness_fractional_cover')

            if frac_cover:
                soil_frac_cover = np.mean(envi_to_array(fractional_cover)[:,:,2])
                if soil_frac_cover > threshold:
                    continue

            # slpit data - g1
            if int(envi_to_array(slpit_ems_records[0])[0, 0, 1]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 1]), 'Filename'].iloc[0]
                slpit_detections.append(mineral_groupings[filename])
            else:
                slpit_detections.append('none')

            # slpit data - g2
            if int(envi_to_array(slpit_ems_records[0])[0, 0, 3]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 3]), 'Filename'].iloc[0]
                slpit_detections.append(mineral_groupings[filename])
            else:
                slpit_detections.append('none')

            # emit data - g1
            if int(envi_to_array(emit_records[0])[0, 0, 1]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(emit_records[0])[0, 0, 1]), 'Filename'].iloc[0]
                emit_detections.append(mineral_groupings[filename])
            else:
                emit_detections.append('none')

            if int(envi_to_array(emit_records[0])[0, 0, 3]) != 0:
                filename = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(emit_records[0])[0, 0, 3]), 'Filename'].iloc[0]
                emit_detections.append(mineral_groupings[filename])

            else:
                emit_detections.append('none')

        print(list(sorted(set(emit_detections + slpit_detections))))
        # write confusion matrix
        label_encoder = LabelEncoder()
        label_encoder.fit(slpit_detections + emit_detections)

        actual_values_encoded = label_encoder.transform(slpit_detections)
        predicted_values_encoded = label_encoder.transform(emit_detections)

        # Get the class labels
        class_labels = label_encoder.classes_
        cm = metrics.confusion_matrix(actual_values_encoded, predicted_values_encoded)

        plt.figure(figsize=(20, 18))
        plt.grid(True)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=90)
        plt.yticks(tick_marks, class_labels)

        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white', size=8)

        plt.xlabel('EMIT')
        plt.ylabel('SLPIT')
        plt.tight_layout()

        plt.savefig(os.path.join(self.fig_directory, 'mineral_confusion_matrix.png'), format="png", dpi=300,
                        bbox_inches="tight")

def run_figure_workflow(base_directory):
    ems = ['soil']
    tc = tetracorder_figures(base_directory=base_directory)
    #tc.confusion_matrix()
    tc.tetracorder_libraries()
    #tc.mineral_validation()
    #tc.mineral_error_soil()
    # for em in ems:
    #    tc.simulation_fig(xaxis=em)
