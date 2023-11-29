import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd
import numpy as np
from utils.envi import envi_to_array, load_band_names
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

def bin_sums(x, y, false_pos, false_neg, bin_width:float):
    mae = []
    x_vals = []
    percent_false_pos = []
    percent_false_neg = []

    for col in range(x.shape[1]):
        fraction = x[0, col]
        vals = y[:, col]
        #vals[vals == 0] = np.nan
        mae_calc = np.mean((vals))
        x_vals.append(fraction)

        mineral_false_neg = false_neg[:, col]
        mineral_false_pos = false_pos[:, col]

        mae.append(mae_calc)
        percent_false_neg.append(np.sum(mineral_false_neg != 0)/mineral_false_neg.shape[0])
        percent_false_pos.append(np.sum(mineral_false_pos != 0) / mineral_false_pos.shape[0])

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
            if np.round(soil_fractions, 2) == 0:
                error_grid[_row, _col, :] = np.absolute(spectral_abundance_array[_row, _col, :] - pure_soil_array[int(soil_index), :])
            else:
                sa_c = spectral_abundance_array[_row, _col, :]/ np.round(soil_fractions, 2)
                error = np.absolute(sa_c - pure_soil_array[int(soil_index), :])
                error_grid[_row, _col, :] = error

            # fill out the detection grid
            for _mineral, mineral in enumerate(pure_soil_array[int(soil_index), :]):
                if pure_soil_array[int(soil_index), _mineral] == 0 and spectral_abundance_array[_row, _col, _mineral] != 0:
                    false_positive_grid[_row, _col, _mineral] = 1
                elif pure_soil_array[int(soil_index), _mineral] != 0 and spectral_abundance_array[_row, _col, _mineral] == 0:
                    false_negative_grid[_row, _col, _mineral] = 1

    return error_grid, false_positive_grid, false_negative_grid


def atmosphere_meta(atmosphere):
    basename = os.path.basename(atmosphere)
    aod = basename.split('_')[-4].replace('-', '.')
    h2o = basename.split('_')[-3].replace('-', '.')
    doy = basename.split('_')[1]

    time_dh = basename.split('_')[-6].replace('-', '.')

    hours = int(float(time_dh))
    minutes = (float(time_dh) * 60) %60
    seconds = (float(time_dh) * 3600) %60
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

        self.bands = load_band_names(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_abun_mineral'))

    def simulation_fig(self, xaxis:str):

        # load simulation data - truncate the sa files from augmentation; unmixing is ignored here!
        sim_index_array = envi_to_array(os.path.join(self.output_directory, f'tetracorder_{xaxis}_index'))
        sim_fractions_array = envi_to_array(os.path.join(self.output_directory, f'tetracorder_{xaxis}_fractions'))

        sim_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{xaxis}_spectra_simulation_augmented_abun_mineral'))[:, 0:11, :]
        soil_sa_sim_pure = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_abun_mineral'))[:, 0, :]

        atmospheres = glob(os.path.join(self.sa_outputs, '*atm_*0_abun_mineral'))

        error_grid, false_positive_grid, false_negative_grid = error_abundance_corrected(spectral_abundance_array=sim_sa_arrary,
                                                                                         pure_soil_array=soil_sa_sim_pure,
                                                                                         fractions=sim_fractions_array, index=sim_index_array)
        #error_grid[error_grid == 0] = np.nan

        atmospheres_abundances_corrected = []
        atmosphere_meta_information = []

        for i in atmospheres:
            aod, h2o, sza = atmosphere_meta(i)
            if float(aod) == 0.05 and float(h2o) == 0.75:

                if int(sza) == 13 or int(sza) == 57:
                    atmos_sa = envi_to_array(i)
                    atmosphere_error_grid = error_abundance_corrected(spectral_abundance_array=atmos_sa,
                                                                      pure_soil_array=soil_sa_sim_pure,
                                                                      fractions=sim_fractions_array, index=sim_index_array)
                    atmospheres_abundances_corrected.append(atmosphere_error_grid[0])
                    atmosphere_meta_information.append([aod, h2o, sza])
                else:
                    pass

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

                x_vals, mae, percent_false_neg, percent_false_pos = bin_sums(x=fractions, y=abs_error, false_pos=mineral_false_positive, false_neg=mineral_false_negative,
                                       bin_width=0.10)
                l1 = ax.plot(x_vals, mae, label='Baseline')
                ax.set_ylim(0.0, 0.25)
                ax.set_xlim(0.0, 1.05)

                ax2 = ax.twinx()
                l2 = ax2.plot(x_vals, np.absolute(percent_false_neg), label='% False Neg', color='r', linestyle='dotted')
                l3 = ax2.plot(x_vals, np.absolute(percent_false_pos), label='% False Pos', color='g', linestyle='dashed')
                ax2.set_ylim(0.0, 1)
                ax2.set_ylabel('% Detection')

                # if xaxis == 'soil':
                #     # plot the atmospheres
                #     for _i, i in enumerate(atmosphere_meta_information):
                #         atmosphere_error_grid = atmospheres_abundances_corrected[_i]
                #         aod, h2o, sza = atmosphere_meta_information[_i]
                #         atmos_abs_error = atmosphere_error_grid[:, :, counter]
                #         x_vals_atm, mae_atm, percent_false_neg, percent_false_pos = bin_sums(x=fractions, y=atmos_abs_error, false_pos=mineral_false_positive, false_neg=mineral_false_negative,bin_width=0.10)
                #         l4 = ax.plot(x_vals_atm, mae_atm, label=f'SZA: {str(sza)}, AOD: {aod}, H$_2$O: {h2o}')

                # added these three lines
                if xaxis == 'soil':
                    lns = l1 + l2 + l3 # + l4
                else:
                    lns = l1 + l2 + l3
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
            abundance_emit = glob(os.path.join(self.sa_outputs, f'*{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_abun_mineral'))
            abundance_contact_probe = os.path.join(self.sa_outputs, f'{plot.replace(" ", "").replace("SPEC", "Spectral")}-emit_ems_augmented_abun_mineral')
            fractional_cover = os.path.join(self.slpit_output, 'sma-best', f'asd-local___{plot.replace(" ", "")}___num-endmembers_20_n-mc_25_normalization_brightness_fractional_cover')

            print(abundance_emit)
            # load arrays
            truth_array = envi_to_array(abundance_contact_probe)[0,0,:]
            #truth_array[truth_array == 0] = np.nan
            estimated_array = envi_to_array(abundance_emit[0])
            #estimated_array[estimated_array == 0] = np.nan

            # load fractional cover
            plot_fractional_cover = np.average(envi_to_array(fractional_cover)[:, :, 2])

            plot_truth_abun = [plot]
            plot_estimated_abun = [plot]
            plot_soils = [plot, plot_fractional_cover]

            for _mineral, mineral in enumerate(self.bands):
                truth_abun = truth_array[_mineral]
                plot_truth_abun.append(truth_abun)
                estimated_abun = np.mean(estimated_array[0:3, 0:3, _mineral])/plot_fractional_cover
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

                p = ax.scatter(df_truth[self.bands[counter]], df_est[self.bands[counter]], c=df_soil['soil_frac'], cmap='viridis', s=8)
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
                #ax.set_aspect('equal', adjustable='box')
                counter += 1

        plt.savefig(os.path.join(self.fig_directory, 'abundance_error_by_soil_cover.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    def tetracorder_libraries(self):

        # TODO: Get these from....direct input?  Configuration file?
        MINERAL_FRACTION_FILES = [ \
            'calcite.group2.txt',
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

        SPECTRAL_REFERENCE_LIBRARY = { \
            'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
            'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi'),
        }

        decoded_expert = tetracorder.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                          log_file=None, log_level='INFO')
        print(list(decoded_expert)[0])
        mff = [os.path.join('utils', 'tetracorder', 'minerals', x) for x in MINERAL_FRACTION_FILES]
        mineral_fractions = tetracorder.read_mineral_fractions(mff)
        unique_file_names, fractions, scaling, library_names, records, reference_band_depths = unique_file_fractions(mineral_fractions, decoded_expert)



        data = {
            'unique_file_names': unique_file_names,
            'scaling': scaling,
            'library_names': library_names,
            'records' : records,
            'reference_band_depths': reference_band_depths
        }

        df = pd.DataFrame(data)
        print(df)
        df.to_csv(r'G:\My Drive\test\decoded_expert.csv', index=False)
        # spectral_reference_library_files = SPECTRAL_REFERENCE_LIBRARY
        # libraries = {}
        # cmap = cm.get_cmap('viridis')
        # ind = 0
        #
        # fig = plt.figure(figsize=(15, 10))
        # gs = gridspec.GridSpec(2, 3, figure=fig)
        # ax1 = fig.add_subplot(gs[0, 0])
        # ax2 = fig.add_subplot(gs[0, 1])
        # ax3 = fig.add_subplot(gs[1, :2])
        # ax4 = fig.add_subplot(gs[:, 2])
        # for key, item in spectral_reference_library_files.items():
        #     ind += 1
        #     library = envi.open(envi_header(item), item)
        #     library_reflectance = library.spectra.copy()
        #     library_records = [int(q) for q in library.metadata['record']]
        #
        #     hdr = envi.read_envi_header(envi_header(item))
        #     wavelengths = np.array([float(q) for q in hdr['wavelength']])
        #
        #     if ';;;' in key:
        #         key = key.replace(';;;', ',')
        #         logging.debug(f'found comma replacement, now: {key}')
        #
        #     libraries[key] = {'reflectance': library_reflectance,
        #                       'library_records': library_records, 'wavelengths': wavelengths}
        #
        #     band_depths = np.zeros(fractions.shape[0])
        #
        #     for _f, (frac, filename, library_name, record) in enumerate(
        #             zip(fractions, unique_file_names, library_names.tolist(), records.tolist())):
        #         if library_name == key:
        #             for cont_feat in decoded_expert[filename.split('.depth.gz')[0]]['features']:
        #                 if np.all(np.array(cont_feat['continuum']) < 0.8):
        #                     cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(record), :],
        #                                         cont_feat['continuum'])
        #                     ax1.plot(wl, cont,
        #                              label=f'{frac[0]} ||| {os.path.basename(unique_file_names[_f]).split(".depth.gz")[0]}')
        #                 if np.all(np.array(cont_feat['continuum']) > 0.7):
        #                     cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(record), :],
        #                                         cont_feat['continuum'])
        #                     ax2.plot(wl, cont, label=frac[0])
        #
        #             ax3.plot(wavelengths, library_reflectance[library_records.index(record), :], label=frac[0])
#
#             handles, labels = ax1.get_legend_handles_labels()
#             order = np.argsort(labels)
#             handles = np.array(handles)[order].tolist()
#             labels = np.array(labels)[order].tolist()
#             ax4.legend(handles, labels)
#             ax4.axis('off')
#
#             ax1.set_title('Continuum Removed R1')
#             ax2.set_title('Continuum Removed R2')
#             ax3.set_title('Full Spectrum')
#             ax4.set_title('XRD Quantity || Mineral Name')
def run_figure_workflow(base_directory):

    ems = ['npv', 'pv', 'soil']
    tc = tetracorder_figures(base_directory=base_directory)
    tc.tetracorder_libraries()
    #tc.mineral_validation()
    #tc.mineral_error_soil()
    # for em in ems:
    #     tc.simulation_fig(xaxis=em)
