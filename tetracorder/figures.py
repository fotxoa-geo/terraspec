import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def bin_sums(x, y, bin_width:float):
    mae = []
    x_vals = []

    for i in np.linspace(0.0, 1.0, int(1/bin_width) + 1):
        i = np.round(i, 2)
        bin_min = x >= i
        bin_max = x <= i + bin_width
        mask = np.logical_and(bin_min, bin_max)
        y_select = y[mask]
        print(y_select)
        y_select = y_select[~np.isnan(y_select)]

        mae_calc = np.mean(y_select)
        mae.append(mae_calc)
        x_vals.append(i)

    return x_vals, mae


def error_abundance_corrected(spectral_abundance_array, pure_soil_array, fractions, index):
    # correct spectral abundance, third dimension is the minerals
    error_grid = np.zeros((np.shape(spectral_abundance_array)[0], np.shape(spectral_abundance_array)[1], np.shape(spectral_abundance_array)[2]))

    for _row, row in enumerate(spectral_abundance_array):
        for _col, col in enumerate(row):
            soil_fractions = fractions[_row, _col, 2]

            soil_index = index[_row, 0, 2]

            if np.round(soil_fractions, 2) == 0:
                pass
            else:
                sa_c = spectral_abundance_array[_row, _col, :] / np.round(soil_fractions, 2)
                error = np.absolute(sa_c - pure_soil_array[int(soil_index), :])
                error_grid[_row, _col, :] = error

    return error_grid


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

        self.bands = load_band_names(os.path.join(self.sa_outputs, 'simulated-soil_simulation_augmented_abun_mineral'))

    def simulation_fig(self, xaxis:str):

        # load simulation data - truncate the sa files from augmentation; unmixing is ignored here!
        sim_index_array = envi_to_array(os.path.join(self.aug_directory, 'tetracorder_index'))
        sim_fractions_array = envi_to_array(os.path.join(self.aug_directory, 'tetracorder_fractions'))

        sim_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, 'tetracorder_spectra_abun_mineral'))
        soil_sa_sim_pure = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_simulation_augmented_abun_mineral'))[:, 0, :]

        soil_sa_sim_pure[soil_sa_sim_pure == 0] = np.nan
        atmospheres = glob(os.path.join(self.sa_outputs, '*atm_*0_abun_mineral'))

        error_grid = error_abundance_corrected(spectral_abundance_array=sim_sa_arrary, pure_soil_array=soil_sa_sim_pure,
                                               fractions=sim_fractions_array, index=sim_index_array)
        error_grid[error_grid == 0] = np.nan

        atmospheres_abundances_corrected = []
        atmosphere_meta_information = []

        # for i in atmospheres:
        #     aod, h2o, sza = atmosphere_meta(i)
        #     atmos_sa = envi_to_array(i)
        #     atmosphere_error_grid = error_abundance_corrected(spectral_abundance_array=atmos_sa,
        #                                                       pure_soil_array=soil_sa_sim_pure,
        #                                                       fractions=sim_fractions_array, index=sim_index_array)
        #     atmospheres_abundances_corrected.append(atmosphere_error_grid)
        #     atmosphere_meta_information.append([aod, h2o, sza])


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
                if xaxis == 'npv':
                    fractions = sim_fractions_array[:, :, 0]

                if xaxis == 'pv':
                    fractions = sim_fractions_array[:, :, 1]

                if xaxis == 'soil':
                    fractions = sim_fractions_array[:, :, 2]

                x_vals, mae = bin_sums(x=fractions, y=abs_error, bin_width=0.10)
                ax.plot(x_vals, mae, label='Baseline')
                ax.set_ylim(0.0, 0.25)
                ax.set_xlim(-0.01, 1.05)

                # # plot the atmospheres
                # for _i, i in enumerate(atmospheres):
                #     atmosphere_error_grid = atmospheres_abundances_corrected[_i]
                #     aod, h2o, sza = atmosphere_meta_information[_i]
                #     atmos_abs_error = atmosphere_error_grid[:, :, counter]
                #     x_vals_atm, mae_atm = bin_sums(x=fractions, y=atmos_abs_error, bin_width=0.10)
                #     ax.plot(x_vals_atm, mae_atm, label=f'SZA: {str(sza)}, AOD: {aod}, H$_2$O: {h2o}')

                ax.legend(prop={'size': 6})
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

            # load arrays
            truth_array = envi_to_array(abundance_contact_probe)
            truth_array[truth_array == 0] = np.nan
            estimated_array = envi_to_array(abundance_emit[0])
            estimated_array[estimated_array == 0] = np.nan

            # load fractional cover
            plot_fractional_cover = np.average(envi_to_array(fractional_cover)[:, :, 2])

            plot_truth_abun = [plot]
            plot_estimated_abun = [plot]
            plot_soils = [plot, plot_fractional_cover]

            # load df for em position key
            em_csv = os.path.join(self.slpit_output, 'spectral_transects', 'endmembers', plot.replace(" ", "").replace('SPEC', 'Spectral') + '-emit.csv')
            df_em = pd.read_csv(em_csv)
            first_soil_index = df_em.index[df_em['level_1'] == 'Soil'].min()

            for _mineral, mineral in enumerate(self.bands):
                truth_abun = np.nanmean(truth_array[first_soil_index:, 0, _mineral])
                plot_truth_abun.append(truth_abun)
                estimated_abun = np.nanmean(estimated_array[:, 0:3, _mineral])
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
                ax.set_xlim(0, 0.15)
                ax.set_ylim(0, 0.15)

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

def run_figure_workflow(base_directory):

    ems = ['npv', 'pv', 'soil']
    tc = tetracorder_figures(base_directory=base_directory)
    #tc.mineral_validation()
    #tc.mineral_error_soil()
    for em in ems:
        #tc.phil_figure(xaxis=em)
        tc.simulation_fig(xaxis=em)
        #tc.mineral_incremenent_fig(xaxis=em)