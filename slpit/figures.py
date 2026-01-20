import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from glob import glob

import pytz
from osgeo import gdal
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.image as mpimg
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.envi import envi_to_array, load_band_names, read_metadata
from datetime import datetime, timezone
from p_tqdm import p_map
from isofit.core.sunposition import sunpos
import geopandas as gp
from utils.results_utils import r2_calculations, load_data, error_metrics
from matplotlib.ticker import FormatStrFormatter
from spectral.io import envi
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.plot import show
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib_scalebar.scalebar import ScaleBar

def duplicate_check_fractions(array):
    seen = set()
    duplicate_flag = 0

    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] in seen:
                array[i][j] = np.nan
                duplicate_flag = 1
            else:
                seen.add(array[i][j])
    
    return array, duplicate_flag


def fraction_file_info(fraction_file):
    name = os.path.basename(fraction_file)
    unmix_mode = os.path.basename(os.path.dirname(fraction_file))

    library_mode = name.split("___")[0].split('-')[1]
    instrument = name.split("___")[0].split('-')[0]
    plot = name.split("___")[1]

    num_cmb_em = name.split("___")[2].split('_')[1]
    num_mc = name.split("___")[2].split('_')[3]
    normalization = name.split("___")[2].split('_')[5]

    fraction_array = envi_to_array(fraction_file)

    unc_path = os.path.join(f'{fraction_file}_uncertainty')
    unc_array = envi_to_array(unc_path)

    mean_fractions = []
    mean_se = []
    mean_sigma = []
    mean_use = []

    for _band, band in enumerate(range(0, fraction_array.shape[2])):
            
            if instrument == 'asd':
                selected_fractions = fraction_array[:, :, _band]
                selected_unc = unc_array[:, :, _band]
            else:
                selected_fractions = fraction_array[:, :, _band]
                selected_unc = unc_array[:, :, _band]

            selected_fractions = np.where(selected_fractions == -9999, np.nan, selected_fractions)
            selected_fractions, duplicate_flag = duplicate_check_fractions(selected_fractions)
            
            selected_unc = np.where(selected_unc == -9999, np.nan, selected_unc)
            mean_fractions.append(np.nanmean(selected_fractions))

            # calculate se
            se = np.nanmean(selected_unc/np.sqrt(int(num_mc)))
            mean_se.append(se)

            # caluclate mean sigma
            sigma = np.nanmean(selected_unc)
            mean_sigma.append(sigma)

            # calculate U_se
            sstd = selected_unc.flatten()
            sum_square_sstd = np.nansum(np.square(sstd))
            use = np.sqrt(sum_square_sstd)/sstd.shape[0]
            mean_use.append(use)

    return [instrument, unmix_mode, plot, library_mode, int(num_cmb_em), int(num_mc), normalization, fraction_array.shape[0], fraction_array.shape[1], duplicate_flag] + mean_fractions + mean_se + mean_sigma + mean_use

class figures:
    def __init__(self, base_directory: str, sensor: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory,  "figures")
        self.gis_directory = os.path.join(base_directory, "gis")

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)
        self.good_sensor_bands = spectra.get_good_bands_mask(self.wvls, wavelength_pairs=None)
        self.wvls[~self.good_sensor_bands] = np.nan

        create_directory(os.path.join(base_directory, "figures"))

        # ems
        self.ems = ['NPV', 'GV', 'Soil']
        self.cmap_kw = 'Accent'

        # figure fonts, font size, etc
        self.major_axis_fontsize = major_axis_fontsize
        self.minor_axis_fontsize = minor_axis_fontsize
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.fig_height = fig_height
        self.fig_width = fig_width
        self.linewidth = linewidth
        self.sig_figs = sig_figs
        self.cmap_kw = 'copper'
        self.axes_limits = {
            'ymin': 0,
            'ymax': 1,
            'xmin': 0,
            'xmax': 1}

    def plot_summary(self):
        # spectral data directories
        create_directory(os.path.join(self.fig_directory, 'plot_stats'))
        spectral_transects_directories = glob(os.path.join(self.output_directory, 'spectral_transects', '**'))

        # load gis data
        df_gis = gp.read_file(os.path.join('gis', "Observation.json"))
        df_gis['longitude'] = df_gis.geometry.x
        df_gis['latitude'] = df_gis.geometry.y
        df_gis = pd.DataFrame(df_gis.drop(columns='geometry'))
        df_gis['Name'] = df_gis['Name'].str.replace(' ', '', regex=False)
        df_gis = df_gis.sort_values('Name')

        # load tetracorder mineral grouping tables
        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))

        for spectral_transect_directory in spectral_transects_directories:
            plot_name = os.path.basename(spectral_transect_directory)

            # load rfl data
            slpit_rfl = envi_to_array(os.path.join(spectral_transect_directory, 'RFL', f'{plot_name}_SLPIT_emit'))
            slpit_rfl[slpit_rfl == -9999] = np.nan
            df_slpit_rfl = pd.read_csv(os.path.join(spectral_transect_directory, 'RFL', f'{plot_name}_SLPIT_emit.csv'))
            df_em_spectra = pd.read_csv(os.path.join(spectral_transect_directory, 'EMS', f'{plot_name}_EMS_emit.csv'))

            # get gis data
            df_transect = df_gis.loc[df_gis['Name'] == plot_name.replace("Spectral", 'SPEC')].copy()

            # Create figure and subplots
            fig = plt.figure(figsize=(14, 10))

            # plot map
            ax_map = plt.subplot2grid((16, 16), (0, 0), colspan=6, rowspan=7,
                                      projection=ccrs.PlateCarree())
            ax_map.set_title('Plot Map')
            ax_map.set_global()
            ax_map.set_xlim(-125, -90)  # Longitude range
            ax_map.set_ylim(25, 45)  # Latitude range
            ax_map.add_feature(cfeature.LAND)
            ax_map.add_feature(cfeature.OCEAN)
            ax_map.add_feature(cfeature.BORDERS, linestyle=':')
            ax_map.coastlines()
            ax_map.plot(np.mean(df_transect.longitude), np.mean(df_transect.latitude), marker='o', color='red',
                        markersize=8, transform=ccrs.PlateCarree())

            # add states
            states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale='50m',
                facecolor='none')
            ax_map.add_feature(states_provinces, edgecolor='gray', linewidth=0.5)

            # Enable gridlines and labels
            gl = ax_map.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
            gl.top_labels = False  # Turn off labels at the top
            gl.right_labels = False

            # plot landsacpe image
            ax_landspace_pic = plt.subplot2grid((16, 16), (0, 6), colspan=3, rowspan=4)
            ax_landspace_pic.set_title('Landscape\nPicture')
            pic_path = os.path.join(spectral_transect_directory, f'{plot_name}_landscape_pic.jpg')
            img = mpimg.imread(pic_path)
            ax_landspace_pic.imshow(img)
            ax_landspace_pic.axis('off')

            # Axes for the reflectance plot (main area)
            ax_rfl_plot = plt.subplot2grid((16, 16), (8, 0), colspan=8, rowspan=8)
            ax_rfl_plot.set_xlabel('Wavelength (nm)')
            ax_rfl_plot.set_ylabel('Reflectance (%)')

            ax_rfl_plot.set_xlim(300, 2550)
            ax_rfl_plot.xaxis.set_major_locator(MultipleLocator(100))
            ax_rfl_plot.xaxis.set_minor_locator(MultipleLocator(50))
            ax_rfl_plot.set_ylim(0, 1)
            ax_rfl_plot.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_rfl_plot.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax_rfl_plot.tick_params(axis='x', rotation=45)

            # # get date and time from slpit
            slpit_date = df_slpit_rfl['date'].unique()[0]
            slpit_datetime = datetime.strptime(df_slpit_rfl['date'].unique()[0], "%Y-%m-%d")

            # Calculate mean time of ASD  collections
            ax_rfl_plot.set_title(f'SLPIT Acquistion Date: {slpit_date}')

            y_mean = np.nanmean(slpit_rfl, axis=(0, 1))
            y_std = np.nanstd(slpit_rfl, axis=(0, 1))
            # plot slpit data
            ax_rfl_plot.plot(self.wvls, y_mean, label=f"SLPIT mean", linewidth=2, color='black')

            # fill 1 sigma
            ax_rfl_plot.fill_between(self.wvls, y_mean - y_std*2, y_mean + y_std*2,
                                      color='grey', alpha=0.2)

            # plot sensor data
            sensor_rfls = glob(os.path.join(spectral_transect_directory, 'EXT', f'*_RFL_*_EXT'))
            sensor_cmap = plt.cm.get_cmap('jet', len(sensor_rfls))

            for _, sensor_rfl_file in enumerate(sensor_rfls):
                acquisition_date = os.path.basename(sensor_rfl_file).split("_")[-2]
                sensor_rfl = envi_to_array(sensor_rfl_file)
                sensor_rfl[sensor_rfl == -9999] = np.nan

                # calculate geometries
                acquisition_datetime_utc = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc)
                geometry_results_sensor = sunpos(acquisition_datetime_utc, np.mean(df_transect.latitude),
                                                 np.mean(df_transect.longitude), np.mean(df_slpit_rfl.elevation))
                acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
                delta = slpit_datetime - acquisition_datetime
                days = np.absolute(delta.days)

                base_label = f'{acquisition_date} (±{days:02d} days)  SZA : {str(int(geometry_results_sensor[1]))}°'
                sensor_mean = np.nanmean(sensor_rfl, axis=(0, 1))
                sensor_std = np.nanstd(sensor_rfl, axis=(0, 1))

                ax_rfl_plot.plot(self.wvls, sensor_mean, label=base_label, linewidth=1, color=sensor_cmap(_))
                ax_rfl_plot.fill_between(self.wvls, sensor_mean - sensor_std*2, sensor_mean + sensor_std*2,
                                         color=sensor_cmap(_), alpha=0.2)

            ax_rfl_plot.legend()

            # plot NPV endmembers
            ax_npv_spectra = plt.subplot2grid((16, 16), (0, 9), colspan=7, rowspan=4)
            df_spectra = df_em_spectra[(df_em_spectra['level_1'] == 'NPV')].copy()
            df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
            num_species = len(sorted(list(df_spectra.species.unique())))
            npv_cmap = plt.cm.get_cmap(self.cmap_kw, num_species)
            unique_species = sorted(df_spectra['species'].unique())

            for i, species in enumerate(unique_species):
                # Filter and extract spectra data in one go
                em_spectra = df_spectra[df_spectra['species'] == species].iloc[:, 11:].to_numpy()

                # Plot all rows at once. Transposing em_spectra (T) allows .plot()
                # to handle all lines in a single call.
                color = npv_cmap(i)
                ax_npv_spectra.plot(self.wvls, em_spectra.T, color=color, alpha=0.5)

                # Get the label and add a single dummy entry for the legend
                common_name = df_species_key.loc[df_species_key['key_value'] == species, 'label'].iloc[0]
                ax_npv_spectra.plot([], [], color=color, label=common_name)

            ax_npv_spectra.set_xlim(320, 2550)
            ax_npv_spectra.xaxis.set_major_locator(MultipleLocator(100))
            ax_npv_spectra.xaxis.set_minor_locator(MultipleLocator(50))
            ax_npv_spectra.get_xaxis().set_ticklabels([])

            ax_npv_spectra.set_ylim(0, 1)
            ax_npv_spectra.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_npv_spectra.yaxis.set_minor_locator(MultipleLocator(0.1))

            ax_npv_spectra.text(385, 0.85, f"NPV (n = {str(df_spectra.shape[0])})", fontsize=12)
            ax_npv_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            # plot PV endmembers
            ax_pv_spectra = plt.subplot2grid((16, 16), (4, 9), colspan=7, rowspan=4)
            df_spectra = df_em_spectra[(df_em_spectra['level_1'] == 'PV')].copy()
            num_species = len(sorted(list(df_spectra.species.unique())))
            pv_cmap = plt.cm.get_cmap(self.cmap_kw, num_species)
            unique_species = sorted(df_spectra['species'].unique())

            for i, species in enumerate(unique_species):
                # Filter and extract spectra data in one go
                em_spectra = df_spectra[df_spectra['species'] == species].iloc[:, 11:].to_numpy()

                # Plot all rows at once. Transposing em_spectra (T) allows .plot()
                # to handle all lines in a single call.
                color = pv_cmap(i)
                ax_pv_spectra.plot(self.wvls, em_spectra.T, color=color, alpha=0.5)

                # Get the label and add a single dummy entry for the legend
                common_name = df_species_key.loc[df_species_key['key_value'] == species, 'label'].iloc[0]
                ax_pv_spectra.plot([], [], color=color, label=common_name)

            ax_pv_spectra.set_xlim(320, 2550)
            ax_pv_spectra.xaxis.set_major_locator(MultipleLocator(100))
            ax_pv_spectra.xaxis.set_minor_locator(MultipleLocator(50))
            ax_pv_spectra.get_xaxis().set_ticklabels([])

            ax_pv_spectra.set_ylim(0, 1)
            ax_pv_spectra.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_pv_spectra.yaxis.set_minor_locator(MultipleLocator(0.1))

            ax_pv_spectra.text(385, 0.85, f"PV (n = {str(df_spectra.shape[0])})", fontsize=12)
            ax_pv_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            # plot soil endmembers
            ax_soil_spectra = plt.subplot2grid((16, 16), (8, 9), colspan=7, rowspan=4)
            df_spectra = df_em_spectra[(df_em_spectra['level_1'] == 'Soil')].copy()
            soil_index_min = min(df_spectra.index[df_spectra['level_1'] == 'Soil'].tolist())
            soil_index_max = max(df_spectra.index[df_spectra['level_1'] == 'Soil'].tolist())
            soil_tetracorder_results = envi_to_array(os.path.join(spectral_transect_directory, 'tetracorder',
                                                                  f'{plot_name}_EMS_emit_augmented_min'))[soil_index_min:soil_index_max + 1, :, :]

            g1_unique = soil_tetracorder_results[:, 0, 1]
            g2_unique = soil_tetracorder_results[:, 0, 3]
            df_spectra.insert(0, "g1_minerals", g1_unique)
            df_spectra.insert(0, "g2_minerals", g2_unique)

            df_spectra['g1_minerals'] = df_spectra['g1_minerals'].map(df_mineral_matrix.set_index('Index')['Name'])
            df_spectra['g2_minerals'] = df_spectra['g2_minerals'].map(df_mineral_matrix.set_index('Index')['Name'])
            df_spectra['g1_minerals'] = df_spectra['g1_minerals'].fillna('No Detection')
            df_spectra['g2_minerals'] = df_spectra['g2_minerals'].fillna('No Detection')

            num_minerals_g1 = len(sorted(list(df_spectra.g1_minerals.unique())))
            g1_minerals_cmap = plt.cm.get_cmap('jet', num_minerals_g1)
            unique_minerals_g1 = sorted(df_spectra['g1_minerals'].unique())

            # plot g1 minerals
            for i, soils_unique in enumerate(unique_minerals_g1):

                # Filter and extract spectra data in one go
                em_spectra = df_spectra[df_spectra['g1_minerals'] == soils_unique].iloc[:, 13:].to_numpy()

                # Plot all rows at once. Transposing em_spectra (T) allows .plot()
                # to handle all lines in a single call.
                color = g1_minerals_cmap(i)
                ax_soil_spectra.plot(self.wvls, em_spectra.T, color=color, alpha=0.5)

                # Get the label and add a single dummy entry for the legend
                ax_soil_spectra.plot([], [], color=color, label=soils_unique)

            ax_soil_spectra.set_xlim(320, 2550)
            ax_soil_spectra.xaxis.set_major_locator(MultipleLocator(100))
            ax_soil_spectra.xaxis.set_minor_locator(MultipleLocator(50))
            ax_soil_spectra.get_xaxis().set_ticklabels([])

            ax_soil_spectra.set_ylim(0, 1)
            ax_soil_spectra.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_soil_spectra.yaxis.set_minor_locator(MultipleLocator(0.1))

            ax_soil_spectra.text(385, 0.85, f"Soil (n = {str(df_spectra.shape[0])})", fontsize=12)
            ax_soil_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            # plot group 2 minerals
            ax_soil_spectra2 = plt.subplot2grid((16, 16), (12, 9), colspan=7, rowspan=4)
            num_minerals_g2 = len(sorted(list(df_spectra.g2_minerals.unique())))
            g2_minerals_cmap = plt.cm.get_cmap('viridis', num_minerals_g2)
            unique_minerals_g2 = sorted(df_spectra['g2_minerals'].unique())

            # plot g2 minerals
            for i, soils_unique in enumerate(unique_minerals_g2):
                # Filter and extract spectra data in one go
                em_spectra = df_spectra[df_spectra['g2_minerals'] == soils_unique].iloc[:, 13:].to_numpy()

                # Plot all rows at once. Transposing em_spectra (T) allows .plot()
                # to handle all lines in a single call.
                color = g2_minerals_cmap(i)
                ax_soil_spectra2.plot(self.wvls, em_spectra.T, color=color, alpha=0.5)

                # Get the label and add a single dummy entry for the legend
                ax_soil_spectra2.plot([], [], color=color, label=soils_unique)

            ax_soil_spectra2.set_xlim(320, 2550)
            ax_soil_spectra2.xaxis.set_major_locator(MultipleLocator(100))
            ax_soil_spectra2.xaxis.set_minor_locator(MultipleLocator(50))
            ax_soil_spectra2.tick_params(axis='x', rotation=45)

            ax_soil_spectra2.set_ylim(0, 1)
            ax_soil_spectra2.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_soil_spectra2.yaxis.set_minor_locator(MultipleLocator(0.1))

            ax_soil_spectra2.text(385, 0.85, f"Soil (n = {str(df_spectra.shape[0])})", fontsize=12)
            ax_soil_spectra2.legend(prop={'size': 6}, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', f'{plot_name}.pdf'), format="pdf", dpi=300,
                        bbox_inches="tight")
            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', f'{plot_name}.png'), format="png", dpi=300,
                        bbox_inches="tight")
            plt.clf()
            plt.close()

    def plot_rmse(self, norm_option):
        df_all = pd.read_csv(os.path.join(self.fig_directory, 'fraction_output.csv'))
        df_all['Team'] = df_all['plot'].str.split('-').str[0].str.strip()
        df_all = df_all[df_all['Team'] != 'THERM']

        # load cpu performance results
        df_cpu = pd.read_csv(os.path.join(self.fig_directory, 'computing_performance_report_emit.csv'))
        df_cpu['normalization'] = df_cpu['normalization'].str.strip('"')
        df_cpu['instrument'] = df_cpu['reflectance_file'].str.strip('"').apply(lambda x: os.path.basename(x).split('___')[0].split('-')[1])
        df_cpu['Team'] = df_cpu['reflectance_file'].str.strip('"').apply(lambda x: os.path.basename(x).split('___')[1].split('-')[0])

        # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        col_map = {0: 'npv', 1: 'pv', 2: 'soil'}

        # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'local') & (df_cpu['mode'] == 'sma') & (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'emit')].copy()
            if row == 1:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'global') & (df_cpu['mode'] == 'sma')& (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'emit')].copy()
            if row == 2:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) &  (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'local') & (df_cpu['mode'] == 'mesma')& (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'emit')].copy()
            if row == 3:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) &  (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'global') & (df_cpu['mode'] == 'mesma') & (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'emit')].copy()

            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                mode = list(df_select['unmix_mode'].unique())[0]
                lib_mode = list(df_select['lib_mode'].unique())[0]
                n_mc = list(df_select['num_mc'].unique())[0]
                n_cmbs = list(df_select['num_cmb_em'].unique())[0]

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 3 and col == 1:
                    ax.set_xlabel("SLPIT", fontsize=self.axis_label_fontsize)

                if row == 3 and col != 0:
                    ax.set_xticklabels([''] + ax.get_xticklabels()[1:])

                if col == 0:
                    if mode == 'sma':
                        mode_to_plot = 'E(MC)$^2$'
                    else:
                        mode_to_plot = "MESMA"
                    # ax.set_ylabel(mode_to_plot.upper() + '$_{' + lib_mode + '}$',
                    #               fontsize=self.axis_label_fontsize)

                    ax.set_ylabel(mode_to_plot.upper(),
                                  fontsize=self.axis_label_fontsize)

                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.2, 0.2))

                if col != 0:
                    ax.set_yticklabels([])

                if row != 3:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                df_x = df_select[(df_select['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y = df_select[(df_select['instrument'] == 'emit')].copy().reset_index(drop=True)

                # plot fractional cover values
                x = df_x[col_map[col]]
                y = df_y[col_map[col]]
                x_u = df_x[f'{col_map[col]}_se']
                y_u = df_y[f'{col_map[col]}_se']

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None',capsize=5)
                ax.scatter(x, y, marker='s', edgecolor='black', color='blue', label='EMIT', zorder=10)

                performance = df_performance['spectra_per_s'].mean()

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae,rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),
                    #r'CPU: %.2f' % (performance,)
                ))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        fig.supylabel('EMIT Fractions', fontsize=self.axis_label_fontsize)
        plt.savefig(os.path.join(self.fig_directory, f'regression_{norm_option}.png'), format="png", dpi=400, bbox_inches="tight")


    def sza_plot(self, norm_option):
        print('loading sza plot...')

        df_rows = []
        # gis shapefile
        gdf = gp.read_file(os.path.join('gis', "Observation.shp"))
        df_gis = gdf.drop(columns='geometry')
        df_gis['latitude'] = gdf['geometry'].apply(lambda geom: geom.y)
        df_gis['longitude'] = gdf['geometry'].apply(lambda geom: geom.x)
        df_gis = df_gis.sort_values('Name')
        df_gis['Team'] = df_gis['Name'].str.split('-').str[0].str.strip()
        df_gis = df_gis[df_gis['Team'] != 'THERM']

        # transect data for elevation
        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-emit.csv'))

        # load fraction outputs
        df_all = pd.read_csv(os.path.join(self.fig_directory, 'fraction_output.csv'))
        df_all['Team'] = df_all['plot'].str.split('-').str[0].str.strip()
        df_all = df_all[df_all['Team'] != 'THERM']

        df_rows = []
        for index, row in df_gis.iterrows():
            plot = row['Name']
            plot_num = int(plot.split('-')[1])

            if plot_num > 60:
                continue

            else:
                emit_filetime = row['EMIT DATE']

                df_transect = transect_data.loc[transect_data['plot_name'] == plot.replace("SPEC", "Spectral")].copy()
                acquisition_datetime_utc = datetime.strptime(emit_filetime, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                geometry_results_emit = sunpos(acquisition_datetime_utc, row['latitude'], row['longitude'], np.mean(df_transect['elevation']))

                df_row = [plot, geometry_results_emit[1]]
                df_rows.append(df_row)

        df = pd.DataFrame(df_rows)
        df.columns = ['plot', 'sza']
        df = df.sort_values('plot')
        df = df.dropna()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.05, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        for row in range(nrows):
            if row == 0:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['normalization'] == norm_option)].copy()

            if row == 1:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['normalization'] == norm_option)].copy()

            if row == 2:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) & (df_all['normalization'] == norm_option)].copy()

            if row == 3:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) & (df_all['normalization'] == norm_option)].copy()

            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], 0.4)
                ax.set_xlim(10, 60)
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                mode = list(df_select['unmix_mode'].unique())[0]
                lib_mode = list(df_select['lib_mode'].unique())[0]

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 3 and col == 1:
                    ax.set_xlabel("Solar Zenith Angles (°)", fontsize=self.axis_label_fontsize)

                if row == 3 and col != 0:
                    ax.set_xticklabels([''] + ax.get_xticklabels()[1:])

                if col == 0:
                    if mode == 'sma':
                        mode = 'E(MC)$^2$'

                    ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$',
                                  fontsize=self.axis_label_fontsize)

                ax.set_yticks(np.arange(self.axes_limits['ymin'], 0.4 + 0.05, 0.05))

                if col != 0:
                    ax.set_yticklabels([])

                if row != 3:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                df_x = df_select[(df_select['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_x = df_x.sort_values('plot')
                df_y = df_select[(df_select['instrument'] == 'emit')].copy().reset_index(drop=True)
                df_y = df_y.sort_values('plot')

                # plot fractional cover values
                x = df_x[col_map[col]].values
                y = df_y[col_map[col]].values

                abs_error = np.absolute(x-y)
                sza_vals = df['sza'].values

                ax.scatter(sza_vals, abs_error)
                r2 = r2_calculations(sza_vals, abs_error)
                txtstr = '\n'.join((
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x))))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        fig.supylabel('Absolute Error', fontsize=self.axis_label_fontsize)
        plt.savefig(os.path.join(self.fig_directory, f'sza_mae_{norm_option}.png'), format="png", dpi=300, bbox_inches="tight")


    def local_slpit(self):
        # load all fraction files
        df_all = pd.read_csv(os.path.join(self.fig_directory, 'fraction_output.csv'))
        df_all['Team'] = df_all['plot'].str.split('-').str[0].str.strip()
        df_all = df_all[df_all['Team'] != 'THERM']

        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        col_map = {0: 'npv', 1: 'pv', 2: 'soil'}

        # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local')].copy()
                
            if row == 1:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100)].copy()
                
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.grid('on', linestyle='--')
                ax.set_xlabel('SLPIT Fractions')
                ax.set_ylabel("EMIT Fractions")

                ax.set_aspect(1. / ax.get_data_ratio())

                ax.set_title(f'{self.ems[col]}')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # plot 1 to 1 line
                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red')

                df_x = df_select[(df_select['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y = df_select[(df_select['instrument'] == 'emit')].copy().reset_index(drop=True)

                # plot fractional cover values
                x = df_x[col_map[col]]
                y = df_y[col_map[col]]

                cmap = plt.get_cmap('viridis') 
                c = list(range(1, len(df_x['plot'].values) + 1))
                #ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', markersize=4, zorder=1)
                scatter = ax.scatter(x, y, c=c, cmap=cmap, edgecolor='black')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae,rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x))))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
        
        fig.colorbar(scatter, label='Plot Number')
        plt.savefig(os.path.join(self.fig_directory, 'regression_local.png'), format="png", dpi=300, bbox_inches="tight")




def run_figures(base_directory, sensor):
    base_directory = base_directory
    major_axis_fontsize = 14
    minor_axis_fontsize = 12
    title_fontsize = 22
    axis_label_fontsize = 20
    fig_height = 12
    fig_width = 12
    linewidth = 1
    sig_figs = 2


    fig = figures(base_directory=base_directory, sensor=sensor, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs)

    fig.plot_summary()
    #fig.plot_rmse(norm_option='brightness')
    #fig.plot_rmse(norm_option='none')
    #fig.local_slpit()
    #fig.sza_plot(norm_option='brightness')
    #fig.sza_plot(norm_option='none')
    #fig.map_detail_figure()
