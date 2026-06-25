import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.gridspec as gridspec
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.image as mpimg
from sympy.abc import alpha
from transformers.utils import add_start_docstrings_to_model_forward

from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.envi import envi_to_array, load_band_names, read_metadata
from datetime import datetime, timezone
from isofit.core.sunposition import sunpos
import geopandas as gp
from utils.results_utils import r2_calculations, load_data, error_metrics
from matplotlib.ticker import FormatStrFormatter
from spectral.io import envi
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from p_tqdm import p_map
from scipy.spatial import ConvexHull

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

    if unmix_mode == 'mesma':
        num_cmb_em = 100
    else:
        num_cmb_em = 20

    library_mode = name.split("_")[0]
    instrument = name.split("_")[2]
    plot = name.split("_")[1]

    num_mc = 25
    normalization = name.split("_")[-4]
    fraction_array = envi_to_array(fraction_file)

    unc_path = os.path.join(f'{fraction_file}_uncertainty')
    unc_array = envi_to_array(unc_path)

    mean_fractions = []
    mean_se = []
    mean_sigma = []
    mean_use = []

    distances = np.arange(fraction_array.shape[0]) * 2.5

    # 2. Calculate the Normalized Radial Weighting Factors (Sum of weights equals 1.0)
    radial_weights = distances / np.sum(distances)

    # Initialize storage lists for your radial metrics
    radial_weighted_means = []

    for _band, band in enumerate(range(0, fraction_array.shape[2])):
            
            if instrument == 'SLPIT':
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
            se = np.nanmean(selected_unc/np.sqrt(int(25)))
            mean_se.append(se)

            # caluclate mean sigma
            sigma = np.nanmean(selected_unc)
            mean_sigma.append(sigma)

            # calculate U_se
            sstd = selected_unc.flatten()
            sum_square_sstd = np.nansum(np.square(sstd))
            use = np.sqrt(sum_square_sstd)/sstd.shape[0]
            mean_use.append(use)

            # radial means
            if instrument == 'SLPIT':
                weighted_columns = selected_fractions * radial_weights[:, np.newaxis]
                column_sums = np.nansum(weighted_columns, axis=0)
                radial_weighted_sum = np.nanmean(column_sums)
                radial_weighted_means.append(radial_weighted_sum)
            else:
                radial_weighted_means.append(np.nanmean(selected_fractions))



    return [instrument, unmix_mode, plot, library_mode, int(num_cmb_em), int(num_mc), normalization, fraction_array.shape[0], fraction_array.shape[1], duplicate_flag] + mean_fractions + radial_weighted_means + mean_se + mean_sigma + mean_use

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

        # load asd wvls
        self.asd_wvls = spectra.load_asd_wavelenghts()
        self.good_asd_bands = spectra.get_good_bands_mask(self.asd_wvls, wavelength_pairs=None)
        self.asd_wvls[~self.good_asd_bands] = np.nan
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

            if plot_name in ['unmix_tc_outlogs', 'THERM-001', 'THERM-003', 'THERM-002']:
                continue
            print(plot_name)

            # Create figure and subplots
            fig = plt.figure(figsize=(14, 8))
            # Axes for the reflectance plot (main area)
            ax_rfl_plot = plt.subplot2grid((16, 16), (7, 0), colspan=9, rowspan=9)
            ax_rfl_plot.set_xlabel('Wavelength (nm)')
            ax_rfl_plot.set_ylabel('Reflectance (%)')

            ax_rfl_plot.set_xlim(300, 2550)
            ax_rfl_plot.xaxis.set_major_locator(MultipleLocator(100))
            ax_rfl_plot.xaxis.set_minor_locator(MultipleLocator(50))
            ax_rfl_plot.set_ylim(0, 1)
            ax_rfl_plot.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_rfl_plot.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax_rfl_plot.tick_params(axis='x', rotation=45)

            # load rfl data
            try:
                slpit_rfl = envi_to_array(os.path.join(spectral_transect_directory, 'RFL', f'{plot_name}_SLPIT_asd'))
                slpit_rfl[slpit_rfl == -9999] = np.nan
                df_slpit_rfl = pd.read_csv(os.path.join(spectral_transect_directory, 'RFL', f'{plot_name}_SLPIT_asd.csv'))

                # # get date and time from slpit
                slpit_date = df_slpit_rfl['date'].unique()[0]
                slpit_datetime = datetime.strptime(df_slpit_rfl['date'].unique()[0], "%Y-%m-%d")

                # Calculate mean time of ASD  collections
                ax_rfl_plot.set_title(f'SLPIT Acquistion Date: {slpit_date}')

                y_mean = np.nanmean(slpit_rfl, axis=(0, 1))
                y_std = np.nanstd(slpit_rfl, axis=(0, 1))

                # plot slpit data
                ax_rfl_plot.plot(self.asd_wvls, y_mean, label=f"SLPIT mean", linewidth=2, color='black')

                # fill 1 sigma
                ax_rfl_plot.fill_between(self.asd_wvls, y_mean - y_std * 2, y_mean + y_std * 2,
                                         color='grey', alpha=0.2)

            except:
                print(f'RFL {plot_name} not found!')

            try:
                df_em_spectra = pd.read_csv(os.path.join(spectral_transect_directory, 'EMS', f'{plot_name}_EMS_emit.csv'))
            except:
                print(os.path.join(spectral_transect_directory, 'EMS', f'{plot_name}_EMS_emit.csv'), "not found!")

            # get gis data
            df_transect = df_gis.loc[df_gis['Name'] == plot_name.replace("Spectral", 'SPEC')].copy()

            # plot map
            ax_map = plt.subplot2grid((16, 16), (0, 0), colspan=5, rowspan=7,
                                      projection=ccrs.PlateCarree())
            ax_map.set_title(f'{plot_name} Plot Map')
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
            ax_landspace_pic = plt.subplot2grid((16, 16), (0, 5), colspan=4, rowspan=7)
            ax_landspace_pic.set_title('Landscape\nPicture')
            pic_path = os.path.join(spectral_transect_directory, f'{plot_name}_landscape_pic.jpg')
            img = mpimg.imread(pic_path)
            ax_landspace_pic.imshow(img)
            ax_landspace_pic.axis('off')

            # plot sensor data
            try:
                sensor_date = df_transect['EMIT DATE'].unique()[0]
                sensor_rfl_file = os.path.join(spectral_transect_directory, 'EXT', f'{plot_name}_RFL_{sensor_date}_EXT')
                acquisition_date = os.path.basename(sensor_rfl_file).split("_")[-2]
                sensor_rfl = envi_to_array(sensor_rfl_file)

                # calculate geometries
                acquisition_datetime_utc = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc)
                geometry_results_sensor = sunpos(acquisition_datetime_utc, np.mean(df_transect.latitude),
                                                 np.mean(df_transect.longitude), np.mean(df_slpit_rfl.elevation))
                acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
                delta = slpit_datetime - acquisition_datetime
                days = np.absolute(delta.days)

                base_label = f'{acquisition_date} (±{days:02d} days)  SZA : {str(int(geometry_results_sensor[1]))}°; Spatial Res: 60 m'
                sensor_std = np.nanstd(sensor_rfl, axis=(0, 1))
                sensor_mean = np.nanmean(sensor_rfl, axis=(0, 1))
                ax_rfl_plot.plot(self.wvls, sensor_mean, label=base_label, linewidth=1, color='blue')
                ax_rfl_plot.fill_between(self.wvls, sensor_mean - sensor_std*2, sensor_mean + sensor_std*2,
                                     color='skyblue', alpha=0.2)
            except:
                print(f'sensor_rfl_file: {sensor_rfl_file} not found')

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
            ax_npv_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

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
            ax_pv_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

            # plot soil endmembers
            try:
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
                ax_soil_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

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
                ax_soil_spectra2.set_xlabel('Wavelength (nm)')

            except:
                print('Soil not found!')
                pass

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', f'{plot_name}.pdf'), format="pdf", dpi=300,
                        bbox_inches="tight")
            plt.clf()
            plt.close()

    def sza_plot(self, norm_option):
        print('loading sza plot...')

        df_rows = []
        # gis shapefile
        gdf = gp.read_file(os.path.join('gis', "Observation.json"))
        df_gis = gdf.drop(columns='geometry')
        df_gis['latitude'] = gdf['geometry'].apply(lambda geom: geom.y)
        df_gis['longitude'] = gdf['geometry'].apply(lambda geom: geom.x)
        df_gis = df_gis.sort_values('Name')
        df_gis['Team'] = df_gis['Name'].str.split('-').str[0].str.strip()
        df_gis = df_gis[df_gis['Team'] != 'THERM']

        # transect data for elevation
        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-SLPIT-emit.csv'))

        # load fraction outputs
        df_all = pd.read_csv(os.path.join(self.fig_directory, 'fraction_output.csv'))
        df_all['Team'] = df_all['plot'].str.split('-').str[0].str.strip()
        df_all = df_all[df_all['Team'] != 'THERM']
        df_all['plot_num'] = df_all['plot'].str.split('-').str[1].str.strip().astype(int)
        df_all = df_all[df_all['plot_num'] <= 60]

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
                df_select = df_all[(df_all['unmix_mode'] == 'emc2') & (df_all['lib_mode'] == 'local') & (df_all['normalization'] == norm_option)].copy()

            if row == 1:
                df_select = df_all[(df_all['unmix_mode'] == 'emc2') & (df_all['lib_mode'] == 'global') & (df_all['normalization'] == norm_option)].copy()

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
                    if mode == 'emc2':
                        mode = 'E(MC)$^2$'

                    ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$',
                                  fontsize=self.axis_label_fontsize)

                ax.set_yticks(np.arange(self.axes_limits['ymin'], 0.4 + 0.05, 0.05))

                if col != 0:
                    ax.set_yticklabels([])

                if row != 3:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                df_x = df_select[(df_select['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
                df_x = df_x.sort_values('plot')
                df_y = df_select[(df_select['instrument'] == 'RFL')].copy().reset_index(drop=True)
                df_y = df_y.sort_values('plot')
                # plot fractional cover values
                x = df_x[f'{col_map[col]}_r'].values
                y = df_y[col_map[col]].values

                abs_error = np.absolute(x-y)
                sza_vals = df['sza'].values

                ax.scatter(sza_vals, abs_error, marker='s', color='blue', edgecolor='black', label='EMIT', zorder=10)
                r2, bias = r2_calculations(sza_vals, abs_error)
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
        df_all['plot_num'] = df_all['plot'].str.split('-').str[1].str.strip().astype(int)
        df_all = df_all[df_all['plot_num'] <= 60]

        #plots_to_avoid_20_days = ['Spectral-001', 'Spectral-002', 'Spectral-003' , 'Spectral-023', 'Spectral-024', 'Spectral-026',
        #                  'Spectral-038', 'Spectral-044', 'Spectral-059', 'Spectral-060']

        plots_to_avoid_10_days = ['Spectral-001', 'Spectral-002', 'Spectral-003', 'Spectral-023', 'Spectral-024',
                                  'Spectral-026',
                                  'Spectral-038', 'Spectral-044', 'Spectral-059', 'Spectral-060', 'Spectral-053', 'Spectral-052',
                                  'Spectral-037', 'Spectral-039', 'Spectral-040', 'Spectral-041', 'Spectral-032', 'Spectral-033',
                                  'Spectral-034', 'Spectral-035', 'Spectral-036', 'Spectral-029', 'Spectral-030', 'Spectral-031',
                                  'Spectral-022', 'Spectral-017', 'Spectral-018', 'Spectral-019', 'Spectral-020']

        df_all = df_all[~df_all['plot'].isin(plots_to_avoid_10_days)]

        for lib_mode in df_all['lib_mode'].unique():
            # create figure
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            ncols = 3
            nrows = 2
            gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

            col_map = {0: 'npv', 1: 'pv', 2: 'soil'}

            # loop through figure columns
            for row in range(nrows):
                if row == 0:
                    df_select = df_all[(df_all['unmix_mode'] == 'emc2') & (df_all['lib_mode'] == lib_mode) & (df_all['normalization'] == 'brightness')].copy()

                if row == 1:
                    df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == lib_mode) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) & (df_all['normalization'] == 'brightness')].copy()

                for col in range(ncols):
                    ax = fig.add_subplot(gs[row, col])
                    ax.grid('on', linestyle='--')

                    if row == 0:
                        ax.set_xlabel('SLPIT - E(MC)$^2$- Fractions')
                        ax.set_ylabel("EMIT - E(MC)$^2$- Fractions")
                    if row ==1:
                        ax.set_xlabel('SLPIT - MESMA - Fractions')
                        ax.set_ylabel("EMIT - MESMA - Fractions")

                    ax.set_aspect(1. / ax.get_data_ratio())
                    ax.set_title(f'{self.ems[col]}')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                    # plot 1 to 1 line
                    one_line = np.linspace(0, 1, 101)
                    ax.plot(one_line, one_line, color='red')

                    df_x = df_select[(df_select['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
                    df_y = df_select[(df_select['instrument'] == 'RFL')].copy().reset_index(drop=True)

                    # plot fractional cover values
                    x = df_x[f'{col_map[col]}_r']
                    y = df_y[col_map[col]]

                    cmap = plt.get_cmap('viridis')
                    c = list(range(1, len(df_x['plot'].values) + 1))
                    #ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', markersize=4, zorder=1)
                    scatter = ax.scatter(x, y, c=c, cmap=cmap, edgecolor='black')

                    # Add error metrics
                    rmse = root_mean_squared_error(x, y)
                    mae = mean_absolute_error(x, y)
                    r2, bias = r2_calculations(x, y)

                    txtstr = '\n'.join((
                        r'MAE(RMSE): %.2f(%.2f)' % (mae,rmse),
                        r'R$^2$: %.2f' % (r2,),
                        r'n = ' + str(len(x))))

                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                    ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)

            #fig.colorbar(scatter, label='Plot Number')
            plt.savefig(os.path.join(self.fig_directory, f'{lib_mode}_regression.png'), format="png", dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()


    def map_detail_figure(self):
        emit_rfl = envi_to_array(os.path.join(self.fig_directory, 'sedgwick_boundary_approx_RFL_20230927T214531_EXT'))[:,:, [39,24,10]]
        emit_fractional_cover = envi_to_array(os.path.join(self.fig_directory, 'EMIT_L2A_RFL_001_20230927T214531_2327014_002_fractional_cover'))[:,:, :-1]
        landscape_pic = os.path.join(self.output_directory, 'spectral_transects', 'Spectral-051', 'Spectral-051_landscape_pic.jpg')
        slpit_rfl = envi_to_array(os.path.join(self.output_directory, 'spectral_transects', 'Spectral-051', 'RFL', 'Spectral-051_SLPIT_asd'))
        df_slpit_rfl = pd.read_csv(os.path.join(self.output_directory, 'spectral_transects', 'Spectral-051', 'RFL',
                                                'Spectral-051_SLPIT_asd.csv'))
        sensor_rfl = envi_to_array(os.path.join(self.output_directory, 'spectral_transects', 'Spectral-051', 'EXT', 'Spectral-051_RFL_20230927T214531_EXT'))

        # get spatial data
        img = envi.open(os.path.join(self.fig_directory, 'sedgwick_boundary_approx_RFL_20230927T214531_EXT.hdr'))
        metadata = img.metadata

        # Spectral doesn't calculate extent automatically,
        # but it organizes the 'map info' into a list for you
        map_info = metadata.get('map info', [])

        # MapX is index 3, MapY is index 4, DX is 5, DY is 6
        ul_x = float(map_info[3])
        ul_y = float(map_info[4])
        dx = float(map_info[5])
        dy = float(map_info[6])

        extent = [ul_x, ul_x + (img.ncols * dx), ul_y - (img.nrows * dy), ul_y]

        # load gis data
        df_gis = gp.read_file(os.path.join('gis', "Observation.json"))
        df_gis['longitude'] = df_gis.geometry.x
        df_gis['latitude'] = df_gis.geometry.y
        df_gis = pd.DataFrame(df_gis.drop(columns='geometry'))
        df_gis['Name'] = df_gis['Name'].str.replace(' ', '', regex=False)
        df_gis = df_gis.sort_values('Name')


        # Create figure and subplots
        fig = plt.figure(figsize=(8, 6))

        # Axes for the reflectance plot (main area)
        ax_rfl_plot = plt.subplot2grid((12, 12), (6, 0), colspan=12, rowspan=6)
        ax_rfl_plot.set_xlabel('Wavelength (nm)', fontsize=12)
        ax_rfl_plot.set_ylabel('Reflectance (%)', fontsize=12)

        ax_rfl_plot.set_xlim(300, 2550)
        ax_rfl_plot.xaxis.set_major_locator(MultipleLocator(100))
        ax_rfl_plot.xaxis.set_minor_locator(MultipleLocator(50))
        ax_rfl_plot.set_ylim(0, 1)
        ax_rfl_plot.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_rfl_plot.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax_rfl_plot.tick_params(axis='x', rotation=45)

        # load rfl data
        slpit_rfl[slpit_rfl == -9999] = np.nan

        # # get date and time from slpit
        slpit_date = df_slpit_rfl['date'].unique()[0]
        slpit_datetime = datetime.strptime(df_slpit_rfl['date'].unique()[0], "%Y-%m-%d")

        df_transect = df_gis.loc[df_gis['Name'] == 'Spectral-051'.replace("Spectral", 'SPEC')].copy()
        acquisition_date = '20230927T214531'

        # calculate geometries
        acquisition_datetime_utc = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc)
        geometry_results_sensor = sunpos(acquisition_datetime_utc, np.mean(df_transect.latitude),
                                         np.mean(df_transect.longitude), np.mean(df_slpit_rfl.elevation))
        acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
        delta = slpit_datetime - acquisition_datetime
        days = np.absolute(delta.days)

        #base_label = f'EMIT Acquistion Date:{acquisition_date} (±{days:02d} days)  SZA : {str(int(geometry_results_sensor[1]))}°'
        base_label = f'EMIT Acquisition Date: 2023-09-27'
        sensor_std = np.nanstd(sensor_rfl, axis=(0, 1))
        sensor_mean = np.nanmean(sensor_rfl, axis=(0, 1))
        ax_rfl_plot.plot(self.wvls, sensor_mean, label=base_label, linewidth=1, color='blue')
        ax_rfl_plot.fill_between(self.wvls, sensor_mean - sensor_std * 2, sensor_mean + sensor_std * 2,
                                 color='skyblue', alpha=0.2)

        # Calculate mean time of ASD  collections
        y_mean = np.nanmean(slpit_rfl, axis=(0, 1))
        y_std = np.nanstd(slpit_rfl, axis=(0, 1))

        # plot slpit data
        ax_rfl_plot.plot(self.asd_wvls, y_mean, label=f'SLPIT Acquisition Date: {slpit_date}', linewidth=2, color='black')

        # fill 1 sigma
        ax_rfl_plot.fill_between(self.asd_wvls, y_mean - y_std * 2, y_mean + y_std * 2,
                                 color='grey', alpha=0.2)
        ax_rfl_plot.legend(fontsize=12)

        # get gis data
        lon = df_transect['longitude'].values[0]
        lat = df_transect['latitude'].values[0]

        gdf = gpd.read_file(os.path.join('gis', 'sedgwick_boundary_approx.geojson'))

        # # plot emit fractional cover
        ax_map = plt.subplot2grid((12, 12), (0, 0), colspan=4, rowspan=5)
        gdf.plot(ax=ax_map, facecolor='none', edgecolor='cyan', linewidth=2)
        ax_map.set_title('EMIT Fractional Cover')
        vmin, vmax = np.percentile(emit_fractional_cover, [2, 98])
        img_display = np.clip((emit_fractional_cover - vmin) / (vmax - vmin), 0, 1)
        ax_map.imshow(img_display, extent=extent, aspect='auto')
        ax_map.axis('off')
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
        ax_map.scatter(lon, lat, color='yellow', marker='*', s=150, zorder=9, label='SLPIT')

        ax_map.annotate('N', xy=(0.05, 0.995), xytext=(0.05, 0.85),
                         arrowprops=dict(facecolor='white', width=2, headwidth=7.5),
                         ha='center', va='center', fontsize=12, color='white',
                         xycoords='axes fraction')

        red_patch = mpatches.Patch(color='red', label='R: NPV')
        green_patch = mpatches.Patch(color='green', label='G: GV')
        blue_patch = mpatches.Patch(color='blue', label='B: Soil')

        # Get the existing legend handles and labels from the map
        handles, labels = ax_map.get_legend_handles_labels()
        all_handles = handles + [red_patch, green_patch, blue_patch]


        ax_map.legend(handles=all_handles, loc='center left', bbox_to_anchor=(-0.40, 0.5),
                      fontsize=8, frameon=True, borderaxespad=0, facecolor='wheat',
                      title_fontsize=8, framealpha=0.65)

        # plot RGB image
        ax_map2 = plt.subplot2grid((12, 12), (0, 4), colspan=4, rowspan=5)
        gdf.plot(ax=ax_map2, facecolor='none', edgecolor='cyan', linewidth=2, label='Boundary')
        ax_map2.set_title('EMIT RGB')
        vmin, vmax = np.percentile(emit_rfl, [2.5, 95])
        img_display = np.clip((emit_rfl - vmin) / (vmax - vmin), 0, 1)
        ax_map2.imshow(img_display, extent=extent, aspect='auto')
        ax_map2.axis('off')
        ax_map2.set_xlim(extent[0], extent[1])
        ax_map2.set_ylim(extent[2], extent[3])
        ax_map2.scatter(lon, lat, color='yellow', marker='*', s=150, zorder=9)

        ax_map2.annotate('N', xy=(0.05, 0.995), xytext=(0.05, 0.85),
                        arrowprops=dict(facecolor='white', width=2, headwidth=7.5),
                        ha='center', va='center', fontsize=12, color='white',
                        xycoords='axes fraction')

        # # plot landsacpe image
        ax_landspace_pic = plt.subplot2grid((12, 12), (0, 8), colspan=4, rowspan=5)
        ax_landspace_pic.set_title('Landscape Picture', fontsize=14)
        img = mpimg.imread(landscape_pic)
        ax_landspace_pic.imshow(img, aspect='auto')
        ax_landspace_pic.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_directory, 'map_detailed_slpit.png'), format="png", dpi=600)
        plt.clf()
        plt.close()





    def methods_diagram(self):
        import matplotlib.patches as patches
        import matplotlib.patheffects as pe

        """
            Generates a 2x3 methods diagram comparing EMIT radial transects 
            (180m scale) and SHIFT SLPIT transects (8m scale).
            """
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), layout='constrained')

        (ax_slpit_emit, ax_rfl_plot_emit, ax_slpit_photo_emit), \
            (ax_slpit_shift, ax_rfl_plot_shift, ax_slpit_photo_shift) = axes

        # Formatting helper to keep diagrams square
        for ax in [ax_slpit_emit, ax_slpit_shift]:
            ax.set_aspect('equal', adjustable='box')

        for ax in [ax_rfl_plot_emit, ax_rfl_plot_shift]:
            ax.set_aspect('auto')  # This lets the data fill the space
            ax.set_box_aspect(1)  # This makes the actual plot box a square

        # ---------------------------------------------------------
        # 1. EMIT RADIAL DIAGRAM (Top Left - 180m x 180m)
        # ---------------------------------------------------------
        side_emit = 180
        center_emit = side_emit / 2
        spoke_length = 60

        # Orientations (Cartesian)
        angles_deg = [135, 15, 255]  # Leg 1 (NW), Leg 2 (ENE), Leg 3 (SSW)
        angles_rad = np.radians(angles_deg)

        ax_slpit_emit.set_title('SLPIT for EMIT Sites Diagram', fontweight='bold', fontsize=10)

        for i, angle in enumerate(angles_rad):
            x_end = center_emit + spoke_length * np.cos(angle)
            y_end = center_emit + spoke_length * np.sin(angle)
            ax_slpit_emit.plot([center_emit, x_end], [center_emit, y_end], color='black', lw=1.5, zorder=1)

            # Plot ASD GIFOVs
            distances = np.linspace(0, spoke_length, 25)
            for d in distances:
                px = center_emit + d * np.cos(angle)
                py = center_emit + d * np.sin(angle)
                circle = plt.Circle((px, py), 1, facecolor='red', alpha=0.4,
                                    edgecolor='red', linewidth=0.5, zorder=2)
                ax_slpit_emit.add_patch(circle)

        # Grid Lines (EMIT Pixel boundaries)
        for pos in [60, 120]:
            ax_slpit_emit.axvline(pos, color='blue', linestyle='--', alpha=0.5, lw=0.8)
            ax_slpit_emit.axhline(pos, color='blue', linestyle='--', alpha=0.5, lw=0.8)

        offset_dist = 14
        brace_angle = np.radians(15)
        perp_angle = brace_angle - np.pi / 2  # Subtracting flips it to the "South-East" side

        b_start_x = center_emit + offset_dist * np.cos(perp_angle)
        b_start_y = center_emit + offset_dist * np.sin(perp_angle)
        b_end_x = b_start_x + spoke_length * np.cos(brace_angle)
        b_end_y = b_start_y + spoke_length * np.sin(brace_angle)

        ax_slpit_emit.plot([b_start_x, b_end_x], [b_start_y, b_end_y], color='black', lw=1)

        # Tick marks at ends
        tick_size = 3
        for tx, ty in [(b_start_x, b_start_y), (b_end_x, b_end_y)]:
            ax_slpit_emit.plot([tx - tick_size * np.cos(perp_angle), tx + tick_size * np.cos(perp_angle)],
                               [ty - tick_size * np.sin(perp_angle), ty + tick_size * np.sin(perp_angle)],
                               color='black', lw=1)

        # Label below the brace
        ax_slpit_emit.text((b_start_x + b_end_x) / 2 + 4 * np.cos(perp_angle),
                           (b_start_y + b_end_y) / 2 + 4 * np.sin(perp_angle),
                           '60 m', fontsize=8, ha='center', va='top',
                           rotation=15, fontweight='bold')

        # Arc (120°)
        arc = patches.Arc((center_emit, center_emit), 45, 45, theta1=135, theta2=255, edgecolor='black', ls='--')
        ax_slpit_emit.add_patch(arc)
        ax_slpit_emit.text(center_emit + 38 * np.cos(np.radians(195)),
                           center_emit + 38 * np.sin(np.radians(195)),
                           '120°', fontsize=9, ha='right', va='center', fontweight='bold')

        sun_compass_deg = 45
        leg1_compass_deg = 315

        # Converting Compass to Cartesian for Matplotlib (0 deg is East)
        # Formula: Cartesian = (90 - Compass) % 360
        sun_cart = (90 - sun_compass_deg) % 360  # 45 deg
        leg1_cart = (90 - leg1_compass_deg) % 360  # 135 deg

        # Sun
        # --- SUN ANNOTATION (Corrected) ---
        # --- SUN: Yellow Star ---
        sun_rad = np.radians(sun_cart)
        sx = center_emit + 85 * np.cos(sun_rad)
        sy = center_emit + 85 * np.sin(sun_rad)

        # Yellow Star
        ax_slpit_emit.plot(sx, sy, marker='*', markersize=22,
                          color='yellow', markeredgecolor='orange',
                          zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        # Yellow arrow pointing from Sun to Center
        ax_slpit_emit.annotate('Sun', xy=(center_emit, center_emit),
                               xytext=(center_emit + 95 * np.cos(sun_rad),
                                       center_emit + 95 * np.sin(sun_rad)),
                               arrowprops=dict(arrowstyle='->', color=(0, 0, 0, 0.5), lw=1, shrinkA=12),
                               fontweight='bold',
                               color=(0, 0, 0, 0.5),  # Black with 0.5 alpha
                               fontsize=11,
                               ha='left', va='bottom',
                               annotation_clip=False)

        # --- THE RIGHT ANGLE MARKER ---
        # Placed between Leg 1 (135 Cartesian) and the Solar Vector (45 Cartesian)
        ra_size = 7
        leg1_rad = np.radians(leg1_cart)

        x1 = center_emit + ra_size * np.cos(leg1_rad)
        y1 = center_emit + ra_size * np.sin(leg1_rad)

        # Point 2: On the Solar Path
        x2 = center_emit + ra_size * np.cos(sun_rad)
        y2 = center_emit + ra_size * np.sin(sun_rad)

        # Point 3: The "Corner" of the square
        # Since 45 and 135 are symmetric around North (90), the corner is directly above the center
        xc = center_emit + (ra_size * np.cos(leg1_rad)) + (ra_size * np.cos(sun_rad))
        yc = center_emit + (ra_size * np.sin(leg1_rad)) + (ra_size * np.sin(sun_rad))

        # Plot the red square marker
        ax_slpit_emit.plot([x1, xc, x2], [y1, yc, y2], color='black', lw=1, zorder=10, alpha=0.5)

        # Legend for EMIT
        circle_proxy = mlines.Line2D([], [], color='red', marker='o', ls='None', markersize=8,
                                     mfc=(1, 0, 0, 0.4), mec='red', label='ASD GIFOV')
        pixel_proxy = mlines.Line2D([], [], color='blue', ls='--', alpha=0.5, label='EMIT Pixel')
        ax_slpit_emit.legend(handles=[circle_proxy, pixel_proxy], loc='lower right', fontsize=7, framealpha=0.75)

        ax_slpit_emit.set_xlim(0, 180)
        ax_slpit_emit.set_ylim(0, 180)
        ax_slpit_emit.set_xlabel('180 m')
        ax_slpit_emit.set_ylabel('180 m')
        ax_slpit_emit.set_xticks([])
        ax_slpit_emit.set_yticks([])

        # ---------------------------------------------------------
        # 2. SLPIT SHIFT DIAGRAM (Bottom Left - 8m)
        # ---------------------------------------------------------
        ax_slpit_shift.set_title('SLPIT for SHIFT Sites Diagram', fontweight='bold', fontsize=10)
        grid_limit = 8
        ax_slpit_shift.axvline(1.5, color='black', linestyle='--', alpha=0.5)
        ax_slpit_shift.axvline(5+1.5, color='black', linestyle='--', alpha=0.5)
        ax_slpit_shift.axhline(1.5, color='black', linestyle='--', alpha=0.5)
        ax_slpit_shift.axhline(5+1.5, color='black', linestyle='--', alpha=0.5)

        for x in [2, 6]:
            for y in np.arange(0, 8.01, 0.33):
                circle = plt.Circle((x + 0.25, y), 0.07, facecolor='red', alpha=0.3,
                                    edgecolor='red', linewidth=0.5)
                ax_slpit_shift.add_patch(circle)

        # Legend for EMIT
        circle_proxy = mlines.Line2D([], [], color='red', marker='o', ls='None', markersize=8,
                                     mfc=(1, 0, 0, 0.4), mec='red', label='ASD GIFOV')
        pixel_proxy = mlines.Line2D([], [], color='black', ls='--', alpha=0.5, label='AVIRIS-NG Pixel')
        ax_slpit_shift.legend(handles=[circle_proxy, pixel_proxy], loc='lower center', fontsize=7, framealpha=0.75)

        ax_slpit_shift.set_xlim(0, 8)
        ax_slpit_shift.set_ylim(0, 8)
        ax_slpit_shift.set_xlabel('8 m', fontsize=9)
        ax_slpit_shift.set_ylabel('8 m', fontsize=9)
        ax_slpit_shift.set_xticks([])
        ax_slpit_shift.set_yticks([])

        # North Arrow for BOTH diagrams (Top-Left of their axes)
        for ax in [ax_slpit_emit, ax_slpit_shift]:
            ax.annotate('N', xy=(-0.05, 0.98), xytext=(-0.05, 0.775),
                        arrowprops=dict(facecolor='black', width=1, headwidth=5.5),
                        ha='center', va='center', fontsize=10, color='black',
                        xycoords='axes fraction', annotation_clip=False)



        # ---------------------------------------------------------
        # 3. REFLECTANCE PLOTS (Middle Column)
        # ---------------------------------------------------------
        rfl_axes = [ax_rfl_plot_emit, ax_rfl_plot_shift]
        rfl_paths = [
            os.path.join(self.output_directory, 'spectral_transects', 'Spectral-007', 'RFL',
                         'Spectral-007_SLPIT_asd'),
            os.path.join('terraspec_output', 'shift', 'output', 'spectral_transects', 'DPB-020_SPRING', 'RFL',
                         'DPB-020_SPRING_SLPIT_asd')
        ]
        titles_rfl = ['SLPIT(EMIT) Reflectance', 'SLPIT(SHIFT) Reflectance']

        for i, ax in enumerate(rfl_axes):
            ax.set_title(titles_rfl[i], fontweight='bold', fontsize=10)
            # Assuming envi_to_array and asd_wvls are defined class methods/attributes
            slpit_rfl = envi_to_array(rfl_paths[i])
            slpit_rfl[slpit_rfl == -9999] = np.nan
            y_mean = np.nanmean(slpit_rfl, axis=(0, 1))
            y_std = np.nanstd(slpit_rfl, axis=(0, 1))

            ax.plot(self.asd_wvls, y_mean, color='red', lw=2, label='Mean')
            ax.fill_between(self.asd_wvls, y_mean - y_std, y_mean + y_std, color='red', alpha=0.2, label='1σ')
            ax.set_xlim(300, 2550)
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.xaxis.set_minor_locator(MultipleLocator(100))
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.set_xlabel('Wavelength (nm)', fontsize=8)
            ax.set_ylabel('Reflectance (%)', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(fontsize=8)

        # ---------------------------------------------------------
        # 4. PHOTOS (Right Column)
        # ---------------------------------------------------------
        photo_axes = [ax_slpit_photo_emit, ax_slpit_photo_shift]
        photo_dir = os.path.join('objects', 'shift_photos')
        landscape_paths = [os.path.join(self.output_directory, 'spectral_transects', 'Spectral-007',
                         'Spectral-007_landscape_pic.jpg'), os.path.join(photo_dir, 'SLPIT.jpg')]
        titles_photo = ["SLPIT for EMIT Sites", "SLPIT for SHIFT Sites"]

        for i, ax in enumerate(photo_axes):
            ax.set_title(titles_photo[i], fontweight='bold', fontsize=10)
            try:
                img = mpimg.imread(landscape_paths[i])
                ax.imshow(img, aspect='equal')
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'Photo Not Found', ha='center', va='center')
            ax.axis('off')

        # Save and Cleanup
        plt.savefig(os.path.join(self.fig_directory, 'methods_diagram_paper2.png'), dpi=600)
        plt.show()
        plt.close()

    def spectral_variance(self):
        df_global = pd.read_csv(os.path.join('terraspec_output', 'simulation', 'output', 'endmember_libraries', 'convex_hull__n_dims_4_sensor_emit_geofilter_True_unmix_library.csv'))
        df_global = df_global.sort_values('level_1')

        df_all = pd.read_csv(os.path.join(self.fig_directory, 'fraction_output.csv'))
        df_all['Team'] = df_all['plot'].str.split('-').str[0].str.strip()
        df_all = df_all[df_all['Team'] != 'THERM']
        df_all['plot_num'] = df_all['plot'].str.split('-').str[1].str.strip().astype(int)
        df_all = df_all[df_all['plot_num'] <= 60]


        col_map = {0: 'npv', 1: 'pv', 2: 'soil'}

        df_select = df_all[(df_all['unmix_mode'] == 'emc2') & (df_all['lib_mode'] == 'local') & (
                            df_all['normalization'] == 'brightness')].copy()

        local_map = {'npv': ['005', '044'], 'pv': ['029', '014'], 'soil': ['036', '019']}

        for col in range(3):
            df_x = df_select[(df_select['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
            df_y = df_select[(df_select['instrument'] == 'RFL')].copy().reset_index(drop=True)

            plots = df_x['plot'].values
            x = df_x[f'{col_map[col]}_r'].values
            y = df_y[col_map[col]].values

            error = np.absolute(x-y)
            df_new = pd.DataFrame({
                'plots': plots,
                'error': error
            })
            df_new = df_new.sort_values('error')

        local_map = {'npv': ['005','044'], 'pv': ['029', '014'], 'soil': ['036', '019']}
        fig, axes = plt.subplots(1, 3, figsize=(10, 6.5), layout='constrained')

        for col in range(3):
            current_em = col_map[col]

            # 1. Isolate Global Target Data
            df_global_em = df_global.loc[(df_global['level_1'] == col_map[col])].copy().reset_index(drop=True)
            df_spectra_global = df_global_em.iloc[:, 7:].to_numpy()
            df_global_em['dataset'] = 'global'
            meta_global = df_global_em['dataset']

            # 2. Isolate Local - Best Target Data
            df_best_local = pd.read_csv(
                os.path.join(self.output_directory, 'spectral_transects', f'Spectral-{local_map[current_em][0]}',
                             f'unmix_Spectral-{local_map[current_em][0]}_EMS_emit.csv'))
            df_best_local['level_1'] = df_best_local['level_1'].str.lower()
            df_best_local = df_best_local.loc[(df_best_local['level_1'] == current_em)].copy().reset_index(drop=True)
            df_spectra_local_best = df_best_local.iloc[:, 11:].to_numpy()
            meta_best = df_best_local['plot_name']

            # 3. Isolate Local - Worst Target Data
            df_worst_local = pd.read_csv(
                os.path.join(self.output_directory, 'spectral_transects', f'Spectral-{local_map[current_em][1]}',
                             f'unmix_Spectral-{local_map[current_em][1]}_EMS_emit.csv'))
            df_worst_local['level_1'] = df_worst_local['level_1'].str.lower()
            df_worst_local = df_worst_local.loc[(df_worst_local['level_1'] == current_em)].copy().reset_index(drop=True)
            df_spectra_local_worst = df_worst_local.iloc[:, 11:].to_numpy()
            meta_worst = df_worst_local['plot_name']

            meta_combined = pd.concat([meta_global, meta_best, meta_worst], axis=0).reset_index(drop=True)

            # 4. Vertical Stack Concatenation & Normalization Execution
            df_all_spectra = np.concatenate((df_spectra_global, df_spectra_local_best, df_spectra_local_worst), axis=0)

            norm = p_map(spectra.vector_normalize_spectrum, df_all_spectra,
                         **{"desc": f"\t\t\tnormalizing spectrum: d = {col_map[col]}...", "ncols": 150})

            df_norm = pd.DataFrame(norm)
            df_norm.columns = df_global_em.columns[7:]
            df_norm.insert(0, 'info', meta_combined)
            df_norm.insert(0, 'level_1', current_em)

            # 5. Dimensionality Reduction (PCA Engine)
            pc_components = spectra.pca_analysis(df_norm, spectra_starting_col=3, em=col_map[col])
            pc_array = np.asarray(pc_components)[:, 3: 3 + 4]
            pc_array_2d = pc_array[:, :2]

            # 6. Map Coordinate Spatial Bounds to Build Uniform, Equal Square Footprints
            x_vals = pc_array_2d[:, 0]
            y_vals = pc_array_2d[:, 1]

            x_center, y_center = (x_vals.max() + x_vals.min()) / 2, (y_vals.max() + y_vals.min()) / 2
            max_range = max(x_vals.max() - x_vals.min(), y_vals.max() - y_vals.min()) / 2

            # Include a 10% outer cushion padding to keep hulls cleanly in boundary view
            half_side = max_range + (max_range * 0.1)

            axes[col].set_xlim(x_center - half_side, x_center + half_side)
            axes[col].set_ylim(y_center - half_side, y_center + half_side)
            axes[col].set_aspect('equal', adjustable='box')

            # 7. Generate Target Library Class Trackers
            source_labels = (
                    ['Global'] * len(df_spectra_global) +
                    [f'SPEC-{local_map[current_em][0]}'] * len(df_spectra_local_best) +
                    [f'SPEC-{local_map[current_em][1]}'] * len(df_spectra_local_worst)
            )

            df_plot_groups = pd.DataFrame({
                'PC1': x_vals,
                'PC2': y_vals,
                'source': source_labels
            })

            style_map = {
                'Global': {'color': '#d3d3d3', 'alpha': 0.5, 'zorder': 1},
                f'SPEC-{local_map[current_em][0]}': {'color': '#1f77b4', 'alpha': 0.9, 'zorder': 3},
                f'SPEC-{local_map[current_em][1]}': {'color': '#ff7f0e', 'alpha': 0.9, 'zorder': 2}
            }

            # 8. Render Plot Layers via Rigid Explicit Sequence Iteration
            target_draw_order = [
                'Global',
                f'SPEC-{local_map[current_em][0]}',
                f'SPEC-{local_map[current_em][1]}'
            ]

            for label in target_draw_order:
                group = df_plot_groups[df_plot_groups['source'] == label]

                if group.empty:
                    continue

                pts = group[['PC1', 'PC2']].to_numpy()
                cfg = style_map[label]

                # Draw categorical dispersion cluster
                axes[col].scatter(
                    pts[:, 0], pts[:, 1],
                    color=cfg['color'], alpha=cfg['alpha'], s=15,
                    label=label, zorder=cfg['zorder']
                )

                # Draw corresponding customized Convex Hull
                if len(pts) >= 3:
                    ch_sub = ConvexHull(pts)
                    for simplex in ch_sub.simplices:
                        axes[col].plot(
                            pts[simplex, 0], pts[simplex, 1],
                            color=cfg['color'], linestyle='-', lw=1.5,
                            zorder=cfg['zorder']
                        )

            # 9. Format Labels and Annotations
            if current_em == 'pv':
                current_em = 'GV'

            axes[col].set_title(f"{current_em.upper()}")
            axes[col].set_ylim(-.35,.35)
            axes[col].set_xlim(-.35,.35)
            axes[col].set_xlabel("PC - 1")
            if col == 0:
                axes[col].set_ylabel("PC - 2")
            if col != 0:
                axes[col].set_yticklabels([])
            axes[col].legend(loc='lower left', frameon=True, fontsize=8)

        # Export and display visual layout output
        plt.savefig(os.path.join(self.fig_directory, 'spectral_variance.png'), dpi=600)
        plt.clf()
        plt.close()

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

    #fig.plot_summary()
    #fig.methods_diagram()
    fig.local_slpit()
    #fig.sza_plot(norm_option='brightness')
    #fig.map_detail_figure()
    fig.spectral_variance()
