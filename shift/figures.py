import time
import pandas as pd
from glob import glob
from p_tqdm import p_map
from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from scipy.stats import alpha
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from utils.spectra_utils import spectra
from utils.create_tree import create_directory
from utils.envi import envi_to_array
import geopandas as gp
from utils.results_utils import r2_calculations, load_data
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timezone
from isofit.core.sunposition import sunpos
import spectral.io.envi as envi


scene_key = {'DPA-004_FALL': {'flightline': 'ang20220915t195816', 'version': '003'},
             'DPB-003_FALL': {'flightline': 'ang20220915t195816', 'version': '003'},
             'DPB-004_FALL': {'flightline': 'ang20220915t200714', 'version': '000'},
             'DPB-005_FALL': {'flightline': 'ang20220915t195816', 'version': '003'},
             'DPB-020_SPRING': {'flightline': 'ang20220322t204749', 'version': '000'},
             'DPB-027_SPRING': {'flightline': 'ang20220412t205405', 'version': '001'},
             'SRA-007_FALL': {'flightline': 'ang20220914t184300', 'version': '000'},
             'SRA-008_FALL': {'flightline': 'ang20220914t184300', 'version': '000'},
             'SRA-019_SPRING': {'flightline': 'ang20220308t204043', 'version': '008'},
             'SRA-020_SPRING': {'flightline': 'ang20220308t205512', 'version': '002'},
             'SRA-021_SPRING': {'flightline': 'ang20220308t204043', 'version': '008'},
             'SRA-033_SPRING': {'flightline': 'ang20220316t210303', 'version': '002'},
             'SRA-034_SPRING': {'flightline': 'ang20220316t210303', 'version': '002'},
             'SRA-056_FALL': {'flightline': 'ang20220914t184300', 'version': '000'},
             'SRA-109_SPRING': {'flightline': 'ang20220511t190344', 'version': '002'},
             'SRB-010_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
             'SRB-021_SPRING': {'flightline': 'ang20220308t205512', 'version': '002'},
             'SRB-026_SPRING': {'flightline': 'ang20220308t204043', 'version': '007'},
             'SRB-045_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
             'SRB-046_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
             'SRB-084_SPRING': {'flightline': 'ang20220511t191813', 'version': '007'},
             'SRB-100_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
             'SRB-050_FALL': {'flightline': 'ang20220914t184300', 'version': '000'},
             'SRB-047_SPRING': {'flightline': 'ang20220405t201359', 'version': '002'},
             'SRB-004_FALL': {'flightline': 'ang20220914t184300', 'version': '000'},
             'SRB-200_FALL': {'flightline': 'ang20220914t184300', 'version': '000'}}

class figures:
    def __init__(self, base_directory: str, sensor: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")
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

        # load emit slpit
        terraspec_base = os.path.dirname(base_directory)
        self.slpit_figures = os.path.join(terraspec_base, 'slpit', 'figures')
        self.slpit_dir = os.path.join(terraspec_base, 'slpit')



    def plot_summary(self):


        # spectral data directories
        create_directory(os.path.join(self.fig_directory, 'plot_stats'))
        spectral_transects_directories = glob(os.path.join(self.output_directory, 'spectral_transects', '**'))

        # load gis data
        df_gis = gp.read_file(os.path.join('gis', "shift_transects_centroid.geojson"))

        df_gis['longitude'] = df_gis.geometry.x
        df_gis['latitude'] = df_gis.geometry.y
        df_gis = pd.DataFrame(df_gis.drop(columns='geometry'))

        # load tetracorder mineral grouping tables
        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))

        for spectral_transect_directory in spectral_transects_directories:
            plot_name = os.path.basename(spectral_transect_directory)

            if plot_name in ['unmix_tc_outlogs', 'DPA-9999_FALL', 'SRA-9999_FALL', 'SRA-000_SPRING']:
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
                slpit_rfl = envi_to_array(
                    os.path.join(spectral_transect_directory, 'RFL', f'{plot_name}_SLPIT_asd'))
                slpit_rfl[slpit_rfl == -9999] = np.nan
                df_slpit_rfl = pd.read_csv(
                    os.path.join(spectral_transect_directory, 'RFL', f'{plot_name}_SLPIT_asd.csv'))

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
                raise


            try:
                df_em_spectra = pd.read_csv(os.path.join(spectral_transect_directory, 'EMS', f'{plot_name}_EMS_aviris_ng.csv'))
            except:
                print(os.path.join(spectral_transect_directory, 'EMS', f'{plot_name}_EMS_aviris_ng.csv'), "not found!")

            # get gis data
            df_transect = df_gis.loc[df_gis['plot'] == plot_name].copy()

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
            try:
                pic_path = os.path.join(spectral_transect_directory, f'{plot_name.split("_")[0]}_landscape_pic.jpg')
                img = mpimg.imread(pic_path)
                ax_landspace_pic.imshow(img)
                ax_landspace_pic.axis('off')
            except:
                print(f'Landscape {plot_name} not found!')


            # plot sensor data
            sensor_rfl_file = os.path.join(spectral_transect_directory, 'EXT',
                                   f'{plot_name}_RFL_{scene_key[plot_name]["flightline"]}_{scene_key[plot_name]["version"]}_EXT')
            acquisition_date = os.path.basename(sensor_rfl_file).split("_")[3][3:]
            version = os.path.basename(sensor_rfl_file).split("_")[4]
            sensor_rfl = envi_to_array(sensor_rfl_file)

            # calculate geometries
            acquisition_datetime_utc = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            geometry_results_sensor = sunpos(acquisition_datetime_utc, np.mean(df_transect.latitude),
                                             np.mean(df_transect.longitude), np.mean(df_slpit_rfl.elevation))
            acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
            delta = slpit_datetime - acquisition_datetime
            days = np.absolute(delta.days)

            if days > 3:
                continue

            # Open the image using the header file
            img = envi.open(f'{sensor_rfl_file}.hdr')

            # Access the raw metadata dictionary
            metadata = img.metadata
            map_info = metadata['map info']
            x_res = map_info[5]
            y_res = map_info[6]

            if float(x_res) < 1.5:
                continue

            base_label = f'{acquisition_date} (±{days:02d} days); version: {version}; SZA : {str(int(geometry_results_sensor[1]))}°; Spatial Res: {x_res} m'
            sensor_std = np.nanstd(sensor_rfl, axis=(0, 1))
            sensor_mean = np.nanmean(sensor_rfl, axis=(0, 1))
            ax_rfl_plot.plot(self.wvls, sensor_mean, label=base_label, linewidth=1, color='skyblue')
            ax_rfl_plot.fill_between(self.wvls, sensor_mean - sensor_std * 2, sensor_mean + sensor_std * 2,
                                    color='skyblue', alpha=0.2)

            ax_rfl_plot.legend()

            # plot NPV endmembers
            ax_npv_spectra = plt.subplot2grid((16, 16), (0, 9), colspan=7, rowspan=4)
            df_spectra = df_em_spectra[(df_em_spectra['level_1'] == 'npv')].copy()
            df_spectra['species'] = df_spectra['species'].fillna('UNK-')
            df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
            num_species = len(sorted(list(df_spectra.species.unique())))
            npv_cmap = plt.cm.get_cmap(self.cmap_kw, num_species)
            unique_species = sorted(df_spectra['species'].unique())

            for i, species in enumerate(unique_species):
                # Filter and extract spectra data in one go
                em_spectra = df_spectra[df_spectra['species'] == species].iloc[:, 12:].to_numpy()

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
            df_spectra = df_em_spectra[(df_em_spectra['level_1'] == 'pv')].copy()
            df_spectra['species'] = df_spectra['species'].fillna('UNK-')
            num_species = len(sorted(list(df_spectra.species.unique())))
            pv_cmap = plt.cm.get_cmap(self.cmap_kw, num_species)
            unique_species = sorted(df_spectra['species'].unique())

            for i, species in enumerate(unique_species):
                # Filter and extract spectra data in one go
                em_spectra = df_spectra[df_spectra['species'] == species].iloc[:, 12:].to_numpy()

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

            # # plot soil endmembers
            try:
                ax_soil_spectra = plt.subplot2grid((16, 16), (8, 9), colspan=7, rowspan=4)
                df_spectra = df_em_spectra[(df_em_spectra['level_1'] == 'soil')].copy()
                soil_index_min = min(df_spectra.index[df_spectra['level_1'] == 'soil'].tolist())
                soil_index_max = max(df_spectra.index[df_spectra['level_1'] == 'soil'].tolist())
                soil_tetracorder_results = envi_to_array(os.path.join(spectral_transect_directory, 'tetracorder',
                                                                      f'{plot_name}_EMS_aviris_ng_augmented_min'))[
                    soil_index_min:soil_index_max + 1, :, :]

                g1_unique = soil_tetracorder_results[:, 0, 1]
                g2_unique = soil_tetracorder_results[:, 0, 3]
                df_spectra.insert(0, "g1_minerals", g1_unique)
                df_spectra.insert(0, "g2_minerals", g2_unique)

                df_spectra['g1_minerals'] = df_spectra['g1_minerals'].map(
                    df_mineral_matrix.set_index('Index')['Name'])
                df_spectra['g2_minerals'] = df_spectra['g2_minerals'].map(
                    df_mineral_matrix.set_index('Index')['Name'])
                df_spectra['g1_minerals'] = df_spectra['g1_minerals'].fillna('No Detection')
                df_spectra['g2_minerals'] = df_spectra['g2_minerals'].fillna('No Detection')

                num_minerals_g1 = len(sorted(list(df_spectra.g1_minerals.unique())))
                g1_minerals_cmap = plt.cm.get_cmap('jet', num_minerals_g1)
                unique_minerals_g1 = sorted(df_spectra['g1_minerals'].unique())

                # plot g1 minerals
                for i, soils_unique in enumerate(unique_minerals_g1):
                    # Filter and extract spectra data in one go
                    em_spectra = df_spectra[df_spectra['g1_minerals'] == soils_unique].iloc[:, 14:].to_numpy()
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
                ax_soil_spectra.legend(prop={'size': 6}, bbox_to_anchor=(1.01, 1), loc='upper left',
                                       borderaxespad=0.)

                # plot group 2 minerals
                ax_soil_spectra2 = plt.subplot2grid((16, 16), (12, 9), colspan=7, rowspan=4)
                num_minerals_g2 = len(sorted(list(df_spectra.g2_minerals.unique())))
                g2_minerals_cmap = plt.cm.get_cmap('viridis', num_minerals_g2)
                unique_minerals_g2 = sorted(df_spectra['g2_minerals'].unique())

                # plot g2 minerals
                for i, soils_unique in enumerate(unique_minerals_g2):
                    # Filter and extract spectra data in one go
                    em_spectra = df_spectra[df_spectra['g2_minerals'] == soils_unique].iloc[:, 14:].to_numpy()

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
                ax_soil_spectra2.legend(prop={'size': 6}, bbox_to_anchor=(1.01, 1), loc='upper left',
                                        borderaxespad=0.)
                ax_soil_spectra2.set_xlabel('Wavelength (nm)')

            except:
                print('Soil not found!')
                pass

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', f'{plot_name}.pdf'), format="pdf", dpi=300,
                        bbox_inches="tight")
            plt.clf()
            plt.close()

    def plot_rmse(self, norm_option):

        skip = ['SRA-000-SPRING', 'SRB-047-SPRING', 'SRB-004-FALL', 'SRB-050-FALL', 'SRB-200-FALL']
        df_all = pd.read_csv(os.path.join(self.figure_directory, 'shift_fraction_output.csv'))
        df_all = df_all[~df_all['plot'].isin(skip)]

        # load cpu performance results
        df_cpu = pd.read_csv(os.path.join(self.figure_directory, 'shift_computing_performance_report.csv'))
        df_cpu['normalization'] = df_cpu['normalization'].str.strip('"')
        df_cpu['mode'] = df_cpu['mode'].str.strip('"')
        df_cpu['instrument'] = df_cpu['reflectance_file'].str.strip('"').apply(lambda x: os.path.basename(x).split('___')[0].split('-')[1])
        df_cpu['plot'] = df_cpu['reflectance_file'].str.strip('"').apply(lambda x: os.path.basename(x).split('___')[1])
        df_cpu = df_cpu[~df_cpu['plot'].isin(skip)]

        # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'local') & (df_cpu['mode'] == 'sma') & (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'aviris')].copy()
            if row == 1:
                df_select = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'global') & (df_cpu['mode'] == 'sma') & (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'aviris')].copy()
            if row == 2:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) & (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'local') & (df_cpu['mode'] == 'mesma') & (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'aviris')].copy()
            if row == 3:
                df_select = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 100) & (df_all['normalization'] == norm_option)].copy()
                df_performance = df_cpu[(df_cpu['library'] == 'global') & (df_cpu['mode'] == 'mesma') & (df_cpu['normalization'] == norm_option) & (df_cpu['instrument'] == 'aviris')].copy()

            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                mode = list(df_select['unmix_mode'].unique())[0]
                lib_mode = list(df_select['lib_mode'].unique())[0]

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 3 and col == 1:
                    ax.set_xlabel("SLPIT", fontsize=self.axis_label_fontsize)

                if row == 3 and col != 0:
                    ax.set_xticklabels([''] + ax.get_xticklabels()[1:])

                if col == 0:
                    ax.set_ylabel(mode.upper() + '$_{'+lib_mode +'}$', fontsize=self.axis_label_fontsize)

                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.2, 0.2))

                if col != 0:
                    ax.set_yticklabels([])

                if row != 3:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                df_x = df_select[(df_select['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y = df_select[(df_select['instrument'] == 'aviris')].copy().reset_index(drop=True)

                # plot fractional cover values
                x = df_x[col_map[col]]
                y = df_y[col_map[col]]
                x_u = df_x[f'{col_map[col]}_se']
                y_u = df_y[f'{col_map[col]}_se']

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                performance = df_performance['spectra_per_s'].mean()

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None', capsize=5)
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10)

                # for i, label in enumerate(df_x['plot'].values):
                #     ax.text(x[i], y[i], label, fontsize=12, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),
                    #r'CPU: %.2f' % (performance,),
                ))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        fig.supylabel('AVIRIS$_{NG}$ Fractions', fontsize=self.axis_label_fontsize)
        plt.savefig(os.path.join(self.figure_directory, f'shift_regression_{norm_option}.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()

    def plot_combined(self, norm_option):

        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['plot_num'] = df_emit['plot'].str.split('-').str[1].str.strip().astype(int)
        df_emit = df_emit[df_emit['plot_num'] <= 60]
        df_emit['campaign'] = 'emit'

        #skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL', 'SRB-047_SPRING', 'SRB-050_FALL']
        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL']
        df_aviris = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        row_configs = [
            {'unmix_mode': 'emc2', 'lib_mode': 'local', 'num_cmb_em': 20},
            {'unmix_mode': 'emc2', 'lib_mode': 'global', 'num_cmb_em': 20},
            {'unmix_mode': 'mesma', 'lib_mode': 'local', 'num_cmb_em': 100},
            {'unmix_mode': 'mesma', 'lib_mode': 'global', 'num_cmb_em': 100}
        ]

        fig = plt.figure(figsize=(6.5, 8.5))  # Adjusted height for 4 rows
        ncols = 3
        nrows = 4
        grid_size = (12, 12)
        r_height = 12 // nrows
        c_width = 12 // ncols

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        for row, config in enumerate(row_configs):
            # Filter data for the entire row once
            df_row_emit = df_all[
                (df_all['unmix_mode'] == config['unmix_mode']) &
                (df_all['lib_mode'] == config['lib_mode']) &
                (df_all['campaign'] == 'emit') &
                (df_all['normalization'] == norm_option) &
                (df_all['num_mc'] == 25) &
                (df_all['num_cmb_em'] == config['num_cmb_em'])
                ].copy()

            df_row_shift = df_all[
                (df_all['unmix_mode'] == config['unmix_mode']) &
                (df_all['lib_mode'] == config['lib_mode']) &
                (df_all['campaign'] == 'shift') &
                (df_all['normalization'] == norm_option) &
                (df_all['num_mc'] == 25) &
                (df_all['num_cmb_em'] == config['num_cmb_em'])
                ].copy()

            for col in range(ncols):
                r_start = row * r_height
                c_start = col * c_width

                ax = plt.subplot2grid(grid_size, (r_start, c_start), colspan=c_width, rowspan=r_height)

                ax.set_ylim(self.axes_limits['ymin'], 1)
                ax.set_xlim(self.axes_limits['xmin'], 1)
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{self.sig_figs}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{self.sig_figs}f'))

                ax.yaxis.set_major_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.xaxis.set_major_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='both', which='major', width=2, length=5)
                ax.tick_params(axis='both', which='minor', width=1, length=2.5)

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=14)
                if row == 3 and col == 1:
                    ax.set_xlabel("SLPIT", fontsize=10)


                if col == 0:
                    mode_label = 'E(MC)$^2$' if config['unmix_mode'] == 'emc2' else config['unmix_mode'].upper()
                    ax.set_ylabel(f"{mode_label}$_{{{config['lib_mode']}}}$", fontsize=10)
                else:
                    ax.set_yticklabels([])


                if row != 3:
                    ax.set_xticklabels([])
                if row != 3 and col == 0:
                    ax.set_yticklabels([''] + [l.get_text() for l in ax.get_yticklabels()[1:]])

                df_x_emit = df_row_emit[df_row_emit['instrument'] == 'SLPIT'].reset_index(drop=True)
                df_y_emit = df_row_emit[df_row_emit['instrument'] == 'RFL'].reset_index(drop=True)
                df_x_shift = df_row_shift[df_row_shift['instrument'] == 'SLPIT'].reset_index(drop=True)
                df_y_shift = df_row_shift[df_row_shift['instrument'] == 'RFL'].reset_index(drop=True)

                col_name = col_map[col]
                x = list(df_x_emit[col_name]) + list(df_x_shift[col_name])
                y = list(df_y_emit[col_name]) + list(df_y_shift[col_name])
                x_u = list(df_x_emit[f'{col_name}_sigma']) + list(df_x_shift[f'{col_name}_sigma'])
                y_u = list(df_y_emit[f'{col_name}_sigma']) + list(df_y_shift[f'{col_name}_sigma'])

                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red', zorder=1, linewidth=1)

                if len(x) > 0:
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                    ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', ecolor='gray', alpha=0.5, zorder=9)
                    ax.scatter(df_x_emit[col_name], df_y_emit[col_name], marker='s', color='blue', edgecolor='black',
                               label='EMIT', zorder=10)
                    ax.scatter(df_x_shift[col_name], df_y_shift[col_name], marker='^', color='orange',
                               edgecolor='black', label='AVIRIS$_{NG}$', zorder=10)

                    rmse = root_mean_squared_error(x, y)
                    mae = mean_absolute_error(x, y)
                    r2 = r2_calculations(x, y)[0]

                    txtstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR$^2$: {r2:.2f}'
                    ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=7, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

                if col == 1 and row == 2:
                    ax.legend(loc='lower right', fontsize=6, facecolor='lightgray', framealpha=0.33)

        fig.supylabel(r'Spaceborne/Airborne Fractions', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_directory, f'Fig-4_regression_combined_{norm_option}.png'), dpi=600)
        plt.clf()
        plt.close()

    def mesma_vs_emc2(self, norm_option):

        # load fraction plots
        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['plot_num'] = df_emit['plot'].str.split('-').str[1].str.strip().astype(int)
        df_emit = df_emit[df_emit['plot_num'] <= 60]
        df_emit['campaign'] = 'emit'

        #skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL', 'SRB-050_FALL', 'SRB-200_FALL']
        skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL']
        df_aviris = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        #  create figure
        fig = plt.figure(figsize=(6.5, 3))  # Adjust size for single row
        grid_size = (9, 9)
        ncols = 3
        col_map = {0: 'npv', 1: 'pv', 2: 'soil'}

        df_base = df_all[(df_all['lib_mode'] == 'global') &
                        (df_all['num_mc'] == 25) &
                        (df_all['normalization'] == norm_option)].copy()

        for col in range(ncols):
            c_width = 9 // ncols
            c_start = col * c_width
            ax = plt.subplot2grid(grid_size, (0, c_start), colspan=c_width, rowspan=9)
            ax.set_aspect('equal', adjustable='box')
            ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
            ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
            ax.set_title(self.ems[col], fontsize=14)
            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{self.sig_figs}f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{self.sig_figs}f'))

            ax.yaxis.set_major_locator(MultipleLocator(0.25))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.xaxis.set_major_locator(MultipleLocator(0.25))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', which='major', width=2, length=5)
            ax.tick_params(axis='both', which='minor', width=1, length=2.5)

            # Axis Labels
            ax.set_xlabel("SLPIT - MESMA", fontsize=10)
            if col == 0:
                ax.set_ylabel(r'E(MC)$^2_{global}$', fontsize=10)
            else:
                ax.set_yticklabels([])
                ax.set_xticklabels([''] + [l.get_text() for l in ax.get_xticklabels()[1:]])


            # EMIT Comparison
            d_emit = df_base[df_base['campaign'] == 'emit']
            x_emit_df = d_emit[(d_emit['instrument'] == 'SLPIT') & (d_emit['unmix_mode'] == 'mesma')].reset_index(
                drop=True)
            y_emit_df = d_emit[(d_emit['instrument'] == 'RFL') & (d_emit['unmix_mode'] == 'emc2')].reset_index(
                drop=True)

            # AVIRIS Comparison
            d_shift = df_base[df_base['campaign'] == 'shift']
            x_shift_df = d_shift[(d_shift['instrument'] == 'SLPIT') & (d_shift['unmix_mode'] == 'mesma')].reset_index(
                drop=True)
            y_shift_df = d_shift[(d_shift['instrument'] == 'RFL') & (d_shift['unmix_mode'] == 'emc2')].reset_index(
                drop=True)

            target_col = col_map[col]
            x = list(x_emit_df[target_col]) + list(x_shift_df[target_col])
            y = list(y_emit_df[target_col]) + list(y_shift_df[target_col])
            x_u = list(x_emit_df[f'{target_col}_sigma']) + list(x_shift_df[f'{target_col}_sigma'])
            y_u = list(y_emit_df[f'{target_col}_sigma']) + list(y_shift_df[f'{target_col}_sigma'])


            one_line = np.linspace(0, 1, 101)
            ax.plot(one_line, one_line, color='red', zorder=1)  # 1:1 line

            if len(x) > 0:
                m, b = np.polyfit(x, y, 1)
                ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', ecolor='gray', zorder=9, alpha=0.4)
                ax.scatter(x_emit_df[target_col], y_emit_df[target_col], marker='s', color='blue', edgecolor='black',
                           label='EMIT', zorder=10)
                ax.scatter(x_shift_df[target_col], y_shift_df[target_col], marker='^', color='orange',
                           edgecolor='black', label='AVIRIS$_{NG}$', zorder=10)

                rmse = root_mean_squared_error(x, y)
                mae = mean_absolute_error(x, y)
                r2, _ = r2_calculations(x, y)
                txtstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR$^2$: {r2:.2f}'
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=7, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

            if col == 1:
                ax.legend(loc='lower right', fontsize=6, facecolor='lightgray', framealpha=0.25)

        fig.supylabel(r'Spaceborne\Airborne Fractions', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_directory, f'Fig-6_mesma_vs_emc2_{norm_option}.png'), dpi=600)
        plt.clf()
        plt.close()


    def  error_vs_time(self, norm_option):

        # days table
        df_days = pd.read_csv(os.path.join('slpit', f'overpass_table.csv'))
        df_days['Site ID1'] = df_days['Site ID1'].str.replace(" ", "")
        df_days_emit = df_days[df_days['Sensor'] == 'EMIT']
        df_days_emit['Site ID1'] = df_days_emit['Site ID1'].str.replace("SPEC", "Spectral")
        df_days_avr = df_days[df_days['Sensor'] == 'AVIRISNG']

        # load fraction plots
        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['plot_num'] = df_emit['plot'].str.split('-').str[1].str.strip().astype(int)
        df_emit = df_emit[df_emit['plot_num'] <= 60]
        df_emit['campaign'] = 'emit'

        #skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL', 'SRB-050_FALL', 'SRB-200_FALL']
        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL']
        df_aviris = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        #  create figure
        fig = plt.figure(figsize=(6.5, 3))
        grid_size = (9, 9)
        ncols = 3

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        df_select_emit = df_all[(df_all['unmix_mode'] == 'emc2') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()
        df_select_emit = df_days_emit.merge(df_select_emit, left_on='Site ID1', right_on='plot', how='inner')

        df_select_shift = df_all[(df_all['unmix_mode'] == 'emc2') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()
        df_select_shift = df_days_avr.merge(df_select_shift, left_on='Site ID1', right_on='plot', how='inner')

        mode = list(df_select_emit['unmix_mode'].unique())[0]
        lib_mode = list(df_select_emit['lib_mode'].unique())[0]

        # emit variables
        df_x_emit = df_select_emit[(df_select_emit['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
        df_y_emit = df_select_emit[(df_select_emit['instrument'] == 'RFL')].copy().reset_index(drop=True)

        # aviris variables
        df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
        df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'RFL')].copy().reset_index(drop=True)

        for col in range(0, 3):
            c_width = 9 // ncols
            c_start = col * c_width
            ax = plt.subplot2grid(grid_size, (0, c_start), colspan=c_width, rowspan=9)
            ax.set_box_aspect(1)
            ax.set_ylim(self.axes_limits['ymin'], 0.25)
            ax.set_xlim(-95, 95)

            ax.set_title(self.ems[col], fontsize=14)
            #ax.set_yticks(np.arange(self.axes_limits['ymin'], 0.25 + 0.05, 0.05))

            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(0)}f'))

            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            #ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.xaxis.set_major_locator(MultipleLocator(25))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='both', which='major', width=2, length=5)
            ax.tick_params(axis='both', which='minor', width=1, length=2.5)

            if col == 1:
                ax.set_xlabel('', fontsize=1)

            if mode == 'emc2':
                mode = 'E(MC)$^2$'

            if col == 0:
                ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$ Absolute Error', fontsize=10)

            if col != 0:
                ax.set_yticklabels([])

            # plot fractional cover values
            x_emit = df_x_emit[col_map[col]]
            y_emit = df_y_emit[col_map[col]]
            emit_days = df_x_emit['Day Difference3']
            emit_error = np.absolute(x_emit - y_emit)

            x_shift = df_x_shift[col_map[col]]
            y_shift = df_y_shift[col_map[col]]
            avr_days = df_x_shift['Day Difference3']
            avr_error = np.absolute(x_shift - y_shift)

            x = list(x_emit.values) + list(x_shift.values)
            y = list(y_emit.values) + list(y_shift.values)
            days = list(emit_days.values) + list(avr_days.values)

            error = np.absolute(np.array(x) - np.array(y))

            mae = []
            for day in sorted(list(set(days))):
                indices = [i for i, val in enumerate(days) if val == day]
                selected_error = np.mean([error[i] for i in indices])
                mae.append(selected_error)

            ax.scatter(emit_days, emit_error, marker='s', color='blue', edgecolor='black', label='EMIT', zorder=10)
            ax.scatter(avr_days, avr_error, marker='^', color='orange', edgecolor='black', label='AVIRIS$_{NG}$', zorder=10)

            r2, bias = r2_calculations(days, error)
            print(f'r^2 = {r2:.3f} and bias = {bias:.3f} for days vs. error for each overpass')

            # Adjust subplot to make space for arrows
            plt.subplots_adjust(bottom=0.3)

            # Add arrows spanning the specified ranges below the x-axis
            ax.annotate('', xy=(-90, -0.20), xytext=(-25, -0.20),
                        arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10, ha='center',
                        xycoords=('data', 'axes fraction'))

            ax.annotate('', xy=(90, -0.20), xytext=(25, -0.20),
                        arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10, ha='center',
                        xycoords=('data', 'axes fraction'))

            # Add labels at the center of the arrows
            ax.text((-90 + -25) / 2, -0.35, 'Days before\noverpass', fontsize=10, ha='center', va='center',
                    transform=ax.get_xaxis_transform())

            ax.text((90 + 25) / 2, -0.35, 'Days after\noverpass', fontsize=10, ha='center', va='center',
                    transform=ax.get_xaxis_transform())

            if col == 2:
                ax.legend(loc='upper right', fontsize=8, facecolor='lightgray', framealpha=0.25)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_directory, f'Sup-A_error_vs_time_{norm_option}.png'), dpi=600)# load all fraction files
        plt.clf()
        plt.close()

    def cross_norm(self, mode):
        # load fraction plots
        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['plot_num'] = df_emit['plot'].str.split('-').str[1].str.strip().astype(int)
        df_emit = df_emit[df_emit['plot_num'] <= 60]
        df_emit['campaign'] = 'emit'

        #skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL', 'SRB-050_FALL', 'SRB-200_FALL']
        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL']
        df_aviris = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        fig = plt.figure(figsize=(6.5, 4.67))
        ncols = 3
        nrows = 2

        col_map = {0: 'npv', 1: 'pv', 2: 'soil'}
        str_mode = 'E(MC)$^2$' if mode == 'emc2' else 'MESMA'

        # Pre-filter base data to reduce overhead inside the loop
        df_base_emit = df_all[
            (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['campaign'] == 'emit')].copy()
        df_base_shift = df_all[
            (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['campaign'] == 'shift')].copy()

        for row in range(nrows):

            norm_x = 'brightness' if row == 0 else 'none'
            norm_y = 'none' if row == 0 else 'brightness'
            label_x = "Vector normalization" if row == 0 else "No normalization"
            label_y = "No normalization" if row == 0 else "Vector normalization"

            for col in range(ncols):
                # Using a standard grid is cleaner here
                ax = plt.subplot2grid((nrows, ncols), (row, col))

                ax.set_box_aspect(1)
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{self.sig_figs}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{self.sig_figs}f'))
                ax.yaxis.set_major_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.xaxis.set_major_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='both', which='major', width=2, length=5)
                ax.tick_params(axis='both', which='minor', width=1, length=2.5)

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=14)
                    ax.set_xticklabels([])

                if col == 1:
                    ax.set_xlabel(f"SLPIT {str_mode}\n({label_x})", fontsize=10)

                if col == 0:
                    ax.set_ylabel(f"{str_mode}\n({label_y})", fontsize=10)
                else:
                    ax.set_yticklabels([])
                    # Clean up overlapping x-labels
                    ax.set_xticklabels([''] + [l.get_text() for l in ax.get_xticklabels()[1:]])

                df_x_emit = df_base_emit[
                    (df_base_emit['instrument'] == 'SLPIT') & (df_base_emit['unmix_mode'] == mode) & (
                                df_base_emit['normalization'] == norm_x)].reset_index(drop=True)
                df_y_emit = df_base_emit[
                    (df_base_emit['instrument'] == 'RFL') & (df_base_emit['unmix_mode'] == mode) & (
                                df_base_emit['normalization'] == norm_y)].reset_index(drop=True)

                df_x_shift = df_base_shift[
                    (df_base_shift['instrument'] == 'SLPIT') & (df_base_shift['unmix_mode'] == mode) & (
                                df_base_shift['normalization'] == norm_x)].reset_index(drop=True)
                df_y_shift = df_base_shift[
                    (df_base_shift['instrument'] == 'RFL') & (df_base_shift['unmix_mode'] == mode) & (
                                df_base_shift['normalization'] == norm_y)].reset_index(drop=True)

                target = col_map[col]
                x = list(df_x_emit[target]) + list(df_x_shift[target])
                y = list(df_y_emit[target]) + list(df_y_shift[target])
                x_u = list(df_x_emit[f'{target}_sigma']) + list(df_x_shift[f'{target}_sigma'])
                y_u = list(df_y_emit[f'{target}_sigma']) + list(df_y_shift[f'{target}_sigma'])

                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red', zorder=1, linewidth=1, alpha=0.8)

                if len(x) > 0:
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                    ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', ecolor='gray', alpha=0.4, zorder=9)
                    ax.scatter(df_x_emit[target], df_y_emit[target], marker='s', color='blue', edgecolor='black',
                               label='EMIT', zorder=10, s=25)
                    ax.scatter(df_x_shift[target], df_y_shift[target], marker='^', color='orange', edgecolor='black',
                               label='AVIRIS$_{NG}$', zorder=10, s=25)

                    rmse = root_mean_squared_error(x, y)
                    mae = mean_absolute_error(x, y)
                    r2, _ = r2_calculations(x, y)

                    txtstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR$^2$: {r2:.2f}'
                    ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=7, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

                if col == 2 and row == 0:
                    ax.legend(loc='lower right', fontsize=6, facecolor='lightgray', framealpha=0.25)

        fig.supylabel(r'Spaceborne/Airborne Fractions', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        save_path = os.path.join(self.fig_directory, f'Fig-7_cross_normalization_{mode}.png')
        plt.savefig(save_path, dpi=600)
        plt.clf()
        plt.close()


    def uncertainty_table(self):
        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['plot_num'] = df_emit['plot'].str.split('-').str[1].str.strip().astype(int)
        df_emit = df_emit[df_emit['plot_num'] <= 60]
        df_emit['campaign'] = 'emit'

        #skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL', 'SRB-050_FALL', 'SRB-200_FALL']
        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL']
        df_aviris = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        for mode in ['emc2', 'mesma']:
            for lib_mode in ['local', 'global']:
                # # loop through figure columns
                df_select_emit = df_all[(df_all['num_mc'] == 25) & (df_all['campaign'] == 'emit') & (df_all['unmix_mode'] == mode) & (df_all['lib_mode'] == lib_mode)].copy()
                df_select_shift = df_all[(df_all['num_mc'] == 25) & (df_all['campaign'] == 'shift') & (df_all['unmix_mode'] == mode) & (df_all['lib_mode'] == lib_mode)].copy()

                for em in ['npv', 'pv', 'soil']:
                    df_x_emit = df_select_emit[(df_select_emit['instrument'] == 'SLPIT') & (df_select_emit['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_emit = df_select_emit[(df_select_emit['instrument'] == 'RFL') & (df_select_emit['normalization'] == 'brightness')].copy().reset_index(drop=True)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'SLPIT') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'RFL') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                    # plot fractional cover values
                    x_emit = df_x_emit[em]
                    y_emit = df_y_emit[em]
                    x_u_emit = df_x_emit[f'{em}_sigma']
                    y_u_emit = df_y_emit[f'{em}_sigma']


                    x_shift = df_x_shift[em]
                    y_shift = df_y_shift[em]
                    x_u_shift = df_x_shift[f'{em}_sigma']
                    y_u_shift = df_y_shift[f'{em}_sigma']

                    x = list(x_emit.values) + list(x_shift.values)
                    y = list(y_emit.values) + list(y_shift.values)
                    x_u = list(x_u_emit.values) + list(x_u_shift.values)
                    y_u = list(y_u_emit.values) + list(y_u_shift.values)

                    x_u = np.array(x_u)
                    y_u = np.array(y_u)

                    mu_t = 1/2 * np.sum((x_u + y_u))/(len(x_u) + len(y_u))
                    rmsu_t = 1/2 * np.sqrt(np.sum((x_u + y_u)**2)/(len(x_u) + len(y_u)))
                    mae = mean_absolute_error(x, y)
                    rmse = root_mean_squared_error(x,y)

                    mut_emit = 1/2 * np.sum((x_u_emit + y_u_emit))/(len(x_u_emit) + len(y_u_emit))
                    rmsu_t_emit = 1 / 2 * np.sqrt(np.sum((x_u_emit + y_u_emit) ** 2) / (len(x_u_emit) + len(y_u_emit)))
                    mae_emit = mean_absolute_error(x_emit, y_emit)
                    rmse_emit = root_mean_squared_error(x_emit, y_emit)

                    mut_aviris = 1 / 2 * np.sum((x_u_shift + y_u_shift)) / (len(x_u_shift) + len(y_u_shift))
                    rmsu_t_aviris = 1 / 2 * np.sqrt(np.sum((x_u_shift + y_u_shift) ** 2) / (len(x_u_shift) + len(y_u_shift)))
                    mae_aviris = mean_absolute_error(x_shift, y_shift)
                    rmse_aviris = root_mean_squared_error(x_shift, y_shift)


                    print(f"EMIT: unmix mode: {mode}, lib: {lib_mode}, em: {em}, {np.round(mae_emit, 2)}, {np.round(rmse_emit, 2)}, {np.round(mut_emit, 2)}, {np.round(rmsu_t_emit, 2)}")
                    print(f"AVIRIS: unmix mode: {mode}, lib: {lib_mode}, em: {em}, {np.round(mae_aviris,2)}, {np.round(rmse_aviris,2)}, {np.round(mut_aviris,2)}, {np.round(rmsu_t_aviris,2)}")
                    print(f"Combined: unmix mode: {mode}, lib: {lib_mode}, em: {em}, {np.round(mae, 2)}, {np.round(rmse, 2)}, {np.round(mu_t, 2)}, {np.round(rmsu_t, 2)}")
                    print()


    def supplemental_combined(self, norm_option):

        # load fraction plots
        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['plot_num'] = df_emit['plot'].str.split('-').str[1].str.strip().astype(int)
        df_emit = df_emit[df_emit['plot_num'] <= 60]
        df_emit['campaign'] = 'emit'

        #skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL', 'SRB-050_FALL', 'SRB-200_FALL']
        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL']
        df_aviris = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        #  create figure
        fig = plt.figure(figsize=(6.5, 4.75))
        ncols = 3
        nrows = 2


        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        # # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'emc2_best') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'emc2_best') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()

            if row == 1:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'emc2_best') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'emc2_best') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()

            for col in range(ncols):
                ax = plt.subplot2grid((nrows, ncols), (row, col))
                ax.set_box_aspect(1)
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                ax.yaxis.set_major_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.xaxis.set_major_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='both', which='major', width=2, length=5)
                ax.tick_params(axis='both', which='minor', width=1, length=2.5)

                mode = list(df_select_emit['unmix_mode'].unique())[0]
                lib_mode = list(df_select_emit['lib_mode'].unique())[0]

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=14)

                if row == 1 and col == 1:
                    ax.set_xlabel("SLPIT", fontsize=10)

                if row == 1 and col != 0:
                    ax.set_xticklabels([''] + ax.get_xticklabels()[1:])

                if col == 0:
                    if mode == 'emc2' or mode == 'emc2_best':
                        mode = 'E(MC)$^2$'

                    ax.set_ylabel(mode.upper() + '$_{'+lib_mode +'}$', fontsize=10)


                if col != 0:
                    ax.set_yticklabels([])

                if row != 1:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                # emit variables
                df_x_emit = df_select_emit[(df_select_emit['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
                df_y_emit = df_select_emit[(df_select_emit['instrument'] == 'RFL')].copy().reset_index(drop=True)

                # aviris variables
                df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'SLPIT')].copy().reset_index(drop=True)
                df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'RFL')].copy().reset_index(drop=True)

                # plot fractional cover values
                x_emit = df_x_emit[col_map[col]]
                y_emit = df_y_emit[col_map[col]]
                x_u_emit = df_x_emit[f'{col_map[col]}_sigma']
                y_u_emit = df_y_emit[f'{col_map[col]}_sigma']

                x_shift = df_x_shift[col_map[col]]
                y_shift = df_y_shift[col_map[col]]
                x_u_shift = df_x_shift[f'{col_map[col]}_sigma']
                y_u_shift = df_y_shift[f'{col_map[col]}_sigma']

                x = list(x_emit.values) + list(x_shift.values)
                y = list(y_emit.values) + list(y_shift.values)
                x_u = list(x_u_emit.values) + list(x_u_shift.values)
                y_u = list(y_u_emit.values) + list(y_u_shift.values)

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                ax.plot(one_line, one_line, color='red', zorder=1)
                ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', ecolor='gray', alpha=0.4, zorder=9)
                ax.scatter(x_emit, y_emit, marker='s', color='blue', edgecolor='black', label='EMIT', zorder=10)
                ax.scatter(x_shift,y_shift, marker='^', color='orange', edgecolor='black', label='AVIRIS$_{NG}$', zorder=10)

                if col == 1 and row == 1:
                    ax.legend(loc='lower right', fontsize=6, facecolor='lightgray', framealpha=0.33)

                # Add error metrics
                rmse = root_mean_squared_error(x, y)
                mae = mean_absolute_error(x, y)
                r2, bias = r2_calculations(x, y)

                txtstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR$^2$: {r2:.2f}'

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=7,
                        verticalalignment='top', bbox=props)

        fig.supylabel(r'Spaceborne\Airborne Fractions', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0.12)
        #plt.show()
        plt.savefig(os.path.join(self.fig_directory, f'Sup-B_regression_combined_{norm_option}_emc2-best.png'), format="png", dpi=600)# load all fraction files
        plt.clf()
        plt.close()

    def local_slpit(self):

        # load all fraction files
        #skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL', 'SRB-047_SPRING', 'SRB-050_FALL']
        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL']
        df_all = pd.read_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'))
        df_all = df_all[~df_all['plot'].isin(skip)]

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
                        ax.set_xlabel('SLPIT -emc2- Fractions')
                        ax.set_ylabel("EMIT Fractions")
                    if row == 1:
                        ax.set_xlabel('SLPIT -mesma- Fractions')
                        ax.set_ylabel("EMIT Fractions")

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
                    x = df_x[col_map[col]]
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

            fig.colorbar(scatter, label='Plot Number')
            plt.savefig(os.path.join(self.fig_directory, f'{lib_mode}_regression.png'), format="png", dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

def run_figures(base_directory):
    base_directory = base_directory
    sensor = 'aviris_ng'
    major_axis_fontsize = 12
    minor_axis_fontsize = 10
    title_fontsize = 12
    axis_label_fontsize = 12
    fig_height = 10
    fig_width = 12
    linewidth = 1
    sig_figs = 2

    fig = figures(base_directory=base_directory, sensor=sensor, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs)
    fig.plot_summary()
    fig.plot_combined(norm_option='brightness')
    fig.plot_combined(norm_option='none')
    fig.mesma_vs_emc2(norm_option='brightness')
    fig.cross_norm(mode='mesma')
    fig.cross_norm(mode='emc2')
    fig.supplemental_combined(norm_option='brightness')
    fig.error_vs_time(norm_option='brightness')
    fig.local_slpit()
    fig.uncertainty_table()


