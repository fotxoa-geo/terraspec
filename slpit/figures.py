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
from mpl_toolkits.basemap import Basemap
import matplotlib.image as mpimg
from pypdf import PdfMerger
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.envi import envi_to_array, load_band_names, read_metadata
from datetime import datetime, timezone
from p_tqdm import p_map
from isofit.core.sunposition import sunpos
import geopandas as gp
from utils.results_utils import r2_calculations, load_data, error_metrics
from matplotlib.ticker import FormatStrFormatter


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
    for _band, band in enumerate(range(0, fraction_array.shape[2])):
            
            selected_fractions = fraction_array[:, :, _band]
            selected_fractions, duplicate_flag = duplicate_check_fractions(selected_fractions)
            #if instrument == 'asd':
             #   mean_fractions.append(np.mean(fraction_array[:, :, _band]))
            #else:
            #    mean_fractions.append(np.mean(fraction_array[:, :, _band]))
            
            mean_fractions.append(np.nanmean(selected_fractions))

            se = np.mean(unc_array[:, :, _band]/np.sqrt(int(num_mc)))
            mean_se.append(se)

    return [instrument, unmix_mode, plot, library_mode, int(num_cmb_em), int(num_mc), normalization, fraction_array.shape[0], fraction_array.shape[1], duplicate_flag] + mean_fractions + mean_se


class figures:
    def __init__(self, base_directory: str, sensor: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory,  "figures")
        self.gis_directory = os.path.join(base_directory, "gis")

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='emit')
        create_directory(os.path.join(base_directory, "figures"))

        # ems
        self.ems = ['NPV', 'GV', 'Soil']

        # load teatracorder directory
        terraspec_base = os.path.dirname(base_directory)
        self.tetracorder_output_directory = os.path.join(terraspec_base, 'tetracorder', 'output', 'spectral_abundance')
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
        # spectral data
        create_directory(os.path.join(self.fig_directory, 'plot_stats'))
        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-emit.csv'))
        em_data = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-emit.csv'))

        asd_wavelengths = np.array(transect_data.columns[9:]).astype(float)
        good_bands = spectra.get_good_bands_mask(asd_wavelengths, wavelength_pairs=None)
        asd_wavelengths[~good_bands] = np.nan

        self.good_emit_bands = spectra.get_good_bands_mask(self.wvls, wavelength_pairs=None)
        self.wvls[~self.good_emit_bands] = np.nan

        # plot summary - merged
        merger = PdfMerger()

        # gis shapefile
        df_gis = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
        df_gis = df_gis.sort_values('Name')

        # fractional cover
        for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):
            fig = plt.figure(figsize=(14, 8))

            fig.suptitle(plot)
            gs1 = gridspec.GridSpec(2, 2)
            gs1.update(left=0.05, right=0.49, wspace=0.05, hspace=0.1)
            ax1 = plt.subplot(gs1[:-1, 0])
            ax2 = plt.subplot(gs1[:-1, 1])
            ax3 = plt.subplot(gs1[-1, :])

            gs2 = gridspec.GridSpec(4, 4)
            gs2.update(left=0.53, right=0.98, hspace=0.1, wspace=0.05)

            ax4 = plt.subplot(gs2[0, :])
            ax5 = plt.subplot(gs2[1, :])
            ax6 = plt.subplot(gs2[2, :])
            ax7 = plt.subplot(gs2[3, :])

            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            for ax in axes:
                if ax == ax1:
                    ax.set_title('Plot Map')
                    df_transect = transect_data.loc[transect_data['plot_name'] == plot].copy()
                    df_transect = df_transect[df_transect.longitude != 'unk']

                    m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=45,
                                llcrnrlon=-125, urcrnrlon=-100, ax=ax1, epsg=4326)
                    m.arcgisimage(service='World_Imagery', xpixels=1000, ypixels=1000, dpi=300, verbose=True)
                    ax.scatter(np.mean(df_transect.longitude), np.mean(df_transect.latitude), color='red', s=12)

                if ax == ax2:
                    ax.set_title('Landscape\nPicture')
                    pic_path = os.path.join(self.output_directory, 'plot_pictures', 'spectral_transects', plot + '.jpg')
                    img = mpimg.imread(pic_path)
                    ax.imshow(img)
                    ax.axis('off')

                if ax == ax3:
                    # plot average of all EMIT files across time
                    reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip', '*' + plot.replace("Spectral", "SPEC").replace(" ", "") + '_RFL_' + '*[!.xml][!.csv][!.hdr]')))

                    if not reflectance_files:
                        reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip',
                                                 '*' + plot.replace("Thermal", "THERM").replace(" ",
                                                                                                "") + '_RFL_' + '*[!.xml][!.csv][!.hdr]')))


                    # plot average of SLPIT
                    refl_file_asd = glob(os.path.join(self.output_directory, 'spectral_transects', 'transect',
                                                      "*" + plot.replace(" ", "")))
                    print(plot)
                    print(refl_file_asd)
                    df_refl_asd = gdal.Open(refl_file_asd[0], gdal.GA_ReadOnly)
                    refl_array_asd = df_refl_asd.ReadAsArray().transpose((1, 2, 0))
                    y = np.mean(refl_array_asd, axis=0).ravel()

                    # get date and time from slpit
                    df_transect = transect_data.loc[transect_data['plot_name'] == plot].copy()
                    slpit_date = df_transect['date'].unique()[0]
                    slpit_datetime = datetime.strptime(df_transect['date'].unique()[0], "%Y-%m-%d")

                    # Convert the time strings to datetime objects
                    df_transect['utc_time'] = pd.to_datetime(df_transect['utc_time'], format='%H:%M:%S')
                    df_transect['total_seconds'] = df_transect['utc_time'].dt.hour * 3600 + df_transect['utc_time'].dt.minute * 60 + df_transect['utc_time'].dt.second

                    # minimum and maximum total time in seconds for split
                    min_time_seconds = df_transect['total_seconds'].min()
                    max_time_seconds = df_transect['total_seconds'].max()

                    min_time_hhmmss = f"{min_time_seconds // 3600:02.0f}{(min_time_seconds % 3600) // 60:02.0f}{min_time_seconds % 60:02.0f}"
                    max_time_hhmmss = f"{max_time_seconds // 3600:02.0f}{(max_time_seconds % 3600) // 60:02.0f}{max_time_seconds % 60:02.0f}"

                    # Calculate mean time of ASD  collections
                    field_slpit_date_min = datetime.strptime(slpit_date + " " + min_time_hhmmss, "%Y-%m-%d %H%M%S").replace(tzinfo=timezone.utc)
                    field_slpit_date_max = datetime.strptime(slpit_date + " " + max_time_hhmmss, "%Y-%m-%d %H%M%S").replace(tzinfo=timezone.utc)

                    ax.set_title(f'Field Sample Date: {slpit_date}')
                    df_gis_select = df_gis.loc[df_gis['Name'] == plot.replace("Spectral", "SPEC")].copy().reset_index(drop=True)
                    #field_emit_date = df_gis_select.at[0, 'EMIT Overp']
                    #field_emit_date = datetime.strptime(field_emit_date, "%b %d, %Y at %I:%M:%S %p")
                    #field_emit_date = field_emit_date.strftime("%Y%m%dT%H%M")

                    for _i, i in enumerate(reflectance_files):
                        acquisition_date = os.path.basename(i).split("_")[2]

                        df_refl = gdal.Open(i, gdal.GA_ReadOnly)
                        refl_array = df_refl.ReadAsArray().transpose((1, 2, 0))
                        y_hat = np.mean(refl_array, axis=(0, 1))

                        acquisition_datetime_utc = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                        geometry_results_emit = sunpos(acquisition_datetime_utc, np.mean(df_transect.latitude),
                                                       np.mean(df_transect.longitude), np.mean(df_transect.elevation))

                        acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
                        formatted_datetime = acquisition_datetime.strftime("%Y-%m-%d %I:%M %p")
                        delta = slpit_datetime - acquisition_datetime
                        days = np.absolute(delta.days)

                        cloud_mask = glob(os.path.join(self.gis_directory, 'emit-data-clip', f'*{plot.replace("Spectral", "SPEC").replace(" ", "")}_MASK_{acquisition_date}'))

                        if not cloud_mask:
                            cloud_mask = glob(os.path.join(self.gis_directory, 'emit-data-clip',
                                                           f'*{plot.replace("Thermal", "THERM").replace(" ", "")}_MASK_{acquisition_date}'))

                        ds_cloud = gdal.Open(cloud_mask[0], gdal.GA_ReadOnly)
                        cloud_array = ds_cloud.ReadAsArray().transpose((1, 2, 0))

                        if days <= 100:
                            cloud_check = np.any(cloud_array == 1)

                            if cloud_check:
                                pass
                            else:
                                base_label = f'{acquisition_date} (±{days:02d} days)  SZA : {str(int(geometry_results_emit[1]))}°'
                                ax.plot(self.wvls,  np.mean(refl_array, axis=(0, 1)), label=base_label, linewidth=1)

                                # if field_emit_date == acquisition_date[:-2]:
                                #     bold_label = base_label

                    geometry_results_slpit_min = sunpos(field_slpit_date_min, np.mean(df_transect.latitude),
                                                       np.mean(df_transect.longitude), np.mean(df_transect.elevation))

                    geometry_results_slpit_max = sunpos(field_slpit_date_max, np.mean(df_transect.latitude),
                                                        np.mean(df_transect.longitude), np.mean(df_transect.elevation))

                    slpit_label = f'SLPIT (SZA: {str(int(geometry_results_slpit_min[1]))} - {str(int(geometry_results_slpit_max[1]))}°)'
                    ax.plot(self.wvls, np.mean(refl_array_asd, axis=0).ravel(), color='black', label=slpit_label,
                            linewidth=1.5)

                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('Reflectance (%)')
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)

                    legend = ax.legend()

                    # # make emit field date bold
                    # for handle, text in zip(legend.legendHandles, legend.get_texts()):
                    #     if text.get_text() == bold_label:
                    #         text.set_fontweight('bold')

                if ax == ax4:
                    df_spectra = em_data[(em_data['plot_name'] == plot) & ((em_data['level_1'] == 'NPV'))].copy()
                    df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
                    num_species = len(sorted(list(df_spectra.species.unique())))
                    npv_cmap = plt.cm.get_cmap(self.cmap_kw, num_species + 1)

                    for _species, species in enumerate(sorted(list(df_spectra.species.unique()))):
                        df_species = df_spectra[df_spectra['species'] == species].copy()
                        em_spectra = df_species.iloc[:, 10:].to_numpy()

                        for _row, row in enumerate(em_spectra):
                            ax.plot(self.wvls, row, color=npv_cmap(_species))

                        common_name = df_species_key.loc[df_species_key['key_value'] == species, ['label']].values[0][0]
                        ax.plot(self.wvls, row, c=npv_cmap(_species), label=common_name)

                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"NPV (n = {str(df_spectra.shape[0])})", fontsize=12)
                    ax.legend(prop={'size': 6})

                if ax == ax5:
                    df_spectra = em_data[(em_data['plot_name'] == plot) & ((em_data['level_1'] == 'PV'))].copy()
                    df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
                    num_species = len(sorted(list(df_spectra.species.unique())))
                    pv_cmap = plt.cm.get_cmap(self.cmap_kw, num_species + 1)

                    for _species, species in enumerate(sorted(list(df_spectra.species.unique()))):
                        df_species = df_spectra[df_spectra['species'] == species].copy()
                        em_spectra = df_species.iloc[:, 10:].to_numpy()

                        for _row, row in enumerate(em_spectra):

                            ax.plot(self.wvls, row, color=pv_cmap(_species))

                        common_name = df_species_key.loc[df_species_key['key_value'] == species, ['label']].values[0][0]
                        ax.plot(self.wvls, row, c=pv_cmap(_species), label=common_name)

                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"PV (n = {str(df_spectra.shape[0])})", fontsize=12)
                    ax.legend(prop={'size': 6})

                if ax == ax6:
                    df_spectra = em_data[(em_data['plot_name'] == plot) & ((em_data['level_1'] == 'Soil'))].copy()
                    em_spectra = df_spectra.iloc[:, 10:].to_numpy()
                    for _row, row in enumerate(em_spectra):
                        ax.plot(self.wvls, row, color='blue')
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(2100, 0.85, f"Soil (n = {str(em_spectra.shape[0])})", fontsize=12)

                if ax == ax7:
                    try:
                        slpit_ems_abundance = glob(os.path.join(self.tetracorder_output_directory, '*' + plot.replace(" ", "") +
                                         '*emit_ems_augmented_abun_mineral'))

                        slpit_transect_abundance = glob(os.path.join(self.tetracorder_output_directory, '*' + plot.replace(" ", "") +
                                         '*transect_augmented_abun_mineral'))

                        emit_spectral_abundance = glob(os.path.join(self.tetracorder_output_directory, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
                                         '*pixels_augmented_abun_mineral'))

                        # load df for em position key
                        em_csv = os.path.join(self.output_directory, 'spectral_transects', 'endmembers', plot.replace(" ", "") + '-emit.csv')
                        df_em = pd.read_csv(em_csv)
                        first_soil_index = df_em.index[df_em['level_1'] == 'Soil'].min()

                        # load data
                        split_abundance_array = envi_to_array(slpit_ems_abundance[0])[0,0,:]
                        #split_abundance_array[split_abundance_array == 0] = np.nan
                        emit_abundance_array = envi_to_array(emit_spectral_abundance[0])
                        #emit_abundance_array[emit_abundance_array == 0] = np.nan
                        split_transect_array = envi_to_array(slpit_transect_abundance[0])[0,0,:]
                        #split_transect_array[split_transect_array == 0] = np.nan

                        mineral_bands = load_band_names(slpit_ems_abundance[0])
                        mineral_bands = [item.replace('+', '\n') for item in mineral_bands]

                        ax.set_ylabel('Spectral Abundance')
                        ax.set_ylim(0, .25)

                        for _mineral, mineral in enumerate(mineral_bands):
                            avg_slpit_em = np.nanmean(split_abundance_array[_mineral])
                            avg_split_transect = np.nanmean(split_transect_array[_mineral])
                            avg_emit = np.mean(emit_abundance_array[0:3, 0:3, _mineral])

                            ax.bar(_mineral, avg_slpit_em, color='green', label='Contact Probe', edgecolor="black", width=0.2)
                            ax.bar(_mineral - 0.1, avg_split_transect, color='black', label='Fiber Optic', edgecolor="black",
                                   width=0.2)
                            ax.bar(_mineral + 0.1, avg_emit, color='blue', label='EMIT', edgecolor="black", width=0.2)

                        # Get handles and labels from the axes
                        handles, labels = ax.get_legend_handles_labels()

                        # Create a dictionary to keep track of unique labels
                        unique_labels = {}
                        unique_handles = []

                        # Iterate through the labels and handles and add them to the unique_labels dictionary
                        for i, label in enumerate(labels):
                            if label not in unique_labels:
                                unique_labels[label] = handles[i]
                                unique_handles.append(handles[i])

                        ax.legend(unique_handles, unique_labels.keys())
                        ax.set_xticks(np.arange(0, len(mineral_bands), step=1), minor=False)
                        ax.set_xticklabels(mineral_bands, fontdict=None, minor=False)
                        ax.tick_params(axis='x', labelsize=8)

                    except:
                        ax.set_ylabel('Spectral Abundance')

            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', plot + '.pdf'), format="pdf", dpi=300,
                        bbox_inches="tight")
            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', plot + '.png'), format="png", dpi=300,
                        bbox_inches="tight")
            plt.clf()
            plt.close()
            merger.append(os.path.join(self.fig_directory, 'plot_stats', plot + '.pdf'))

        # write pdf
        merger.write(os.path.join(self.fig_directory, 'plot_stats', 'plot_summary.pdf'))
        merger.close()

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
                    ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$\n\nEMIT',
                                  fontsize=self.axis_label_fontsize)

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
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

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                ax.scatter(x, y, marker='s', edgecolor='black', label='EMIT', zorder=10)

                performance = df_performance['spectra_per_s'].mean()

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae,rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),
                ))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.fig_directory, f'regression_{norm_option}.png'), format="png", dpi=400, bbox_inches="tight")


    def sza_plot(self):
        print('loading sza plot...')

        df_rows = []
        # gis shapefile
        gdf = gp.read_file(os.path.join('gis', "Observation.shp"))
        df_gis = gdf.drop(columns='geometry')
        df_gis['latitude'] = gdf['geometry'].apply(lambda geom: geom.y)
        df_gis['longitude'] = gdf['geometry'].apply(lambda geom: geom.x)
        df_gis = df_gis.sort_values('Name')

        # transect data for elevation
        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-emit.csv'))

        for index, row in df_gis.iterrows():
            plot = row['Name']
            emit_filetime = row['EMIT DATE']

            df_transect = transect_data.loc[transect_data['plot_name'] == plot.replace("SPEC", "Spectral")].copy()
            acquisition_datetime_utc = datetime.strptime(emit_filetime, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            geometry_results_emit = sunpos(acquisition_datetime_utc, row['latitude'], row['longitude'], np.mean(df_transect['elevation']))

            frac_emit_sma = glob(os.path.join(self.output_directory, 'sma', f'emit-local__*{plot.replace("Spectral", "SPEC").replace(" ", "")}__*_fractional_cover'))
            frac_slpit_sma = glob(os.path.join(self.output_directory, 'sma', f'asd-local__*{plot.replace("Spectral", "SPEC").replace(" ", "")}__*_fractional_cover'))

            sma_slipt_array, sma_emit_array = load_data(frac_slpit_sma[0], frac_emit_sma[0])

            frac_slpit_mesma = glob(os.path.join(self.output_directory, 'mesma', f'asd-local__*{plot.replace("Spectral", "SPEC").replace(" ", "")}__*_fractional_cover'))
            frac_emit_mesma = glob(os.path.join(self.output_directory, 'mesma',
                                                f'emit-local__*{plot.replace("Spectral", "SPEC").replace(" ", "")}__*_fractional_cover'))
            mesma_slipt_array, mesma_emit_array = load_data(frac_slpit_mesma[0], frac_emit_mesma[0])

            df_row = [plot, geometry_results_emit[1]]
            for arrays in [(sma_slipt_array, sma_emit_array), (mesma_slipt_array, mesma_emit_array)]:
                for _em, em in enumerate(self.ems):
                    y = arrays[0][:, :, _em]
                    y_hat = arrays[1][:,:, _em]

                    mae = np.absolute(np.mean(y)) - np.absolute(np.mean(y_hat))
                    df_row.append(np.absolute(mae))

            df_rows.append(df_row)

        df = pd.DataFrame(df_rows)
        df.columns = ['plot', 'sza', 'sma-npv', 'sma-pv', 'sma-soil', 'mesma-npv', 'mesma-pv', 'mesma-soil']
        df = df.sort_values('sza')
        df = df.dropna()
        # # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.grid('on', linestyle='--')
                ax.set_ylabel("Absolute Error")
                ax.set_xlabel('Solar Zenith Angle (°)')

                ax.set_xlim(0, 60)
                ax.set_ylim(0, 0.5)

                if col == 0 and row == 0:
                    ax.set_title(f'NPV')
                    l = df['sma-npv']
                    marker = '.'

                if col == 1 and row == 0:
                    ax.set_title(f'GV')
                    l = df['sma-pv']
                    marker = '+'

                if col == 2 and row == 0:
                    ax.set_title(f'Soil')
                    l = df['sma-soil']
                    marker = 'x'

                if col == 0 and row == 1:
                    l = df['mesma-npv']
                    marker = '.'

                if col == 1 and row == 1:
                    l = df['mesma-pv']
                    marker = '+'

                if col == 2 and row == 1:
                    l = df['mesma-soil']
                    marker = 'x'

                x = df['sza']
                ax.scatter(x, l, marker=marker)
                r2 = r2_calculations(x, l)
                txtstr = '\n'.join((
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x))))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                ax.set_aspect(1. / ax.get_data_ratio())

        plt.savefig(os.path.join(self.fig_directory, 'sza_mae.png'), format="png", dpi=300, bbox_inches="tight")


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

def run_figures(base_directory):
    base_directory = base_directory
    sensor = 'emit'
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
    fig.plot_rmse(norm_option='brightness')
    fig.plot_rmse(norm_option='none')
    fig.local_slpit()
    fig.sza_plot()
