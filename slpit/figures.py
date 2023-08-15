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
def fraction_file_info(fraction_file):
    name = os.path.basename(fraction_file)
    mode = os.path.basename(os.path.dirname(fraction_file))
    plot = name.split("_")[1]
    library_mode = name.split("_")[0].split('-')[1]

    ds = gdal.Open(fraction_file, gdal.GA_ReadOnly)
    array = ds.ReadAsArray().transpose((1, 2, 0))

    mean_fractions = []
    for _band, band in enumerate(range(0, array.shape[2])):
            mean_fractions.append(np.mean(array[:, :, _band]))

    return [name.split("_")[0].split("-")[0], mode, plot, library_mode] + mean_fractions

class figures:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory,  "figures")
        self.gis_directory = os.path.join(base_directory, "gis")

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='emit')
        create_directory(os.path.join(base_directory, "figures"))

        # ems
        self.ems = ['npv', 'pv', 'soil']

        # load teatracorder directory
        terraspec_base = os.path.join(base_directory, "..")
        self.tetracorder_output_directory = os.path.join(terraspec_base, 'tetracorder', 'output', 'spectral_abundance')

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
        df_gis = pd.DataFrame(gp.read_file(os.path.join(self.gis_directory, "Observation.shp")))
        df_gis = df_gis.sort_values('Name')

        # fractional cover
        for plot in sorted(list(transect_data.plot_name.unique())):
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
                    reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip', '*'+ plot.replace("Spectral", "SPEC").replace(" ", "") + '_RFL_' + '*[!.xml][!.csv][!.hdr]')))

                    # plot average of SLPIT
                    refl_file_asd = glob(os.path.join(self.output_directory, 'spectral_transects', 'transect',
                                                      "*" + plot.replace(" ", "")))

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
                    field_emit_date = df_gis_select.at[0, 'EMIT Overp']
                    field_emit_date = datetime.strptime(field_emit_date, "%b %d, %Y at %I:%M:%S %p")
                    field_emit_date = field_emit_date.strftime("%Y%m%dT%H%M")

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

                        if days <= 40:
                            base_label = f'{formatted_datetime} (±{days:02d} days)  SZA : {str(int(geometry_results_emit[1]))}°'
                            ax.plot(self.wvls,  np.mean(refl_array, axis=(0, 1)), label=base_label, linewidth=0.75)

                            if field_emit_date == acquisition_date[:-2]:
                                bold_label = base_label

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

                    # make emit field date bold
                    for handle, text in zip(legend.legendHandles, legend.get_texts()):
                        if text.get_text() == bold_label:
                            text.set_fontweight('bold')

                if ax == ax4:
                    df_spectra = em_data[(em_data['plot_name'] == plot) & ((em_data['level_1'] == 'NPV'))].copy()
                    em_spectra = df_spectra.iloc[:, 10:].to_numpy()
                    for _row, row in enumerate(em_spectra):
                        ax.plot(self.wvls, row, color='red')
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(2100, 0.85, f"NPV (n = {str(em_spectra.shape[0])})", fontsize=12)

                if ax == ax5:
                    df_spectra = em_data[(em_data['plot_name'] == plot) & ((em_data['level_1'] == 'PV'))].copy()
                    em_spectra = df_spectra.iloc[:, 10:].to_numpy()
                    for _row, row in enumerate(em_spectra):
                        ax.plot(self.wvls, row, color='green')
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(2100, 0.85, f"PV (n = {str(em_spectra.shape[0])})", fontsize=12)

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
                    # slpit_ems_abundance = glob(os.path.join(self.tetracorder_output_directory, '*' + plot.replace(" ", "") +
                    #                  '*emit_ems_augmented_sa_mineral'))
                    #
                    # slpit_transect_abundance = glob(os.path.join(self.tetracorder_output_directory, '*' + plot.replace(" ", "") +
                    #                  '*transect_augmented_sa_mineral'))
                    #
                    # emit_spectral_abundance = glob(os.path.join(self.tetracorder_output_directory, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
                    #                  '*pixels_augmented_sa_mineral'))
                    #
                    # # load df for em position key
                    # em_csv = os.path.join(self.output_directory, 'spectral_transects', 'endmembers', plot.replace(" ", "") + '-emit.csv')
                    # df_em = pd.read_csv(em_csv)
                    # first_soil_index = df_em.index[df_em['level_1'] == 'Soil'].min()
                    #
                    # # load data
                    # split_abundance_array = envi_to_array(slpit_ems_abundance[0])[first_soil_index: ,: ,:]
                    # split_abundance_array[split_abundance_array == 0] = np.nan
                    # emit_abundance_array = envi_to_array(emit_spectral_abundance[0])
                    # emit_abundance_array[emit_abundance_array == 0] = np.nan
                    # split_transect_array = envi_to_array(slpit_transect_abundance[0])
                    # split_transect_array[split_transect_array == 0] = np.nan
                    #
                    # mineral_bands = load_band_names(slpit_ems_abundance[0])
                    # mineral_bands = [item.replace('+', '\n') for item in mineral_bands]
                    #
                    ax.set_ylabel('Spectral Abundance')
                    ax.set_ylim(0, .25)
                    #
                    # for _mineral, mineral in enumerate(mineral_bands):
                    #     avg_slpit_em = np.nanmean(split_abundance_array[:, 0, _mineral])
                    #     avg_split_transect = np.nanmean(split_transect_array[:, 0, _mineral])
                    #     avg_emit = np.nanmean(emit_abundance_array[:, 0:3, _mineral])
                    #
                    #     ax.bar(_mineral, avg_slpit_em, color='green', label='Contact Probe', edgecolor="black", width=0.2)
                    #     ax.bar(_mineral - 0.1, avg_split_transect, color='black', label='Fiber Optic', edgecolor="black",
                    #            width=0.2)
                    #     ax.bar(_mineral + 0.1, avg_emit, color='blue', label='EMIT', edgecolor="black", width=0.05)
                    #
                    # # Get handles and labels from the axes
                    # handles, labels = ax.get_legend_handles_labels()
                    #
                    # # Create a dictionary to keep track of unique labels
                    # unique_labels = {}
                    # unique_handles = []
                    #
                    # # Iterate through the labels and handles and add them to the unique_labels dictionary
                    # for i, label in enumerate(labels):
                    #     if label not in unique_labels:
                    #         unique_labels[label] = handles[i]
                    #         unique_handles.append(handles[i])
                    #
                    # ax.legend(unique_handles, unique_labels.keys())
                    # ax.set_xticks(np.arange(0, len(mineral_bands), step=1), minor=False)
                    # ax.set_xticklabels(mineral_bands, fontdict=None, minor=False)
                    # ax.tick_params(axis='x', labelsize=8)

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

    def plot_rmse(self):
        # load all fraction files
        fraction_files_sma = sorted(glob(os.path.join(self.output_directory, 'sma-best', '*fractional_cover')))
        fraction_files_mesma = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover')))
        all_files = fraction_files_sma + fraction_files_mesma

        results = p_map(fraction_file_info, all_files, **{"desc": "\t\t retrieving mean fractional cover: ...", "ncols": 150})
        df_all = pd.DataFrame(results)
        df_all.columns = ['name', 'mode', 'plot', 'lib_mode', 'npv', 'pv', 'soil', 'shade']

        # load all uncertainty files
        uncer_files_sma = sorted(glob(os.path.join(self.output_directory, 'sma-best', '*fractional_cover_uncertainty')))
        uncer_files_mesma = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover_uncertainty')))
        all_uncer_files = uncer_files_sma + uncer_files_mesma

        results_uncer = p_map(fraction_file_info, all_uncer_files, **{"desc": "\t\t retrieving mean uncertainty: ...", "ncols": 150})
        df_all_uncer = pd.DataFrame(results_uncer)
        df_all_uncer.columns = ['name', 'mode', 'plot', 'lib_mode', 'npv', 'pv', 'soil', 'shade']

        # # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select = df_all[(df_all['mode'] == 'sma-best') & (df_all['lib_mode'] == 'local')].copy()
                df_uncer = df_all_uncer[(df_all_uncer['mode'] == 'sma-best') & (df_all_uncer['lib_mode'] == 'local')].copy()
            if row == 1:
                df_select = df_all[(df_all['mode'] == 'sma-best')].copy()
                df_uncer = df_all_uncer[(df_all_uncer['mode'] == 'sma-best')].copy()
            if row == 2:
                df_select = df_all[(df_all['mode'] == 'mesma') & (df_all['lib_mode'] == 'local')].copy()
                df_uncer = df_all_uncer[(df_all_uncer['mode'] == 'mesma') & (df_all_uncer['lib_mode'] == 'local')].copy()
            if row == 3:
                df_select = df_all[(df_all['mode'] == 'mesma')].copy()
                df_uncer = df_all_uncer[(df_all_uncer['mode'] == 'mesma')].copy()

            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_xlabel('SLPIT')
                ax.set_ylabel("EMIT Fractions")
                mode = list(df_select['mode'].unique())[0]
                lib_mode = list(df_select['lib_mode'].unique())[0]

                ax.set_title(f'{self.ems[col]} - {mode} - {lib_mode}')
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)

                # plot 1 to 1 line
                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red')

                if row == 0 or row == 2:
                    df_x = df_select[(df_select['name'] == 'asd')].copy().reset_index(drop=True)
                    df_y = df_select[(df_select['name'] == 'emit')].copy().reset_index(drop=True)
                    df_x_u = df_uncer[(df_uncer['name'] == 'asd')].copy().reset_index(drop=True)
                    df_y_u = df_uncer[(df_uncer['name'] == 'emit')].copy().reset_index(drop=True)
                else:
                    df_x = df_select[(df_select['name'] == 'asd') & (df_select['lib_mode'] == 'local')].copy().reset_index(drop=True)
                    df_y = df_select[(df_select['name'] == 'emit') & (df_select['lib_mode'] == 'global')].copy().reset_index(drop=True)
                    df_x_u = df_uncer[(df_uncer['name'] == 'asd') & (df_uncer['lib_mode'] == 'local')].copy().reset_index(drop=True)
                    df_y_u = df_uncer[(df_uncer['name'] == 'emit') & (df_uncer['lib_mode'] == 'global')].copy().reset_index(drop=True)

                # plot fractional cover values
                if col == 0:
                    x = df_x['npv']
                    y = df_y['npv']
                    x_u = df_x_u['npv']
                    y_u = df_y_u['npv']

                elif col == 1:
                    x = df_x['pv']
                    y = df_y['pv']
                    x_u = df_x_u['pv']
                    y_u = df_y_u['pv']
                else:
                    x = df_x['soil']
                    y = df_y['soil']
                    x_u = df_x_u['soil']
                    y_u = df_y_u['soil']

                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='o')

                # # Add labels to each point
                # for xi, yi, label in zip(x, y, df_x['plot']):
                #     plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_score(x, y)

                txtstr = '\n'.join((
                    r'RMSE: %.2f' % (rmse,),
                    r'MAE: %.2f' % (mae,),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x))))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.fig_directory, 'regression.png'), format="png", dpi=300, bbox_inches="tight")

def run_figures(base_directory):
    fig = figures(base_directory=base_directory)
    fig.plot_summary()
    #fig.plot_rmse()