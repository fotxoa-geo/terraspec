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
from spectral.io import envi
from rasterio.plot import show
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, InsetPosition
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

    # def full_spectrum_plots(self):
    #     # spectral data
    #     create_directory(os.path.join(self.fig_directory, 'plot_stats'))
    #     transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-emit.csv'))
    #
    #     asd_wavelengths = np.array(transect_data.columns[9:]).astype(float)
    #     good_bands = spectra.get_good_bands_mask(asd_wavelengths, wavelength_pairs=None)
    #     asd_wavelengths[~good_bands] = np.nan
    #
    #     self.good_emit_bands = spectra.get_good_bands_mask(self.wvls, wavelength_pairs=None)
    #     self.wvls[~self.good_emit_bands] = np.nan
    #
    #     # gis shapefile
    #     df_gis = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
    #     df_gis = df_gis.sort_values('Name')
    #     df_gis['Team'] = df_gis['Name'].str.split('-').str[0].str.strip()
    #     df_gis = df_gis[df_gis['Team'] != 'THERM']
    #
    #     fig, axs = plt.subplot_mosaic([['rgb', 'frac'],
    #                                    ['photo', 'spec']], layout='constrained')
    #     plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    #
    #     rgb_tif = r'G:\My Drive\terraspec\slpit\figures\frac_map_rgb data\20230927_rgb_sr.tif'
    #     frac_tif = r'G:\My Drive\terraspec\slpit\figures\frac_map_rgb data\20230927_frac_sr.tif'
    #     shp_path = r'G:\My Drive\terraspec\slpit\figures\frac_map_rgb data\sr_boundary_gcs.shp'
    #
    #     for name, ax in axs.items():
    #         if name in ['rgb', 'frac']:
    #             # plot the spatial data
    #             if name == 'rgb':
    #                     raster_path = rgb_tif
    #                     plot_title = 'EMIT RGB'
    #             else:
    #                 raster_path = frac_tif
    #                 plot_title = 'EMIT Fractional Cover'
    #
    #             with rasterio.open(raster_path) as src:
    #                 raster_data = src.read([1,2,3])  # Read the first band
    #                 raster_transform = src.transform
    #                 raster_crs = src.crs
    #                 raster_meta = src.meta
    #                 nodata_value = src.nodata
    #
    #                 if nodata_value is None:
    #                     nodata_value = 0  # or choose another value that doesn't appear in your data
    #                     raster_meta.update(nodata=nodata_value)
    #
    #                 shapefile_data = gpd.read_file(shp_path)
    #                 shapes = [mapping(geom) for geom in shapefile_data.geometry]
    #                 mask = geometry_mask(shapes, transform=raster_transform, invert=True,
    #                                      out_shape=raster_data.shape[1:])
    #
    #                 rgba_data = np.zeros((4, raster_data.shape[1], raster_data.shape[2]), dtype=raster_data.dtype)
    #                 rgba_data[:3, :, :] = raster_data  # RGB channels
    #                 rgba_data[3, :, :] = 255  # Alpha channel set to 255 (fully opaque)
    #
    #                 # Apply the mask to the alpha channel
    #                 rgba_data[3, ~mask] = 0  # Set alpha to 0 (fully transparent) outside the shapefile
    #
    #                 ax.imshow(np.moveaxis(rgba_data, 0, -1), extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
    #
    #                 shapefile_data.plot(ax=ax, facecolor='none', edgecolor='red')
    #                 ax.set_title(plot_title)
    #                 ax.axis('off')
    #
    #                 # add scale bar to rgb and north arrow
    #                 if name == 'rgb':
    #                     # Adding scale bar
    #                     # Calculate scale bar length and midpoint in kilometers at the specified latitude
    #                     scalebar_length_km = 4  # 4 kilometers
    #                     midpoint_km = 2  # 2 kilometers
    #
    #                     # Conversion factor from degrees to kilometers at given latitude
    #                     km_per_degree = 111.32 * np.cos(np.deg2rad(latitude))
    #
    #                     scalebar_length_deg = scalebar_length_km / km_per_degree
    #                     midpoint_deg = midpoint_km / km_per_degree
    #
    #                     scalebar = ScaleBar(scalebar_length_deg, location='lower right', units='km', scale_loc='bottom',
    #                                         frameon=False, midpoint=midpoint_deg)  # Scale bar length in degrees
    #                     ax.add_artist(scalebar)
    #
    #                 ax_inset = inset_axes(ax, width="10%", height="10%", loc='upper right')
    #                     ax_inset.imshow([[1, 1], [0, 0]], cmap=plt.cm.gray, interpolation='nearest')
    #                     ax_inset.axis('off')
    #                     mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=0.5)
    #
    #                 else:
    #                     print('skip')
    #
    #             # plot the plot data
    #         elif name in ['photo']:
    #             ax.set_title('Landscape Picture')
    #             pic_path = os.path.join(self.output_directory, 'plot_pictures', 'spectral_transects', 'Spectral - 051.jpg')
    #             img = mpimg.imread(pic_path)
    #             ax.imshow(img)
    #
    #             ax.axis('off')
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.fig_directory, f'rgb-frac-map.png'), format="png", dpi=300)


        # for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):
        #
        #

        #
        #     # plot average of all EMIT files across time
        #     reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip',
        #                                                  '*' + plot.replace("Spectral", "SPEC").replace(" ",
        #                                                                                                 "") + '_RFL_' + '*[!.xml][!.csv][!.hdr]')))
        #
        #     if not reflectance_files:
        #         reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip',
        #                                                      '*' + plot.replace("Thermal", "THERM").replace(" ",
        #                                                                                                     "") + '_RFL_' + '*[!.xml][!.csv][!.hdr]')))
        #
        #     # plot average of SLPIT
        #     refl_file_asd = glob(os.path.join(self.output_directory, 'spectral_transects', 'transect',
        #                                       "*" + plot.replace(" ", "")))
        #
        #     df_refl_asd = gdal.Open(refl_file_asd[0], gdal.GA_ReadOnly)
        #     refl_array_asd = df_refl_asd.ReadAsArray().transpose((1, 2, 0))
        #     y = np.mean(refl_array_asd, axis=0).ravel()
        #
        #     # get date and time from slpit
        #     df_transect = transect_data.loc[transect_data['plot_name'] == plot].copy()
        #     slpit_date = df_transect['date'].unique()[0]
        #     slpit_datetime = datetime.strptime(df_transect['date'].unique()[0], "%Y-%m-%d")
        #
        #     # Convert the time strings to datetime objects
        #     df_transect['utc_time'] = pd.to_datetime(df_transect['utc_time'], format='%H:%M:%S')
        #     df_transect['total_seconds'] = df_transect['utc_time'].dt.hour * 3600 + df_transect['utc_time'].dt.minute * 60 + \
        #                                    df_transect['utc_time'].dt.second
        #
        #     # minimum and maximum total time in seconds for split
        #     min_time_seconds = df_transect['total_seconds'].min()
        #     max_time_seconds = df_transect['total_seconds'].max()
        #
        #     min_time_hhmmss = f"{min_time_seconds // 3600:02.0f}{(min_time_seconds % 3600) // 60:02.0f}{min_time_seconds % 60:02.0f}"
        #     max_time_hhmmss = f"{max_time_seconds // 3600:02.0f}{(max_time_seconds % 3600) // 60:02.0f}{max_time_seconds % 60:02.0f}"
        #
        #     # Calculate mean time of ASD  collections
        #     field_slpit_date_min = datetime.strptime(slpit_date + " " + min_time_hhmmss, "%Y-%m-%d %H%M%S").replace(
        #         tzinfo=timezone.utc)
        #     field_slpit_date_max = datetime.strptime(slpit_date + " " + max_time_hhmmss, "%Y-%m-%d %H%M%S").replace(
        #         tzinfo=timezone.utc)
        #
        #     ax.set_title(f'Field Sample Date: {slpit_date}')
        #     df_gis_select = df_gis.loc[df_gis['Name'] == plot.replace("Spectral", "SPEC")].copy().reset_index(drop=True)
        #     # field_emit_date = df_gis_select.at[0, 'EMIT Overp']
        #     # field_emit_date = datetime.strptime(field_emit_date, "%b %d, %Y at %I:%M:%S %p")
        #     # field_emit_date = field_emit_date.strftime("%Y%m%dT%H%M")
        #
        #     for _i, i in enumerate(reflectance_files):
        #         acquisition_date = os.path.basename(i).split("_")[2]
        #
        #         df_refl = gdal.Open(i, gdal.GA_ReadOnly)
        #         refl_array = df_refl.ReadAsArray().transpose((1, 2, 0))
        #         y_hat = np.mean(refl_array, axis=(0, 1))
        #
        #         acquisition_datetime_utc = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        #         geometry_results_emit = sunpos(acquisition_datetime_utc, np.mean(df_transect.latitude),
        #                                        np.mean(df_transect.longitude), np.mean(df_transect.elevation))
        #
        #         acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
        #         formatted_datetime = acquisition_datetime.strftime("%Y-%m-%d %I:%M %p")
        #         delta = slpit_datetime - acquisition_datetime
        #         days = np.absolute(delta.days)
        #
        #         cloud_mask = glob(os.path.join(self.gis_directory, 'emit-data-clip',
        #                                        f'*{plot.replace("Spectral", "SPEC").replace(" ", "")}_MASK_{acquisition_date}'))
        #
        #         if not cloud_mask:
        #             print('helloooooo')
        #             cloud_mask = glob(os.path.join(self.gis_directory, 'emit-data-clip',
        #                                            f'*{plot.replace("Thermal", "THERM").replace(" ", "")}_MASK_{acquisition_date}'))
        #             if not cloud_mask:
        #                 continue
        #
        #         ds_cloud = gdal.Open(cloud_mask[0], gdal.GA_ReadOnly)
        #         cloud_array = ds_cloud.ReadAsArray().transpose((1, 2, 0))
        #
        #         if days <= 15:
        #             cloud_check = np.any(cloud_array == 1)
        #
        #             if cloud_check:
        #                 pass
        #             else:
        #                 base_label = f'{acquisition_date} (±{days:02d} days)  SZA : {str(int(geometry_results_emit[1]))}°'
        #                 ax.plot(self.wvls, np.mean(refl_array, axis=(0, 1)), label=base_label, linewidth=1)
        #
        #                 # if field_emit_date == acquisition_date[:-2]:
        #                 #     bold_label = base_label
        #
        #     geometry_results_slpit_min = sunpos(field_slpit_date_min, np.mean(df_transect.latitude),
        #                                         np.mean(df_transect.longitude), np.mean(df_transect.elevation))
        #
        #     geometry_results_slpit_max = sunpos(field_slpit_date_max, np.mean(df_transect.latitude),
        #                                         np.mean(df_transect.longitude), np.mean(df_transect.elevation))
        #
        #     slpit_label = f'SLPIT (SZA: {str(int(geometry_results_slpit_min[1]))} - {str(int(geometry_results_slpit_max[1]))}°)'
        #     ax.plot(self.wvls, np.mean(refl_array_asd, axis=0).ravel(), color='black', label=slpit_label,
        #             linewidth=1.5)
        #
        #     ax.set_xlabel('Wavelength (nm)')
        #     ax.set_ylabel('Reflectance (%)')
        #     ax.set_ylim(0, 1)
        #     ax.set_xlim(320, 2550)
        #
        #     legend = ax.legend()
        #
        #     # make emit field date bold
        #     for handle, text in zip(legend.legendHandles, legend.get_texts()):
        #         if text.get_text() == bold_label:
        #             text.set_fontweight('bold')
        #
        #     plt.savefig(os.path.join(self.fig_directory, 'plot_stats', f'full-spectrum{plot}.png'), format="png", dpi=300,
        #                 bbox_inches="tight")
        #     plt.clf()
        #     plt.close()


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
        # df_gis['Team'] = df_gis['Name'].str.split('-').str[0].str.strip()
        # df_gis = df_gis[df_gis['Team'] != 'THERM']

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

                    df_refl_asd = gdal.Open(refl_file_asd[0], gdal.GA_ReadOnly)
                    refl_array_asd = df_refl_asd.ReadAsArray().transpose((1, 2, 0))
                    refl_array_asd[refl_array_asd == -9999] = np.nan
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
                            if not cloud_mask:
                                continue

                        ds_cloud = gdal.Open(cloud_mask[0], gdal.GA_ReadOnly)
                        cloud_array = ds_cloud.ReadAsArray().transpose((1, 2, 0))

                        if days <= 30:
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

                    ax.plot(self.wvls, np.nanmean(refl_array_asd, axis=(0, 1)), color='black', label=slpit_label,
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


    def map_detail_figure(self):
        rgb_img_path = os.path.join(self.gis_directory, 'rgb-quick-look', 'emit_20230927T214531.tif')

        rgb_img = envi_to_array()
        frac_img = envi_to_array()


    def av3_comparisson(self):
        # spectral data
        create_directory(os.path.join(self.fig_directory, 'plot_stats'))
        sr_plots = ['SPEC - 045', 'SPEC - 046', 'SPEC - 047', 'SPEC - 048', 'SPEC - 049', 'SPEC - 050', 'SPEC - 051',
                    'SPEC - 052', 'SPEC - 053', 'SPEC - 054', 'SPEC - 055']

        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-emit.csv'))
        transect_data['plot_name'] = transect_data['plot_name'].str.replace('Spectral', 'SPEC')
        transect_data = transect_data[transect_data['plot_name'].isin(sr_plots)]

        asd_wavelengths = np.array(transect_data.columns[9:]).astype(float)
        good_bands = spectra.get_good_bands_mask(asd_wavelengths, wavelength_pairs=None)
        asd_wavelengths[~good_bands] = np.nan

        self.good_emit_bands = spectra.get_good_bands_mask(self.wvls, wavelength_pairs=None)
        self.wvls[~self.good_emit_bands] = np.nan

        for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):
            fig, axs = plt.subplot_mosaic([['plot_pic', 'map'],
                                           ['av1', 'av2']], figsize=(10, 10))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            fig.suptitle(plot)
            df_transect = transect_data.loc[transect_data['plot_name'] == plot].copy()
            df_transect = df_transect[df_transect.longitude != 'unk']

            for name, ax in axs.items():
                if name in ['plot_pic']:
                    ax.set_title('Landscape Picture')
                    pic_path = os.path.join(self.output_directory, 'plot_pictures', 'spectral_transects', f'{plot.replace("SPEC","Spectral")}.jpg')
                    img = mpimg.imread(pic_path)
                    ax.imshow(img)
                    ax.axis('off')

                if name in ['map']:
                    ax.set_title('Plot Map')
                    # update to SR Boundaries
                    m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=45,
                                llcrnrlon=-125, urcrnrlon=-100, ax=ax, epsg=4326)
                    m.arcgisimage(service='World_Imagery', xpixels=1000, ypixels=1000, dpi=300, verbose=True)
                    ax.scatter(np.mean(df_transect.longitude), np.mean(df_transect.latitude), color='red', s=12)

                if name in ['av1', 'av2']:

                    if name == 'av1':
                        av_flightline = 'AV320230926t200400'
                    else:
                        av_flightline = 'AV320231026t191939'

                    ax.set_title(f'AVIRIS_$3$ Flightline: {av_flightline}')

                    # get all emit rfl files
                    emit_reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip', f'*{plot.replace(" ","")}_RFL_*[!.xml][!.csv][!.hdr]')))

                    # plot average of av3
                    av3_file = os.path.join(self.gis_directory, 'av3_sr', f'{plot}_RFL_{av_flightline}')
                    av3_array = envi_to_array(av3_file)
                    av3_array[av3_array == 0] = np.nan
                    av3_array[av3_array == -0.01] = np.nan
                    flight_datetime = datetime.strptime(av_flightline, "AV3%Y%m%dt%H%M%S")

                    av3_label = f'AVIRIS_3: {av_flightline[3:]}'

                    av3_meta = envi.read_envi_header(f'{av3_file}.hdr')
                    av3_wvls = np.array(av3_meta['wavelength']).astype(float)
                    ax.plot(av3_wvls, np.mean(av3_array, axis=(0, 1)).ravel(), color='blue', label=av3_label,
                            linewidth=1, linestyle=':')

                    # plot average of SLPIT
                    refl_file_asd = os.path.join(self.output_directory, 'spectral_transects', 'transect',
                                                 f'{plot.replace("SPEC", "Spectral").replace(" ", "")}')
                    slpit_array = envi_to_array(refl_file_asd)

                    slpit_field_date = sorted(df_transect.date.unique())[0]
                    slpit_datetime = datetime.strptime(slpit_field_date, "%Y-%m-%d")
                    delta = slpit_datetime - flight_datetime
                    days = delta.days
                    slpit_label = f'SLPIT Field Date: {slpit_field_date}  ({days:02d} days)'

                    ax.plot(self.wvls, np.mean(slpit_array, axis=0).ravel(), color='black', label=slpit_label,
                            linewidth=1.5, linestyle='-')

                    for _i, i in enumerate(emit_reflectance_files):
                        acquisition_date = os.path.basename(i).split("_")[2]
                        refl_array = envi_to_array(i)

                        acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%dT%H%M%S")
                        delta = acquisition_datetime - flight_datetime
                        days = delta.days

                        cloud_mask = glob(os.path.join(self.gis_directory, 'emit-data-clip', f'*{plot.replace(" ", "")}_MASK_{acquisition_date}'))
                        cloud_array = envi_to_array(cloud_mask[0])

                        if -15 <= days <= 15:
                            cloud_check = np.any(cloud_array == 1)

                            if cloud_check:
                                pass
                            else:
                                base_label = f'EMIT: {acquisition_date} ({days:02d} days)'
                                ax.plot(self.wvls, np.mean(refl_array, axis=(0, 1)), label=base_label, linewidth=1, linestyle='--')

                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('Reflectance (%)')
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)

                    ax.legend(prop={'size': 6})

            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', f'av3_{plot}.png'), format="png", dpi=300)
            plt.clf()
            plt.close()




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

    #fig.full_spectrum_plots()
    #fig.plot_summary()
    #fig.plot_rmse(norm_option='brightness')
    #fig.plot_rmse(norm_option='none')
    #fig.local_slpit()
    fig.sza_plot(norm_option='brightness')
    #fig.sza_plot(norm_option='none')
    #fig.map_detail_figure()
    #fig.av3_comparisson()
