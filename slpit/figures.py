import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from osgeo import gdal
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.basemap import Basemap
from p_tqdm import p_map
import matplotlib.image as mpimg
from pypdf import PdfMerger
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from datetime import datetime

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
                    reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data-clip', '*'+ plot.replace("Spectral", "SPEC").replace(" ", "") + '*[!.xml][!.csv][!.hdr]')))

                    # plot average of SLPIT
                    refl_file_asd = glob(os.path.join(self.output_directory, 'spectral_transects', 'transect',
                                                      plot.replace(" ", "") + '-reflectance**[!.csv][!.hdr]'))
                    df_refl_asd = gdal.Open(refl_file_asd[0], gdal.GA_ReadOnly)
                    refl_array_asd = df_refl_asd.ReadAsArray().transpose((1, 2, 0))
                    y = np.mean(refl_array_asd, axis=0).ravel()

                    # get date from slpit
                    df_transect = transect_data.loc[transect_data['plot_name'] == plot].copy()
                    slpit_date = df_transect['date'].unique()[0]
                    slpit_datetime = datetime.strptime(df_transect['date'].unique()[0], "%Y-%m-%d")

                    ax.set_title(f'Field Sample Date: {slpit_date}')
                    for i in reflectance_files:
                        acquisition_date = os.path.basename(i).split("_")[2][:8]

                        df_refl = gdal.Open(i, gdal.GA_ReadOnly)
                        refl_array = df_refl.ReadAsArray().transpose((1, 2, 0))
                        y_hat = np.mean(refl_array, axis=(0, 1))
                        mae = mean_absolute_error(y[self.good_emit_bands], y_hat[self.good_emit_bands])
                        if mae <= 0.05:
                            acquisition_datetime = datetime.strptime(acquisition_date, "%Y%m%d")
                            delta = slpit_datetime - acquisition_datetime
                            days = np.absolute(delta.days)
                            if days <= 50:
                                ax.plot(self.wvls, np.mean(refl_array, axis=(0, 1)), label=f'{acquisition_date} (± {days})  MAE: {mae:.2f}', linewidth=0.75)
                        else:
                            pass

                    ax.plot(self.wvls, np.mean(refl_array_asd, axis=0).ravel(), color='black', label=f'ASD',
                            linewidth=1.5)

                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('Reflectance (%)')
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.legend()

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
                    pass
                    # fraction_files = glob(
                    #     os.path.join(self.output_directory, 'sma-best', '*asd-' + plot.replace(" ", "") +
                    #                  '*fractional_cover'))
                    # ds_fractions = gdal.Open(fraction_files[0], gdal.GA_ReadOnly)
                    # estimated_array = ds_fractions.ReadAsArray().transpose((1, 2, 0))
                    # df_spectra = em_data[(em_data['plot_name'] == plot)].copy()
                    # plot_ems = list(sorted(df_spectra.level_1.unique()))
                    #
                    # # add emit fraction estimates
                    # emit_file = sorted(glob(os.path.join(self.output_directory, 'sma-best',
                    #                                      '*emit-' + plot.replace(" ", "") + '*fractional_cover')))
                    # ds_emit = gdal.Open(emit_file[0], gdal.GA_ReadOnly)
                    # emit_array = ds_emit.ReadAsArray().transpose((1, 2, 0))
                    #
                    # for _em, em in enumerate(plot_ems):
                    #     avg_em = np.average(estimated_array[:, :, _em])
                    #     avg_emit = np.average(emit_array[:, :, _em])
                    #
                    #     ax.set_ylabel('Fractional\nCover')
                    #     ax.set_ylim(0, 1)
                    #
                    #     if em == 'PV':
                    #         color = 'g'
                    #         ax.bar(1, avg_em, color='green', label='SLPIT', edgecolor="black", width=0.2)
                    #         ax.bar(1 + 0.05, avg_emit, color='blue', label='EMIT', edgecolor="black", width=0.2)
                    #     elif em == 'NPV':
                    #         color = 'r'
                    #         ax.bar(0, avg_em, color='green', edgecolor="black", width=0.2)
                    #         ax.bar(0 + 0.05, avg_emit, color='blue', edgecolor="black", width=0.2)
                    #     elif em == 'Soil':
                    #         color = 'b'
                    #         ax.bar(2, avg_em, color='green', edgecolor="black", width=0.2)
                    #         ax.bar(2 + 0.05, avg_emit, color='blue', edgecolor="black", width=0.2)
                    #
                    #     ax.legend()
                    #     ax.set_xticks(np.arange(0, len(self.ems), step=1), minor=False)
                    #     ax.set_xticklabels(self.ems, fontdict=None, minor=False)

            plt.savefig(os.path.join(self.fig_directory, 'plot_stats', plot + '.pdf'), format="pdf", dpi=300,
                        bbox_inches="tight")
            plt.clf()
            plt.close()
            merger.append(os.path.join(self.fig_directory, 'plot_stats', plot + '.pdf'))

        # write pdf
        merger.write(os.path.join(self.fig_directory, 'plot_stats', 'plot_summary.pdf'))
        merger.close()

    def plot_rmse(self):
        # load fraction files
        fraction_files = sorted(glob(os.path.join(self.output_directory, 'sma-best', '*asd-' + '*fractional_cover')))
        # arrays for data points
        npv_emit = []
        npv_slpit = []
        pv_emit = []
        pv_slpit = []
        soil_emit = []
        soil_slpit = []

        # loop through plot files
        for file in fraction_files:
            plot_num = os.path.basename(file).split("_")[0].split("-")[2]
            em_file = os.path.join(self.output_directory, 'spectral_transects', 'endmembers',
                                   'Spectral-' + plot_num + "-emit.csv")
            df_em = pd.read_csv(em_file)
            local_ems = sorted(list(df_em['level_1'].unique()))
            emit_file = sorted(glob(
                os.path.join(self.output_directory, 'sma-best', '*emit-Spectral-' + plot_num + '*fractional_cover')))

            # read ground data
            slpit_ds = gdal.Open(file, gdal.GA_ReadOnly)
            slpit_array = slpit_ds.ReadAsArray().transpose((1, 2, 0))

            # read emit data
            emit_ds = gdal.Open(emit_file[0], gdal.GA_ReadOnly)
            emit_array = emit_ds.ReadAsArray().transpose((1, 2, 0))

            # seperate data means by class
            for _em, em in enumerate(local_ems):
                if em == 'NPV':
                    npv_emit.append(np.mean(emit_array[1, 1, _em]))
                    npv_slpit.append(np.mean(slpit_array[:, :, _em]))
                elif em == 'PV':
                    pv_emit.append(np.mean(emit_array[1, 1, _em]))
                    pv_slpit.append(np.mean(slpit_array[:, :, _em]))
                else:
                    soil_emit.append(np.mean(emit_array[1, 1, _em]))
                    soil_slpit.append(np.mean(slpit_array[:, :, _em]))

        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        ncols = 3
        nrows = 1
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_xlabel('SPLIT')
                ax.set_ylabel("EMIT Fractions")

                # plot npv
                if col == 0:
                    ax.scatter(npv_slpit, npv_emit)
                    ax.set_title("NPV")
                    x = npv_slpit
                    y = npv_emit

                elif col == 1:
                    ax.scatter(pv_slpit, pv_emit)
                    ax.set_title("PV")
                    x = pv_slpit
                    y = pv_emit

                else:
                    ax.scatter(soil_slpit, soil_emit)
                    ax.set_title("Soil")
                    x = soil_slpit
                    y = soil_emit

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_score(x, y)
                m, b = np.polyfit(x, y, 1)

                # plot 1 to 1 line
                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red')

                txtstr = '\n'.join((
                    r'RMSE: %.2f' % (rmse,),
                    r'MAE: %.2f' % (mae,),
                    r'R$^2$: %.2f' % (r2,),
                    r'y = %.2fx + %.2f' % (m, b),
                    r'n = ' + str(len(x))))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                ax.set_aspect(1 / ax.get_data_ratio())

        plt.savefig(os.path.join(self.fig_directory, 'regression.png'), format="png", dpi=300, bbox_inches="tight")

def run_figures(base_directory):
    fig = figures(base_directory=base_directory)
    fig.plot_summary()