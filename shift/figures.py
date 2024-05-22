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
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.spectra_utils import spectra
from utils.create_tree import create_directory
from pypdf import PdfMerger
from utils.envi import envi_to_array
from datetime import datetime
import geopandas as gpd
from utils.results_utils import r2_calculations, load_data
from matplotlib.ticker import FormatStrFormatter

# quadrat groupings
quad_phenophase_key = {'early leaf out': 'pv',
                       'early senescence': 'npv',
                       'flowers': 'pv',
                       'full leaf out': 'pv',
                       'full senescence': 'npv',
                       'last year senescence': 'npv',
                       'seeds': 'npv',
                       'yellow flower': 'pv'}


class figures:
    def __init__(self, base_directory: str, sensor: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs):

        self.base_directory = base_directory
        self.figure_directory = os.path.join(base_directory, 'figures')
        self.output_directory = os.path.join(base_directory, 'output')
        create_directory(self.figure_directory)

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='aviris_ng')
        self.exclude = ['.hdr', '.csv', '.ini', '.xml']

        # ems
        self.ems = ['NPV', 'GV', 'Soil']

        self.ems_short = ['npv', 'GV', 'soil']

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


    def plot_summary(self):
        # spectral data
        create_directory(os.path.join(self.figure_directory, 'plot_stats'))
        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-asd.csv'))
        em_data = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-aviris-ng.csv'))

        # load asd wavelengths
        asd_wavelengths = spectra.load_asd_wavelenghts()
        good_bands = spectra.get_good_bands_mask(asd_wavelengths, wavelength_pairs=None)
        asd_wavelengths[~good_bands] = np.nan

        # load instrument wavelengths
        ins_wvls = self.wvls
        good_ins_band = spectra.get_good_bands_mask(ins_wvls, wavelength_pairs=None)
        ins_wvls[~good_ins_band] = np.nan

        # plot summary - merged
        merger = PdfMerger()

        # fractional cover
        for plot in sorted(list(transect_data.plot_name.unique())):

            plot = plot.upper()
            fig = plt.figure(figsize=(14, 8))

            fig.suptitle(f"SHIFT: {plot}")
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

            site = plot.split("-")[0] + '-' + plot.split("-")[1]
            season = plot.split("-")[-1]
            df_coords = pd.read_csv('plot_coordinates.csv')
            df_coords['Date'] = pd.to_datetime(df_coords['Date'])
            date = df_coords.loc[(df_coords['Plot Name'] == site) & (df_coords['Season'] == season.upper()), 'Date'].iloc[0]
            date = datetime.strptime(str(date).split(' ')[0], "%Y-%m-%d")

            em_data = pd.read_csv(os.path.join(self.output_directory, 'spectral_transects', 'endmembers',
                                               f'{site}_{season.upper()}_original_{self.instrument}.csv'))
            for ax in axes:
                if ax == ax1:
                    ax.set_title('Plot Map')
                    long = df_coords.loc[(df_coords['Plot Name'] == site) & (df_coords['Season'] == season.upper()), 'longitude'].iloc[0]
                    lat = df_coords.loc[(df_coords['Plot Name'] == site) & (df_coords['Season'] == season.upper()), 'latitude'].iloc[0]

                    if site[:2] == 'DP':
                        lleft_lat = 34.4086695
                        uright_lat = 34.7477056
                        lleft_lon = -120.6851337
                        uright_lon = -120.0943748
                    else:
                        lleft_lat = 34.5747161
                        uright_lat = 34.7916992
                        lleft_lon = -120.2198138
                        uright_lon = -119.8417282

                    m = Basemap(projection='merc', llcrnrlat=lleft_lat, urcrnrlat=uright_lat,
                                llcrnrlon=lleft_lon, urcrnrlon=uright_lon, ax=ax1, epsg=4326)
                    m.arcgisimage(service='World_Imagery', xpixels=1000, ypixels=1000, dpi=300, verbose=True)
                    ax.scatter(float(long), float(lat), color='red', s=12)

                if ax == ax2:

                    flip = ['SRA-019-SPRING', 'DPA-004-FALL', 'DPB-005-FALL', 'SRB-010-FALL', 'SRB-045-FALL', 'SRB-050-FALL',
                            'SRB-004-FALL', 'SRA-056-FALL', 'SRA-007-FALL', 'SRA-008-FALL']
                    flip_90 = ['SRA-034-SPRING', 'SRB-026-SPRING', 'SRB-046-FALL', 'SRB-010-FALL']

                    ax.set_title('Landscape\nPicture')
                    pics = glob(os.path.join(self.base_directory, 'field', 'pictures', f'*{site}*'))

                    if not pics:
                        ax.text(0.5, 0.5, 'No Plot Picture Available', transform=ax.transAxes, fontsize=8,
                                verticalalignment='top')
                    else:
                        img = mpimg.imread(pics[0])
                        if plot in flip:
                            ax.imshow(img, origin='lower')
                        elif plot in flip_90:
                            flipped_img = np.flip(np.transpose(img, axes=(1, 0, 2)), axis=0)
                            ax.imshow(flipped_img, origin='lower')
                        else:
                            ax.imshow(img)
                    ax.axis('off')

                if ax == ax7:
                    ax.set_title('Mineralogy')

                if ax == ax4:
                    df_spectra = em_data[em_data['level_1'] == 'npv'].copy()
                    specific_string = 'UNK-'
                    df_spectra['species'] = df_spectra['species'].fillna(specific_string)
                    df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
                    num_species = len(sorted(list(df_spectra.species.unique())))
                    colors = np.random.rand(num_species, 3)

                    for _species, species in enumerate(sorted(list(df_spectra.species.unique()))):
                        df_species = df_spectra[df_spectra['species'] == species].copy()
                        em_spectra = df_species.iloc[:, 10:].to_numpy()

                        for _row, row in enumerate(em_spectra):
                            ax.plot(ins_wvls, row, color=colors[_species])

                        common_name = df_species_key.loc[df_species_key['key_value'] == species, ['label']].values[0][0]
                        ax.plot(ins_wvls, row, c=colors[_species], label=common_name)

                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"NPV (n = {str(df_spectra.shape[0])})", fontsize=12)
                    ax.legend(prop={'size': 6})

                if ax == ax5:
                    df_spectra = em_data[em_data['level_1'] == 'pv'].copy()
                    specific_string = 'UNK-'
                    df_spectra['species'] = df_spectra['species'].fillna(specific_string)
                    df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
                    num_species = len(sorted(list(df_spectra.species.unique())))
                    colors = np.random.rand(num_species, 3)

                    for _species, species in enumerate(sorted(list(df_spectra.species.unique()))):
                        df_species = df_spectra[df_spectra['species'] == species].copy()
                        em_spectra = df_species.iloc[:, 10:].to_numpy()

                        for _row, row in enumerate(em_spectra):
                            ax.plot(ins_wvls, row, color=colors[_species])

                        common_name = df_species_key.loc[df_species_key['key_value'] == species, ['label']].values[0][0]
                        ax.plot(ins_wvls, row, c=colors[_species], label=common_name)

                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"PV (n = {str(df_spectra.shape[0])})", fontsize=12)
                    ax.legend(prop={'size': 6})

                if ax == ax6:
                    df_spectra = em_data[em_data['level_1'] == 'soil'].copy()
                    em_spectra = df_spectra.iloc[:, 10:].to_numpy()
                    for _row, row in enumerate(em_spectra):
                        ax.plot(ins_wvls, row, color='blue')
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"Soil (n = {str(df_spectra.shape[0])})", fontsize=12)

                if ax == ax3:
                    # plot average spectra
                    df_spectra = transect_data[transect_data['plot_name'] == site + '-' + season.upper()].copy()
                    transect_spectra = df_spectra.iloc[:, 9:].to_numpy()

                    refl_data = sorted(glob(os.path.join(self.base_directory, 'gis', 'shift-data-clip', f"*{plot}_*")))
                    exclude = ['.hdr', '.aux', '.xml']

                    for i in refl_data:
                        if os.path.splitext(i)[1] in exclude:
                            pass
                        else:
                            img_date = datetime.strptime(i.split("_")[-1], "ang%Y%m%dt%H%M%S")
                            refl = envi_to_array(i)
                            refl = np.mean(refl, axis=(0, 1))
                            img_diff = img_date - date
                            img_diff = img_diff.days

                            if np.absolute(img_diff) > 10:
                                pass
                            else:
                                ax.plot(ins_wvls, refl, label=f"{img_date}: {img_diff}")

                    # plot average of SLPIT
                    ax.set_title(f'Field Sample Date: {date}')
                    ax.plot(asd_wavelengths, np.mean(transect_spectra, axis=0), color='black', label=f"ASD Mean Refl")
                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.legend(fontsize='8')

            plt.savefig(os.path.join(self.figure_directory, 'plot_stats', plot + '.pdf'), format="pdf", dpi=300)
            plt.clf()
            plt.close()
            merger.append(os.path.join(self.figure_directory, 'plot_stats', plot + '.pdf'))

        # write pdf
        merger.write(os.path.join(self.figure_directory, 'plot_stats', 'plot_summary.pdf'))
        merger.close()

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

        # load fraction plots
        df_emit = pd.read_csv(os.path.join(self.slpit_figures, 'fraction_output.csv'))
        df_emit['Team'] = df_emit['plot'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']
        df_emit['campaign'] = 'emit'

        skip = ['SRA-000-SPRING', 'SRB-047-SPRING', 'SRB-004-FALL', 'SRB-050-FALL', 'SRB-200-FALL']
        df_aviris = pd.read_csv(os.path.join(self.figure_directory, 'shift_fraction_output.csv'))
        df_aviris = df_aviris[~df_aviris['plot'].isin(skip)]
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        #  create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.05, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        # # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()

            if row == 1:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_mc'] == 25) & (df_all['num_cmb_em'] == 20)].copy()

            if row == 2:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['num_mc'] == 25) & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_cmb_em'] == 100)].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_cmb_em'] == 100)].copy()

            if row == 3:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['campaign'] == 'emit') & (df_all['normalization'] == norm_option) & (df_all['num_cmb_em'] == 100)].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)& (df_all['campaign'] == 'shift') & (df_all['normalization'] == norm_option) & (df_all['num_cmb_em'] == 100)].copy()

            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                mode = list(df_select_emit['unmix_mode'].unique())[0]
                lib_mode = list(df_select_emit['lib_mode'].unique())[0]

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

                # emit variables
                df_x_emit = df_select_emit[(df_select_emit['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y_emit = df_select_emit[(df_select_emit['instrument'] == 'emit')].copy().reset_index(drop=True)

                # aviris variables
                df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris')].copy().reset_index(drop=True)

                # plot fractional cover values
                x_emit = df_x_emit[col_map[col]]
                y_emit = df_y_emit[col_map[col]]
                x_u_emit = df_x_emit[f'{col_map[col]}_se']
                y_u_emit = df_y_emit[f'{col_map[col]}_se']

                x_shift = df_x_shift[col_map[col]]
                y_shift = df_y_shift[col_map[col]]
                x_u_shift = df_x_shift[f'{col_map[col]}_se']
                y_u_shift = df_y_shift[f'{col_map[col]}_se']

                x = list(x_emit.values) + list(x_shift.values)
                y = list(y_emit.values) + list(y_shift.values)
                x_u = list(x_u_emit.values) + list(x_u_shift.values)
                y_u = list(y_u_emit.values) + list(y_u_shift.values)

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                ax.plot(one_line, one_line, color='red', zorder=1)
                ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', markersize=4, linestyle='None', zorder=9)
                ax.scatter(x_emit, y_emit, marker='s', color='blue', edgecolor='black', label='EMIT', zorder=10)
                ax.scatter(x_shift,y_shift, marker='^', color='orange', edgecolor='black', label='AVIRIS$_{NG}$', zorder=10)

                # Add labels to each point
                # for xi, yi,xu,yu, label in zip(x, y, x_u, y_u, df_x['plot']):
                #     ax.errorbar(xi, yi, yerr=yu, xerr=xu, fmt='o')
                #     plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

                if col == 2 and row == 0:
                    ax.legend(loc='lower right')

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

        fig.supylabel('Spaceborne\Airborne Fractions', fontsize=self.axis_label_fontsize)
        plt.savefig(os.path.join(self.figure_directory, f'regression_combined_{norm_option}.png'), format="png", dpi=400, bbox_inches="tight")# load all fraction files


def run_figures(base_directory):
    base_directory = base_directory
    sensor = 'aviris_ng'
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
    fig.plot_combined(norm_option='brightness')
    fig.plot_combined(norm_option='none')
    #fig.local_slpit()
