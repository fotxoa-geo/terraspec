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

    def load_frac_data(self):
        skip = ['SRA-000-SPRING', 'SRB-047-SPRING', 'SRB-004-FALL', 'SRB-050-FALL', 'SRB-200-FALL', 'SRA-056-SPRING',
                'DPA-004-FALL', 'DPB-027-SPRING', 'SRA-008-FALL', 'SRB-026-SPRING', 'SRA-109-SPRING'] # excluding shrubland plots

        df_all = pd.read_csv(os.path.join(self.figure_directory, 'shift_fraction_output.csv'))
        df_all = df_all[~df_all['plot'].isin(skip)]

        df_wonderpole = pd.read_excel(os.path.join(self.figure_directory, 'wonderpole.xlsx'))
        df_wonderpole['plot'] = df_wonderpole["plot_name"] + '-' + df_wonderpole["season"]
        df_wonderpole = df_wonderpole.dropna()
        df_wonderpole = df_wonderpole[~df_wonderpole['plot'].isin(skip)]
        df_wonderpole = df_wonderpole.sort_values('plot')

        df_quad = pd.read_csv(os.path.join(self.output_directory, 'quad_cover.csv'))
        df_quad['date'] = pd.to_datetime(df_quad['date'], format='%m/%d/%Y')
        df_quad['season'] = df_quad['date'].apply(lambda x: 'FALL' if x.month == 9 else 'SPRING')
        df_quad['plot'] = df_quad["plot"] + '-' + df_quad["season"]
        df_quad = df_quad[~df_quad['plot'].isin(skip)]
        df_quad = df_quad.sort_values('plot')

        return df_all, df_wonderpole, df_quad

    def slpit_fig(self, norm_option):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)
        df_all = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['normalization'] == norm_option)].copy()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 3
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        col_map_wp = {
            0: 'npv_sh',
            1: 'pv_sh',
            2: 'soil_sh'}

        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                # titles
                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                # x-axis
                if row == 2:
                    ax.set_xlabel("SLPIT", fontsize=self.axis_label_fontsize)

                # y axis
                if row == 0 and col == 0:
                    ax.set_ylabel(f'AVIRIS', fontsize=18)

                if row == 1 and col == 0:
                    ax.set_ylabel(f'Wonderpole', fontsize=18)

                if row == 2 and col == 0:
                    ax.set_ylabel(f'Quadrats', fontsize=18)

                ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

                # set blank ticks for col 1 and 2
                if col != 0:
                    ax.set_yticklabels([])

                # set blank ticks for for row 0 and 1
                if row != 2:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                df_x = df_all[(df_all['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_x = df_x.sort_values('plot')
                x = df_x[col_map[col]]

                # plot aviris vs slpit
                if row == 0:
                    df_y = df_all[(df_all['instrument'] == 'aviris')].copy().reset_index(drop=True)
                    df_y = df_y.sort_values('plot')
                    # plot fractional cover values
                    y = df_y[col_map[col]]


                    # plot uncertainty
                    # x_u = df_x[f'{col_map[col]}_se']
                    # y_u = df_y[f'{col_map[col]}_se']

                if row == 1:
                    y = df_wonderpole[col_map_wp[col]]/100

                if row == 2:
                    y = df_quad[col_map[col]]

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                #ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None', capsize=5)
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10)

                #for i, label in enumerate(df_x['plot'].values):
                #    ax.text(x[i], y[i], label, fontsize=12, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, f'methods_fig_1.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()

    def wonderpole_fig(self, norm_option):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)
        df_all = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (
                    df_all['normalization'] == norm_option)].copy()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.05, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        col_map_wp = {
            0: 'npv_sh',
            1: 'pv_sh',
            2: 'soil_sh'}

        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                # titles
                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                # x-axis
                if row == 1:
                    ax.set_xlabel("Wonderpole", fontsize=self.axis_label_fontsize)

                # y axis
                if row == 0 and col == 0:
                    ax.set_ylabel(f'AVIRIS', fontsize=18)

                if row == 1 and col == 0:
                    ax.set_ylabel(f'Quadrats', fontsize=18)

                ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

                # set blank ticks for col 1 and 2
                if col != 0:
                    ax.set_yticklabels([])

                # set blank ticks for for row 0 and 1
                if row != 1:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                x = df_wonderpole[col_map_wp[col]] / 100

                # plot aviris vs slpit
                if row == 0:
                    df_y = df_all[(df_all['instrument'] == 'aviris')].copy().reset_index(drop=True)
                    df_y = df_y.sort_values('plot')
                    # plot fractional cover values
                    y = df_y[col_map[col]]

                    # plot uncertainty
                    # x_u = df_x[f'{col_map[col]}_se']
                    # y_u = df_y[f'{col_map[col]}_se']

                if row == 1:
                    y = df_quad[col_map[col]]

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                # ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None', capsize=5)
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10)

                # for i, label in enumerate(df_x['plot'].values):
                #    ax.text(x[i], y[i], label, fontsize=12, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, f'methods_fig_2.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()


    def quad_fig(self, norm_option):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)
        df_all = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (
                df_all['normalization'] == norm_option)].copy()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 1
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.05, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}


        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                # titles
                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                # x-axis
                ax.set_xlabel("Quadrats", fontsize=self.axis_label_fontsize)

                # y axis
                if col == 0:
                    ax.set_ylabel(f'AVIRIS', fontsize=18)

                ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

                # set blank ticks for col 1 and 2
                if col != 0:
                    ax.set_yticklabels([])

                # set blank ticks for for row 0 and 1
                if row != 1:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                x = df_quad[col_map[col]]

                # plot aviris vs quads
                df_y = df_all[(df_all['instrument'] == 'aviris')].copy().reset_index(drop=True)
                df_y = df_y.sort_values('plot')
                # plot fractional cover values
                y = df_y[col_map[col]]

                    # plot uncertainty
                    # x_u = df_x[f'{col_map[col]}_se']
                    # y_u = df_y[f'{col_map[col]}_se']

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                # ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None', capsize=5)
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10)

                # for i, label in enumerate(df_x['plot'].values):
                #    ax.text(x[i], y[i], label, fontsize=12, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, f'methods_fig_3.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()


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
    fig.slpit_fig(norm_option='brightness')
    fig.wonderpole_fig(norm_option='brightness')
    fig.quad_fig(norm_option='brightness')