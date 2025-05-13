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
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs, legend_text):

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
        self.legend_text = legend_text
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

        self.col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        self.col_map_wp = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

    def quad_cover(self):
        # load quadrat tallies
        df_quad = pd.read_csv(os.path.join(self.base_directory, 'field', 'SHIFT_vegetation_quadrat_tallies.csv'))
        #df_quad = df_quad.dropna(subset=['Phenophase'])
        for i in sorted(list(df_quad.Species_or_type.unique())):
            print(i)

        df_quad['Phenophase'] = df_quad['Phenophase'].replace(np.nan, 'na')
        df_quad['Phenophase'] = df_quad['Phenophase'].str.strip()
        df_quad['Phenophase'] = df_quad['Phenophase'].apply(str.lower)

        df_quad['Species_or_type'] = df_quad['Species_or_type'].apply(str.lower)
        df_quad['Species_or_type'] = df_quad['Species_or_type'].str.strip()
        df_quad['cover'] = ''

        df_quad['Date'] = pd.to_datetime(df_quad['Date'], format='%Y-%m-%d')

        # df coords with data and dates
        df_coords = pd.read_csv('gis/shift_plot_coordinates.csv')

        df_quads_to_merge = []
        for specie in sorted(list(df_quad.Species_or_type.unique())):
            df_quad_select = df_quad.loc[df_quad['Species_or_type'] == specie].copy()

            if specie in ['soil', 'rock', 'npv']:
                if specie == 'rock':
                    df_quad_select['cover'] = 'soil'
                else:
                    df_quad_select['cover'] = specie

            elif specie in ['water']:
                continue
            else:
                df_quad_select['cover'] = df_quad_select['Phenophase'].replace(quad_phenophase_key)
            df_quads_to_merge.append(df_quad_select)

        df_quad_cover = pd.concat(df_quads_to_merge, ignore_index=True)

        df_agg_rows = []
        for _plot, plot in enumerate(sorted(list(df_quad_cover['Plot_name'].unique()))):
            df_plot = df_quad_cover.loc[df_quad_cover['Plot_name'] == plot].copy()
            df_meta = df_coords.loc[df_coords['Plot Name'] == plot]

            if df_meta.empty:
                pass
            else:
                plot_date = df_meta['Date'].values[0]

                df_plot = df_plot.loc[df_plot['Date'] == plot_date].copy()
                df_plot = df_plot.drop(columns=['Date'])

                df_agg = df_plot.groupby(['cover']).sum().reset_index()

                df_agg['frac_cover'] = df_agg['Count'] / df_agg['Count'].sum()

                row = [plot, plot_date, ]
                for cover in sorted(list(df_agg.cover.unique())):
                    frac = df_agg.loc[df_agg['cover'] == cover, 'frac_cover'].iloc[0]
                    row.append((cover, frac))

                df_agg_rows.append(row)

        # aggregate all rows
        df_to_concat = []
        for row in df_agg_rows:
            plot = row[0]
            date = row[1]
            frac_covers = row[2:]
            cols = ['plot', 'date']
            values = [plot, date]
            for cover in frac_covers:
                cols.append(cover[0])
                values.append(cover[1])
            df = pd.DataFrame(values).T
            df.columns = cols
            df_to_concat.append(df)

        df_concat = pd.concat(df_to_concat)
        df_concat = df_concat.fillna(0)
        df_concat.to_csv(os.path.join(self.output_directory, 'quad_cover.csv'), index=False)

    def load_frac_data(self):
        skip = ['SRA-000-SPRING', 'SRB-047-SPRING', 'SRB-004-FALL', 'SRB-050-FALL', 'SRB-200-FALL', 'SRA-056-SPRING',
                'DPA-004-FALL', 'DPB-027-SPRING', 'SRA-008-FALL', 'SRB-026-SPRING'] # excluding shrubland plots

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

    def slpit_fig(self, norm_option, mode, lib):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)
        df_all = df_all[(df_all['unmix_mode'] == mode) & (df_all['lib_mode'] == lib) & (df_all['normalization'] == norm_option)].copy()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 3
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.15, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        df_cols = []

        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.set_aspect('auto')

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
                    ax.set_ylabel(f'AVIRIS', fontsize=self.axis_label_fontsize)

                if row == 1 and col == 0:
                    ax.set_ylabel(f'Wonderpole', fontsize=self.axis_label_fontsize)

                if row == 2 and col == 0:
                    ax.set_ylabel(f'Quadrats', fontsize=self.axis_label_fontsize)

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
                x = df_x[self.col_map[col]]

                # plot aviris vs slpit
                if row == 0:
                    df_y = df_all[(df_all['instrument'] == 'aviris')].copy().reset_index(drop=True)
                    df_y = df_y.sort_values('plot')
                    # plot fractional cover values
                    y = df_y[self.col_map[col]]


                    # plot uncertainty
                    # x_u = df_x[f'{col_map[col]}_se']
                    # y_u = df_y[f'{col_map[col]}_se']

                if row == 1:
                    y = df_wonderpole[self.col_map_wp[col]]/100

                if row == 2:
                    y = df_quad[self.col_map[col]]

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10,
                           s=150)
                ax.tick_params(axis='both', labelsize=self.legend_text)

                for i, label in enumerate(df_wonderpole['plot'].values):
                   ax.text(x.values[i], y.values[i], label, fontsize=8, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2, bias = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                if row == 0:
                    error = np.absolute(x - y)
                    df_error = pd.DataFrame({'Plot': df_y['plot'].values,
                                             self.ems[col] : error})
                    df_error = df_error.sort_values(by=self.ems[col]).reset_index(drop=True)
                    df_error.to_csv(os.path.join(self.figure_directory, f'error_quality_{self.ems[col]}_{mode}_{norm_option}_{lib}.csv'), index=False)


                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, f'methods_fig_1_{mode}_{norm_option}_{lib}.png'), format="png", dpi=400,
                  bbox_inches="tight")
        plt.clf()
        plt.close()



    def wonderpole_fig(self, norm_option, mode, lib):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)
        df_all = df_all[(df_all['unmix_mode'] == mode) & (df_all['lib_mode'] == lib) & (
                    df_all['normalization'] == norm_option)].copy()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.15, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)


        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.set_aspect('auto')

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
                    ax.set_ylabel(f'AVIRIS', fontsize=self.axis_label_fontsize)

                if row == 1 and col == 0:
                    ax.set_ylabel(f'Quadrats', fontsize=self.axis_label_fontsize)

                ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

                # set blank ticks for col 1 and 2
                if col != 0:
                    ax.set_yticklabels([])

                # set blank ticks for for row 0 and 1
                if row != 1:
                    ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
                    ax.set_xticklabels([])

                x = df_wonderpole[self.col_map_wp[col]] / 100

                # plot aviris vs slpit
                if row == 0:
                    df_y = df_all[(df_all['instrument'] == 'aviris')].copy().reset_index(drop=True)
                    df_y = df_y.sort_values('plot')
                    # plot fractional cover values
                    y = df_y[self.col_map[col]]

                    # plot uncertainty
                    # x_u = df_x[f'{col_map[col]}_se']
                    # y_u = df_y[f'{col_map[col]}_se']

                if row == 1:
                    y = df_quad[self.col_map[col]]

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                # ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None', capsize=5)
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10, s=150)
                ax.tick_params(axis='both', labelsize=self.legend_text)

                #for i, label in enumerate(df_wonderpole['plot'].values):
                #   ax.text(x.values[i], y.values[i], label, fontsize=8, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2, bias = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, f'methods_fig_2_{mode}_{norm_option}_{lib}.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()


    def quad_fig(self, norm_option, mode, lib):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)
        df_all = df_all[(df_all['unmix_mode'] == mode) & (df_all['lib_mode'] == lib) & (
                df_all['normalization'] == norm_option)].copy()

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.15, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)


        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                if row == 1:
                    continue
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.set_aspect('auto')

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                # titles
                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                # x-axis
                ax.set_xlabel("Quadrats", fontsize=self.axis_label_fontsize)

                # y axis
                if col == 0:
                    ax.set_ylabel(f'AVIRIS', fontsize=self.axis_label_fontsize)

                ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

                # set blank ticks for col 1 and 2
                if col != 0:
                    ax.set_yticklabels([])

                # set blank ticks for for row 0 and 1
                ax.set_yticklabels([''] + ax.get_yticklabels()[1:])

                x = df_quad[self.col_map[col]].values

                # plot aviris vs quads
                df_y = df_all[(df_all['instrument'] == 'aviris')].copy().reset_index(drop=True)
                df_y = df_y.sort_values('plot')
                # plot fractional cover values
                y = df_y[self.col_map[col]]


                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')
                ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10,
                           s=150)
                ax.tick_params(axis='both', labelsize=self.legend_text)

                # for i, label in enumerate(df_quad['plot'].values):
                #    ax.text(x[i], y[i], label, fontsize=12, ha='center', va='bottom')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2,bias = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, f'methods_fig_3_{mode}_{norm_option}_{lib}.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()

    def mesma_vs_emc2_1(self):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)

        #  create figure
        fig = plt.figure(figsize=(20, 20))
        ncols = 3
        nrows = 6
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.30, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        df_select_shift = df_all[(df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)].copy()

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if row == 0:
                    ax.set_xlabel("SLPIT - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - MESMA - Bright", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma')  & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                if row == 1:
                    ax.set_xlabel("SLPIT - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - MESMA - Bright", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                if row == 2:
                    ax.set_xlabel("SLPIT - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - EMC2 - NN", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 3:
                    ax.set_xlabel("SLPIT - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT -MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 4:
                    ax.set_xlabel("SLPIT - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - Bright", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                if row == 5:
                    ax.set_xlabel("AVIRIS - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - Bright", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.2, 0.2))

                x_shift = df_x_shift[col_map[col]]
                y_shift = df_y_shift[col_map[col]]
                x_u_shift = df_x_shift[f'{col_map[col]}_sigma']
                y_u_shift = df_y_shift[f'{col_map[col]}_sigma']

                x =  list(x_shift.values)
                y =  list(y_shift.values)
                x_u = list(x_u_shift.values)
                y_u = list(y_u_shift.values)

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                ax.plot(one_line, one_line, color='red', zorder=1)
                ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                ax.scatter(x_shift, y_shift, marker='^', color='orange', edgecolor='black', label='AVIRIS$_{NG}$',
                           zorder=10)

                # Add labels to each point
                # for xi, yi,xu,yu, label in zip(x, y, x_u, y_u, df_x['plot']):
                #     ax.errorbar(xi, yi, yerr=yu, xerr=xu, fmt='o')
                #     plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

                if col == 2 and row == 0:
                    ax.legend(loc='lower right')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2, bias = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'Bias: %.2f ' % (bias),
                ))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

        fig.supylabel('Spaceborne\Airborne Fractions', fontsize=self.axis_label_fontsize)
        plt.savefig(os.path.join(self.figure_directory, f'mesma_vs_emc2_1.png'), format="png",
                    dpi=400, bbox_inches="tight")  # load all fraction files


    def mesma_vs_emc2_2(self):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)

        #  create figure
        fig = plt.figure(figsize=(40, 40))
        ncols = 3
        nrows = 18
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.35, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        df_select_shift = df_all[(df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)].copy()

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if row == 0:
                    ax.set_xlabel("SLPIT - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - EMC2 - NN", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma')  & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 1:
                    ax.set_xlabel("SLPIT - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - MESMA - NN", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 2:
                    ax.set_xlabel("SLPIT - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - NN", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 3:
                    ax.set_xlabel("SLPIT - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS -MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 4:
                    ax.set_xlabel("AVIRIS - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 5:
                    ax.set_xlabel("AVIRIS - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 6:
                    ax.set_xlabel("AVIRIS - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 7:
                    ax.set_xlabel("AVIRIS - MESMA - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 8:
                    ax.set_xlabel("AVIRIS - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 9:
                    ax.set_xlabel("AVIRIS - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("SLPIT - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 10:
                    ax.set_xlabel("AVIRIS - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 11:
                    ax.set_xlabel("AVIRIS - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)


                if row == 12:
                    ax.set_xlabel("SLPIT - EMC2 - NN", fontsize=8)
                    ax.set_ylabel("SLPIT - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)


                if row == 13:
                    ax.set_xlabel("SLPIT - EMC2 - NN", fontsize=8)
                    ax.set_ylabel("AVIRIS - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 14:
                    ax.set_xlabel("SLPIT - MESMA - NN", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 15:
                    ax.set_xlabel("AVIRIS - MESMA - NN", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 16:
                    ax.set_xlabel("SLPIT - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - EMC2 - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (
                                df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)


                if row == 17:
                    ax.set_xlabel("SLPIT - EMC2 - Bright", fontsize=8)
                    ax.set_ylabel("AVIRIS - MESMA - NN", fontsize=8)

                    # aviris variables
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (
                                df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.2, 0.2))

                x_shift = df_x_shift[col_map[col]]
                y_shift = df_y_shift[col_map[col]]
                x_u_shift = df_x_shift[f'{col_map[col]}_sigma']
                y_u_shift = df_y_shift[f'{col_map[col]}_sigma']

                x =  list(x_shift.values)
                y =  list(y_shift.values)
                x_u = list(x_u_shift.values)
                y_u = list(y_u_shift.values)

                m, b = np.polyfit(x, y, 1)
                one_line = np.linspace(0, 1, 101)

                ax.plot(one_line, one_line, color='red', zorder=1)
                ax.plot(one_line, m * one_line + b, color='black', zorder=2)
                ax.scatter(x_shift, y_shift, marker='^', color='orange', edgecolor='black', label='AVIRIS$_{NG}$',
                           zorder=10)

                # Add labels to each point
                # for xi, yi,xu,yu, label in zip(x, y, x_u, y_u, df_x['plot']):
                #     ax.errorbar(xi, yi, yerr=yu, xerr=xu, fmt='o')
                #     plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

                if col == 2 and row == 0:
                    ax.legend(loc='lower right')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2, bias = r2_calculations(x, y)

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'Bias: %.2f ' % (bias),
                ))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

        fig.supylabel('Spaceborne\Airborne Fractions', fontsize=self.axis_label_fontsize)
        plt.savefig(os.path.join(self.figure_directory, f'mesma_vs_emc2_2.png'), format="png",
                    dpi=400, bbox_inches="tight")  # load all fraction file

    def slpit_fig_by_site(self):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)

        # # # create figure
        fig = plt.figure(figsize=(self.fig_width, 20))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.15, hspace=0.35, width_ratios=[1] * ncols, height_ratios=[1] * nrows)


        df_select_shift = df_all[(df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)].copy()

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


                if row == 0:
                    ax.set_ylabel("EMC2 - Normalized", fontsize=8)
                    # aviris variables
                    df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                if row == 1:
                    ax.set_ylabel("EMC2 - NN", fontsize=8)
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'sma') & (
                                    df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                if row == 2:
                    ax.set_ylabel("MESMA - Normalized", fontsize=8)
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                df_select_shift['normalization'] == 'brightness')].copy().reset_index(drop=True)

                if row == 3:
                    ax.set_ylabel("MESMA - NN", fontsize=8)
                    df_x_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'asd') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)
                    df_y_shift = df_select_shift[
                        (df_select_shift['instrument'] == 'aviris') & (df_select_shift['unmix_mode'] == 'mesma') & (
                                df_select_shift['normalization'] == 'none')].copy().reset_index(drop=True)

                x_shift = df_x_shift[col_map[col]]
                y_shift = df_y_shift[col_map[col]]

                x = list(x_shift.values)
                y = list(y_shift.values)

                error = np.array(x) - np.array(y)
                m, b = np.polyfit(x, error, 1)
                one_line = np.linspace(0, 1, 101)

                # plot 1 to 1 line
                ax.plot(one_line, one_line, color='red')
                ax.plot(one_line, m * one_line + b, color='black')

                ax.scatter(x_shift, error, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10,
                           s=150)

                r2, bias = r2_calculations(x, error)

                txtstr = '\n'.join((
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                        verticalalignment='top', bbox=props)

                #ax.tick_params(axis='both', labelsize=self.legend_text)
                #ax.set_xticklabels(df_x_shift['plot'].values, rotation=90, fontsize=8)

        plt.savefig(os.path.join(self.figure_directory, f'error_quality_all.png'), format="png", dpi=400,
                  bbox_inches="tight")
        plt.clf()
        plt.close()






def run_figures(base_directory):
    base_directory = base_directory
    sensor = 'aviris_ng'
    major_axis_fontsize = 22
    minor_axis_fontsize = 20
    title_fontsize = 30
    axis_label_fontsize = 26
    fig_height = 10
    fig_width = 17
    linewidth = 3
    sig_figs = 2
    legend_text = 20

    fig = figures(base_directory=base_directory, sensor=sensor, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs, legend_text=legend_text)
    #fig.plot_summary()
    fig.quad_cover()
    fig.mesma_vs_emc2_1()
    fig.mesma_vs_emc2_2()
    fig.slpit_fig_by_site()
    for mode in ['sma', 'mesma']:
        for norm in ['brightness', 'none']:
            for lib in ['global']:
                fig.slpit_fig(norm_option=norm, mode=mode, lib=lib)
                fig.wonderpole_fig(norm_option=norm, mode=mode, lib=lib)
                fig.quad_fig(norm_option=norm, mode=mode, lib=lib)
