import os
import time

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sys import platform
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial import ConvexHull
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if not "win32" in platform:
    plt.switch_backend('Agg')

class figures:
    def __init__(self, base_directory: str, sensor: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")

        # check for figure directory
        create_directory(self.fig_directory)

        # set wvls
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)

        # em_labels
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
        self.axes_limits = {'ymin': 0.04,
                            'ymax': 0.20,
                            'xmin':1.8,
                            'xmax': 6.2}


    def merge_sma_mesma(self):
        df_sma_error = pd.read_csv(os.path.join(self.fig_directory, 'sma_unmix_error_report.csv'))
        df_sma_error['mode'] = 'sma'
        df_mesma_error = pd.read_csv(os.path.join(self.fig_directory, 'mesma_unmix_error_report.csv'))
        df_mesma_error['mode'] = 'mesma'
        df_error = pd.concat([df_sma_error, df_mesma_error], ignore_index=True)
        df_error = df_error.replace('brightness', "Brightness")
        df_error = df_error.replace('1500', "1500 nm")
        df_error = df_error.replace('none', 'No Normalization')

        return df_error

    def load_sma_error(self):
        df_error = pd.read_csv(os.path.join(self.fig_directory, 'sma_unmix_error_report.csv'))
        df_error['mode'] = 'sma'
        df_error = df_error.replace('brightness', "Brightness")
        df_error = df_error.replace('none', 'No Normalization')

        return df_error


    def size_endmembers_figure(self, cmap_kw):
        df_error = figures.load_sma_error(self)

        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 1 and col == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.02, 0.02))

                if col != 0:
                    ax.set_yticklabels([])

                if row == 0:
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks[1:])
                    ax.set_xticklabels([])

        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]

        # color gradient: warm
        # get number of lines
        num_lines = 5
        cmap = plt.cm.get_cmap(cmap_kw, num_lines)
        counter = 0

        df_error = df_error.dropna(subset=['num_em'])
        # filter the data by normalization
        for _em, em in enumerate(sorted(list(df_error.num_em.unique()))):
            if em == np.nan:
                continue

            df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['num_em'] == em) & (df_error['mc_runs'] == 25) & (df_error['mode'] == 'sma')].copy()

            # filter by scenario
            for scenario in df_select.scenario.unique():
                df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                df_select1 = df_select1.sort_values('dims')  # sort by dimensions, lower to greater

                error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                capsize = [8, 6, 4]

                linestyle = 'solid'

                # loop thorough figure and plot
                for row in range(nrows):
                    for col in range(ncols):
                        label = str(int(em)) + ' EM'
                        ax = axes_2d[row][col]

                        if scenario == 'latin' and row == 0:

                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(counter))

                            if col == 0:
                                ax.set_ylabel('Latin Hypercube\n\nMAE', fontsize=self.axis_label_fontsize)

                        if scenario == 'convex' and row == 1:
                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(counter))

                            if col == 0:
                                ax.set_ylabel('Convex Hull\n\nMAE', fontsize=self.axis_label_fontsize)

                        if row == 0 and col == 1:
                            ax.legend(prop={'size': 12})

            counter += 1
        plt.savefig(os.path.join(self.fig_directory, 'endmember_selection_figure.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def uncertainty_figure(self, cmap_kw):
        df_error = figures.merge_sma_mesma(self)

        # create 3x2 figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        num_lines = 4
        cmap = plt.cm.get_cmap(cmap_kw, num_lines)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 1 and col == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.02, 0.02))
                if col != 0:
                    ax.set_yticklabels([])

                if row == 0:
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks[1:])
                    ax.set_xticklabels([])

        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]
        # color gradient: warm
        # get number of lines

        counter = 0
        df_error = df_error[df_error['mc_runs'] != 1]

        # filter the data by normalization
        for _mc, mc_runs in enumerate(sorted(list(df_error.mc_runs.unique()))):

            for mode in df_error['mode'].unique():
                if mode == 'sma':
                    df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['num_em'] == 30) & (df_error['mc_runs'] == mc_runs)].copy()
                    mode_label = 'SMA'
                    linestyle = 'solid'
                if mode == 'mesma':
                    df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['cmbs'] == 100) & (df_error['mc_runs'] == mc_runs)].copy()
                    mode_label = "MESMA"
                    linestyle = 'dotted'

                # filter by scenario
                for scenario in df_select.scenario.unique():
                    df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                    df_select1 = df_select1.sort_values('dims')  # sort by dimensions, lower to greater

                    error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                    error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                    capsize = [8, 6, 4]

                    # loop thorough figure and plot
                    for row in range(nrows):
                        for col in range(ncols):
                            ax = axes_2d[row][col]

                            label = f"{mc_runs}  MC Runs ({mode_label})"
                            if scenario == 'latin' and row == 0:
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{label}", solid_capstyle='projecting', capsize=capsize[col], linestyle= linestyle, color=cmap(_mc))

                                if col == 0:
                                    ax.set_ylabel('Latin Hypercube\n\nMAE', fontsize=self.axis_label_fontsize)

                            if scenario == 'convex' and row == 1:
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{label}", solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(_mc))

                                if col == 0:
                                    ax.set_ylabel('Convex Hull\n\nMAE', fontsize=self.axis_label_fontsize)

                            if row == 0 and col == 1:
                                ax.legend(prop={'size': 12})


                counter += 1

        plt.savefig(os.path.join(self.fig_directory, 'uncertainty_options.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def atmosphere(self, cmap_kw):
        df_error = figures.merge_sma_mesma(self)

        df_atmos = pd.read_csv(os.path.join(self.fig_directory, "atmosphere_error_report.csv"))
        df_atmos = df_atmos.loc[df_atmos['mode'] != 'sma-best'].copy()

        # call the uncertainty data; mc = 25, em =30
        df_optimized_sma = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['num_em'] == 30) &
                                    (df_error['mc_runs'] == 25) & (df_error['scenario'] == 'convex')
                                    & (df_error['dims'] == 4)].copy()

        df_optimized_mesma = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['cmbs'] == 100) &
                                        (df_error['mc_runs'] == 25) & (df_error['scenario'] == 'convex')
                                        & (df_error['dims'] == 4)].copy()

        # create 3x3 figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        df_select = df_atmos.loc[(df_atmos['azimuth'] == 0) & (df_atmos['sensor_zenith'] == 180) & (df_atmos['aod'] != 0) & (df_atmos['h2o'] != 0)].copy()

        # cmap
        num_lines = 4
        cmap = plt.cm.get_cmap(cmap_kw, num_lines)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])

                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.02, 0.02))
                ax.set_xticks(np.arange(10, 60 + 10, 10))
                ax.set_xlim(8, 60)
                capsize = [8, 6, 4]

                if row == 0:  # aod
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks[1:])

                    ax.set_xticklabels([])

                    for _aod, aods in enumerate(sorted(list(df_select.aod.unique()))):
                        for mode in df_error['mode'].unique():

                            if mode == 'sma':
                                linestyle = 'solid'
                                label_mode = "SMA"
                                df_optimized = df_optimized_sma

                            elif mode == 'mesma':
                                linestyle = 'dotted'
                                label_mode = "MESMA"
                                df_optimized = df_optimized_mesma

                            df_results1 = df_select.loc[(df_select['aod'] == aods) & (df_select['h2o'] == 0.75) & (df_select['mode'] == mode)].copy()
                            df_results1 = df_results1.sort_values('solar_zenith')

                            if col == 0:
                                ax.set_ylabel('H$_2$O (g/cm$^2$) = 0.75\n\nMAE', fontsize=self.axis_label_fontsize)
                                plot_error = df_results1.npv_mae
                                plot_uncer = df_results1['npv_stde']

                            elif col == 1:
                                plot_error = df_results1.pv_mae
                                plot_uncer = df_results1['pv_stde']

                            else:
                                plot_error = df_results1.soil_mae
                                plot_uncer = df_results1['soil_stde']

                            ax.errorbar(df_results1.solar_zenith, plot_error, yerr=plot_uncer,
                                        label=f'AOD: {aods:.2f} ({label_mode})', solid_capstyle='projecting', capsize=capsize[_aod], linestyle=linestyle, color=cmap(_aod))


                    for mode in df_error['mode'].unique():
                        if mode == 'sma':
                            label_mode = "SMA"
                            df_optimized = df_optimized_sma
                            linestyle = 'solid'
                        elif mode == 'mesma':
                            label_mode = "MESMA"
                            linestyle = 'dotted'
                            df_optimized = df_optimized_mesma

                        if col == 0:
                            optimized_error = df_optimized['npv_mae']
                            optimized_uncer = df_optimized['npv_stde']

                        elif col == 1:
                            optimized_error = df_optimized['pv_mae']
                            optimized_uncer = df_optimized['pv_stde']


                        else:
                            optimized_error = df_optimized['soil_mae']
                            optimized_uncer = df_optimized['soil_stde']

                        ax.errorbar(df_results1.solar_zenith,
                                    [optimized_error.values[0]] * df_results1.solar_zenith.shape[0],
                                    yerr=optimized_uncer, solid_capstyle='projecting', capsize=2,
                                    label=f'{label_mode} (no atmosphere)', linestyle=linestyle,
                                    color=cmap(_aod + 1))

                        ax.set_xticklabels([])

                    if col == 1:
                        ax.legend(prop={'size': 12})

                if row == 1:  # h20

                    if col == 1:
                        ax.set_xlabel("Solar Zenith Angle (°)", fontsize=self.axis_label_fontsize)

                    for _h2o, h2os in enumerate(sorted(list(df_select.h2o.unique()))):

                        for mode in df_error['mode'].unique():

                            if mode == 'sma':
                                linestyle = 'solid'
                                label_mode = "SMA"
                                df_optimized = df_optimized_sma

                            elif mode == 'mesma':
                                linestyle = 'dotted'
                                label_mode = "MESMA"
                                df_optimized = df_optimized_mesma

                            df_results = df_select.loc[(df_select['h2o'] == h2os) & (df_select['aod'] == 0.05) & (df_select['mode'] == mode)].copy()
                            df_results = df_results.sort_values('solar_zenith')
                            if col == 0:
                                ax.set_ylabel('AOD = 0.05\n\nMAE', fontsize=self.axis_label_fontsize)
                                plot_error = df_results.npv_mae
                                plot_uncer = df_results['npv_stde']

                            elif col == 1:
                                plot_error = df_results.pv_mae
                                plot_uncer = df_results['pv_stde']

                            else:
                                plot_error = df_results.soil_mae
                                plot_uncer = df_results['soil_stde']

                            ax.errorbar(df_results.solar_zenith, plot_error, yerr=plot_uncer,
                                        label=f'H$_2$O: {h2os:.2f} ({label_mode})', solid_capstyle='projecting', capsize=capsize[_aod], linestyle=linestyle, color=cmap(_h2o))


                    for mode in df_error['mode'].unique():
                        if mode == 'sma':
                            label_mode = "SMA"
                            df_optimized = df_optimized_sma
                            linestyle = 'solid'
                        elif mode == 'mesma':
                            label_mode = "MESMA"
                            linestyle = 'dotted'
                            df_optimized = df_optimized_mesma

                        if col == 0:
                            optimized_error = df_optimized['npv_mae']
                            optimized_uncer = df_optimized['npv_stde']

                        elif col == 1:
                            optimized_error = df_optimized['pv_mae']
                            optimized_uncer = df_optimized['pv_stde']

                        else:
                            optimized_error = df_optimized['soil_mae']
                            optimized_uncer = df_optimized['soil_stde']

                        ax.errorbar(df_results.solar_zenith,
                                    [optimized_error.values[0]] * df_results.solar_zenith.shape[0],
                                    yerr=optimized_uncer, solid_capstyle='projecting', capsize=2,
                                    label=f'{label_mode} (no atmosphere)', linestyle=linestyle, color=cmap(_h2o + 1))


                    # add legend
                    if col == 1:
                        ax.legend(prop={'size': 12})

                if col != 0:
                    ax.set_yticklabels([])

                ax.set_box_aspect(1)
        plt.savefig(os.path.join(self.fig_directory, "atmosphere_error_sun_angles.png"), bbox_inches="tight", dpi=400)


    def combinations_figure(self,  cmap_kw):
        df_error = pd.read_csv(os.path.join(self.fig_directory, 'mesma_unmix_error_report.csv'))
        df_error = df_error.replace('brightness', "Brightness")
        df_error = df_error.replace('none', 'Default')

        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 1 and col == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.02, 0.02))

                if col != 0:
                    ax.set_yticklabels([])

                if row == 0:
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks[1:])
                    ax.set_xticklabels([])

        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]

        # color gradient: warm
        # get number of lines
        num_lines = 6
        cmap = plt.cm.get_cmap(cmap_kw, num_lines)
        counter = 0

        # filter the data by normalization
        for _em, em in enumerate(sorted(list(df_error.cmbs.unique()))):
            df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['cmbs'] == em) & (
                        df_error['mc_runs'] == 5)].copy()

            # filter by scenario
            for scenario in df_select.scenario.unique():
                df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                df_select1 = df_select1.sort_values('dims')  # sort by dimensions, lower to greater

                error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                capsize = [8, 6, 4]

                linestyle = 'solid'

                # loop thorough figure and plot
                for row in range(nrows):
                    for col in range(ncols):
                        ax = axes_2d[row][col]

                        label = str(int(em)) + ' Models'
                        if scenario == 'latin' and row == 0:
                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], color=cmap(counter), linestyle=linestyle)

                            if col == 0:
                                ax.set_ylabel('Latin Hypercube\n\nMAE', fontsize=self.axis_label_fontsize)

                        if scenario == 'convex' and row == 1:
                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], color=cmap(counter), linestyle=linestyle)

                            if col == 0:
                                ax.set_ylabel('Convex Hull\n\nMAE', fontsize=self.axis_label_fontsize)

                        if row == 0 and col == 1:
                            ax.legend(prop={'size': 12})

            counter += 1

        plt.savefig(os.path.join(self.fig_directory, 'mesma_cmbs_selection_figure.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def normalization_figure(self, cmap_kw):
        df_error = figures.merge_sma_mesma(self)

        # create 3x2 figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows,  wspace=0.05, hspace=0.05, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_ylim(self.axes_limits['ymin'], self.axes_limits['ymax'])
                ax.set_xlim(self.axes_limits['xmin'], self.axes_limits['xmax'])

                if row == 0:
                    ax.set_title(self.ems[col], fontsize=self.title_fontsize)

                if row == 1 and col == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.set_yticks(np.arange(self.axes_limits['ymin'], self.axes_limits['ymax'] + 0.02, 0.02))

                if col != 0:
                    ax.set_yticklabels([])

                if row == 0:
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks[1:])
                    ax.set_xticklabels([])

        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]

        # color gradient: warm
        num_lines = 3
        cmap = plt.cm.get_cmap(cmap_kw, num_lines)

        # filter the data
        for _norm_option, norm_option in enumerate(sorted(list(df_error.normalization.unique()))):

            for mode in df_error['mode'].unique():
                if mode == 'sma':
                    df_select = df_error.loc[(df_error['normalization'] == norm_option) & (df_error['num_em'] == 30) & (df_error['mc_runs'] == 25)].copy()
                    label = 'SMA'
                    linestyle = 'solid'
                if mode == 'mesma':
                    df_select = df_error.loc[(df_error['normalization'] == norm_option) & (df_error['cmbs'] == 100) & (df_error['mc_runs'] == 25)].copy()
                    label = "MESMA"
                    linestyle = 'dotted'

                # filter by scenario
                for scenario in df_select.scenario.unique():
                    df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                    df_select1 = df_select1.sort_values('dims') # sort by dimensions, lower to greater

                    error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                    error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                    capsize = [8, 6, 4]

                    # loop thorough figure and plot
                    for row in range(nrows):
                        for col in range(ncols):
                            ax = axes_2d[row][col]
                            if norm_option == 'Brightness':
                                norm_label = 'Length of Vector'
                            else:
                                norm_label = norm_option

                            if scenario == 'latin' and row == 0:
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{norm_label} ({label})", solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(_norm_option))

                                if col == 0:
                                    ax.set_ylabel('Latin Hypercube\n\nMAE', fontsize=self.axis_label_fontsize)

                            if scenario == 'convex' and row == 1:
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{norm_label} ({label})", solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle,  color=cmap(_norm_option))
                                if col == 0:
                                    ax.set_ylabel('Convex Hull\n\nMAE', fontsize=self.axis_label_fontsize)
                            if row == 0 and col == 0:
                                ax.legend(prop={'size': 10})

        plt.savefig(os.path.join(self.fig_directory, 'normalization_figure.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def em_reduction_visulatization(self):
        # load global library
        df = pd.read_csv(os.path.join(self.output_directory, 'convolved', 'geofilter_convolved.csv'))
        df = df.sort_values('level_1')

        # run soil PCA
        pc_components = spectra.pca_analysis(df, spectra_starting_col=7)
        pc_array = np.asarray(pc_components)[:, 7: 9]

        # create new figure
        ncols = 2  # for each em
        nrows = 1  # top row normal, middle water, aod; x-axis solar angle

        fig, axs = plt.subplots(nrows, ncols, figsize=(8, 12))
        fig.subplots_adjust(wspace=0.02, hspace=0)

        for row in range(nrows):
            for col in range(ncols):
                ax = axs[col]
                ax.grid('on', linestyle='--')
                ax.set_xlabel("Principal Component 1", fontsize=self.axis_label_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect('equal', adjustable='box')
                ax.set_ylim(-3,3)
                ax.set_xlim(-7,7)

                if col == 0:
                    ax.set_title('Latin Hypercube', fontsize=self.title_fontsize)
                    ax.set_ylabel("Principal Component 2", fontsize=self.axis_label_fontsize)

                    # get the latin hypercube
                    cubes = spectra.latin_hypercubes(points=pc_array, get_quadrants_index=False)
                    colors = ['blue', 'green', 'orange', 'magenta']

                    for _cube, cube in enumerate(cubes):
                        ax.scatter(cube[:, 0], cube[:, 1], s=4, c=colors[_cube])

                    # plot the quadrant divisions
                    ax.axvline(x=0, color='r', linestyle='--', lw=2)
                    ax.axhline(y=0, color='r', linestyle='--', lw=2)

                if col == 1:
                    ax.set_title('Convex Hull', fontsize=self.title_fontsize)
                    ax.set_yticklabels([])

                    # get the convex hull of a 2-dimensions
                    ch = ConvexHull(pc_array)

                    # plot the connecting lines of the hull; simplices
                    for simplex in ch.simplices:
                        ax.plot(pc_array[simplex, 0], pc_array[simplex, 1], 'c', lw=0.75)

                    #

                    # plot the vertices
                    ax.plot(pc_array[ch.vertices, 0], pc_array[ch.vertices, 1], 'o', mec='r', color='none', lw=0.75,
                            markersize=14)

                    ax.scatter(pc_array[:, 0], pc_array[:, 1], s=4)

        plt.savefig(os.path.join(self.fig_directory, 'em_reduction_visualization.png'), bbox_inches="tight",
                    dpi=400)
        plt.clf()
        plt.close()


def run_figures(base_directory, sensor):
    base_directory = base_directory
    sensor = sensor
    major_axis_fontsize = 14
    minor_axis_fontsize = 12
    title_fontsize = 26
    axis_label_fontsize = 22
    fig_height = 8
    fig_width = 12
    linewidth = 1
    sig_figs = 2

    fig_class = figures(base_directory=base_directory, sensor=sensor, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs)

    fig_class.em_reduction_visulatization()
    fig_class.normalization_figure(cmap_kw='brg')
    fig_class.size_endmembers_figure(cmap_kw='brg')
    fig_class.combinations_figure(cmap_kw='brg')
    fig_class.uncertainty_figure(cmap_kw='brg')
    fig_class.atmosphere(cmap_kw='brg')
