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
        self.ems = ['non-photosynthetic\nvegetation', 'photosynthetic\nvegetation', 'soil']
        self.ems_short = ['npv', 'pv', 'soil']

        # figure fonts, font size, etc
        self.major_axis_fontsize = major_axis_fontsize
        self.minor_axis_fontsize = minor_axis_fontsize
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.fig_height = fig_height
        self.fig_width = fig_width
        self.linewidth = linewidth
        self.sig_figs = sig_figs
        self.cmap_kw = 'viridis'

    def merge_sma_mesma(self):
        df_sma_error = pd.read_csv(os.path.join(self.fig_directory, 'sma_unmix_error_report.csv'))
        df_sma_error['mode'] = 'sma'
        df_mesma_error = pd.read_csv(os.path.join(self.fig_directory, 'mesma_unmix_error_report.csv'))
        df_mesma_error['mode'] = 'mesma'
        df_error = pd.concat([df_sma_error, df_mesma_error], ignore_index=True)
        df_error = df_error.replace('brightness', "Brightness")
        df_error = df_error.replace('1500', "1500 nm")
        df_error = df_error.replace('none', 'No Normalization')

        df_sma_uncer = pd.read_csv(os.path.join(self.fig_directory, 'sma_unmix_uncertainty_report.csv'))
        df_sma_uncer['mode'] = 'sma'
        df_mesma_uncer = pd.read_csv(os.path.join(self.fig_directory, 'mesma_unmix_uncertainty_report.csv'))
        df_mesma_uncer['mode'] = 'mesma'
        df_uncer = pd.concat([df_sma_uncer, df_mesma_uncer], ignore_index=True)
        df_uncer = df_uncer.replace('brightness', "Brightness")
        df_uncer = df_uncer.replace('1500', "1500 nm")
        df_uncer = df_uncer.replace('none', 'No Normalization')

        return df_error, df_uncer

    def load_sma_error(self):
        df_error = pd.read_csv(os.path.join(self.fig_directory, 'sma_unmix_error_report.csv'))
        df_error['mode'] = 'sma'
        df_error = df_error.replace('brightness', "Brightness")
        df_error = df_error.replace('none', 'No Normalization')

        df_uncer = pd.read_csv(os.path.join(self.fig_directory, 'sma_unmix_uncertainty_report.csv'))
        df_uncer['mode'] = 'sma'
        df_uncer = df_uncer.replace('brightness', "Brightness")
        df_uncer = df_uncer.replace('none', 'No Normalization')

        return df_error, df_uncer


    def size_endmembers_figure(self):
        df_error, df_uncer = figures.load_sma_error(self)

        fig = plt.figure(constrained_layout=True, figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.001, hspace=0.0001, figure=fig)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.grid('on', linestyle='--')
                ax.set_ylim(0, 0.22)
                ax.set_xlim(1, 6.25)

                if row == 0:
                    ax.set_title(self.ems[col].title(), fontsize=self.title_fontsize)

                if row == 3:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect(1. / ax.get_data_ratio())
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if col != 0:
                    ax.set_yticklabels([])


        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]

        # color gradient: warm
        # get number of lines
        num_lines = 5
        cmap = plt.cm.get_cmap(self.cmap_kw, num_lines)

        # filter the data by normalization
        for _em, em in enumerate(sorted(list(df_error.num_em.unique()))):
            if em == np.nan:
                continue

            df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['num_em'] == em) & (df_error['mc_runs'] == 25) & (df_error['mode'] == 'sma')].copy()
            df_select_unc = df_uncer.loc[(df_uncer['normalization'] == 'Brightness') & (df_uncer['num_em'] == em) & (df_uncer['mc_runs'] == 25) & (df_uncer['mode'] == 'sma')].copy()

            # filter by scenario
            for scenario in df_select.scenario.unique():
                df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                df_select1 = df_select1.sort_values('dims')  # sort by dimensions, lower to greater

                df_select_unc1 = df_select_unc.loc[(df_select_unc['scenario'] == scenario)].copy()
                df_select_unc1 = df_select_unc1.sort_values('dims')  # sort by dimensions, lower to greater

                error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                capsize = [8, 6, 4]

                if em == 30:
                    linestyle = 'solid'
                elif em == 20:
                    linestyle = 'dashed'
                elif em == 10:
                    linestyle = 'dotted'
                elif em == 5:
                    linestyle = 'dashed'
                elif em == 3:
                    linestyle = 'dotted'

                # loop thorough figure and plot
                for row in range(nrows):
                    for col in range(ncols):
                        label = str(int(em)) + ' EM'
                        ax = axes_2d[row][col]

                        if scenario == 'latin' and row == 0:

                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(_em))

                            if col == 0:
                                ax.set_ylabel('SMA MAE\n(Latin Hypercube)', fontsize=self.axis_label_fontsize)

                        if scenario == 'convex' and row == 1:
                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(_em))

                            if col == 0:
                                ax.set_ylabel('SMA MAE\n(Convex Hull)', fontsize=self.axis_label_fontsize)

                        if row == 1 and col == 1:
                            ax.legend(prop={'size': 8})

        plt.savefig(os.path.join(self.fig_directory, 'endmember_selection_figure.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def uncertainty_figure(self):
        df_error, df_uncer = figures.merge_sma_mesma(self)

        # create 3x2 figure
        fig = plt.figure(constrained_layout=True, figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.001, hspace=0.0001, figure=fig)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.grid('on', linestyle='--')
                ax.set_ylim(0, 0.22)
                ax.set_xlim(1, 6.25)

                if row == 0:
                    ax.set_title(self.ems[col].title(), fontsize=self.title_fontsize)

                if row == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect(1. / ax.get_data_ratio())
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                if col != 0:
                    ax.set_yticklabels([])

        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]
        # color gradient: warm
        # get number of lines
        num_lines = 10
        cmap = plt.cm.get_cmap(self.cmap_kw, num_lines)

        # filter the data by normalization
        for _mc, mc_runs in enumerate(sorted(list(df_error.mc_runs.unique()))):

            for mode in df_error['mode'].unique():
                if mode == 'sma':
                    df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['num_em'] == 30) & (df_error['mc_runs'] == mc_runs)].copy()
                    df_select_unc = df_uncer.loc[(df_uncer['normalization'] == 'Brightness') & (df_uncer['num_em'] == 30) & (df_uncer['mc_runs'] == mc_runs)].copy()
                    mode_label = 'SMA'
                    linestyle = 'solid'
                if mode == 'mesma':
                    df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['cmbs'] == 100) & (df_error['mc_runs'] == mc_runs)].copy()
                    df_select_unc = df_uncer.loc[(df_uncer['normalization'] == 'Brightness') & (df_uncer['cmbs'] == 100) & (df_uncer['mc_runs'] == mc_runs)].copy()
                    mode_label = "MESMA"
                    linestyle = 'dotted'


                # filter by scenario
                for scenario in df_select.scenario.unique():
                    df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                    df_select1 = df_select1.sort_values('dims')  # sort by dimensions, lower to greater

                    df_select_unc1 = df_select_unc.loc[(df_select_unc['scenario'] == scenario)].copy()
                    df_select_unc1 = df_select_unc1.sort_values('dims')  # sort by dimensions, lower to greater

                    error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                    error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                    capsize = [8, 6, 4]

                    if mc_runs == 25:
                        linestyle = 'dashed'
                    elif mc_runs == 5:
                        linestyle = 'dashed'
                    elif mc_runs == 10:
                        linestyle = 'dotted'
                    elif mc_runs == 50:
                        linestyle = 'dotted'

                    # loop thorough figure and plot
                    for row in range(nrows):
                        for col in range(ncols):
                            ax = axes_2d[row][col]

                            if mc_runs == 25 and mode == 'sma':
                                linestyle = 'solid'

                            label = str(int(mc_runs)) + ' MC Runs'
                            if scenario == 'latin' and row == 0:
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{label} - {mode_label}", solid_capstyle='projecting', capsize=capsize[col], linestyle= linestyle, color=cmap(_mc))

                                if col == 0:
                                    ax.set_ylabel('MAE\n(Latin Hypercube)', fontsize=self.axis_label_fontsize)

                            if scenario == 'convex' and row == 1:
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{label} - {mode_label}", solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle, color=cmap(_mc))

                                if col == 0:
                                    ax.set_ylabel('MAE\n(Convex Hull)', fontsize=self.axis_label_fontsize)

                            if row == 1 and col == 1:
                                ax.legend(prop={'size': 8})

        plt.savefig(os.path.join(self.fig_directory, 'uncertainty_options.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def atmosphere(self):
        df_error, df_uncer = figures.merge_sma_mesma(self)

        df_atmos = pd.read_csv(os.path.join(self.fig_directory, "atmosphere_error_report.csv"))
        df_atmos = df_atmos.loc[df_atmos['mode'] != 'sma-best'].copy()

        # call the optimization data; mc = 25, em = 30,
        df_uncer_opt = df_uncer.loc[(df_uncer['normalization'] == 'Brightness') & (df_uncer['num_em'] == 30) &
                                    (df_uncer['mc_runs'] == 25) & (df_uncer['scenario'] == 'convex')
                                    & (df_uncer['dims'] == 4)].copy()

        # call the uncertainty data; mc = 25, em =30
        df_optimized_sma = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['num_em'] == 30) &
                                    (df_error['mc_runs'] == 25) & (df_error['scenario'] == 'convex')
                                    & (df_error['dims'] == 4)].copy()

        df_optimized_mesma = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['cmbs'] == 100) &
                                        (df_error['mc_runs'] == 25) & (df_error['scenario'] == 'convex')
                                        & (df_error['dims'] == 4)].copy()

        # create 3x3 figure
        ncols = 3  # for each em
        nrows = 2  # top row zenith, middle water, aod; x-axis solar angle

        fig, axs = plt.subplots(nrows, ncols, figsize=(self.fig_width, self.fig_height))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        df_select = df_atmos.loc[(df_atmos['azimuth'] == 0) & (df_atmos['sensor_zenith'] == 180) & (df_atmos['aod'] != 0) & (df_atmos['h2o'] != 0)].copy()

        # cmap
        num_lines = 9
        cmap = plt.cm.get_cmap(self.cmap_kw, num_lines)

        for row in range(nrows):
            for col in range(ncols):
                ax = axs[row, col]

                ax.grid('on', linestyle='--')
                ax.set_ylim(0, 0.22)
                ax.set_xlim(0, 60)
                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect(1. / ax.get_data_ratio())
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                capsize = [8, 6, 4]

                if row == 0:  # aod
                    ax.set_title(self.ems[col].title(), fontsize=self.title_fontsize)

                    count = 0
                    for _aod, aods in enumerate(sorted(list(df_select.aod.unique()))):
                        for mode in df_error['mode'].unique():

                            if mode == 'sma':
                                linestyle = 'dashed'
                                label_mode = "SMA"
                                df_optimized = df_optimized_sma

                            elif mode == 'mesma':
                                linestyle = 'dotted'
                                label_mode = "MESMA"
                                df_optimized = df_optimized_mesma

                            df_results1 = df_select.loc[(df_select['aod'] == aods) & (df_select['h2o'] == 0.75) & (df_select['mode'] == mode)].copy()
                            df_results1 = df_results1.sort_values('solar_zenith')

                            if col == 0:
                                ax.set_ylabel('MAE', fontsize=self.axis_label_fontsize)
                                plot_error = df_results1.npv_mae
                                plot_uncer = df_results1['npv_sma-uncertainty']

                            elif col == 1:
                                plot_error = df_results1.pv_mae
                                plot_uncer = df_results1['pv_sma-uncertainty']

                            else:
                                plot_error = df_results1.soil_mae
                                plot_uncer = df_results1['soil_sma-uncertainty']

                            count += 1
                            ax.errorbar(df_results1.solar_zenith, plot_error, yerr=plot_uncer / 25,
                                        label=f'AOD: {aods:.2f} - {label_mode}', solid_capstyle='projecting', capsize=capsize[_aod], linestyle=linestyle, color=cmap(count))

                    for mode in df_error['mode'].unique():
                        if mode == 'sma':
                            label_mode = "SMA"
                            df_optimized = df_optimized_sma
                            linestyle = 'solid'
                        elif mode == 'mesma':
                            label_mode = "MESMA"
                            linestyle = 'dashed'
                            df_optimized = df_optimized_mesma

                        if col == 0:
                            optimized_error = df_optimized['npv_mae']
                            optimized_uncex = df_uncer_opt['npv_uncer']

                        elif col == 1:
                            optimized_error = df_optimized['pv_mae']
                            optimized_uncex = df_uncer_opt['pv_uncer']

                        else:
                            optimized_error = df_optimized['soil_mae']
                            optimized_uncex = df_uncer_opt['soil_uncer']

                        count += 1
                        ax.errorbar(df_results1.solar_zenith,
                                    [optimized_error.values[0]] * df_results1.solar_zenith.shape[0],
                                    yerr=optimized_uncex / 25, solid_capstyle='projecting', capsize=2,
                                    label=f'{label_mode} (no atmosphere)', linestyle=linestyle,
                                    color=cmap(count))

                        ax.set_xticklabels([])

                    if col == 1:
                        ax.legend(prop={'size': 8}, fontsize=self.axis_label_fontsize, title='H$_2$O (g/cm$^2$): 0.75')

                if row == 1:  # h20
                    ax.set_xlabel("Solar Zenith Angle (°)", fontsize=self.axis_label_fontsize)

                    count = 0
                    for _h2o, h2os in enumerate(sorted(list(df_select.h2o.unique()))):

                        for mode in df_error['mode'].unique():

                            if mode == 'sma':
                                linestyle = 'dashed'
                                label_mode = "SMA"
                                df_optimized = df_optimized_sma

                            elif mode == 'mesma':
                                linestyle = 'dotted'
                                label_mode = "MESMA"
                                df_optimized = df_optimized_mesma

                            df_results = df_select.loc[(df_select['h2o'] == h2os) & (df_select['aod'] == 0.05) & (df_select['mode'] == mode)].copy()
                            df_results = df_results.sort_values('solar_zenith')
                            if col == 0:
                                ax.set_ylabel('MAE', fontsize=self.axis_label_fontsize)
                                plot_error = df_results.npv_mae
                                plot_uncer = df_results['npv_sma-uncertainty']

                            elif col == 1:
                                plot_error = df_results.pv_mae
                                plot_uncer = df_results['pv_sma-uncertainty']

                            else:
                                plot_error = df_results.soil_mae
                                plot_uncer = df_results['soil_sma-uncertainty']

                            count += 1
                            ax.errorbar(df_results.solar_zenith, plot_error, yerr=plot_uncer / 25,
                                        label=f'H$_2$O (g/cm$^2$) : {h2os:.2f} - {label_mode}', solid_capstyle='projecting', capsize=capsize[_aod], linestyle=linestyle, color=cmap(count))

                    for mode in df_error['mode'].unique():
                        if mode == 'sma':
                            label_mode = "SMA"
                            df_optimized = df_optimized_sma
                            linestyle = 'solid'
                        elif mode == 'mesma':
                            label_mode = "MESMA"
                            linestyle = 'dashed'
                            df_optimized = df_optimized_mesma

                        if col == 0:
                            optimized_error = df_optimized['npv_mae']
                            optimized_uncex = df_uncer_opt['npv_uncer']

                        elif col == 1:
                            optimized_error = df_optimized['pv_mae']
                            optimized_uncex = df_uncer_opt['pv_uncer']

                        else:
                            optimized_error = df_optimized['soil_mae']
                            optimized_uncex = df_uncer_opt['soil_uncer']

                        count += 1
                        ax.errorbar(df_results.solar_zenith,
                                    [optimized_error.values[0]] * df_results.solar_zenith.shape[0],
                                    yerr=optimized_uncex / 25, solid_capstyle='projecting', capsize=2,
                                    label=f'{label_mode} (no atmosphere)', linestyle=linestyle, color=cmap(count))

                    # add legend
                    if col == 1:
                        ax.legend(prop={'size': 8}, fontsize=self.axis_label_fontsize, title='AOD: 0.05')

                if col != 0:
                    ax.set_yticklabels([])
        plt.savefig(os.path.join(self.fig_directory, "atmosphere_error_sun_angles.png"), bbox_inches="tight", dpi=400)






    def combinations_figure(self):
        df_error = pd.read_csv(os.path.join(self.fig_directory, 'mesma_unmix_error_report.csv'))
        df_error = df_error.replace('brightness', "Brightness")
        df_error = df_error.replace('none', 'Default')

        df_uncer = pd.read_csv(os.path.join(self.fig_directory, 'mesma_unmix_uncertainty_report.csv'))
        df_uncer = df_uncer.replace('brightness', "Brightness")

        # create 3x2 figure
        ncols = 3  # for each em
        nrows = 2

        fig, axs = plt.subplots(nrows, ncols, figsize=(self.fig_width, self.fig_height))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        for row in range(nrows):
            for col in range(ncols):
                ax = axs[row, col]
                ax.grid('on', linestyle='--')
                ax.set_ylim(0, 0.22)
                ax.set_xlim(1, 6.25)

                if row == 0:
                    ax.set_title(self.ems[col].title(), fontsize=self.title_fontsize)

                if row == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect(1. / ax.get_data_ratio())
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if col != 0:
                    ax.set_yticklabels([])

        # color gradient: warm
        # get number of lines
        num_lines = 6
        cmap = plt.cm.get_cmap(self.cmap_kw, num_lines)

        # filter the data by normalization
        for _em, em in enumerate(sorted(list(df_error.cmbs.unique()))):
            df_select = df_error.loc[(df_error['normalization'] == 'Brightness') & (df_error['cmbs'] == em) & (
                        df_error['mc_runs'] == 5)].copy()
            df_select_unc = df_uncer.loc[(df_uncer['normalization'] == 'Brightness') & (df_uncer['cmbs'] == em) & (
                        df_uncer['mc_runs'] == 5)].copy()

            # filter by scenario
            for scenario in df_select.scenario.unique():
                df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                df_select1 = df_select1.sort_values('dims')  # sort by dimensions, lower to greater

                df_select_unc1 = df_select_unc.loc[(df_select_unc['scenario'] == scenario)].copy()
                df_select_unc1 = df_select_unc1.sort_values('dims')  # sort by dimensions, lower to greater

                error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                capsize = [8, 6, 4]

                if em == 100:
                    linestyle = 'solid'
                elif em == 10:
                    linestyle = 'dashed'
                elif em == 25:
                    linestyle = 'dotted'
                elif em == 50:
                    linestyle = 'dashed'
                elif em == 500:
                    linestyle = 'dotted'
                elif em == 1000:
                    linestyle = 'dashed'

                # loop thorough figure and plot
                for row in range(nrows):
                    for col in range(ncols):
                        ax = axs[row, col]

                        label = str(int(em)) + ' CMBS'
                        if scenario == 'latin' and row == 0:
                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], color=cmap(_em), linestyle=linestyle)

                        if scenario == 'convex' and row == 1:
                            ax.errorbar(df_select1.dims, df_select1[error_mae[col]],
                                        yerr=df_select1[error_options[col]],
                                        label=label, solid_capstyle='projecting', capsize=capsize[col], color=cmap(_em), linestyle=linestyle)

                        if row == 1 and col == 1:
                            ax.legend()

        plt.savefig(os.path.join(self.fig_directory, 'mesma_cmbs_selection_figure.png'), bbox_inches="tight", dpi=400)
        plt.clf()
        plt.close()

    def normalization_figure(self):
        df_error, df_uncer = figures.merge_sma_mesma(self)

        # create 3x2 figure
        fig = plt.figure(constrained_layout=True, figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.001, hspace=0.0001, figure=fig)

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.grid('on', linestyle='--')
                ax.set_ylim(0, 0.22)
                ax.set_xlim(1, 6.25)

                if row == 0:
                    ax.set_title(self.ems[col].title(), fontsize=self.title_fontsize)

                if row == 1:
                    ax.set_xlabel("Principal Components", fontsize=self.axis_label_fontsize)

                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect(1. / ax.get_data_ratio())
                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                if col != 0:
                    ax.set_yticklabels([])

        axes_2d = fig.get_axes()
        axes_2d = [axes_2d[i:i + ncols] for i in range(0, len(axes_2d), ncols)]
        color_guide = {"Brightness": 'green', "1500 nm": "red", "No Normalization": "blue"}

        # color gradient: warm
        num_lines = 6
        cmap = plt.cm.get_cmap(self.cmap_kw, num_lines)

        # filter the data
        for _norm_option, norm_option in enumerate(sorted(list(df_error.normalization.unique()))):

            for mode in df_error['mode'].unique():
                if mode == 'sma':
                    df_select = df_error.loc[(df_error['normalization'] == norm_option) & (df_error['num_em'] == 30) & (df_error['mc_runs'] == 25)].copy()
                    df_select_unc = df_uncer.loc[(df_uncer['normalization'] == norm_option) & (df_uncer['num_em'] == 30) & (df_uncer['mc_runs'] == 25)].copy()
                    label = 'SMA'
                    linestyle = 'dashed'
                if mode == 'mesma':
                    df_select = df_error.loc[(df_error['normalization'] == norm_option) & (df_error['cmbs'] == 100) & (df_error['mc_runs'] == 25)].copy()
                    df_select_unc = df_uncer.loc[(df_uncer['normalization'] == norm_option) & (df_uncer['cmbs'] == 100) & (df_uncer['mc_runs'] == 25)].copy()
                    label = "MESMA"
                    linestyle = 'dotted'

                # filter by scenario
                for scenario in df_select.scenario.unique():
                    df_select1 = df_select.loc[(df_select['scenario'] == scenario)].copy()
                    df_select1 = df_select1.sort_values('dims') # sort by dimensions, lower to greater

                    df_select_unc1 = df_select_unc.loc[(df_select_unc['scenario'] == scenario)].copy()
                    df_select_unc1 = df_select_unc1.sort_values('dims')  # sort by dimensions, lower to greater

                    error_options = ['npv_stde', 'pv_stde', 'soil_stde']
                    error_mae = ['npv_mae', 'pv_mae', 'soil_mae']
                    capsize = [8, 6, 4]

                    # loop thorough figure and plot
                    for row in range(nrows):
                        for col in range(ncols):
                            ax = axes_2d[row][col]

                            if scenario == 'latin' and row == 0:
                                if norm_option == 'Brightness' and mode == 'sma':
                                    linestyle ='solid'

                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{norm_option} - {label}", solid_capstyle='projecting', capsize=capsize[col], linestyle= linestyle, color=cmap(_norm_option))

                                if col == 0:
                                    ax.set_ylabel('MAE\n(Latin Hypercube)', fontsize=self.axis_label_fontsize)

                            if scenario == 'convex' and row == 1:
                                if norm_option == 'Brightness' and mode == 'sma':
                                    linestyle ='solid'
                                ax.errorbar(df_select1.dims, df_select1[error_mae[col]], yerr=df_select1[error_options[col]],
                                            label=f"{norm_option} - {label}", solid_capstyle='projecting', capsize=capsize[col], linestyle=linestyle,  color=cmap(_norm_option))
                                if col == 0:
                                    ax.set_ylabel('MAE\n(Convex Hull)', fontsize=self.axis_label_fontsize)

                            if row == 1 and col == 1:
                                ax.legend(prop={'size': 8})

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

        fig, axs = plt.subplots(nrows, ncols, figsize=(self.fig_width, self.fig_height))
        fig.subplots_adjust(wspace=0.02, hspace=0)

        for row in range(nrows):
            for col in range(ncols):
                ax = axs[col]
                ax.grid('on', linestyle='--')
                ax.set_xlabel("Principal Component 1", fontsize=self.axis_label_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=self.major_axis_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=self.minor_axis_fontsize)
                ax.set_aspect(1. / ax.get_data_ratio())
                ax.set_ylim(-3,3)
                ax.set_xlim(-7,7)

                ax.scatter(pc_array[:, 0], pc_array[:, 1], s=8)

                if col == 0:
                    ax.set_title('Latin Hypercube', fontsize=self.title_fontsize)
                    ax.set_ylabel("Principal Component 2", fontsize=self.axis_label_fontsize)

                    # get the latin hypercube
                    cubes = spectra.latin_hypercubes(points=pc_array, get_quadrants_index=False)
                    colors = ['blue', 'green', 'orange', 'magenta']

                    for _cube, cube in enumerate(cubes):
                        ax.scatter(cube[:, 0], cube[:, 1], s=8, c=colors[_cube])

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
                        ax.plot(pc_array[simplex, 0], pc_array[simplex, 1], 'c')

                    # plot the vertices
                    ax.plot(pc_array[ch.vertices, 0], pc_array[ch.vertices, 1], 'o', mec='r', color='none', lw=1,
                            markersize=14)

        plt.savefig(os.path.join(self.fig_directory, 'em_reduction_visualization.png'), bbox_inches="tight",
                    dpi=400)
        plt.clf()
        plt.close()


def run_figures(base_directory, sensor):
    base_directory = base_directory
    sensor = sensor
    major_axis_fontsize = 10
    minor_axis_fontsize = 10
    title_fontsize = 12
    axis_label_fontsize = 10
    fig_height = 12
    fig_width = 12
    linewidth = 1.5
    sig_figs = 3

    fig_class = figures(base_directory=base_directory, sensor=sensor, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs)

    #fig_class.em_reduction_visulatization()
    fig_class.normalization_figure()
    fig_class.size_endmembers_figure()
    fig_class.combinations_figure()
    fig_class.uncertainty_figure()
    fig_class.atmosphere()
