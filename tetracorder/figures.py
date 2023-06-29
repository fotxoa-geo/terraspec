import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from utils.envi import envi_to_array, load_band_names
import os


def bin_sums(x, y):
    mae = []
    x_vals = []
    for i in np.linspace(0.0, 1.0, 21):
        mask_1 = x >= i
        mask_2 = x <= i + 0.05
        mask_3 = np.logical_and(mask_1, mask_2)
        y_select = y[mask_3]
        y_select = y_select[~np.isnan(y_select)]

        if i == 1.0:
            continue
        else:
            mae_calc = np.mean(y_select)
            mae.append(mae_calc)
            mid_point = (i + (i + 0.05) / 2)
            x_vals.append(mid_point)

    return x_vals, mae


class tetracorder_figures:
    def __init__(self, base_directory: str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'tetracorder', 'output')
        self.sim_output = os.path.join(base_directory, 'simulation', 'output')
        self.slpit_output = os.path.join(base_directory, 'slpit', 'output')
        self.sa_outputs = os.path.join(base_directory, 'tetracorder', 'output', 'spectral_abundance')
        self.fig_directory = os.path.join(base_directory, 'tetracorder', 'figures')

        self.bands = load_band_names(os.path.join(self.sa_outputs, 'simulated_soil_sa_mineral'))

    def simulation_fig(self, xaxis:str):

        # load simulation data - truncate the sa files from augmentation; no unmix here!
        sim_index_array = envi_to_array(os.path.join(self.sim_output, 'convex_hull__n_dims_4_index'))
        sim_fractions_array = envi_to_array(os.path.join(self.sim_output, 'convex_hull__n_dims_4_fractions'))
        sim_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_spectra_sa_mineral'))[:, 0, :]
        soil_sa_sim_pure = envi_to_array(os.path.join(self.sa_outputs, 'convex_hull__n_dims_4_simulation_library_sa_mineral'))[:, 0, :]

        # correct spectral abundance- y dimension is the minerals
        error_grid = np.zeros((np.shape(sim_sa_arrary)[0], np.shape(sim_sa_arrary)[1]))

        for _row, row in enumerate(sim_sa_arrary):
            soil_fractions = sim_fractions_array[_row, 0, 2]

            if soil_fractions >= 0.99:
                for _mineral, mineral in enumerate(row):

                    if sim_sa_arrary[_row, _mineral] == 0:
                        sim_sa_arrary[_row, _mineral] = np.nan

                    if soil_sa_sim_pure[_row, _mineral] == 0:
                        soil_sa_sim_pure[_row, _mineral] = np.nan

            soil_index = sim_index_array[_row, 0, 2]
            sa_c = sim_sa_arrary[_row, :]/soil_fractions
            error = np.absolute(sa_c - soil_sa_sim_pure[int(soil_index), :])
            error_grid[_row, :] = error

        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        ncols = 5
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0, hspace=0, figure=fig)

        counter = 0
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(self.bands[counter])
                ax.set_xlabel(f'{xaxis}')
                ax.set_ylabel('MAE')
                abs_error = error_grid[:, counter]
                if xaxis == 'npv':
                    fractions = sim_fractions_array[:, 0, 0]

                if xaxis == 'pv':
                    fractions = sim_fractions_array[:, 0, 1]

                if xaxis == 'soil':
                    fractions = sim_fractions_array[:, 0, 2]

                x_vals, mae = bin_sums(x=fractions, y=abs_error)
                ax.plot(x_vals, mae)
                ax.set_ylim(0.0, 0.3)
                ax.set_xlim(-0.01, 1.05)
                ax.set_aspect(1. / ax.get_data_ratio())

                counter += 1
        plt.savefig(os.path.join(self.fig_directory, 'tetracorder_mae_' + xaxis + '.png'), dpi=300, bbox_inches='tight')


def run_figure_workflow(base_directory):
    ems = ['npv', 'pv', 'soil']
    tc = tetracorder_figures(base_directory=base_directory)
    for em in ems:
        tc.simulation_fig(xaxis=em)