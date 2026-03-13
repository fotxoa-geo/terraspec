import os
from glob import glob
import pandas as pd
from utils.create_tree import create_directory
from p_tqdm import p_umap, p_map
import numpy as np
from utils.envi import envi_to_array


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

    if unmix_mode == 'mesma':
        num_cmb_em = 100
    else:
        num_cmb_em = 20

    library_mode = name.split("_")[0]
    instrument = name.split("_")[3]
    plot = f'{name.split("_")[1]}_{name.split("_")[2]}'

    num_mc = 25
    normalization = name.split("_")[-4]
    fraction_array = envi_to_array(fraction_file)

    unc_path = os.path.join(f'{fraction_file}_uncertainty')
    unc_array = envi_to_array(unc_path)

    mean_fractions = []
    mean_se = []
    mean_sigma = []
    mean_use = []

    for _band, band in enumerate(range(0, fraction_array.shape[2])):

        if instrument == 'SLPIT':
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
        se = np.nanmean(selected_unc / np.sqrt(int(25)))
        mean_se.append(se)

        # caluclate mean sigma
        sigma = np.nanmean(selected_unc)
        mean_sigma.append(sigma)

        # calculate U_se
        sstd = selected_unc.flatten()
        sum_square_sstd = np.nansum(np.square(sstd))
        use = np.sqrt(sum_square_sstd) / sstd.shape[0]
        mean_use.append(use)

    return [instrument, unmix_mode, plot, library_mode, int(num_cmb_em), int(num_mc), normalization,
            fraction_array.shape[0], fraction_array.shape[1],
            duplicate_flag] + mean_fractions + mean_se + mean_sigma + mean_use


class tables:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")
        # check for figure directory
        create_directory(self.fig_directory)

        # em_labels
        self.ems = ['non-photosynthetic\nvegetation', 'photosynthetic\nvegetation', 'soil']
        self.ems_short = ['npv', 'pv', 'soil']


    def error_tables(self):
        fraction_files = sorted(glob(os.path.join(self.output_directory, 'spectral_transects', '**', '*fractional_cover'), recursive=True))

        results = p_map(fraction_file_info, fraction_files,
                        ** {"desc": "\t\t retrieving mean fractional cover: ...", "ncols": 150})

        df_all = pd.DataFrame(results)
        df_all.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization', 'rows', 'cols', 'duplicate_flag', 'npv', 'pv', 'soil', 'shade', 'npv_se', 'pv_se', 'soil_se', 'shade_se', 'npv_sigma', 'pv_sigma', 'soil_sigma', 'shade_sigma', 'npv_use', 'pv_use', 'soil_use', 'shade_use']
        df_all.to_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'), index=False)


def run_tables(base_directory):
    tb = tables(base_directory=base_directory)
    tb.error_tables()
