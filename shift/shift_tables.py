import os
from glob import glob
import pandas as pd
from p_tqdm import p_umap
from utils.results_utils import performance_log
from utils.create_tree import create_directory
from slpit.figures import fraction_file_info
from p_tqdm import p_umap, p_map

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

    def performance_table(self):
        outfiles = glob(os.path.join(self.output_directory, 'outlogs', '*.out'))

        dfs = p_umap(performance_log, outfiles,
                     **{"desc": f"\t\t processing performance tables...", "ncols": 150})

        df_performance = pd.concat(dfs)
        df_performance['library'] = df_performance['output_file_base'].apply(os.path.basename).str.split("_").apply(lambda x: x[0]).str.split('-').apply(lambda x: x[0])

        df_performance.to_csv(os.path.join(self.fig_directory, "shift_computing_performance_report.csv"), index=False)

    def error_tables(self):
        fraction_files_sma = sorted(glob(os.path.join(self.output_directory, 'sma', '*fractional_cover')))
        fraction_files_mesma = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover')))
        all_files = fraction_files_sma + fraction_files_mesma

        results = p_map(fraction_file_info, all_files,** {"desc": "\t\t retrieving mean fractional cover: ...",
                        "ncols": 150})

        df_all = pd.DataFrame(results)

        df_all.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization',
                                                 'npv', 'pv', 'soil', 'shade']

        df_all.to_csv(os.path.join(self.fig_directory, 'shift_fraction_output.csv'), index=False)


def run_tables(base_directory):
    tb = tables(base_directory=base_directory)
    tb.performance_table()
    tb.error_tables()