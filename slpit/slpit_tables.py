from glob import glob
import os
from p_tqdm import p_umap
import pandas as pd
from utils.results_utils import performance_log
from utils.create_tree import create_directory

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
        df_performance['library'] = df_performance['endmember_file'].apply(os.path.basename).str.split("___").apply(lambda x: x[0]).str.split('-').apply(lambda x: x[1])

        df_performance.to_csv(os.path.join(self.fig_directory, "computing_performance_report_emit.csv"), index=False)


def run_tables(base_directory):
    tb = tables(base_directory=base_directory)
    tb.performance_table()
