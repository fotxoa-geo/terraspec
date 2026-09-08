from glob import glob
import os
from p_tqdm import p_umap, p_map
import pandas as pd
from utils.create_tree import create_directory
from slpit.figures import fraction_file_info


class Tables:
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
                        **{"desc": "\t\t retrieving mean fractional cover: ...", "ncols": 150})
        df_all = pd.DataFrame(results)
        df_all.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization', 'rows', 'cols', 'duplicate_flag', 'npv', 'pv', 'soil', 'shade', 'npv_r', 'pv_r', 'soil_r', 'shade_r','npv_se', 'pv_se', 'soil_se', 'shade_se', 'npv_sigma', 'pv_sigma', 'soil_sigma', 'shade_sigma', 'npv_use', 'pv_use', 'soil_use', 'shade_use']
        df_all.to_csv(os.path.join(self.fig_directory, 'fraction_output.csv'), index=False)
        
def run_tables(base_directory):
    tb = Tables(base_directory=base_directory)
    tb.error_tables()
