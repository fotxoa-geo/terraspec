import time

import pandas as pd
import numpy as np
import os
import ast
from sklearn.metrics import r2_score


# Custom formatter for two decimal places
def format_float(val):
    if isinstance(val, (int, float)):
        return "{:.2f}".format(val)
    return val

class latex:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")

        # em_labels
        self.ems = ['non-photosynthetic\nvegetation', 'photosynthetic\nvegetation', 'soil']
        self.ems_short = ['npv', 'pv', 'soil']

    def optimal_parameters(self):
        sma_table = os.path.join(self.fig_directory, "sma-best_unmix_error_report.csv")
        mesma_table = os.path.join(self.fig_directory, "mesma_unmix_error_report.csv")
        df_sma = pd.read_csv(sma_table)
        df_sma.insert(0, 'mode', 'SMA')
        df_mesma = pd.read_csv(mesma_table)
        df_mesma.insert(0, 'mode', 'MESMA')
        df = pd.concat([df_sma, df_mesma], ignore_index=True)

        df = df.replace('convex', "Convex Hull")
        df = df.replace('latin', 'Latin Hypercube')
        df = df.replace('brightness', 'Brightness')

        # select parameters for mesma and sma
        df_select = df.loc[(df['normalization'] == 'Brightness') & (df['mc_runs'] == 25)].copy()
        df_select = df_select.loc[(df['num_em'] == 30) | (df['cmbs'] == 100)].copy()
        df_select['cmbs'] = df_select['cmbs'].fillna(1)
        df_select['num_em'] = df_select['num_em'].fillna(3)

        for i in ["SMA", "MESMA"]:
            df_mode = df_select.loc[df_select['mode'] == i].copy()
            df_mode = df_mode.sort_values('dims')
            df_mode = df_mode.sort_values('scenario', ascending=False)

            # combine mae(rmse) for table
            df_mode['combined_npv'] = df_mode['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mode['npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_mode['combined_pv'] = df_mode['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mode['pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_mode['combined_soil'] = df_mode['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mode['soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'

            # remove reduntant information
            cols = df_mode.columns.tolist()
            to_remove = cols[7:22]
            df_mode = df_mode.loc[:, ~df_mode.columns.isin(to_remove)]

            df_mode = df_mode[['mode', 'scenario', 'dims', 'cmbs', 'normalization', 'num_em', 'mc_runs', 'combined_npv', 'combined_pv', 'combined_soil']]
            df_mode = df_mode.astype({'num_em': 'int', 'mc_runs': 'int', 'cmbs': 'int'})
            print(df_mode.to_latex(index=False))

    def baseline_setings(self):
        sma_table = os.path.join(self.fig_directory, "sma-best_unmix_error_report.csv")
        mesma_table = os.path.join(self.fig_directory, "mesma_unmix_error_report.csv")
        df_sma = pd.read_csv(sma_table)
        df_sma.insert(0, 'mode', 'SMA')
        df_mesma = pd.read_csv(mesma_table)
        df_mesma.insert(0, 'mode', 'MESMA')
        df = pd.concat([df_sma, df_mesma], ignore_index=True)

        df = df.replace('convex', "Convex Hull")
        df = df.replace('latin', 'Latin Hypercube')
        df = df.replace('brightness', 'Brightness')

        # select parameters for mesma and sma
        df_select = df.loc[(df['normalization'] == 'none') & (df['mc_runs'].isnull())].copy()
        df_select = df_select.loc[(df_select['cmbs'] == 10) | (df_select['cmbs'].isnull())].copy()
        df_select = df_select.loc[(df_select['num_em'].isnull())].copy()

        modes = ['SMA', "MESMA"]
        scenarios = ['Latin Hypercube', 'Convex Hull']
        for i in modes:
            df_mode = df_select.loc[(df_select['mode'] == i)].copy()

            for scenario in scenarios:
                df_scenario = df_mode.loc[(df_mode['scenario'] == scenario)].copy()

                y_vars = ['npv_mae', 'pv_mae', 'soil_mae']

                for y_var in y_vars:
                    x = df_scenario['dims']
                    y = df_scenario[y_var]
                    print(i, scenario, y_var, 'r$^2$ = ', np.round(r2_score(x, y),2))

    def atmosphere_table(self):
        table = os.path.join(self.fig_directory, "atmosphere_error_report.csv")
        df = pd.read_csv(table)
        df = df.replace('convex', "Convex Hull")
        df = df.replace('latin', 'Latin Hypercube')
        df = df.replace('brightness', 'Brightness')
        df = df.replace('sma-best', "SMA")
        df = df.replace('mesma', "MESMA")
        df.solar_zenith = df.solar_zenith.round()
        df['solar_zenith'] = df['solar_zenith'].astype('int')

        atmospheres_to_plot = [[0.05, 0.75], [0.05, 4.0], [0.4, 0.75], [0.4, 4.0]]
        atmosphere_scenarios = ['Clear Atmosphere and Low Water Content', 'Clear Atmosphere and High Water Content',
                                'Non-Clear Atmosphere and Low Water Content', 'Non-Clear Atmosphere and High Water Content']

        for _table, table in enumerate(atmospheres_to_plot):
            df_select = df.loc[(df['aod'] == table[0]) & (df['h2o'] == table[1])].copy()
            df_select = df_select.sort_values('solar_zenith',  ascending=False)

            # combine mae(rmse) for table
            df_select['combined_npv'] = df_select['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_select['combined_pv'] = df_select['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_select['combined_soil'] = df_select['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'

            df_select = df_select[['mode','aod', 'h2o', 'solar_zenith', 'combined_npv', 'combined_pv', 'combined_soil']]
            df_select = df_select.sort_values('mode', ascending=False)
            df_select.insert(0, 'Scenario', atmosphere_scenarios[_table])
            formatted_df = df_select.reset_index(drop=True).applymap(format_float)
            formatted_df.loc[1:, 'Scenario'] = ''

            print(formatted_df.to_latex(index=False, escape=False))

    def summary_table(self):
        df_unmix = pd.read_csv(os.path.join(self.fig_directory, "sma-best_unmix_error_report.csv"))
        df_uncertainty = pd.read_csv(os.path.join(self.fig_directory, "sma-best_unmix_uncertainty_report.csv"))
        df_unmix_mesma = pd.read_csv(os.path.join(self.fig_directory, "mesma_unmix_error_report.csv"))
        df_uncertainty_mesma = pd.read_csv(os.path.join(self.fig_directory, "mesma_unmix_uncertainty_report.csv"))
        df_atmos = pd.read_csv(os.path.join(self.fig_directory, "atmosphere_error_report.csv"))
        df_atmos.solar_zenith = df_atmos.solar_zenith.round()
        df_atmos['solar_zenith'] = df_atmos['solar_zenith'].astype('int')

        rows = []

        # sma
        df_select_unmix = df_unmix.loc[(df_unmix['normalization'] == 'brightness') &
                                       (df_unmix['num_em'] == 30) & (df_unmix['mc_runs'] == 25) &
                                       (df_unmix['scenario'] == 'convex') & (df_unmix['dims'] == 4)].copy()
        # get mae from unmix- sma
        df_select_unmix['combined_npv'] = df_select_unmix['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select_unmix[
            'npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
        df_select_unmix['combined_pv'] = df_select_unmix['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select_unmix[
            'pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
        df_select_unmix['combined_soil'] = df_select_unmix['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select_unmix[
            'soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'

        # get uncertainty
        df_select_uncer = df_uncertainty.loc[(df_uncertainty['normalization'] == 'brightness') &
                                       (df_uncertainty['num_em'] == 30) & (df_uncertainty['mc_runs'] == 25) &
                                       (df_uncertainty['scenario'] == 'convex') & (df_uncertainty['dims'] == 4)].copy()

        rows.append(['No Atmosphere-SMA',df_select_unmix.combined_npv.values[0], '± '+ str(np.round(df_select_uncer.npv_uncer.values[0],2)),
                     df_select_unmix.combined_pv.values[0], '± ' + str(np.round(df_select_uncer.pv_uncer.values[0],2)),
                     df_select_unmix.combined_soil.values[0], '± '+ str(np.round(df_select_uncer.soil_uncer.values[0],2))])

        # mesma --------------------------------------------------
        df_mesma = df_unmix_mesma.loc[(df_unmix_mesma['normalization'] == 'brightness') &
                                        (df_unmix_mesma['mc_runs'] == 25) & (df_unmix_mesma['scenario'] == 'convex') &
                                        (df_unmix_mesma['dims'] == 4) & (df_unmix_mesma['cmbs'] == 100)].copy()

        df_mesma['combined_npv'] = df_mesma['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mesma[
            'npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
        df_mesma['combined_pv'] = df_mesma['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mesma[
            'pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
        df_mesma['combined_soil'] = df_mesma['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mesma[
            'soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'

        # get uncertainty mesma
        df_select_uncer_mesma = df_uncertainty_mesma.loc[(df_uncertainty_mesma['normalization'] == 'brightness') &
                                       (df_uncertainty_mesma['cmbs'] == 100) & (df_uncertainty_mesma['mc_runs'] == 25) &
                                       (df_uncertainty_mesma['scenario'] == 'convex') & (df_uncertainty_mesma['dims'] == 4)].copy()

        rows.append(['No Atmosphere-MESMA', df_mesma.combined_npv.values[0], '± '+ str(np.round(df_select_uncer_mesma.npv_uncer.values[0],2)),
                     df_mesma.combined_pv.values[0], '± '+ str(np.round(df_select_uncer_mesma.pv_uncer.values[0],2)),
                     df_mesma.combined_soil.values[0], '± '+ str(np.round(df_select_uncer_mesma.soil_uncer.values[0],2))])

        # get atmospheric runs
        atmopspheres = [[0.05, 0.75], [0.4, 4.0]]
        atmosphere_scenarios = ['Clear Atmosphere and Low Water Content', 'Non-Clear Atmosphere and High Water Content']

        for _atmos, atmos in enumerate(atmopspheres):
            if atmos == 'Clear Atmosphere and Low Water Content':
                sza = 13
            else:
                sza = 57
            df_select = df_atmos.loc[(df_atmos['aod'] == atmos[0]) & (df_atmos['h2o'] == atmos[1]) & (df_atmos['solar_zenith'] == sza)].copy()

            # combine mae(rmse) for table
            df_select['combined_npv'] = df_select['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_select['combined_pv'] = df_select['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_select['combined_soil'] = df_select['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'

            for _row, row in df_select.iterrows():
                rows.append([atmosphere_scenarios[_atmos] + ' - ' + row['mode'].upper(),
                             row.combined_npv, '± '+ str(np.round(row['npv_sma-uncertainty'],2)),
                             row.combined_pv, '± ' + str(np.round(row['pv_sma-uncertainty'],2)),
                             row.combined_soil, '± ' + str(np.round(row['soil_sma-uncertainty'],2))])

        df_table = pd.DataFrame(rows)
        formatted_df = df_table.reset_index(drop=True).applymap(format_float)
        print(formatted_df.to_latex(index=False, escape=False))

    def time_table(self):
        df_time = pd.read_csv(os.path.join(self.fig_directory, "computing_performance_report.csv"))
        df_time.columns = ['reflectance_file', 'endmember_file', 'endmember_class_header', 'output_file_base',
                           'spectral_starting_column', 'truncate_end_columns', 'reflectance_uncertainty_file',
                           'n_mc', 'mode', 'refl_nodata', 'refl_scale', 'normalization', 'combination_type',
                           'max_combinations', 'num_endmembers', 'write_complete_fractions', 'optimizer', 'start_line',
                           'end_line', 'endmember_classes', 'log_file', 'elapsed_time', 'worker_time']

        df_time = df_time.replace('"', '', regex=True)
        df_time['reflectance_file'] = df_time['reflectance_file'].apply(os.path.basename)
        df_time.drop(columns=['endmember_file', 'endmember_class_header', 'output_file_base',
                                'spectral_starting_column', 'truncate_end_columns', 'reflectance_uncertainty_file',
                                'refl_nodata', 'refl_scale', 'write_complete_fractions', 'optimizer', 'start_line',
                              'end_line', 'endmember_classes', 'log_file',], inplace=True)
        df_time['scenario'] = df_time['reflectance_file'].apply(lambda path: os.path.basename(path).split('__')[0])
        df_time['elapsed_time'] = df_time['elapsed_time'].astype(float)
        df_time['num_endmembers'] = df_time['num_endmembers'].apply(lambda x: ast.literal_eval(x)[0])
        values_to_keep = ['latin_hypercube', 'convex_hull']
        filtered_df = df_time.loc[df_time['scenario'].isin(values_to_keep)]
        filtered_df['dims'] = filtered_df['reflectance_file'].apply(lambda path: int(os.path.basename(path).split('_')[-2]))
        df_select = filtered_df.loc[(filtered_df['normalization'] == 'brightness') & (filtered_df['n_mc'] == 25) & (filtered_df['dims'] == 4)].copy()
        df_select = df_select.loc[(df_select['num_endmembers'] == 30) | (df_select['num_endmembers'] == 3)].copy()
        df_select = df_select.loc[(df_select['max_combinations'] == 100) | (df_select['max_combinations'] == -1)].copy()

        filtered_df = df_select.loc[~((df_select['mode'] == 'sma-best') & (df_select['max_combinations'] == -1) & (
                    df_select['num_endmembers'] == 3))]
        filtered_df['elapsed_time'] = filtered_df['elapsed_time'].round(2)

        print(filtered_df)



def run_latex_tables(base_directory: str):
    latex_class = latex(base_directory=base_directory)
    #latex_class.optimal_parameters()
    latex_class.atmosphere_table()
    latex_class.summary_table()
    latex_class.time_table()
    latex_class.baseline_setings()
