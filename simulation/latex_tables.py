import pandas as pd
import numpy as np
import os


class latex:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")

        # em_labels
        self.ems = ['non-photosynthetic\nvegetation', 'photosynthetic\nvegetation', 'soil']
        self.ems_short = ['npv', 'pv', 'soil']

    def optimal_parameters(self, mode:str):
        table = os.path.join(self.fig_directory, mode + "_unmix_error_report.csv")
        df = pd.read_csv(table)
        df = df.replace('convex', "Convex Hull")
        df = df.replace('latin', 'Latin Hypercube')
        df = df.replace('brightness', 'Brightness')
        df_select = df.loc[(df['normalization'] == 'Brightness') & (df['num_em'] == 30) & (df['mc_runs'] == 25)].copy()
        df_select = df_select.sort_values('dims')
        df_select = df_select.sort_values('scenario', ascending=False)

        # combine mae(rmse) for table
        df_select['combined_npv'] = df_select['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select['npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
        df_select['combined_pv'] = df_select['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select['pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
        df_select['combined_soil'] = df_select['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select['soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'

        # remove reduntant information
        cols = df_select.columns.tolist()
        to_remove = cols[5:17]
        df_select = df_select.loc[:, ~df_select.columns.isin(to_remove)]
        df_select = df_select[['scenario', 'dims', 'normalization', 'num_em', 'mc_runs', 'combined_npv', 'combined_pv', 'combined_soil']]
        df_select = df_select.astype({'num_em': 'int', 'mc_runs': 'int'})
        print(df_select.to_latex(index=False))

    def atmosphere_table(self):
        table = os.path.join(self.fig_directory, "atmosphere_error_report.csv")
        df = pd.read_csv(table)
        df = df.replace('convex', "Convex Hull")
        df = df.replace('latin', 'Latin Hypercube')
        df = df.replace('brightness', 'Brightness')
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

            df_select = df_select[['aod', 'h2o', 'solar_zenith', 'combined_npv', 'combined_pv', 'combined_soil']]
            df_select.insert(0, 'Scenario', atmosphere_scenarios[_table])
            print(df_select.to_latex(index=False))

    def summary_table(self):
        df_unmix = pd.read_csv(os.path.join(self.fig_directory, "unmix_error_report.csv"))
        df_uncertainty = pd.read_csv(os.path.join(self.fig_directory, "unmix_uncertainty_report.csv"))
        df_atmos = pd.read_csv(os.path.join(self.fig_directory, "atmosphere_error_report.csv"))
        df_atmos.solar_zenith = df_atmos.solar_zenith.round()
        df_atmos['solar_zenith'] = df_atmos['solar_zenith'].astype('int')

        rows = []

        df_select_unmix = df_unmix.loc[(df_unmix['normalization'] == 'brightness') &
                                       (df_unmix['num_em'] == 30) & (df_unmix['mc_runs'] == 25) &
                                       (df_unmix['scenario'] == 'convex') & (df_unmix['dims'] == 4)].copy()
        # get mae from unmix
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

        rows.append(['No Atmosphere',df_select_unmix.combined_npv.values[0], np.round(df_select_uncer.npv_uncer.values[0],2),
                     df_select_unmix.combined_pv.values[0], np.round(df_select_uncer.pv_uncer.values[0],2),
                     df_select_unmix.combined_soil.values[0], np.round(df_select_uncer.soil_uncer.values[0],2)])

        # get atmospheric runs
        atmopspheres = [[0.05, 0.75], [0.4, 4.0]]
        atmosphere_scenarios = ['Clear Atmosphere and Low Water Content', 'Non-Clear Atmosphere and High Water Content']

        for _atmos, atmos in enumerate(atmopspheres):
            df_select = df_atmos.loc[(df_atmos['aod'] == atmos[0]) & (df_atmos['h2o'] == atmos[1]) & (df_atmos['solar_zenith'] == 13)].copy()
            # combine mae(rmse) for table
            df_select['combined_npv'] = df_select['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_select['combined_pv'] = df_select['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            df_select['combined_soil'] = df_select['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_select[
                'soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            rows.append([atmosphere_scenarios[_atmos], df_select.combined_npv.values[0], np.round(df_select['npv_sma-uncertainty'].values[0],2),
                         df_select.combined_pv.values[0], np.round(df_select['pv_sma-uncertainty'].values[0],2),
                         df_select.combined_soil.values[0], np.round(df_select['soil_sma-uncertainty'].values[0],2)])

        df_table = pd.DataFrame(rows)
        print(df_table.to_latex(index=False))


def run_latex_tables(base_directory: str):
    latex_class = latex(base_directory=base_directory)
    latex_class.optimal_parameters(mode='sma-best')
    latex_class.atmosphere_table()
    latex_class.summary_table()