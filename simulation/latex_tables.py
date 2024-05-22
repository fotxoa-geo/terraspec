import time

import pandas as pd
import numpy as np
import os
import ast
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from utils.results_utils import r2_calculations

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
        sma_table = os.path.join(self.fig_directory, "sma_unmix_error_report.csv")
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
        df_select = df.loc[(df['normalization'] == 'Brightness') & (df['dims'] == 4)].copy()
        df_select = df_select.loc[(df['num_em'] == 30) | (df['cmbs'] == 100)].copy()
        df_select['cmbs'] = df_select['cmbs'].fillna(1)
        df_select['num_em'] = df_select['num_em'].fillna(3)

        for i in ["SMA", "MESMA"]:
            df_mode = df_select.loc[df_select['mode'] == i].copy()
            df_mode = df_mode.sort_values('mc_runs')
            df_mode = df_mode.sort_values('scenario', ascending=False)

            # combine mae(rmse) for table
            #df_mode['combined_npv'] = df_mode['npv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mode['npv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            #df_mode['combined_pv'] = df_mode['pv_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mode['pv_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            #df_mode['combined_soil'] = df_mode['soil_mae'].apply('{:,.2f}'.format).astype(str) + '(' + df_mode['soil_rmse'].apply('{:,.2f}'.format).astype(str) + ')'
            
            # mc unc avg std error for table 
            df_mode['combined_npv'] =  df_mode['npv_mc_unc'].apply('{:,.3f}'.format).astype(str)
            df_mode['combined_pv'] = df_mode['pv_mc_unc'].apply('{:,.3f}'.format).astype(str)
            df_mode['combined_soil'] = df_mode['soil_mc_unc'].apply('{:,.3f}'.format).astype(str)



            # remove reduntant information
            cols = df_mode.columns.tolist()
            to_remove = cols[7:22]
            df_mode = df_mode.loc[:, ~df_mode.columns.isin(to_remove)]

            df_mode = df_mode[['mode', 'scenario', 'dims', 'cmbs', 'normalization', 'num_em', 'mc_runs', 'combined_npv', 'combined_pv', 'combined_soil']]
            df_mode = df_mode.astype({'num_em': 'int', 'mc_runs': 'int', 'cmbs': 'int'})
            print(df_mode.to_latex(index=False))

    def baseline_setings(self):
        sma_table = os.path.join(self.fig_directory, "sma_unmix_error_report.csv")
        mesma_table = os.path.join(self.fig_directory, "mesma_unmix_error_report.csv")
        sma_uncert_table = os.path.join(self.fig_directory, "sma_unmix_uncertainty_report.csv")
        mesma_uncer_table = os.path.join(self.fig_directory, "mesma_unmix_uncertainty_report.csv")

        df_sma = pd.read_csv(sma_table)
        df_sma.insert(0, 'mode', 'SMA')
        df_mesma = pd.read_csv(mesma_table)
        df_mesma.insert(0, 'mode', 'MESMA')
        df = pd.concat([df_sma, df_mesma], ignore_index=True)

        df_sma_uncer = pd.read_csv(sma_uncert_table)
        df_sma_uncer.insert(0, 'mode', 'SMA')
        df_mesma_uncer = pd.read_csv(mesma_uncer_table)
        df_mesma_uncer.insert(0, 'mode', 'MESMA')
        df_uncer = pd.concat([df_sma_uncer, df_mesma_uncer], ignore_index=True)

        df_uncer = df_uncer.replace('convex', "Convex Hull")
        df_uncer = df_uncer.replace('latin', 'Latin Hypercube')
        df_uncer = df_uncer.replace('brightness', 'Brightness')

        df = df.replace('convex', "Convex Hull")
        df = df.replace('latin', 'Latin Hypercube')
        df = df.replace('brightness', 'Brightness')

        # select parameters for mesma and sma
        df_base = df.loc[((df['normalization'] == 'none') & (df['mc_runs']== 25)) & ((df['cmbs'] == 100) | (df['cmbs'].isnull())) & ((df['num_em']==3) | df['num_em'].isnull())].copy()
        df_opt = df.loc[((df['normalization'] == 'Brightness') & (df['mc_runs']== 25)) & ((df['cmbs'] == 100) | (df['cmbs'].isnull())) & ((df['num_em']==30) | df['num_em'].isnull())].copy()
        modes = ['SMA', "MESMA"]
        scenarios = ['Latin Hypercube', 'Convex Hull']

        for i in modes:
            df_base_mode = df_base.loc[(df_base['mode'] == i)].copy()
            df_opt_mode = df_opt.loc[(df_opt['mode'] == i)].copy()
            #print(f'Baseline {i} MAE :', np.round(np.average(df_base_mode['npv_mae']), 2), np.round(np.average(df_base_mode['pv_mae']),2),
                  #np.round(np.average(df_base_mode['soil_mae']), 2))
            print(f'Optimized {i} MAE :', np.round(np.average(df_opt_mode['npv_mae']), 2),
                  np.round(np.average(df_opt_mode['pv_mae']), 2),
                  np.round(np.average(df_opt_mode['soil_mae']), 2))

        df_sma_num_ems = df.loc[(df['normalization'] == 'Brightness') & (df['mc_runs']== 25) & (df['mode']=='SMA')].copy()
        num_ems = []
        npv_mae = []
        pv_mae = []
        soil_mae = []

        for i in sorted(list(df_sma_num_ems.num_em.unique())):
            df_num_em = df_sma_num_ems.loc[(df_sma_num_ems['num_em'] == i)].copy()
            num_ems.append(i)
            npv_mae.append(np.round(np.average(df_num_em['npv_mae'] ),2))
            pv_mae.append(np.round(np.average(df_num_em['pv_mae'] ), 2))
            soil_mae.append(np.round(np.average(df_num_em['soil_mae'] ), 2))

        print('The relationship between n and MAE for NPV OPT: ', r2_calculations(num_ems, npv_mae))
        print('The relationship between n and MAE for PV OPT: ',  r2_calculations(num_ems, pv_mae))
        print('The relationship between n and MAE for Soil OPT: ',  r2_calculations(num_ems, soil_mae))

        dims = []
        npv_mae = []
        pv_mae = []
        soil_mae = []

        for i in sorted(list(df_base.dims.unique())):
            df_num_em = df_base.loc[(df_base['dims'] == i)].copy()
            dims.append(i)
            npv_mae.append(np.round(np.average(df_num_em['npv_mae']), 2))
            pv_mae.append(np.round(np.average(df_num_em['pv_mae']), 2))
            soil_mae.append(np.round(np.average(df_num_em['soil_mae']), 2))

        print('The relationship between d and MAE for NPV BASE: ', r2_calculations(dims, npv_mae))
        print('The relationship between d and MAE for PV BASE: ', r2_calculations(dims, pv_mae))
        print('The relationship between d and MAE for Soil BASE: ', r2_calculations(dims, soil_mae))
        print('_______________________________________________________________')
        print('\n')
        print('This begins the Brightness adjustments section')
        df_opt_norm = df.loc[(df['mc_runs'] == 25) & ((df['cmbs'] == 100) | (df['cmbs'].isnull())) & ((df['num_em'] == 30) | df['num_em'].isnull())].copy()


        for x in modes:
            df_opt_norm_mode = df_opt_norm.loc[(df_opt_norm['mode'] == x)].copy()

            for norm in sorted(list(df_opt_norm_mode.normalization.unique())):
                df_norm_select = df_opt_norm_mode.loc[(df_opt_norm_mode['normalization'] == norm)].copy()

                print(f'Optimized {x,norm} MAE :', np.round(np.average(df_norm_select['npv_mae']), 2),
                      np.round(np.average(df_norm_select['pv_mae']), 2),
                      np.round(np.average(df_norm_select['soil_mae']), 2))
        print('_______________________________________________________________')
        print('\n')
        print('This begins the Monte Carlo uncertainty sections')
        df_mc = df.loc[((df['normalization'] == 'Brightness')) & ((df['cmbs'] == 100) | (df['cmbs'].isnull())) & ((df['num_em'] == 30) | df['num_em'].isnull())].copy()

        for x in sorted(list(df_mc.mc_runs.unique())):
            dc_mc_select = df_mc.loc[(df_mc['mc_runs'] == x)].copy()

            for i in modes:
                df_mc_mode_select = dc_mc_select.loc[(dc_mc_select['mode'] == i)].copy()
                print(f'Optimized {i} ; MC Runs{x} MAE :', np.round(np.average(df_mc_mode_select['npv_mae']), 2),
                      np.round(np.average(df_mc_mode_select['pv_mae']), 2),
                      np.round(np.average(df_mc_mode_select['soil_mae']), 2))

        df_mc_unc = df_uncer.loc[((df_uncer['normalization'] == 'Brightness')) & ((df_uncer['cmbs'] == 100) | (df_uncer['cmbs'].isnull())) & ((df_uncer['num_em'] == 30) | df_uncer['num_em'].isnull())].copy()
        for x in sorted(list(df_mc_unc.mc_runs.unique())):
            dc_mc_unc_select = df_mc_unc.loc[(df_mc_unc['mc_runs'] == x)].copy()

            for i in modes:
                df_mc_unc_mode_select = dc_mc_unc_select.loc[(dc_mc_unc_select['mode'] == i)].copy()
                print(f'Optimized {i} ; MC Runs {x} UNCER :', np.round(np.average(df_mc_unc_mode_select['npv_uncer']), 2),
                      np.round(np.average(df_mc_unc_mode_select['pv_uncer']), 2),
                      np.round(np.average(df_mc_unc_mode_select['soil_uncer']), 2))

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

        sma_uncert_table = os.path.join(self.fig_directory, "sma-best_unmix_uncertainty_report.csv")
        mesma_uncer_table = os.path.join(self.fig_directory, "mesma_unmix_uncertainty_report.csv")

        df_sma_uncer = pd.read_csv(sma_uncert_table)
        df_sma_uncer.insert(0, 'mode', 'SMA')
        df_mesma_uncer = pd.read_csv(mesma_uncer_table)
        df_mesma_uncer.insert(0, 'mode', 'MESMA')
        df_uncer = pd.concat([df_sma_uncer, df_mesma_uncer], ignore_index=True)

        df_uncer = df_uncer.replace('convex', "Convex Hull")
        df_uncer = df_uncer.replace('latin', 'Latin Hypercube')
        df_uncer = df_uncer.replace('brightness', 'Brightness')

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
            formatted_df = df_select.reset_index(drop=True).applymap(format_float)

            print(formatted_df.to_latex(index=False, escape=False))

            print('_______________________________________________________________')
            print('\n')
            print('This begins the atmosphere parameters')

        for aod in sorted(list(df.aod.unique())):
            df_aod = df.loc[(df['aod'] == aod)]
            print(f'AOD {aod} MAE;', np.round(np.average(df_aod['npv_mae']), 2),
                  np.round(np.average(df_aod['pv_mae']), 2),
                  np.round(np.average(df_aod['soil_mae']), 2))

        for h2o in sorted(list(df.h2o.unique())):
            df_h2o = df.loc[(df['h2o'] == h2o)]
            print(f'h2o {h2o} MAE;', np.round(np.average(df_h2o['npv_mae']), 2),
                  np.round(np.average(df_h2o['pv_mae']), 2),
                  np.round(np.average(df_h2o['soil_mae']), 2))

        for sza in sorted(list(df.solar_zenith.unique())):
            df_sza = df.loc[(df['solar_zenith'] == sza)]
            print(f'sza {sza} MAE;', np.round(np.average(df_sza['npv_mae']), 2),
                  np.round(np.average(df_sza['pv_mae']), 2),
                  np.round(np.average(df_sza['soil_mae']), 2))

        print('\n')
        print('This is related to uncertainty....')

        for aod in sorted(list(df.aod.unique())):
            df_aod = df.loc[(df['aod'] == aod)]
            print(f'AOD {aod} MAE;', np.round(np.average(df_aod['npv_sma-uncertainty']), 2),
                  np.round(np.average(df_aod['pv_sma-uncertainty']), 2),
                  np.round(np.average(df_aod['soil_sma-uncertainty']), 2))

        for h2o in sorted(list(df.h2o.unique())):
            df_h2o = df.loc[(df['h2o'] == h2o)]
            print(f'h2o {h2o} MAE;', np.round(np.average(df_h2o['npv_sma-uncertainty']), 2),
                  np.round(np.average(df_h2o['pv_sma-uncertainty']), 2),
                  np.round(np.average(df_h2o['soil_sma-uncertainty']), 2))

        for sza in sorted(list(df.solar_zenith.unique())):
            df_sza = df.loc[(df['solar_zenith'] == sza)]
            print(f'sza {sza} MAE;', np.round(np.average(df_sza['npv_sma-uncertainty']), 2),
                  np.round(np.average(df_sza['pv_sma-uncertainty']), 2),
                  np.round(np.average(df_sza['soil_sma-uncertainty']), 2))

        print('\n')
        print("Uncertainty proportions")
        df_uncer = df_uncer.loc[((df_uncer['mc_runs'] == 25) & (df_uncer['dims'] == 4) & (df_uncer['scenario'] == 'Convex Hull')& (df_uncer['normalization'] == 'Brightness')) & ((df_uncer['cmbs'] == 100) | (df_uncer['cmbs'].isnull())) & ((df_uncer['num_em'] == 30) | df_uncer['num_em'].isnull())]

        for i in ['SMA', 'MESMA']:
            df_uncer_mode = df_uncer.loc[(df_uncer['mode'] == i)]
            df_atmos_mode = df.loc[(df['mode'] == i)]

            # uncertainty from unmixing
            npv_uncer = np.average(df_uncer_mode['npv_uncer'])
            pv_uncer = np.average(df_uncer_mode['pv_uncer'])
            soil_uncer = np.average(df_uncer_mode['soil_uncer'])

            # uncertainty from atmosphere
            npv_uncer_atmos = np.average(df_atmos_mode['npv_sma-uncertainty'])
            pv_uncer_atmos = np.average(df_atmos_mode['pv_sma-uncertainty'])
            soil_uncer_atmos = np.average(df_atmos_mode['soil_sma-uncertainty'])

            atmosphers_uncer = [npv_uncer_atmos, pv_uncer_atmos, soil_uncer_atmos]
            unmix_uncer = [npv_uncer, pv_uncer, soil_uncer]
            ems = ['NPV', 'PV', 'Soil']

            for _x, x in enumerate(atmosphers_uncer):
                atmosp_propotion = (unmix_uncer[_x] - x)/x
                print(f'{i} {ems[_x]} UNCER;', np.round(atmosp_propotion,2))



    def summary_table(self):
        df_unmix = pd.read_csv(os.path.join(self.fig_directory, "sma_unmix_error_report.csv"))
        #df_uncertainty = pd.read_csv(os.path.join(self.fig_directory, "sma_unmix_uncertainty_report.csv"))
        df_unmix_mesma = pd.read_csv(os.path.join(self.fig_directory, "mesma_unmix_error_report.csv"))
        #df_uncertainty_mesma = pd.read_csv(os.path.join(self.fig_directory, "mesma_unmix_uncertainty_report.csv"))
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
        #df_select_uncer = df_uncertainty.loc[(df_uncertainty['normalization'] == 'brightness') &
        #                               (df_uncertainty['num_em'] == 30) & (df_uncertainty['mc_runs'] == 25) &
        #                               (df_uncertainty['scenario'] == 'convex') & (df_uncertainty['dims'] == 4)].copy()

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
                           'end_line', 'endmember_classes', 'log_file', 'spectra_per_s',  'total_time', 'error',
                           'worker_count', 'node']

        df_time = df_time.replace('"', '', regex=True)
        df_time['reflectance_file'] = df_time['reflectance_file'].apply(os.path.basename)
        df_time['endmember_file'] = df_time['endmember_file'].apply(os.path.basename)
        df_time.drop(columns=['endmember_file', 'endmember_class_header', 'output_file_base',
                                'spectral_starting_column', 'truncate_end_columns', 'reflectance_uncertainty_file',
                                'refl_nodata', 'refl_scale', 'write_complete_fractions', 'optimizer', 'start_line',
                              'end_line', 'endmember_classes', 'log_file',], inplace=True)

        df_time['scenario'] = df_time['reflectance_file'].apply(lambda path: '-'.join(os.path.basename(path).split('_')[:2]).split('-')[1])
        df_time['spectra_per_s'] = df_time['spectra_per_s'].astype(float)
        df_time['spectra_per_s'] = df_time['spectra_per_s'].apply(lambda x: round(x, 2))
        df_time['num_endmembers'] = df_time['num_endmembers'].apply(lambda x: ast.literal_eval(x)[0])
        df_time['dims'] = df_time['reflectance_file'].apply(lambda path: os.path.basename(path).split('_')[:5][4]).astype(int)

        df_time = df_time.loc[(df_time['error'] == 0)].copy()
        
        # get optimal settings - both convex and latin hypercube
        df_avg_optimal = df_time.loc[((df_time['normalization'] == 'brightness') & (df_time['dims'] == 4)) &
            ((df_time['max_combinations'] == 100) | (df_time['max_combinations'] == -1)) & ((df_time['num_endmembers'] == 30) | (df_time['num_endmembers'] == 3))].copy()




        for mode in df_time['mode'].unique():
            if mode == 'mesma':
                df_avg_optimal_mesma = df_avg_optimal.loc[(df_avg_optimal['mode'] == 'mesma')].copy()
                df_avg = df_avg_optimal_mesma.sort_values('n_mc')

            elif mode == 'sma':
                df_avg_optimal_sma = df_avg_optimal.loc[
                    (df_avg_optimal['mode'] == 'sma') & (df_avg_optimal['num_endmembers'] == 30)].copy()
                df_avg = df_avg_optimal_sma.sort_values('n_mc')

            for i in df_time['scenario'].unique():
                df_time_select = df_avg.loc[(df_avg['scenario'] == i) & (df_avg['mode'] == mode)].copy()
                print(df_time_select.to_latex(index=False, escape=False))

        # optimal settings - all values of number of endmembers (n) for sma
        df_avg_sma_num_em = df_time.loc[(df_time['normalization'] == 'brightness') & (df_time['n_mc'] == 25) & (df_time['max_combinations'] == -1)].copy()
        num_em = []
        avg_time_per_em = []
        
                
        for i in sorted(list(df_avg_sma_num_em.num_endmembers.unique())):
            df_select = df_avg_sma_num_em.loc[(df_avg_sma_num_em['num_endmembers'] == i)].copy()
            num_em.append(i)
            avg_elapsed_time = np.round(np.average(df_select['spectra_per_s']), 2)
            avg_time_per_em.append(avg_elapsed_time)
        
        print(num_em)
        print(avg_time_per_em)


        X = np.array(num_em)
        y = np.array(avg_time_per_em)
        X = X.reshape(-1,1)
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        y_pred = model.predict(X)

        print(slope, intercept)
        print('The r^2 value between n and time is : ', np.round(r2_score(avg_time_per_em, y_pred),2))

        # optimal settings - combinations increase for MESMA
        df_avg_mesma_cmb = df_time.loc[(df_time['normalization'] == 'brightness') & (df_time['n_mc'] == 25) & (
                    df_time['max_combinations'] != -1)].copy()

        num_cmb = []
        avg_time_per_cmb = []
        
        
        for i in sorted(list(df_avg_mesma_cmb.max_combinations.unique())):
            df_select = df_avg_mesma_cmb.loc[(df_avg_mesma_cmb['max_combinations'] == i)].copy()
            num_cmb.append(i)
            avg_elapsed_time = np.round(np.average(df_select['spectra_per_s']), 2)
            avg_time_per_cmb.append(avg_elapsed_time)
        
        print(num_cmb)
        print(avg_time_per_cmb)


        X = np.array(num_cmb)
        y = np.array(avg_time_per_cmb)
        X = X.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        y_pred = model.predict(X)
        
        print(slope, intercept)
        print('The r^2 value between number of combinations and time is : ', np.round(r2_score(avg_time_per_cmb, y_pred), 2))

        # optimal settings - find variations in normalization methods ; SMA
        df_avg_sma_normalization = df_time.loc[(df_time['n_mc'] == 25) & (df_time['num_endmembers'] == 30)].copy()
        for i in sorted(list(df_avg_sma_normalization.normalization.unique())):
            df_select = df_avg_sma_normalization.loc[(df_avg_sma_normalization['normalization'] == i)].copy()
            print(f'Average Normalization {i} SMA (spec/s) : ', np.round(np.average(df_select['spectra_per_s']), 2),
                  's; Worker Count: ', np.average(df_select['worker_count']))

        # optimal settings - find variations in normalization methods MESMA
        df_avg_mesma_normalization = df_time.loc[(df_time['n_mc'] == 25) & (df_time['max_combinations'] == 100)].copy()
        for i in sorted(list(df_avg_mesma_normalization.normalization.unique())):
            df_select = df_avg_mesma_normalization.loc[(df_avg_mesma_normalization['normalization'] == i)].copy()
            print(f'Average Normalization {i} MESMA (spec/s) : ', np.round(np.average(df_select['spectra_per_s']), 2),
                  's; Worker Count: ', np.average(df_select['worker_count']))

        # monte carlo run increases with opt settings
        df_avg_mc_runs_sma = df_time.loc[(df_time['normalization'] == 'brightness') & (df_time['num_endmembers'] == 30)].copy()
        mc_runs = []
        sma_mc_times = []
        for i in sorted(list(df_avg_mc_runs_sma.n_mc.unique())):
            df_select = df_avg_mc_runs_sma.loc[(df_avg_mc_runs_sma['n_mc'] == i)].copy()
            mc_runs.append(i)
            avg_elapsed_time = np.round(np.average(df_select['spectra_per_s']), 2)
            sma_mc_times.append(avg_elapsed_time)

        print(mc_runs)
        print(sma_mc_times)

        # monte carlo runs increase with opt settings mesma
        df_avg_mc_runs_mesma = df_time.loc[
            (df_time['normalization'] == 'brightness') & (df_time['num_endmembers'] == 3) & (df_time['max_combinations'] == -1)].copy()
        mc_runs = []
        mesma_mc_times = []
        for i in sorted(list(df_avg_mc_runs_mesma.n_mc.unique())):
            df_select = df_avg_mc_runs_mesma.loc[(df_avg_mc_runs_mesma['n_mc'] == i)].copy()
            mc_runs.append(i)
            avg_elapsed_time = np.round(np.average(df_select['spectra_per_s']), 2)
            mesma_mc_times.append(avg_elapsed_time)

        print(mc_runs)
        print(mesma_mc_times)

    def library_results(self):
        df = pd.read_csv(os.path.join(self.output_directory, 'convolved', 'geofilter_convolved.csv'))

        for lib in df.dataset.unique():
            df_lib = df.loc[(df['dataset'] == lib)].copy()
            print(f'Lib {lib}', df_lib.level_1.value_counts())


def run_latex_tables(base_directory: str):
    latex_class = latex(base_directory=base_directory)
    latex_class.optimal_parameters()
    # latex_class.atmosphere_table()
    #latex_class.summary_table()
    latex_class.time_table()
    # latex_class.baseline_setings()

    #latex_class.library_results()
