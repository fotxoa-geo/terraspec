import time
import numpy as np
from glob import glob
import os
import isofit.core.common as isc
import pandas as pd
from p_tqdm import p_map
from functools import partial
import geopandas as gpd
from utils.envi import save_envi, get_meta
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.slpit_download import load_pickle
from slpit.build_slpit import haversine_distance
from utils.text_guide import cursor_print, query_yes_no
import subprocess
from math import radians, sin, cos, sqrt, atan2
import struct


class build_libraries:
    def __init__(self, base_directory: str, sensor:str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)

        # create output directories
        create_directory(os.path.join(self.output_directory, 'spectral_endmembers'))
        create_directory(os.path.join(self.output_directory, 'spectral_transects'))
        create_directory(os.path.join(self.output_directory, 'spectral_transects', 'transect'))
        create_directory(os.path.join(self.output_directory, 'spectral_transects', 'endmembers'))
        create_directory(os.path.join(self.output_directory, 'spectral_transects', 'endmembers-raw'))
        create_directory(os.path.join(self.output_directory, 'plot_pictures'))
        create_directory(os.path.join(self.output_directory, 'plot_pictures', 'spectral_transects'))
        create_directory(os.path.join(self.output_directory, 'plot_pictures', 'spectral_endmembers'))

        # input data directories
        self.spectral_em_directory = os.path.join(self.base_directory, 'data', 'spectral_endmembers')
        self.spectral_transect_directory = os.path.join(self.base_directory, 'data', 'spectral_transects')

        # instrument to indicate wavelengths in output folder
        self.instrument = sensor

        # output data directories
        self.output_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')
        self.output_transect_em_directory = os.path.join(self.output_directory, 'spectral_transects', 'endmembers')
        self.output_transect_em_directory_raw = os.path.join(self.output_directory, 'spectral_transects',
                                                             'endmembers-raw')

        terraspec_base = os.path.join(base_directory, "..")

        # import the simulation outputs
        self.em_sim_directory = os.path.join(terraspec_base, 'simulation', 'output')
        self.emit_global = os.path.join(self.em_sim_directory, 'convolved', 'geofilter_convolved.csv')
        self.convex_global = os.path.join(self.em_sim_directory, 'endmember_libraries',
                                          'convex_hull__n_dims_4_unmix_library.csv') # is this in asd resolution that allows to convolve to EMIT ???
        # import  emit slipt data
        slpit_emit_directories = os.path.join(terraspec_base, 'slpit', 'output')
        self.emit_em_libraries = os.path.join(slpit_emit_directories, 'spectral_transects', 'endmembers-raw')

        self.asd_wvls = spectra.load_asd_wavelenghts()


    def build_transects(self):
        # the transect spectra
        records = load_pickle('shift_slpit')

        print("loading... Spectral Transects")
        for i in records:
            if i['site_list'] == 'JORN' or i['site_list'] == 'SRER' :
                pass

            elif i['plot_survey_type'] == 'slpit':
                plot_name = f"{i['site_list'].upper() + i['team_name'].upper()}-{i['site_num']:03d}"
                season = i['season'].upper()
                date = i['date_taken']
                plot_directory = os.path.join(self.base_directory, 'data', f'SHIFT_{season}', plot_name)

                if os.path.isfile(os.path.join(self.output_transect_directory, f'{plot_name}_{season}.csv')):
                    continue

                # white ref table
                df_white_ref = pd.json_normalize(i['white_ref'])
                df_white_ref = df_white_ref.iloc[:, 14:]

                # get all files in the subdirectories
                all_asd_files = sorted(glob(os.path.join(plot_directory, '**', '*[!.txt][!.log][!.ini]'), recursive=True))

                for old_asd in all_asd_files:

                    if os.path.isfile(old_asd):
                        pass
                    else:
                        all_asd_files.remove(old_asd)

                transect_spectra = []

                for asd_file in all_asd_files:
                    if not os.path.isfile(asd_file):
                        continue
                    file_type = os.path.split(os.path.split(asd_file)[0])[1]
                    if file_type == 'Endmembers':
                        pass

                    else:
                        try:
                            file_num = int(os.path.basename(asd_file).split(".")[0].split("_")[-1])

                        except:
                            # old asd file numbers
                            file_num = int(os.path.basename(asd_file).split(".")[-1])

                        line_num = os.path.split(os.path.split(os.path.split(asd_file)[0])[0])[1]

                        # keep only the good white ref files
                        df_white_ref_select = df_white_ref[(df_white_ref['line_num'] == line_num.lower())].copy()

                        # keep good white spectra and append the transect data
                        if file_num in df_white_ref_select.filenumber.values:
                            good_white_ref = df_white_ref_select[df_white_ref_select['my_element_2'] == 'good'].copy()

                            # ignore bad white ref files
                            if file_num in good_white_ref.filenumber.values:
                                transect_spectra.append(asd_file)
                            else:
                                pass

                        else:
                            transect_spectra.append(asd_file)

                # parallel process the reflectance files
                results_refl = p_map(partial(spectra.get_shift_transect, plot_directory=plot_directory, season=season), transect_spectra,
                                     **{"desc": "\t\t processing plot: " + season + " " + plot_name + " ...", "ncols": 150})

                # make a dataframe of the results
                df_results = pd.DataFrame(results_refl)
                df_results.columns = ["plot_name", "file_name", "line_num", "file_num", "longitude", "latitude", "elevation",
                                      "utc_time"] + list(self.asd_wvls)
                df_results.insert(3, "white_ref", 0)

                # perform white ref corrections on data where we have gps
                df_results = df_results.copy()
                df_results['utc_time'] = pd.to_datetime(df_results['utc_time'], format='%H:%M:%S')
                adjusted_dfs = []

                # sample each line
                for line_num in df_white_ref.line_num.unique():
                    df_select = df_white_ref.loc[(df_white_ref['line_num'] == line_num)
                                                 & (df_white_ref['my_element_2'] == 'good')].copy()
                    df_select = df_select.rename({'filenumber': 'file_num'}, axis=1)

                    # perform white corrections
                    df_query = df_results[(df_results['line_num'] == line_num.upper())].copy()
                    df_query = pd.merge(df_query, df_select, left_on='file_num', right_on='file_num', how='left')
                    df_query['white_ref'] = df_query['white_ref_space']

                    # this drops the join since we already have the values saved
                    df_query = df_query.iloc[:, :-3]

                    # we are not using middle white ref atm
                    df_query = df_query[df_query.white_ref != 'middle']

                    # get white refs ; begin and end
                    df_begin = df_query[df_query['white_ref'] == 'begin'].copy()
                    df_begin_spectra = np.mean(df_begin.iloc[:, 9:].to_numpy(), axis=0)

                    df_end = df_query[df_query['white_ref'] == 'end'].copy()
                    df_end_spectra = np.mean(df_end.iloc[:, 9:].to_numpy(), axis=0)

                    # get times
                    df_begin_time = df_begin.iloc[:, 8].to_frame()
                    df_begin_time['second'] = df_begin_time['utc_time'].dt.strftime('%S').astype(int)
                    df_begin_time['minute'] = df_begin_time['utc_time'].dt.strftime('%M').astype(int)
                    df_begin_time['hour'] = df_begin_time['utc_time'].dt.strftime('%H').astype(int)
                    df_begin_time['total_seconds'] = df_begin_time.second + (df_begin_time.minute * 60) + (
                                df_begin_time.hour * 3600)

                    df_end_time = df_end.iloc[:, 8].to_frame()
                    df_end_time['second'] = df_end_time['utc_time'].dt.strftime('%S').astype(int)
                    df_end_time['minute'] = df_end_time['utc_time'].dt.strftime('%M').astype(int)
                    df_end_time['hour'] = df_end_time['utc_time'].dt.strftime('%H').astype(int)
                    df_end_time['total_seconds'] = df_end_time.second + (df_end_time.minute * 60) + (
                                df_end_time.hour * 3600)

                    # get only transect spectra - no white ref
                    df_spectra = df_query[df_query['white_ref'].isnull()].copy()
                    df_spectra = df_spectra.reset_index(drop=True)
                    df_spectra_array = df_spectra.iloc[:, 9:].to_numpy()

                    # change in white ref
                    delta_white_ref = df_end_spectra - df_begin_spectra
                    delta_time = np.array(
                        np.mean(df_end_time.total_seconds.values) - np.mean(df_begin_time.total_seconds.values).mean())
                    slope = delta_white_ref / delta_time

                    # create an empty zero array to save reflectance
                    spectra_grid = np.zeros((df_spectra_array.shape[0], df_spectra_array.shape[1]))

                    # perform spectra white ref correction
                    for _row, row in enumerate(spectra_grid):
                        for _col, col in enumerate(row):
                            adjustment = df_begin_spectra[_col] + (delta_time * slope[_col])
                            spectra_grid[_row, _col] = df_spectra_array[_row, _col] / adjustment

                    df_corrected = pd.DataFrame(spectra_grid)
                    df_adjusted = pd.concat([df_spectra.iloc[:, :9], df_corrected], axis=1)
                    df_adjusted.columns = df_spectra.columns
                    df_adjusted = df_adjusted.drop('white_ref', axis=1)
                    df_adjusted['utc_time'] = df_adjusted['utc_time'].dt.strftime('%H:%M:%S')
                    df_adjusted.insert(0, "date", date)
                    df_adjusted = df_adjusted.rename({'line_num_x': 'line_num'}, axis=1)  # new method
                    adjusted_dfs.append(df_adjusted)

                df_corrected_all = pd.concat(adjusted_dfs)
                df_corrected_all.to_csv(os.path.join(self.output_transect_directory, plot_name + '_' + season + '.csv'),
                                        index=False)

                # convolve wavelengths to user specified instrument
                results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm, asd_wvl=self.asd_wvls, spectra_starting_col=9), df_corrected_all.iterrows(),
                                         **{"desc": "\t\t\tconvulsing plot: " + plot_name + " ...", "ncols": 150})

                # save files as envi files
                spectra_grid = np.zeros((len(results_convolve), 1, len(self.wvls)))

                # fill spectral data
                for _row, row in enumerate(results_convolve):
                    spectra_grid[_row, :, :] = row

                # save the spectra
                print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
                meta_spectra = get_meta(lines=len(results_convolve), samples=spectra_grid.shape[1], bands=self.wvls,
                                        wvls=True)
                output_raster = os.path.join(self.output_transect_directory, plot_name + '_' + season + ".hdr")
                save_envi(output_raster, meta_spectra, spectra_grid)

            else:
                pass

    def build_endmember_lib(self):
        # the transect spectra
        records = load_pickle('shift_slpit')

        print("loading... Spectral Transects Endmembers")


        for i in records:
            if i['site_list'] == 'JORN' or i['site_list'] == 'SRER':
                pass

            elif i['plot_survey_type'] == 'slpit':
                plot_name = f"{i['site_list'].upper() + i['team_name'].upper()}-{i['site_num']:03d}"
                season = i['season'].upper()
                date = i['date_taken']
                plot_directory = os.path.join(self.base_directory, 'data', 'SHIFT_' + season, plot_name)

                if os.path.isfile(os.path.join(self.output_transect_em_directory_raw, plot_name + '_' + season + '_asd.csv')):
                    continue

                # em table
                df_transect_em = pd.json_normalize(i['em'])
                df_transect_em = df_transect_em.iloc[:, 14:]
                df_transect_em = df_transect_em.loc[df_transect_em['em_condition'] != 'bad'].copy()
                df_transect_em = df_transect_em.loc[df_transect_em['em_classification'] != 'flower'].copy()

                # get all endmembers
                all_asd_files = sorted(glob(os.path.join(plot_directory, '**', '*[!.txt][!.log][!.ini]'), recursive=True))

                transect_spectra = []
                for asd_file in all_asd_files:
                    if not os.path.isfile(asd_file):
                        continue
                    file_type = os.path.split(os.path.split(asd_file)[0])[1]
                    line_num = os.path.split(os.path.split(os.path.split(asd_file)[0])[0])[1]

                    if file_type == 'Transect':
                        continue

                    else:
                        # make sure no bad files go into the master list
                        df_transect_em_select = df_transect_em.loc[df_transect_em['line_num'] == line_num.lower()].copy()
                        df_transect_em_select = df_transect_em_select.loc[df_transect_em_select['em_condition'] != 'bad'].copy()

                        try:
                            file_num = int(os.path.basename(asd_file).split(".")[0].split("_")[-1])

                        except:
                            # old asd file numbers
                            file_num = int(os.path.basename(asd_file).split(".")[-1])

                        if file_num in df_transect_em_select.asd_file_num.values:
                            transect_spectra.append(asd_file)

                # parallel process the reflectance files
                results_refl = p_map(partial(spectra.get_shift_transect, plot_directory=plot_directory, season=season),
                                     transect_spectra,
                                     **{"desc": "\t\t processing plot: " + season + " " + plot_name + " ...",
                                        "ncols": 150})

                # make a dataframe of the results
                df_results = pd.DataFrame(results_refl)
                df_results.columns = ["plot_name", "file_name", "line_num", "file_num", "longitude", "latitude",
                                      "elevation", "utc_time"] + list(self.asd_wvls)
                df_results.insert(0, "date", date)
                df_results.insert(4, "level_1", '')
                df_results.insert(5, "species", '')

                # filter by line
                for line_num in sorted(df_results.line_num.unique()):
                    df_line = df_results[df_results['line_num'] == line_num.upper()].copy()

                    for file_num in df_line.file_num.values:
                        em_clas = df_transect_em.loc[(df_transect_em['asd_file_num'] == file_num) & (df_transect_em['line_num'] == line_num.lower()), 'em_classification'].iloc[0]
                        species = df_transect_em.loc[(df_transect_em['asd_file_num'] == file_num) & (df_transect_em['line_num'] == line_num.lower()), 'species_name'].iloc[0]

                        df_results.loc[(df_results['file_num'] == file_num) & (df_results['line_num'] == line_num.upper()), ['level_1', 'species']] = em_clas, species

                df_results = df_results.sort_values("level_1")
                df_results.to_csv(os.path.join(self.output_transect_em_directory_raw, f'{plot_name}_{season}_asd.csv'),
                                        index=False)

                # convolve wavelengths to user specified instrument
                results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm, asd_wvl=self.asd_wvls,
                                                 spectra_starting_col=11), df_results.iterrows(),
                                         **{"desc": "\t\t\tconvulsing plot: " + plot_name + " ...", "ncols": 150})

                df_convolve = pd.DataFrame(results_convolve)
                df_convolve.columns = list(self.wvls)
                df_convolve = pd.concat([df_results.iloc[:, :10].reset_index(drop=True), df_convolve], axis=1)
                df_convolve = df_convolve.reset_index(drop=True)

                # save original csv endmembers
                df_convolve = df_convolve.sort_values("level_1")
                df_convolve['plot_name'] = plot_name + '-' + season
                df_convolve.to_csv(os.path.join(self.output_transect_em_directory_raw, f'{plot_name}_{season}_{self.instrument}.csv'), index=False)

    def build_em_collection(self):
        print("loading em collection...")
        # merge all endmembers - instrument based wavelengths
        instrument_ems = spectra.get_all_ems(output_directory=self.output_directory, instrument=self.instrument)
        asd_ems = spectra.get_all_ems(output_directory=self.output_directory, instrument='asd')
        emit_convolved_ems =  glob(os.path.join(self.output_directory, 'spectral_transects', 'emit_convolved_endmembers', "*.csv"))

        # dataframes of instrument
        df = pd.concat((pd.read_csv(f) for f in instrument_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-" + self.instrument + ".csv"), index=False)

        # emit convolved ems
        df = pd.concat((pd.read_csv(f) for f in emit_convolved_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-emit-" + self.instrument + ".csv"), index=False)

        # merge all endmembers - asd based wavelengths
        df = pd.concat((pd.read_csv(f) for f in asd_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-asd.csv"), index=False)

        # merge all transect spectra
        instrument_transects = glob(os.path.join(self.output_transect_directory, "*.csv"))
        df = pd.concat((pd.read_csv(f) for f in instrument_transects), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-transect-asd.csv"), index=False)

    def build_gis_data(self):
        print("Building spectral endmember gis shapefile data...", sep=' ', end='', flush=True)
        df = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-' + self.instrument + '.csv'))
        df = df.iloc[:, :9]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, base_directory=self.base_directory, out_name='shift_spectral_lib')

        df = pd.read_csv(os.path.join(self.output_directory, 'all-transect-asd.csv'))
        df_rows = []
        for plot in sorted(list(df.plot_name.unique())):

            df_select = df[df['plot_name'] == plot].copy()
            long_centroid = np.mean(df_select['longitude'])
            lat_centroid = np.mean(df_select['latitude'])
            mean_elevation = np.mean(df_select['elevation'])
            season = plot.split('-')[2]
            df_rows.append([plot, season, long_centroid, lat_centroid, mean_elevation])

        df_centroid = pd.DataFrame(df_rows)
        df_centroid.columns =['plot', 'season', 'longitude', 'latitude', 'mean_elevation']
        spectra.df_to_shapefile(df_centroid, base_directory=self.base_directory, out_name='shift_transects_centroid')

        df = df.iloc[:, :8]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, base_directory=self.base_directory, out_name='shift_transect')
        time.sleep(3)

        # generate centroids of plots using transect data

        print("done")

    def nearest_site(self):
        print("calculating distances")
        # get plot center points from SHIFT
        shapefile_shift = os.path.join('gis', "shift_transects_centroid.shp")

        # get plot points from EMIT
        shapefile_emit = os.path.join('gis', "Observation.shp")
        df_emit = gpd.read_file(shapefile_emit)
        df_emit['latitude'] = df_emit['geometry'].y
        df_emit['longitude'] = df_emit['geometry'].x
        df_emit = df_emit.drop('geometry', axis=1)
        df_emit = df_emit.rename(columns={'Name': 'plot'})
        df_emit['campaign'] = 'emit'

        # load geo dataframe
        df = pd.DataFrame(gpd.read_file(shapefile_shift))
        df = df.drop('geometry', axis=1)
        df = df.sort_values('plot')
        df['campaign'] = 'shift'

        # merge both spatial dataframes
        df_merge = pd.concat([df, df_emit])
        columns_to_keep = ['plot', 'season', 'longitude', 'latitude', 'campaign']

        # Get indices of columns to keep
        columns_to_exclude = [col for col in df_merge.columns if col not in columns_to_keep]

        # Drop the last 4 columns while keeping specified columns
        df_merge = df_merge.drop(columns_to_exclude, axis=1)

        df_min_distance_rows = []
        for index, row in df.iterrows():
            plot = row['plot']
            lon = row['longitude']
            lat = row['latitude']

            results = p_map(partial(haversine_distance, lat, lon), df_merge['latitude'].values, df_merge['longitude'].values, df_merge['plot'].values,
                            **{"desc": f"geographic distance: {plot}", "ncols": 150})

            df_results = pd.DataFrame(results)
            df_results.columns = ['plot_function', 'distance_km']
            df_results = df_results.dropna()
            df_results = df_results.sort_values('distance_km')

            df_results['combined'] = list(zip(df_results['plot_function'], df_results['distance_km']))
            df_results = df_results.drop(columns=['plot_function', 'distance_km']).T
            df_results['shift_plot_analysis'] = plot
            df_to_row = df_results.iloc[0].values

            df_min_distance_rows.append(df_to_row)

        min_dist_df = pd.DataFrame(df_min_distance_rows)
        column_names = min_dist_df.columns.tolist()
        column_names[-1] = 'shift_plot_analysis'
        min_dist_df.columns = column_names
        min_dist_df.to_csv(os.path.join('gis', 'shift_min_dist_to_all_plots.csv'), index=False)

    def convolve_emit_sites(self):
        print("convolving emit sites...")
        # convolve emit sites
        csvs = glob(os.path.join(self.emit_em_libraries, '*-asd.csv'))
        create_directory(os.path.join(self.output_directory, 'spectral_transects', 'emit_convolved_endmembers'))

        for i in csvs:

            df = pd.read_csv(i)
            plot_num = os.path.basename(i).split("-")[1]
            output_file = os.path.join(self.output_directory, 'spectral_transects', 'emit_convolved_endmembers',
                                            f"SPEC-{plot_num}_aviris-ng.csv")

            if os.path.isfile(output_file):
                continue

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm, asd_wvl=self.asd_wvls,
                                             spectra_starting_col=11), df.iterrows(),
                                     **{"desc": f"\t\t\tconvulsing plot: SPEC - {plot_num}...", "ncols": 150})

            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df.iloc[:, :10].reset_index(drop=True), df_convolve], axis=1)
            df_convolve = df_convolve.reset_index(drop=True)

            # save original csv endmembers
            df_convolve = df_convolve.sort_values("level_1")
            df_convolve.to_csv(output_file, index=False)

    def convolve_global_lib(self):
        print("convolving global libraries...")
        output_lib = os.path.join(self.output_directory, f'shift_{os.path.basename(self.convex_global)}')

        if not os.path.isfile(output_lib):
            df_global = pd.read_csv(self.convex_global)
            df_global['dataset'] = df_global['dataset'].str.lower()
            all_data_asd = glob(os.path.join(self.em_sim_directory, 'all_data', "*.csv"))

            list_of_dfs = []

            for csv in all_data_asd:
                ds = os.path.basename(csv).split('.')[0].split('_')[-1].lower()

                # load dataset
                df_all_select = pd.read_csv(csv, low_memory=False)
                df_all_select['dataset'] = df_all_select['dataset'].str.lower()

                # drop duplicates in f_name - these are from all data, we never used these
                df_all_select = df_all_select.drop_duplicates(subset=[df_all_select.columns[4]], keep='first')

                # select current dataset from global library
                df_global_select = df_global[(df_global['dataset'] == ds)].copy()

                merged_df = df_all_select[df_all_select['fname'].isin(df_global_select['fname'])]
                merged_df = merged_df.sort_values("level_1")

                # convolve wavelengths to user specified instrument
                wvls = np.array(merged_df.columns[7:]).astype(float)
                results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm, asd_wvl=wvls,
                                                 spectra_starting_col=7), merged_df.iterrows(),
                                         **{"desc": f"\t\t\tconvulsing {ds} to aviris_ng...", "ncols": 150})

                df_convolve = pd.DataFrame(results_convolve)
                df_convolve.columns = list(self.wvls)
                df_convolve = pd.concat([df_global_select.iloc[:, :7].reset_index(drop=True), df_convolve], axis=1)
                df_convolve = df_convolve.reset_index(drop=True)

                list_of_dfs.append(df_convolve)

            df_global_convolved = pd.concat(list_of_dfs, ignore_index=True)
            df_global_convolved = df_global_convolved.sort_values("level_1")
            fname_index_map = {fname: index for index, fname in enumerate(df_global['fname'])}
            df_global_convolved = df_global_convolved.sort_values(by='fname', key=lambda x: x.map(fname_index_map))
            df_global_convolved.to_csv(os.path.join(self.output_directory, f'shift_{os.path.basename(self.convex_global)}'), index=False)\

        else:
            print(f'{output_lib} has been already created!! Skipping...')

    def em_qty_check(self):
        print('calculating em quantity check')
        # ensures that in each csv there at least 3 classes and n samples for each class
        # will use the nearest site for geographic distance
        em_min_samples = {'pv': 30, 'npv': 30, 'soil': 75}

        shift_ems = sorted(spectra.get_all_ems(output_directory=self.output_directory, instrument=self.instrument))
        df_all_shift_ems = pd.read_csv(os.path.join(self.output_directory, "all-endmembers-" + self.instrument + ".csv"))
        df_all_shift_ems['level_1'] = df_all_shift_ems['level_1'].str.lower()
        df_all_emit_ems = pd.read_csv(os.path.join(self.output_directory, "all-endmembers-emit-" + self.instrument + ".csv"))
        df_all_emit_ems['plot_name'] = df_all_emit_ems['plot_name'].str.replace('Spectral', 'SPEC')
        df_all_emit_ems['level_1'] = df_all_emit_ems['level_1'].str.lower()

        df_all_ems = pd.concat([df_all_shift_ems, df_all_emit_ems], ignore_index=True)

        df_distance = pd.read_csv(os.path.join('gis', 'shift_min_dist_to_all_plots.csv'))
        all_ems = sorted(list(df_all_shift_ems.level_1.unique()))

        for i in shift_ems:
            plot = f"{os.path.basename(i).split('_')[0]}-{os.path.basename(i).split('_')[1]}"

            df_em_site = pd.read_csv(i)
            df_em_site['level_1'] = df_em_site['level_1'].str.lower()
            site_em = sorted(list(df_em_site.level_1.unique()))
            df_nearest_distances = df_distance.loc[df_distance['shift_plot_analysis'] == plot].copy()

            ems_to_append = []

            if len(site_em) == 3:
                # fill in remainder so each class in equal to number of desired samples
                for _em, em in enumerate(site_em):
                    df_em_select = df_em_site.loc[df_em_site['level_1'] == em].copy()

                    if df_em_select.shape[0] == em_min_samples[em]:
                        pass
                    else:
                        remaining_samples = em_min_samples[em] - df_em_select.shape[0]
                        site_counter = 0

                        while remaining_samples > 0:
                            current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0]
                            df_nearest_ems = df_all_ems.loc[(df_all_ems['level_1'] == em) & (df_all_ems['plot_name'] == current_nearest_site)].copy()

                            if remaining_samples > df_nearest_ems.shape[0]:
                                df_rand = df_nearest_ems.sample(n=df_nearest_ems.shape[0], random_state=13, ignore_index=True)
                                remaining_samples -= df_nearest_ems.shape[0]

                            else:
                                df_rand = df_nearest_ems.sample(n=remaining_samples, random_state=13, ignore_index=True)
                                remaining_samples -= remaining_samples

                            ems_to_append.append(df_rand)
                            # update counters
                            site_counter += 1

            else:
                print(i, 'missing 3 em clases!')
                # if not 3 classes add n samlpes
                em_difference = sorted(list(set(all_ems) - set(site_em)))

                # append the missing classes
                for em in em_difference:

                    remaining_samples = em_min_samples[em]
                    site_counter = 0

                    while remaining_samples > 0:
                        current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0]

                        df_nearest_ems = df_all_ems.loc[(df_all_ems['level_1'] == em) & (
                                df_all_ems['plot_name'] == current_nearest_site)].copy()

                        if remaining_samples > df_nearest_ems.shape[0]:
                            df_rand = df_nearest_ems.sample(n=df_nearest_ems.shape[0], random_state=13,
                                                            ignore_index=True)
                            remaining_samples -= df_nearest_ems.shape[0]

                        else:
                            df_rand = df_nearest_ems.sample(n=remaining_samples, random_state=13, ignore_index=True)
                            remaining_samples -= remaining_samples

                        ems_to_append.append(df_rand)
                        # update counters
                        site_counter += 1

                # check that the remaining ems meet the criteria
                for _em, em in enumerate(site_em):
                    df_em_select = df_em_site.loc[df_em_site['level_1'] == em].copy()

                    if df_em_select.shape[0] == em_min_samples[em]:
                        pass
                    else:
                        remaining_samples = em_min_samples[em] - df_em_select.shape[0]
                        site_counter = 0

                        while remaining_samples > 0:
                            current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0]
                            df_nearest_ems = df_all_ems.loc[(df_all_ems['level_1'] == em) & (df_all_ems['plot_name'] == current_nearest_site)].copy()

                            if remaining_samples > df_nearest_ems.shape[0]:
                                df_rand = df_nearest_ems.sample(n=df_nearest_ems.shape[0], random_state=13, ignore_index=True)
                                remaining_samples -= df_nearest_ems.shape[0]

                            else:
                                df_rand = df_nearest_ems.sample(n=remaining_samples, random_state=13, ignore_index=True)
                                remaining_samples -= remaining_samples

                            ems_to_append.append(df_rand)
                            # update counters
                            site_counter += 1

            # if list is empty do nothing
            out_csv = os.path.join(self.output_transect_em_directory, f"{os.path.basename(i)}")

            if not ems_to_append:
                df_em_site.to_csv(out_csv, index=False)
            else:
                df_append = pd.concat(ems_to_append, ignore_index=True)
                df_convolve = pd.concat([df_em_site, df_append], ignore_index=True)
                df_convolve = df_convolve.sort_values('level_1')
                df_convolve.to_csv(out_csv, index=False)

def run_build_workflow(base_directory, sensor):

    lib = build_libraries(base_directory=base_directory, sensor=sensor)
    lib.build_transects()
    if not os.path.isfile(os.path.join('gis', 'shift_min_dist_to_all_plots.csv')):
        lib.nearest_site()
    lib.convolve_emit_sites()
    lib.convolve_global_lib()
    lib.build_endmember_lib()
    lib.build_em_collection()
    lib.build_gis_data()
    lib.em_qty_check()

