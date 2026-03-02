import time
import numpy as np
from glob import glob
import os
import pandas as pd
from p_tqdm import p_map
from functools import partial
import geopandas as gpd
from utils.envi import save_envi, get_meta
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.slpit_download import load_pickle
from slpit.build_slpit import haversine_distance
from utils.slpit_utils import slpit
import requests
import subprocess

class build_libraries:
    def __init__(self, base_directory: str, sensor:str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)
        self.asd_wvls = spectra.load_asd_wavelenghts()

        # create output directories
        create_directory(os.path.join(self.output_directory, 'spectral_transects'))

        # input data directories
        self.spectral_transect_directory = os.path.join(self.base_directory, 'data', 'spectral_transects')

        # instrument to indicate wavelengths in output folder
        self.instrument = sensor

        # output data directories
        self.output_transect_directory = os.path.join(self.output_directory, 'spectral_transects')

        # # import  emit slipt data
        # slpit_emit_directories = os.path.join(terraspec_base, 'slpit', 'output')
        # self.emit_em_libraries = os.path.join(slpit_emit_directories, 'spectral_transects', 'endmembers-raw')


    def build_transects(self):
        # the transect spectra
        records = load_pickle('shift_slpit')

        print("loading... Spectral Transects")
        for i in records:
            if i['site_list'] in ['jorn', 'SRER']:
                continue

            plot_measurements = i['plot_survey_type'].split(",")

            if 'slpit' not in plot_measurements:
                continue

            plot_name = f"{i['site_list'].upper() + i['team_name'].upper()}-{i['site_num']:03d}"
            season = i['season'].upper()
            date = i['date_taken']
            plot_directory = os.path.join(self.base_directory, 'data', f'SHIFT_{season}', plot_name)
            plot_pic_url = i['landscape_pic']

            create_directory(os.path.join(self.output_transect_directory, f'{plot_name}_{season}'))
            create_directory(os.path.join(self.output_transect_directory, f'{plot_name}_{season}', 'RFL'))
            output_transect_directory = os.path.join(self.output_transect_directory, f'{plot_name}_{season}', 'RFL')

            if os.path.isfile(os.path.join(output_transect_directory,  f'{plot_name}_{season}_SLPIT_asd.csv')):
                continue

            try:
                img_data = requests.get(plot_pic_url).content
                with open(os.path.join(self.output_transect_directory, f'{plot_name}_{season}', f'{plot_name}_landscape_pic.jpg'),
                          'wb') as handler:
                    handler.write(img_data)
            except:
                print('plot image not available!')

            print(f'\t loading... {plot_name}-{season}')

            # this gets all files including old non .asd files
            all_asd_files = sorted(glob(os.path.join(plot_directory, '**', '*[!.txt][!.log][!.ini]'), recursive=True))
            create_directory(os.path.join(output_transect_directory, 'individual_spectra_plots'))

            # white ref table
            df_white_ref = slpit.df_white_ref_table(record=i)

            for asd_file in all_asd_files:
                if os.path.isfile(asd_file):
                    pass
                else:
                    all_asd_files.remove(asd_file) # this removes directory paths from all_asd_files

            transect_spectra = []
            for asd_file in all_asd_files:
                if not os.path.isfile(asd_file):
                    continue

                # Get folder names
                parent_dir, filename = os.path.split(asd_file)
                file_type = os.path.split(parent_dir)[1]

                if file_type == 'Endmembers':
                    continue

                # Extract file_num exactly as before
                try:
                    file_num = int(filename.split(".")[0].split("_")[-1])
                except (ValueError, IndexError):
                    file_num = int(filename.split(".")[-1])

                # Extract line_num from two directories up
                line_num = os.path.split(os.path.split(parent_dir)[0])[1].lower()

                # Is this file in the white reference list?
                ref_match = df_white_ref[df_white_ref['line_num'] == line_num]

                if file_num in ref_match['filenumber'].values:
                    # If it is, only add it if it is marked 'good'
                    if file_num in ref_match[ref_match['my_element_2'] == 'good']['filenumber'].values:
                        transect_spectra.append(asd_file)
                else:
                    # If it's not in the white ref list at all, it's a data file we want to keep
                    transect_spectra.append(asd_file)

            if transect_spectra:
                p_map(partial(slpit.plot_shift_file,
                              out_directory=os.path.join(output_transect_directory, 'individual_spectra_plots')),
                      transect_spectra,
                      **{"desc": "\t\t plotting asd files: " + plot_name + "...", "ncols": 150, "num_cpus": 10})


            # parallel process the reflectance files
            results_refl = p_map(partial(spectra.get_shift_transect, plot_directory=plot_directory, season=season),
                                 transect_spectra, **{"desc": f"\t\t processing plot: {season} {plot_name}...", "ncols": 150})

            # # make a dataframe of the results
            df_results = pd.DataFrame(results_refl)
            df_results.columns = ["plot_name", "file_name", "line_num", "file_num", "longitude", "latitude", "elevation",
                                  "utc_time"] + list(self.asd_wvls)

            df_results = df_results.sort_values('file_num')
            df_results.insert(3, "white_ref", 0)
            df_results = df_results.copy()
            df_results['utc_time'] = pd.to_datetime(df_results['utc_time'], format='%H:%M:%S', errors='coerce')
            adjusted_dfs = []

            df_white_ref_good = df_white_ref[df_white_ref['my_element_2'] == 'good'].copy()
            lines_to_correct = df_white_ref_good.groupby('line_num')['white_ref_space'].nunique()
            lines_to_correct = lines_to_correct[lines_to_correct > 1].index.tolist()

            for line_num in lines_to_correct:
                df_select = df_white_ref_good[df_white_ref_good['line_num'] == line_num].copy()
                line_num_max = df_select.filenumber.max()
                line_num_min = df_select.filenumber.min()

                df_select = df_select.rename(columns={'filenumber': 'file_num'})

                df_query = df_results[(df_results['file_num'] >= line_num_min) &
                                      (df_results['file_num'] <= line_num_max) & (df_results['line_num'] == line_num.upper())].copy()

                df_query = pd.merge(df_query, df_select[['file_num', 'white_ref_space']],
                                    on='file_num', how='left')
                df_query['line_num'] = line_num  # Add line_num back
                df_query['white_ref'] = df_query['white_ref_space']
                df_query = df_query[df_query['white_ref_space'] != 'middle']
                df_query = df_query.iloc[:, :-1]  # this drops the join since values are now saved in white_ref

                if df_query.white_ref.nunique() > 1:
                    # get white refs @ t1 and t2
                    df_begin = df_query[df_query['white_ref'] == 'begin']
                    white_reference_spectra_t1 = np.mean(df_begin.iloc[:, 9:].to_numpy(), axis=0)
                    t1 = df_begin.iloc[:, 8].mean()

                    df_end = df_query[df_query['white_ref'] == 'end']
                    white_reference_spectra_t2 = np.mean(df_end.iloc[:, 9:].to_numpy(), axis=0)
                    t2 = df_end.iloc[:, 8].mean()

                    # get spectra to correct
                    df_spectra = df_query[df_query['white_ref'].isnull()].reset_index(drop=True)
                    df_spectra_array = df_spectra.iloc[:, 9:].to_numpy()
                    df_time_array = df_spectra.iloc[:, 8].to_numpy()

                    corrected_reflectance = p_map(partial(slpit.white_ref_correction,
                                                          white_reference_spectra_t1=white_reference_spectra_t1,
                                                          white_reference_spectra_t2=white_reference_spectra_t2,
                                                          time_1=t1, time_2=t2),
                                                  [df_spectra_array[_row, :] for _row in
                                                   range(df_spectra_array.shape[0])],
                                                  [df_time_array[_row] for _row in range(df_time_array.shape[0])],
                                                  **{"desc": f"\t\t processing white reference corrections: "
                                                             f"{plot_name} {line_num}...", "ncols": 150})

                    df_corrected = pd.DataFrame(corrected_reflectance)
                    df_adjusted = pd.concat([df_spectra.iloc[:, :9], df_corrected], axis=1)

                    df_adjusted.columns = df_spectra.columns
                    df_adjusted = df_adjusted.drop('white_ref', axis=1)
                    df_adjusted['utc_time'] = df_adjusted['utc_time'].dt.strftime('%H:%M:%S')
                    df_adjusted.insert(0, "date", date)
                    df_adjusted = df_adjusted.rename({'line_num_x': 'line_num'}, axis=1)  # new method
                    adjusted_dfs.append(df_adjusted)

                else:
                    # get white ref; # this is where we forget to take end white spectra
                    line_num_max = df_select.filenumber.max()

                    # this gets all values if it was line 1 or line 2
                    closest_file_numbers = df_white_ref['filenumber'].values - line_num_max
                    min_index = np.argmin(closest_file_numbers[closest_file_numbers != 0])
                    df_query = df_results[(df_results['file_num'] > line_num_max) & (
                                df_results['file_num'] < df_white_ref['filenumber'].values[min_index])].copy()

                    if df_query.empty:
                        # this gets line 3;
                        df_query = df_results[(df_results['file_num'] > line_num_max)].copy()

                    df_query['line_num'] = line_num
                    df_query = df_query.drop('white_ref', axis=1)
                    df_query.insert(0, "date", date)
                    df_query['utc_time'] = df_query['utc_time'].dt.strftime('%H:%M:%S')
                    adjusted_dfs.append(df_query)
                    print(f"\t\t no white ref correction available on: {plot_name}_{season} {line_num}")

            df_corrected_all = pd.concat(adjusted_dfs)
            df_corrected_all.to_csv(os.path.join(output_transect_directory, f'{plot_name}_{season}_SLPIT_asd.csv'),
                                    index=False)

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm,
                                                                asd_wvl=self.asd_wvls, spectra_starting_col=9),
                                                        list(df_corrected_all.iterrows()),
                                         **{"desc": f"\t\t\tspectral convolve: plot: {plot_name}...", "ncols": 150})

            # save outputs as emit resolutions csv's
            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df_corrected_all.iloc[:, :9].reset_index(drop=True), df_convolve], axis=1)
            df_convolve.to_csv(os.path.join(output_transect_directory, f'{plot_name}_{season}_SLPIT_{self.instrument}.csv'),
                               index=False)

            # get the line counts
            max_line_files = []
            for _line, line in enumerate(sorted(df_convolve.line_num.unique())):
                df_line_select = df_convolve[df_convolve['line_num'] == line].copy()
                df_line_select = df_line_select.sort_values('file_num')
                max_line_files.append(df_line_select.shape[0])

            # save files as envi files
            spectra_grid = np.ones((max(max_line_files), len(df_convolve.line_num.unique()), len(self.wvls))) * -9999

            for _line, line in enumerate(df_convolve.line_num.unique()):
                df_line_select = df_convolve[df_convolve['line_num'] == line].copy()
                df_line_select = df_line_select.sort_values('file_num')

                line_spectra_array = df_line_select.iloc[:, 9:].to_numpy()

                for _row, row in enumerate(line_spectra_array):
                    spectra_grid[_row, _line, :] = line_spectra_array[_row, :]

            # save the spectra
            print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
            meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                    wvls=True)
            meta_spectra['data ignore value'] = -9999
            output_raster = os.path.join(output_transect_directory, f'{plot_name}_{season}_SLPIT_{self.instrument}.hdr')
            save_envi(output_raster, meta_spectra, spectra_grid)

            asd_spectra_grid = np.ones((max(max_line_files), len(df_corrected_all.line_num.unique()), len(self.asd_wvls))) * -9999

            for _line, line in enumerate(df_corrected_all.line_num.unique()):
                df_line_select = df_corrected_all[df_corrected_all['line_num'] == line].copy()
                df_line_select = df_line_select.sort_values('file_num')

                line_spectra_array = df_line_select.iloc[:, 9:].to_numpy()

                for _row, row in enumerate(line_spectra_array):
                    asd_spectra_grid[_row, _line, :] = line_spectra_array[_row, :]

            # save the asd spectra
            print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
            meta_spectra = get_meta(lines=asd_spectra_grid.shape[0], samples=asd_spectra_grid.shape[1], bands=self.asd_wvls,
                                    wvls=True)
            output_raster = os.path.join(output_transect_directory, f'{plot_name}_{season}_SLPIT_asd.hdr')
            save_envi(output_raster, meta_spectra, asd_spectra_grid)

    def build_endmember_lib(self):
        # the transect spectra
        records = load_pickle('shift_slpit')

        print("loading... Spectral Transects Endmembers")

        for i in records:
            if i['site_list'] in ['jorn', 'SRER']:
                continue

            plot_name = f"{i['site_list'].upper() + i['team_name'].upper()}-{i['site_num']:03d}"
            season = i['season'].upper()
            print(f'\t loading... {plot_name}-{season}')
            plot_measurements = i['plot_survey_type'].split(",")

            if 'endmembers' not in plot_measurements:
                continue

            date = i['date_taken']
            plot_pic_url = i['landscape_pic']

            plot_directory = os.path.join(self.base_directory, 'data', f'SHIFT_{season}', plot_name)
            create_directory(os.path.join(self.output_transect_directory, f'{plot_name}_{season}'))
            create_directory(os.path.join(self.output_transect_directory, f'{plot_name}_{season}', 'EMS'))
            plot_em_directory = os.path.join(self.output_transect_directory, f'{plot_name}_{season}', 'EMS')

            if os.path.isfile(os.path.join(plot_em_directory, f'{plot_name}_{season}_EMS_asd.csv')):
                continue

            try:
                img_data = requests.get(plot_pic_url).content
                with open(os.path.join(self.output_transect_directory, f'{plot_name}_{season}', f'{plot_name}_landscape_pic.jpg'),
                          'wb') as handler:
                    handler.write(img_data)
            except:
                print('plot image not available!')




            # em table
            df_transect_em = pd.json_normalize(i['em'])
            df_transect_em = df_transect_em.loc[(df_transect_em['em_condition'] != 'bad') &
                                                (df_transect_em['em_classification'] != 'flower')].copy()
            df_transect_em = df_transect_em.iloc[:, 14:]

            # get all endmembers
            all_asd_files = sorted(glob(os.path.join(plot_directory, '**', '*[!.txt][!.log][!.ini]'), recursive=True))

            for asd_file in all_asd_files:
                if os.path.isfile(asd_file):
                    pass
                else:
                    all_asd_files.remove(asd_file) # this removes directory paths from all_asd_files

            transect_spectra = []
            for asd_file in all_asd_files:
                if not os.path.isfile(asd_file):
                    continue

                # Get folder names
                parent_dir, filename = os.path.split(asd_file)
                file_type = os.path.split(parent_dir)[1]

                if file_type == 'Transect':
                    continue

                # Extract file_num exactly as before
                try:
                    file_num = int(filename.split(".")[0].split("_")[-1])
                except (ValueError, IndexError):
                    file_num = int(filename.split(".")[-1])

                # Extract line_num from two directories up
                line_num = os.path.split(os.path.split(parent_dir)[0])[1].lower()

                ref_match = df_transect_em[df_transect_em['line_num'] == line_num]

                if file_num in ref_match['asd_file_num'].values:
                    # If it is, only add it if it is marked 'good'
                    if file_num in ref_match[ref_match['em_condition'] != 'bad']['asd_file_num'].values:
                        transect_spectra.append(asd_file)

            # parallel process the reflectance files
            results_refl = p_map(partial(spectra.get_shift_transect, plot_directory=plot_directory, season=season),
                             transect_spectra, **{"desc": f"\t\t processing plot: {season} {plot_name}...", "ncols": 150})

            # make a dataframe of the results
            df_results = pd.DataFrame(results_refl)
            df_results.columns = ["plot_name", "file_name", "line_num", "file_num", "longitude", "latitude",
                                  "elevation", "utc_time"] + list(self.asd_wvls)

            df_results = df_results.sort_values('file_num')
            df_results.insert(0, "date", date)
            df_results.insert(4, "level_1", '')
            df_results.insert(5, "species", '')
            df_results.insert(6, "notes", '')

            # filter by line
            for line_num in sorted(df_results.line_num.unique()):
                df_line = df_results[df_results['line_num'] == line_num.upper()].copy()

                for file_num in df_line.file_num.values:
                    em_clas = df_transect_em.loc[(df_transect_em['asd_file_num'] == file_num) & (df_transect_em['line_num'] == line_num.lower()), 'em_classification'].iloc[0]
                    species = df_transect_em.loc[(df_transect_em['asd_file_num'] == file_num) & (df_transect_em['line_num'] == line_num.lower()), 'species_name'].iloc[0]
                    notes = df_transect_em.loc[(df_transect_em['asd_file_num'] == file_num) & (df_transect_em['line_num'] == line_num.lower()), 'notes'].iloc[0]

                    df_results.loc[(df_results['file_num'] == file_num) & (df_results['line_num'] == line_num.upper()), ['level_1', 'species', 'notes']] = em_clas, species, notes

            df_results = df_results.sort_values("level_1")
            df_results.to_csv(os.path.join(plot_em_directory, f'{plot_name}_{season}_EMS_asd.csv'), index=False)

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm, asd_wvl=self.asd_wvls,
                                             spectra_starting_col=12), list(df_results.iterrows()),
                                     **{"desc": f"\t\t\tconvulsing plot: {plot_name}...", "ncols": 150})

            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df_results.iloc[:, :12].reset_index(drop=True), df_convolve], axis=1)
            df_convolve = df_convolve.reset_index(drop=True)

            # save original csv endmembers
            df_convolve = df_convolve.sort_values("level_1")
            df_convolve['plot_name'] = f'{plot_name}_{season}'
            df_convolve.to_csv(os.path.join(plot_em_directory, f'{plot_name}_{season}_EMS_{self.instrument}.csv'), index=False)

            # save files as envi files
            spectra_grid = np.zeros((len(results_convolve), 1, len(self.wvls)))

            # fill spectral data
            for _row, row in enumerate(results_convolve):
                spectra_grid[_row, :, :] = row

            # save the spectra
            print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
            meta_spectra = get_meta(lines=len(results_convolve), samples=spectra_grid.shape[1], bands=self.wvls,
                                    wvls=True)
            output_raster = os.path.join(plot_em_directory, f'{plot_name}_{season}_EMS_{self.instrument}.hdr')
            save_envi(output_raster, meta_spectra, spectra_grid)
            time.sleep(3)
            print("done")

    def build_em_collection(self):
        print("loading em collection...")
        # merge all endmembers - instrument based wavelengths
        instrument_ems = spectra.get_all_ems(output_directory=self.output_directory, instrument=self.instrument)
        asd_ems = spectra.get_all_ems(output_directory=self.output_directory, instrument='asd')

        # # dataframes of instrument
        df = pd.concat((pd.read_csv(f) for f in instrument_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, f"all-endmembers-{self.instrument}.csv"), index=False)
        spectra.df_to_envi(df=df, spectral_starting_column=12, wvls=self.wvls,
                           output_raster=os.path.join(self.output_directory, f'all-endmembers-{self.instrument}.hdr'))

        # merge all endmembers - asd based wavelengths
        df = pd.concat((pd.read_csv(f) for f in asd_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-asd.csv"), index=False)

        # merge all transect spectra
        instrument_transects = glob(os.path.join(self.output_transect_directory, '**', f'*_SLPIT_{self.instrument}.csv'), recursive=True)
        df = pd.concat((pd.read_csv(f) for f in instrument_transects), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, f"all-SLPIT-{self.instrument}.csv"), index=False)
        spectra.df_to_envi(df=df, spectral_starting_column=9, wvls=self.wvls,
                          output_raster=os.path.join(self.output_directory, f'all-SLPIT-{self.instrument}.hdr'))

    def build_gis_data(self):
        print("Building spectral endmember gis shapefile data...", sep=' ', end='', flush=True)
        df = pd.read_csv(os.path.join(self.output_directory, f'all-endmembers-{self.instrument}.csv'))
        df = df.iloc[:, :12]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, out_name=f'{self.instrument}_shift_endmembers')

        df = pd.read_csv(os.path.join(self.output_directory, f'all-SLPIT-{self.instrument}.csv'))

        df_rows = []
        for plot in sorted(list(df.plot_name.unique())):
            df_select = df[df['plot_name'] == plot].copy()
            long_centroid = np.mean(df_select['longitude'])
            lat_centroid = np.mean(df_select['latitude'])
            mean_elevation = np.mean(df_select['elevation'])
            season = plot.split('_')[1]
            df_rows.append([plot, season, long_centroid, lat_centroid, mean_elevation])

        df_centroid = pd.DataFrame(df_rows)
        df_centroid.columns =['plot', 'season', 'longitude', 'latitude', 'mean_elevation']
        spectra.df_to_shapefile(df_centroid, out_name='shift_transects_centroid')

        df = df.iloc[:, :9]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, out_name=f'shift_slpit_{self.instrument}')
        time.sleep(3)

        print("done")
    #
    def nearest_site(self):
        print("calculating distances")
        # get plot center points from SHIFT
        shapefile_shift = os.path.join('gis', "shift_transects_centroid.geojson")

        # get plot points from EMIT
        shapefile_emit = os.path.join('gis', "Observation.json")
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

    def em_qty_check(self):
        print('calculating em quantity check')
        # ensures that in each csv there at least 3 classes and n samples for each class
        # will use the nearest site for geographic distance
        em_min_samples = {'pv': 30, 'npv': 30, 'soil': 75}

        shift_ems = sorted(spectra.get_all_ems(output_directory=self.output_directory, instrument='asd'))
        df_all_shift_ems = pd.read_csv(os.path.join(self.output_directory, f"all-endmembers-asd.csv"), low_memory=False)
        df_all_shift_ems.insert(0, "plot_num", '')

        base_directory = os.path.abspath(os.path.join(self.base_directory, '..'))
        df_all_emit_ems = pd.read_csv(os.path.join(base_directory, 'slpit', 'output', "all-endmembers-asd.csv"), low_memory=False)

        df_all_emit_ems['plot_name'] = df_all_emit_ems['plot_name'].str.replace('Spectral', 'SPEC')
        df_all_emit_ems['level_1'] = df_all_emit_ems['level_1'].str.lower()
        df_all_emit_ems.insert(0, "plot_num", '')
        df_all_emit_ems['plot_num'] = df_all_emit_ems['plot_name'].str.split('-').str[1].astype(int)
        df_all_emit_ems = df_all_emit_ems[df_all_emit_ems['plot_num'] <= 60].copy()
        df_all_ems = pd.concat([df_all_shift_ems, df_all_emit_ems], ignore_index=True)

        df_distance = pd.read_csv(os.path.join('gis', 'shift_min_dist_to_all_plots.csv'), low_memory=False)
        all_ems = sorted(list(df_all_ems.level_1.unique()))

        for i in shift_ems:
            plot = f"{os.path.basename(i).split('_')[0]}_{os.path.basename(i).split('_')[1]}"

            if plot in ['DPA-9999_FALL', 'SRA-9999_FALL']:
                continue

            df_em_site = pd.read_csv(i, low_memory=False)
            df_em_site['level_1'] = df_em_site['level_1'].str.lower()
            site_em = sorted(list(df_em_site.level_1.unique()))

            df_nearest_distances = df_distance.loc[df_distance['shift_plot_analysis'] == f"{plot}"].copy()

            out_csv = os.path.join(self.output_transect_directory, f'{plot}',
                                   f"unmix_{plot}_EMS_{self.instrument}.csv")

            if os.path.isfile(out_csv):
                print(f'{out_csv} exists, skipping!')
                continue

            ems_to_append = []

            for em in ['npv', 'pv', 'soil']:
                # 1. Get current count for this specific EM at the site
                df_em_select = df_em_site.loc[df_em_site['level_1'] == em]
                current_count = df_em_select.shape[0]

                # 2. Check if we need more
                if current_count < em_min_samples[em]:
                    remaining_samples = em_min_samples[em] - current_count
                    site_counter = 0

                    print(f"Site {plot} needs {remaining_samples} more samples of {em}")

                    while remaining_samples > 0:
                        # Safety check: don't exceed distance matrix columns
                        if site_counter >= df_nearest_distances.shape[1]:
                            print(f"Exhausted all sites. Shortfall of {remaining_samples} for {em}")
                            break

                        current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0]

                        # Filter SPEC/THERM plots > 60
                        if current_nearest_site.split('-')[0].strip() in ['SPEC', 'THERM']:
                            current_plot_num = int(current_nearest_site.split('-')[1].strip())
                            if current_plot_num > 60:
                                site_counter += 1
                                continue

                        # Pull candidates from df_all_ems
                        df_nearest_ems = df_all_ems.loc[
                            (df_all_ems['level_1'] == em) &
                            (df_all_ems['plot_name'] == current_nearest_site)
                            ].copy()

                        if df_nearest_ems.empty:
                            site_counter += 1
                            continue

                        # Calculate how many to take from this neighbor
                        take_n = min(len(df_nearest_ems), remaining_samples)
                        df_rand = df_nearest_ems.sample(n=take_n, random_state=13, ignore_index=True)

                        ems_to_append.append(df_rand)

                        remaining_samples -= take_n
                        site_counter += 1

            # Final Save Logic
            if not ems_to_append:
                df_em_site.to_csv(out_csv, index=False)
            else:
                df_append = pd.concat(ems_to_append, ignore_index=True)
                df_combined = pd.concat([df_em_site, df_append], ignore_index=True)
                df_combined = df_combined.sort_values('level_1')
                df_combined = df_combined.drop('plot_num', axis=1)
                results_convolve = p_map(partial(spectra.convolve, wvl=self.wvls, fwhm=self.fwhm,
                                                 asd_wvl=self.asd_wvls, spectra_starting_col=12),
                                         list(df_combined.iterrows()),
                                         **{"desc": f"\t\t\tspectral convolve: plot: {plot}...", "ncols": 150})
                df_convolve = pd.DataFrame(results_convolve)
                df_convolve.columns = list(self.wvls)
                df_convolve = pd.concat([df_combined.iloc[:, :12].reset_index(drop=True), df_convolve], axis=1)
                df_convolve.to_csv(out_csv, index=False)

    def unmix_reflectances(self, sensor):
        # scene overlaps
        scene_key = {'DPA-004_FALL': {'flightline': 'ang20220915t195816', 'version': '003'},
                     'DPB-003_FALL': {'flightline': 'ang20220915t195816', 'version': '003'},
                     'DPB-004_FALL': {'flightline': 'ang20220915t200714', 'version': '000'},
                     'DPB-005_FALL': {'flightline': 'ang20220915t195816', 'version': '003'},
                     'DPB-020_SPRING': {'flightline': 'ang20220322t204749', 'version': '000'},
                     'DPB-027_SPRING': {'flightline': 'ang20220412t205404', 'version': '001'},
                     'SRA-007_FALL': {'flightline': 'ang20220914t183400', 'version': '000'},
                     'SRA-008_FALL': {'flightline': 'ang20220914t183400', 'version': '000'},
                     'SRA-019_SPRING': {'flightline': 'ang20220308t204043', 'version': '008'},
                     'SRA-020_SPRING': {'flightline': 'ang20220308t205512', 'version': '002'},
                     'SRA-021_SPRING': {'flightline': 'ang20220308t204043', 'version': '008'},
                     'SRA-033_SPRING': {'flightline': 'ang20220316t210303', 'version': '002'},
                     'SRA-034_SPRING': {'flightline': 'ang20220316t210303', 'version': '002'},
                     'SRA-056_FALL': {'flightline': 'ang20220914t184300', 'version': '000'},
                     'SRA-109_SPRING': {'flightline': 'ang20220511t190344', 'version': '002'},
                     'SRB-010_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
                     'SRB-021_SPRING': {'flightline': 'ang20220308t205512', 'version': '002'},
                     'SRB-026_SPRING': {'flightline': 'ang20220308t204043', 'version': '007'},
                     'SRB-045_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
                     'SRB-046_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
                     'SRB-084_SPRING': {'flightline': 'ang20220511t191813', 'version': '007'},
                     'SRB-100_FALL': {'flightline': 'ang20220915t203517', 'version': '001'},
                     'SRB-050_FALL': {'flightline': 'ang20220914t184300', 'version': '001'},
                     'SRB-047_SPRING': {'flightline': 'ang20220405t1359', 'version': '002'},
                     }

        # create outlogs for unmix and tc
        create_directory(os.path.join(self.output_transect_directory, 'unmix_tc_outlogs'))
        extract_outlog_directory = os.path.join(self.output_transect_directory, 'unmix_tc_outlogs')

        # get plot center points from ipad - these are the plot centers
        spatial_field_data = os.path.join('gis', "shift_transects_centroid.geojson")
        gdf = gpd.read_file(spatial_field_data)

        # # get reflectance and uncertainty files
        reflectance_slpit_files = sorted(glob(os.path.join(self.output_transect_directory, '**', f'*_SLPIT_{sensor}'), recursive=True))
        
        for i in reflectance_slpit_files:
            plot_name = os.path.basename(i).split("_")[0]
            season = os.path.basename(i).split("_")[1]
            plot_number = f"{plot_name}_{season}"
            
            if plot_number in ['SRB-004_FALL', 'SRB-200_FALL','SRA-000_SPRING', 'DPB-9999_FALL', 'SRB-9999_FALL']:
                continue
            
            plot_base_directory = os.path.join(self.output_transect_directory, plot_number)
            em_file = os.path.join(plot_base_directory, f'unmix_{plot_number}_EMS_{sensor}.csv')
            em_local_rfl = os.path.join(plot_base_directory, 'EMS', f'{plot_number}_EMS_{sensor}')
            rfl_ext = os.path.join(plot_base_directory, 'EXT', f'{plot_number}_RFL_{scene_key[plot_number]["flightline"]}_{scene_key[plot_number]["version"]}_EXT')
            rfl_unc = os.path.join(plot_base_directory, 'EXT', f'{plot_number}_UNC_{scene_key[plot_number]["flightline"]}_{scene_key[plot_number]["version"]}_EXT')
            outfile = os.path.join(extract_outlog_directory, f'{os.path.basename(i)}.out')

            base_call = f'sh {os.path.join("shift", "slpit_shift_image_processing.sh" )} {i} {em_file} {plot_base_directory} {em_local_rfl} {rfl_ext} {rfl_unc}'
            sbatch_cmd = f"sbatch --export=ALL -p patient -N 1 -c 1 --mem 20G --output {outfile} --job-name shift.umix  --wrap='{base_call}'"
            subprocess.run(sbatch_cmd, shell=True, text=True)

def run_build_workflow(base_directory, sensor):

    lib = build_libraries(base_directory=base_directory, sensor=sensor)
    lib.build_transects()
    lib.build_endmember_lib()
    lib.build_em_collection()
    lib.build_gis_data()
    if not os.path.isfile(os.path.join('gis', 'shift_min_dist_to_all_plots.csv')):
        lib.nearest_site()
    lib.em_qty_check()
    lib.unmix_reflectances(sensor=sensor)


