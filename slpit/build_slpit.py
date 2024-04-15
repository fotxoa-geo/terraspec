import time
import numpy as np
from glob import glob
import os
from utils.slpit_download import load_pickle
import pandas as pd
from utils import asdreader, sedreader
from functools import partial
from p_tqdm import p_map
import requests
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.envi import get_meta, save_envi
from utils.text_guide import cursor_print, query_yes_no
from utils.slpit_utils import slpit
from math import radians, sin, cos, sqrt, atan2
import geopandas as gpd


def haversine_distance(lat1, lon1, lat2, lon2, plot):
    R = 6371.0  # Earth radius in kilometers

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c # in km
    if distance == 0:
        distance = np.nan

    return [plot, distance] # returns distance in m


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

        # team names keys - corresponds to suffix in ASD files
        self.team_keys = {
            'spectral': 'SP', 'thermal': 'TM'}

        # input data directories
        self.spectral_em_directory = os.path.join(self.base_directory, 'data', 'spectral_endmembers')
        self.spectral_transect_directory = os.path.join(self.base_directory, 'data', 'spectral_transects')

        # instrument to indicate wavelengths in output folder
        self.instrument = sensor

        # output data directories
        self.output_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')
        self.output_transect_em_directory = os.path.join(self.output_directory, 'spectral_transects', 'endmembers')
        self.output_transect_em_directory_raw = os.path.join(self.output_directory, 'spectral_transects', 'endmembers-raw')

        # import the simulation outputs
        terraspec_base = os.path.join(base_directory, "..")
        em_sim_directory = os.path.join(terraspec_base, 'simulation', 'output')
        self.emit_global = os.path.join(em_sim_directory, 'convolved','geofilter_convolved.csv')
        self.convex_global = os.path.join(em_sim_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv')

    def build_emit_transects(self):
        # the transect spectra
        records = load_pickle('emit_slpit')

        print("loading... Spectral Transects")
        for i in records:
            plot_name = f"{i['team_names'].capitalize()} - {i['plot_num']:03d}"
            plot_directory = os.path.join(self.spectral_transect_directory, plot_name)
            plot_pic_url = i['landscape_pic']
            date = i['sample_date']
            emit_date = i['overpass_date']

            if os.path.isfile(os.path.join(self.output_transect_directory, plot_name + '- transect-' + self.instrument + '.csv')):
                continue

            img_data = requests.get(plot_pic_url).content
            with open(os.path.join(self.output_directory, 'plot_pictures', 'spectral_transects', plot_name + '.jpg'),
                      'wb') as handler:
                handler.write(img_data)

            # white ref table
            df_white_ref = slpit.df_white_ref_table(record=i)

            # # em table
            df_transect_em = slpit.df_em_table(record=i)

            # get all asd files from folder
            all_asd_files = sorted(glob(os.path.join(plot_directory, '*.asd')))
            if not all_asd_files:
                print(".asd files not found! Looking for .sed files...")
                all_asd_files = sorted(glob(os.path.join(plot_directory, '*.sed')))

            transect_spectra = []
            for asd_file in all_asd_files:
                file_num = int(os.path.basename(asd_file).split(".")[0].split("_")[-1])

                # check if file is in white ref or em
                if file_num in df_transect_em.asd_file_num.values:
                    pass
                else:

                    # keep only the good white ref files
                    if file_num in df_white_ref.filenumber.values:
                        good_white_ref = df_white_ref[df_white_ref['my_element_2'] == 'good'].copy()

                        # ignore bad white ref files
                        if file_num in good_white_ref.filenumber.values:
                            transect_spectra.append(asd_file)
                        else:
                            pass
                    else:
                        transect_spectra.append(asd_file)

            results_refl = p_map(partial(spectra.get_reflectance_transect, plot_directory=plot_directory,
                                         team_name_key=self.team_keys[i['team_names']]), transect_spectra,
                                 **{
                                     "desc": "\t\t processing plot: " + plot_name + " ...",
                                     "ncols": 150})

            try:
                asd = asdreader.reader(results_refl[0][1])
            except:
                asd = sedreader.reader(results_refl[0][1])

            df_results = pd.DataFrame(results_refl)
            df_results.columns = ["plot_name", "file_name", "file_num", "longitude", "latitude", "elevation",
                                  "utc_time"] + list(asd.wavelengths)
            df_results = df_results.sort_values('file_num')
            df_results.insert(3, "white_ref", 0)
            df_results.insert(4, "line_num", 0)
            df_results = df_results.copy()
            df_results['utc_time'] = pd.to_datetime(df_results['utc_time'], format='%H:%M:%S')

            adjusted_dfs = []

            for line_num in df_white_ref.line_num.unique():
                df_select = df_white_ref.loc[
                    (df_white_ref['line_num'] == line_num) & (df_white_ref['my_element_2'] == 'good')].copy()

                if len(list(df_select.white_ref_space.unique())) > 1:
                    line_num_max = df_select.filenumber.max()
                    line_num_min = df_select.filenumber.min()

                    df_select = df_select.rename({
                                                     'filenumber': 'file_num'}, axis=1)  # new method

                    df_query = df_results[
                        (df_results['file_num'] >= line_num_min) & (df_results['file_num'] <= line_num_max)].copy()
                    df_query['line_num'] = line_num
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
                    delta_time = np.array(np.mean(df_end_time.total_seconds.values) - np.mean(df_begin_time.total_seconds.values))
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
                    df_adjusted = df_adjusted.rename({
                                                         'line_num_x': 'line_num'}, axis=1)  # new method
                    adjusted_dfs.append(df_adjusted)

                else:
                    # get white ref; # this is where we forget to take end white spectra
                    line_num_max = df_select.filenumber.max()

                    # this gets all values if it was line 1 or line 2
                    closest_file_numbers = df_white_ref['filenumber'].values - line_num_max
                    min_index = np.argmin(closest_file_numbers[closest_file_numbers != 0])
                    df_query = df_results[(df_results['file_num'] > line_num_max) & (df_results['file_num'] < df_white_ref['filenumber'].values[min_index])].copy()

                    if df_query.empty:
                        # this gets line 3;
                        df_query = df_results[(df_results['file_num'] > line_num_max)].copy()

                    df_query['line_num'] = line_num
                    df_query = df_query.drop('white_ref', axis=1)
                    df_query.insert(0, "date", date)
                    df_query['utc_time'] = df_query['utc_time'].dt.strftime('%H:%M:%S')
                    adjusted_dfs.append(df_query)
                    print("\t\t no white ref correction available on: ", plot_name, line_num)


            df_corrected_all = pd.concat(adjusted_dfs)
            df_corrected_all.to_csv(os.path.join(self.output_transect_directory, plot_name + '- transect.csv'),
                                    index=False)

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve_asdfile,  wvl=self.wvls, fwhm=self.fwhm),
                                     df_corrected_all.file_name.values.tolist(),
                                     **{"desc": "\t\t\tconvulsing plot: " + plot_name + " ...", "ncols": 150})

            # save outputs as emit resolutions csv's
            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df_corrected_all.iloc[:, :9].reset_index(drop=True), df_convolve], axis=1)
            df_convolve.to_csv(os.path.join(self.output_transect_directory, plot_name + '- transect-' + self.instrument + '.csv'), index=False)

            # save files as envi files
            spectra_grid = np.zeros((len(results_convolve), 1, len(self.wvls)))

            # fill spectral data
            for _row, row in enumerate(results_convolve):
                spectra_grid[_row, :, :] = row

            # save the spectra
            print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
            meta_spectra = get_meta(lines=len(results_convolve), samples=spectra_grid.shape[1], bands=self.wvls,
                                    wvls=True)
            output_raster = os.path.join(self.output_transect_directory, plot_name.replace(" ", "") + ".hdr")
            save_envi(output_raster, meta_spectra, spectra_grid)
            time.sleep(3)
            print("done")

    def build_emit_endmembers(self):
        # transect endmembers
        records = load_pickle('emit_slpit')

        print("loading... Spectral Transects Endmembers")
        all_ems = ['NPV', 'PV', 'Soil']
        for i in records:
            plot_name = f"{i['team_names'].capitalize()} - {i['plot_num']:03d}"
            plot_directory = os.path.join(self.spectral_transect_directory, plot_name)
            date = i['sample_date']
            if os.path.isfile(os.path.join(self.output_transect_em_directory_raw, f'{plot_name.replace(" ", "")}-{self.instrument}.csv')):
                continue

            # em table
            df_transect_em = slpit.df_em_table(record=i)
            df_transect_em = df_transect_em.loc[df_transect_em['em_condition'] != 'bad'].copy()
            df_transect_em = df_transect_em.loc[df_transect_em['endmembers'] != 'Flower'].copy()

            # get all asd files from folder
            all_asd_files = sorted(glob(os.path.join(plot_directory, '*.asd')))
            if not all_asd_files:
                print(".asd files not found! Looking for .sed files...")
                all_asd_files = sorted(glob(os.path.join(plot_directory, '*.sed')))

            endmember_spectra = []
            for asd_file in all_asd_files:
                file_num = int(os.path.basename(asd_file).split(".")[0].split("_")[-1])

                # check if file is in white ref or em
                if file_num in df_transect_em.asd_file_num.values:
                    endmember_spectra.append(asd_file)

                else:
                    pass

            results = p_map(partial(spectra.get_reflectance_transect, plot_directory=plot_directory,
                                    team_name_key=self.team_keys[i['team_names']]), endmember_spectra,
                            **{"desc": "\t\t processing plot: " + plot_name + " ...", "ncols": 150})

            try:
                asd = asdreader.reader(results[0][1])
            except:
                asd = sedreader.reader(results[0][1])

            df_results = pd.DataFrame(results)
            df_results.columns = ["plot_name", "file_name", "file_num", "longitude", "latitude", "elevation",
                                  "utc_time"] + list(asd.wavelengths)
            df_results.insert(0, "date", date)
            df_results.insert(3, "line_num", '')
            df_results.insert(4, "level_1", '')
            df_results.insert(5, "species", '')

            for file_num in df_results.file_num.values:
                if file_num in df_transect_em.asd_file_num.values:
                    em_clas = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'endmembers'].iloc[0]
                    line_num = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'transect_line_num'].iloc[
                        0]
                    species = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'species'].iloc[0]

                    df_results.loc[df_results['file_num'] == file_num, ['line_num', 'level_1',
                                                                        'species']] = line_num, em_clas, species

            df_results = df_results.sort_values("level_1")
            df_results.to_csv(os.path.join(self.output_transect_em_directory_raw, plot_name.replace(" ", "") + '-asd.csv'),
                              index=False)

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve_asdfile, wvl=self.wvls, fwhm=self.fwhm),
                                     df_results.file_name.values.tolist(),
                                     **{
                                         "desc": "\t\t\tconvulsing plot: " + plot_name + " ...",
                                         "ncols": 150})
            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df_results.iloc[:, :10].reset_index(drop=True), df_convolve], axis=1)

            df_convolve = df_convolve.sort_values("level_1")
            df_convolve.to_csv(os.path.join(self.output_transect_em_directory_raw,
                                            plot_name.replace(" ", "") + '-' + self.instrument + '.csv'), index=False)

            # save files as envi files
            spectra_grid = np.zeros((len(results_convolve), 1, len(self.wvls)))

            # fill spectral data
            for _row, row in enumerate(results_convolve):
                spectra_grid[_row, :, :] = row

            # save the spectra
            print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
            meta_spectra = get_meta(lines=len(results_convolve), samples=spectra_grid.shape[1], bands=self.wvls,
                                    wvls=True)
            output_raster = os.path.join(self.output_transect_em_directory_raw, plot_name.replace(" ", "") + '-' + self.instrument + ".hdr")
            save_envi(output_raster, meta_spectra, spectra_grid)
            time.sleep(3)
            print("done")

    def em_qty_check(self):
        # ensures that in each csv there at least 3 classes and n samples for each class
        # will use the nearest site for geographic distance
        em_min_samples = {'PV': 30, 'NPV': 30, 'Soil': 75}

        emit_ems = sorted(spectra.get_all_ems(output_directory=self.output_directory, instrument=self.instrument))
        df_all_emit_ems = pd.read_csv(os.path.join(self.output_directory, "all-endmembers-" + self.instrument + ".csv"))
        df_distance = pd.read_csv(os.path.join('gis', 'min_dist_to_emit_plots.csv'))
        all_ems = sorted(list(df_all_emit_ems.level_1.unique()))

        for i in emit_ems:
            plot_number = os.path.basename(i).split('-')[1]
            df_em_site = pd.read_csv(i)
            site_em = sorted(list(df_em_site.level_1.unique()))
            df_nearest_distances = df_distance.loc[df_distance['emit_plot_analysis'] == f"SPEC - {plot_number}"].copy()

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
                            current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0].split('-')[1]
                            df_nearest_ems = df_all_emit_ems.loc[(df_all_emit_ems['level_1'] == em) & (df_all_emit_ems['plot_name'] == f'Spectral - {current_nearest_site.strip()}')].copy()

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

                for em in em_difference:

                    remaining_samples = em_min_samples[em]
                    site_counter = 0

                    while remaining_samples > 0:
                        current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0].split('-')[1]
                        df_nearest_ems = df_all_emit_ems.loc[(df_all_emit_ems['level_1'] == em) & (
                                df_all_emit_ems['plot_name'] == f'Spectral - {current_nearest_site.strip()}')].copy()

                        df_rand = df_nearest_ems.sample(n=remaining_samples, random_state=13, ignore_index=True)
                        ems_to_append.append(df_rand)

                        # update counters
                        site_counter += 1
                        remaining_samples -= remaining_samples

            # if list is empty do nothing
            out_csv = os.path.join(self.output_transect_em_directory, f"{os.path.basename(i)}")

            if not ems_to_append:
                df_em_site.to_csv(out_csv, index=False)
            else:
                df_append = pd.concat(ems_to_append, ignore_index=True)
                df_convolve = pd.concat([df_em_site, df_append], ignore_index=True)
                df_convolve = df_convolve.sort_values('level_1')
                df_convolve.to_csv(out_csv, index=False)

    def build_em_collection(self):
        # merge all endmembers - instrument based wavelengths
        emit_ems = spectra.get_all_ems(output_directory=self.output_directory, instrument=self.instrument)
        asd_ems = spectra.get_all_ems(output_directory=self.output_directory, instrument='asd')

        # dataframes of all endmembers
        df = pd.concat((pd.read_csv(f) for f in emit_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-" + self.instrument + ".csv"), index=False)
        spectra.df_to_envi(df=df, spectral_starting_column=10, wvls=self.wvls,
                           output_raster=os.path.join(self.output_directory, "all-endmembers-" + self.instrument + ".hdr"))

        # merge all endmembers - asd based wavelengths
        df = pd.concat((pd.read_csv(f) for f in asd_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-asd.csv"), index=False)

        # merge all transect spectra - emit
        emit_transects = glob(os.path.join(self.output_transect_directory, "*transect-" + self.instrument + ".csv"))
        df_transect = pd.concat((pd.read_csv(f) for f in emit_transects), ignore_index=True)
        df_transect.to_csv(os.path.join(self.output_directory, "all-transect-emit.csv"), index=False)
        spectra.df_to_envi(df=df_transect, spectral_starting_column=9, wvls=self.wvls,
                           output_raster=os.path.join(self.output_directory, "all-transect-" + self.instrument + ".hdr"))

    def build_gis_data(self):
        print("Building spectral endmember gis shapefile data...", sep=' ', end='', flush=True)
        df = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-' + self.instrument + '.csv'))
        df = df.iloc[:, :9]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, base_directory=self.base_directory, out_name='emit_global_spectral_lib')

        df = pd.read_csv(os.path.join(self.output_directory, 'all-transect-' + self.instrument + '.csv'))
        df = df.iloc[:, :8]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, base_directory=self.base_directory, out_name='emit_global_transect')
        time.sleep(3)
        print("done")

    def build_derivative_library(self):
        print('Loading EMIT global library...')
        df_global = pd.read_csv(os.path.join(self.emit_global))
        global_results = p_map(partial(spectra.first_derivative, spectral_starting_col=7, wvls=self.wvls), df_global.iterrows(),
                               **{"desc": "\t\t\tcalculating first derivative: global library", "ncols": 150})
        df_results = pd.DataFrame(global_results)
        df_derivative = pd.concat([df_global.iloc[:, :7].reset_index(drop=True), df_results], axis=1)
        df_derivative.columns = df_global.columns
        df_derivative.to_csv(os.path.join(self.output_directory, 'emit-global_first_derivative.csv'), index=False)

        print('EMIT convex hull 4d...')
        df_convex = pd.read_csv(self.convex_global)
        convex_results = p_map(partial(spectra.first_derivative, spectral_starting_col=7, wvls=self.wvls), df_convex.iterrows(),
                               **{"desc": "\t\t\tcalculating first derivative: convex hull library", "ncols": 150})
        df_results = pd.DataFrame(convex_results)
        df_derivative = pd.concat([df_convex.iloc[:, :7].reset_index(drop=True), df_results], axis=1)
        df_derivative.columns = df_convex.columns
        df_derivative.to_csv(os.path.join(self.output_directory, 'convex-hull_4d_first_derivative.csv'), index=False)

        print("loading plot 3....")
        df_plot = pd.read_csv(os.path.join(self.output_transect_em_directory, 'Spectral-003-emit.csv'))
        plot_results = p_map(partial(spectra.first_derivative, spectral_starting_col=10, wvls=self.wvls), df_plot.iterrows(),
                               **{"desc": "\t\t\tcalculating first derivative: plot 003", "ncols": 150})
        df_results = pd.DataFrame(plot_results)
        df_derivative = pd.concat([df_plot.iloc[:, :10].reset_index(drop=True), df_results], axis=1)
        df_derivative.columns = df_plot.columns
        df_derivative.to_csv(os.path.join(self.output_directory, 'SPEC-003-fd.csv'), index=False)

    def nearest_emit_site(self):

        # get plot center points
        shapefile_emit = os.path.join('gis', "Observation.shp")
        df_emit = gpd.read_file(shapefile_emit)
        df_emit['latitude'] = df_emit['geometry'].y
        df_emit['longitude'] = df_emit['geometry'].x
        df_emit = df_emit.drop('geometry', axis=1)
        df_emit['Team'] = df_emit['Name'].str.split('-').str[0].str.strip()
        df_emit = df_emit[df_emit['Team'] != 'THERM']

        df_emit = df_emit.sort_values('Name')
        df_min_distance_rows = []

        for index, row in df_emit.iterrows():

            plot = row['Name']

            lon = row['longitude']
            lat = row['latitude']

            results = p_map(partial(haversine_distance, lat, lon), df_emit['latitude'].values,
                            df_emit['longitude'].values, df_emit['Name'].values,
                            **{"desc": f"geographic distance: {plot}", "ncols": 150})

            df_results = pd.DataFrame(results)
            df_results.columns = ['emit_plot_function', 'distance_km']
            df_results = df_results.dropna()
            df_results = df_results.sort_values('distance_km')

            df_results['combined'] = list(zip(df_results['emit_plot_function'], df_results['distance_km']))
            df_results = df_results.drop(columns=['emit_plot_function', 'distance_km']).T
            df_results['emit_plot_analysis'] = plot
            df_to_row = df_results.iloc[0].values

            df_min_distance_rows.append(df_to_row)

        min_dist_df = pd.DataFrame(df_min_distance_rows)
        column_names = min_dist_df.columns.tolist()
        column_names[-1] = 'emit_plot_analysis'
        min_dist_df.columns = column_names
        min_dist_df.to_csv(os.path.join('gis', 'min_dist_to_emit_plots.csv'), index=False)


def run_build_workflow(base_directory, sensor):
    #msg = f"Please move all .asd Files from the ASD Computer " \
    #      f"to the following location: {os.path.join(base_directory, 'data')}\n" \
    #      f"Folder names should be based on the following naming convention:\n" \
    #      f"\tTeam_Plot-Number (e.g., Spectral - 001; Team = Spectral; Plot-Number: 001"

    #cursor_print(msg)
    user_input = query_yes_no('\nWould you like plots for all .asd/.sed files?', default="yes")

    if user_input:
        transect_directories = sorted(glob(os.path.join(base_directory, 'data', 'spectral_transects', "*", ""), recursive=True))
        create_directory(os.path.join(base_directory, 'figures', 'asd_file_plots'))

        for directory in transect_directories:
            plot_name = os.path.basename(os.path.dirname(directory))
            if os.path.isdir(os.path.join(base_directory, 'figures', 'asd_file_plots', plot_name)) and glob(os.path.join(base_directory, 'figures', 'asd_file_plots', plot_name, '*.png')):
                continue

            create_directory(os.path.join(base_directory, 'figures', 'asd_file_plots', plot_name))
            asd_files = glob(os.path.join(directory, '*.asd'))

            if asd_files:
                p_map(partial(spectra.plot_asd_file, out_directory=os.path.join(base_directory, 'figures', 'asd_file_plots', plot_name)),
                      asd_files, **{"desc": "\t\t plotting asd files: " + plot_name + "...", "ncols": 150})

            else:
                sed_files = glob(os.path.join(directory, '*.sed'))
                p_map(partial(spectra.plot_sed_file, out_directory=os.path.join(base_directory, 'figures', 'asd_file_plots', plot_name)),
                      sed_files, **{"desc": "\t\t plotting sed files: " + plot_name + "...", "ncols": 150})


    else:
        lib = build_libraries(base_directory=base_directory, sensor=sensor)
        lib.build_emit_transects()
        if not os.path.isfile(os.path.join('gis', 'min_dist_to_emit_plots.csv')):
            lib.nearest_emit_site()
        lib.build_emit_endmembers()
        lib.build_em_collection()
        lib.build_gis_data()
        lib.em_qty_check()
        #lib.build_derivative_library()
