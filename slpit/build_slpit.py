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
        create_directory(os.path.join(self.output_directory, 'spectral_transects'))
        # create_directory(os.path.join(self.output_directory, 'spectral_transects', 'transect'))
        # create_directory(os.path.join(self.output_directory, 'spectral_transects', 'endmembers'))
        # create_directory(os.path.join(self.output_directory, 'spectral_transects', 'endmembers-raw'))
        # create_directory(os.path.join(self.output_directory, 'plot_pictures'))
        # create_directory(os.path.join(self.output_directory, 'plot_pictures', 'spectral_transects'))
        # create_directory(os.path.join(self.output_directory, 'plot_pictures', 'spectral_endmembers'))

        # team names keys - corresponds to suffix in ASD files
        self.team_keys = {
            'spectral': 'SP', 'thermal': 'TM'}

        # input data directories
        self.spectral_em_directory = os.path.join(self.base_directory, 'data', 'spectral_endmembers')
        self.spectral_transect_directory = os.path.join(self.base_directory, 'data', 'spectral_transects')

        # instrument to indicate wavelengths in output folder
        self.instrument = sensor

        # output data directories
        self.output_transect_directory = os.path.join(self.output_directory, 'spectral_transects')
        # self.output_transect_em_directory = os.path.join(self.output_directory, 'spectral_transects', 'endmembers')
        # self.output_transect_em_directory_raw = os.path.join(self.output_directory, 'spectral_transects', 'endmembers-raw')

        # import the simulation outputs
        terraspec_base = os.path.join(base_directory, "..")
        em_sim_directory = os.path.join(terraspec_base, 'simulation', 'output')
        self.emit_global = os.path.join(em_sim_directory, 'convolved','geofilter_convolved.csv')
        self.convex_global = os.path.join(em_sim_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv')

    def build_emit_transects(self):
        # the transect spectra
        records = load_pickle('emit_slpit')

        print(f"loading... {len(records)} Spectral Transects")
        for i in records:
            plot_name = f"{i['team_names'].capitalize()} - {i['plot_num']:03d}"
            plot_directory = os.path.join(self.spectral_transect_directory, plot_name)
            plot_pic_url = i['landscape_pic']
            date = i['sample_date']
            plot_measurements = i['plot_measurements'].split(",")

            if 'slpit' not in plot_measurements:
                continue

            if 'thermal' in i['team_names']:
                continue

            if int(i['plot_num']) in [114,113]:
                continue

            if os.path.isfile(os.path.join(self.output_transect_directory, f'{plot_name} - transect-{self.instrument}.csv')):
                continue

            print(f'\t loading... {plot_name}')

            create_directory(os.path.join(self.output_transect_directory, f'{plot_name}'))
            plot_base_directory = os.path.join(self.output_transect_directory, f'{plot_name}')
            img_data = requests.get(plot_pic_url).content
            with open(os.path.join(plot_base_directory, f'{plot_name}_landscape_picture.jpg'),
                      'wb') as handler:
                handler.write(img_data)

            # white ref table
            df_white_ref = slpit.df_white_ref_table(record=i)

            # # em table
            df_transect_em = slpit.df_em_table(record=i)

            # get all asd files from folder
            all_spectrometer_files = sorted(glob(os.path.join(plot_directory, '*.asd')))
            if not all_spectrometer_files:
                print(".asd files not found! Looking for .sed files...")
                all_spectrometer_files = sorted(glob(os.path.join(plot_directory, '*.sed')))
            # else:
            #     continue

            # white refs from transects
            good_white_ref_numbers = set(df_white_ref[df_white_ref['my_element_2'] == 'good']['filenumber'].values)
            all_white_ref_numbers = set(df_white_ref['filenumber'].values)

            transect_spectra = []

            # keep white refs our of all spectra
            for asd_file in all_spectrometer_files:
                file_name = os.path.basename(asd_file)
                try:
                    file_num = int(file_name.split(".")[0].split("_")[-1])
                except ValueError:
                    print(f"Warning: Could not extract file number from {file_name}. Skipping.")
                    continue

                if file_num not in all_white_ref_numbers or file_num in good_white_ref_numbers:
                    transect_spectra.append(asd_file)

            results_refl = p_map(partial(spectra.get_reflectance_transect, plot_directory=plot_directory,
                                         team_name_key=self.team_keys[i['team_names']]), transect_spectra,
                                 **{"desc": f"\t\t processing plot: {plot_name}...", "ncols": 150})

            df_results = pd.DataFrame(results_refl)
            df_results.columns = ["plot_name", "file_name", "file_num", "longitude", "latitude", "elevation",
                                  "utc_time"] + list(spectra.load_asd_wavelenghts())

            df_results = df_results.sort_values('file_num')
            df_results.insert(3, "white_ref", 0)
            df_results.insert(4, "line_num", 0)
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
                                      (df_results['file_num'] <= line_num_max)].copy()

                df_query = pd.merge(df_query, df_select[['file_num', 'white_ref_space']],
                                    on='file_num', how='left')
                df_query['line_num'] = line_num  # Add line_num back
                df_query['white_ref'] = df_query['white_ref_space']
                df_query = df_query[df_query['white_ref_space'] != 'middle']
                df_query = df_query.iloc[:, :-1] # this drops the join since values are now saved in white_ref

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

                    corrected_reflectance = p_map(partial(spectra.white_ref_correction,
                                                          white_reference_spectra_t1=white_reference_spectra_t1,
                                                          white_reference_spectra_t2=white_reference_spectra_t2,
                                                          time_1=t1, time_2=t2),
                                                  [df_spectra_array[_row, :] for _row in range(df_spectra_array.shape[0])],
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
                    df_query = df_results[(df_results['file_num'] > line_num_max) & (df_results['file_num'] < df_white_ref['filenumber'].values[min_index])].copy()

                    if df_query.empty:
                        # this gets line 3;
                        df_query = df_results[(df_results['file_num'] > line_num_max)].copy()

                    df_query['line_num'] = line_num
                    df_query = df_query.drop('white_ref', axis=1)
                    df_query.insert(0, "date", date)
                    df_query['utc_time'] = df_query['utc_time'].dt.strftime('%H:%M:%S')
                    adjusted_dfs.append(df_query)
                    print(f"\t\t no white ref correction available on: {plot_name} {line_num}")

            df_corrected_all = pd.concat(adjusted_dfs)
            df_corrected_all.to_csv(os.path.join(plot_base_directory, f'{plot_name} - transect.csv'),
                                    index=False)

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve_asdfile,  wvl=self.wvls, fwhm=self.fwhm),
                                     df_corrected_all.file_name.values.tolist(),
                                     **{"desc": f"\t\t\tconvolving plot: {plot_name}...", "ncols": 150})

            # save outputs as emit resolutions csv's
            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df_corrected_all.iloc[:, :9].reset_index(drop=True), df_convolve], axis=1)
            df_convolve.to_csv(os.path.join(plot_base_directory, f'{plot_name} - transect-{self.instrument}.csv'), index=False)

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
            output_raster = os.path.join(plot_base_directory, f'{plot_name.replace(" ", "")}.hdr')
            save_envi(output_raster, meta_spectra, spectra_grid)
            time.sleep(3)

    def build_emit_endmembers(self):
        # transect endmembers
        records = load_pickle('emit_slpit')

        print("loading... Spectral Transects Endmembers")

        for i in records:
            plot_name = f"{i['team_names'].capitalize()} - {i['plot_num']:03d}"
            plot_directory = os.path.join(self.spectral_transect_directory, plot_name)
            date = i['sample_date']
            plot_measurements = i['plot_measurements'].split(",")

            if 'endmembers' not in plot_measurements:
                continue

            if 'thermal' in i['team_names']:
                continue

            if os.path.isfile(os.path.join(self.output_transect_em_directory_raw, f'{plot_name.replace(" ", "")}-{self.instrument}.csv')):
                continue

            if int(i['plot_num']) in [114,113, 119, 114, 113]:
                continue

            print(f'\t loading... {plot_name}')

            # em table
            df_transect_em = slpit.df_em_table(record=i)
            df_transect_em = df_transect_em.loc[(df_transect_em['em_condition'] != 'bad') &
                                                (df_transect_em['endmembers'] != 'Flower')].copy()
            # get all asd files from folder
            all_asd_files = sorted(glob(os.path.join(plot_directory, '*.asd')))

            if not all_asd_files:
                print(".asd files not found! Looking for .sed files...")
                all_asd_files = sorted(glob(os.path.join(plot_directory, '*.sed')))

            endmember_spectra = []
            for asd_file in all_asd_files:
                try:
                    file_num = int(os.path.basename(asd_file).split(".")[0].split("_")[-1])
                except:
                    file_num = int(os.path.basename(asd_file).split(".")[0][-5:])

                # check if file is in white ref or em
                if file_num in df_transect_em.asd_file_num.values:
                    endmember_spectra.append(asd_file)

                else:
                    pass

            results = p_map(partial(spectra.get_reflectance_transect, plot_directory=plot_directory,
                                    team_name_key=self.team_keys[i['team_names']]), endmember_spectra,
                            **{"desc": "\t\t processing plot: " + plot_name + " ...", "ncols": 150})

            df_results = pd.DataFrame(results)
            df_results.columns = ["plot_name", "file_name", "file_num", "longitude", "latitude", "elevation",
                                  "utc_time"] + list(spectra.load_asd_wavelenghts())

            df_results.insert(0, "date", date)
            df_results.insert(3, "line_num", '')
            df_results.insert(4, "level_1", '')
            df_results.insert(5, "species", '')
            df_results.insert(6, "notes", '')

            for file_num in df_results.file_num.values:
                if file_num in df_transect_em.asd_file_num.values:
                    em_clas = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'endmembers'].iloc[0]
                    line_num = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'transect_line_num'].iloc[
                        0]
                    species = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'species'].iloc[0]
                    notes = df_transect_em.loc[df_transect_em['asd_file_num'] == file_num, 'notes'].iloc[0]
                    df_results.loc[df_results['file_num'] == file_num, ['line_num', 'level_1',
                                                                        'species', 'notes']] = line_num, em_clas, species, notes

            df_results = df_results.sort_values("level_1")
            df_results.to_csv(os.path.join(self.output_transect_em_directory_raw, plot_name.replace(" ", "") + '-asd.csv'),
                              index=False)

            # convolve wavelengths to user specified instrument
            results_convolve = p_map(partial(spectra.convolve_asdfile, wvl=self.wvls, fwhm=self.fwhm),
                                     df_results.file_name.values.tolist(),
                                     **{"desc": f"\t\t\tconvulsing plot: {plot_name}...", "ncols": 150})
            df_convolve = pd.DataFrame(results_convolve)
            df_convolve.columns = list(self.wvls)
            df_convolve = pd.concat([df_results.iloc[:, :11].reset_index(drop=True), df_convolve], axis=1)

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
        df_all_emit_ems = pd.read_csv(os.path.join(self.output_directory, f"all-endmembers-{self.instrument}.csv"),
                                      low_memory=False)
        df_distance = pd.read_csv(os.path.join('gis', 'min_dist_to_emit_plots.csv'))
        all_ems = sorted(list(df_all_emit_ems.level_1.unique()))

        for i in emit_ems:
            plot_number = os.path.basename(i).split('-')[1]
            if int(plot_number) > 60:
                continue
            df_em_site = pd.read_csv(i, low_memory=False)
            site_em = sorted(list(df_em_site.level_1.unique()))
            df_nearest_distances = df_distance.loc[df_distance['emit_plot_analysis'] == f"SPEC - {plot_number}"].copy()

            ems_to_append = []

            if sorted(site_em) == ['NPV', 'PV', 'Soil']:
                # Fill in remainder so each class in equal to number of desired samples
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

                # if not 3 classes add n samlpes
                em_difference = sorted(list(set(all_ems) - set(site_em)))

                print(f'{os.path.basename(i)} missing following endmembers: {em_difference}')

                for em in em_difference:
                    if em not in ['NPV', 'PV', 'Soil']:
                        continue

                    remaining_samples = em_min_samples[em]
                    site_counter = 0

                    while remaining_samples > 0:
                        current_nearest_site = eval(df_nearest_distances.iloc[:, site_counter].iloc[0])[0].split('-')[1]
                        df_nearest_ems = df_all_emit_ems.loc[(df_all_emit_ems['level_1'] == em) & (
                                df_all_emit_ems['plot_name'] == f'Spectral - {current_nearest_site.strip()}')].copy()

                        if df_nearest_ems.shape[0] < remaining_samples:
                            remaining_samples = df_nearest_ems.shape[0] # nearest plot has less than desired samples

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
        df.to_csv(os.path.join(self.output_directory, f'all-endmembers-{self.instrument}.csv'), index=False)
        spectra.df_to_envi(df=df, spectral_starting_column=11, wvls=self.wvls,
                           output_raster=os.path.join(self.output_directory, f'all-endmembers-{self.instrument}.hdr'))

        # merge all endmembers - asd based wavelengths
        df = pd.concat((pd.read_csv(f) for f in asd_ems), ignore_index=True)
        df.to_csv(os.path.join(self.output_directory, "all-endmembers-asd.csv"), index=False)

        # merge all transect spectra - emit
        emit_transects = glob(os.path.join(self.output_transect_directory, "*transect-" + self.instrument + ".csv"))
        df_transect = pd.concat((pd.read_csv(f) for f in emit_transects), ignore_index=True)
        df_transect.to_csv(os.path.join(self.output_directory, "all-transect-emit.csv"), index=False)
        spectra.df_to_envi(df=df_transect, spectral_starting_column=9, wvls=self.wvls,
                           output_raster=os.path.join(self.output_directory, f'all-transect-{self.instrument}.hdr'))

    def build_gis_data(self):
        print("Building spectral endmember gis shapefile data...", sep=' ', end='', flush=True)
        df = pd.read_csv(os.path.join(self.output_directory, f'all-endmembers-{self.instrument}.csv'),
                         low_memory=False)
        df = df.iloc[:, :10]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, out_name=f'{self.instrument}_endmembers_slpit')

        df = pd.read_csv(os.path.join(self.output_directory, f'all-transect-{self.instrument}.csv'))
        df = df.iloc[:, :8]
        df = df.replace('unk', np.nan)
        df = df.interpolate(method='nearest')
        spectra.df_to_shapefile(df, out_name=f'{self.instrument}_slpit')
        time.sleep(3)
        print("done")


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
        #if not os.path.isfile(os.path.join('gis', 'min_dist_to_emit_plots.csv')):
        #   lib.nearest_emit_site()
        #lib.build_emit_endmembers()
        #lib.build_em_collection()
        #lib.build_gis_data()
        #lib.em_qty_check()
