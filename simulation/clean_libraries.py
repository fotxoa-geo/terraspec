import pandas as pd
import numpy as np
import os
from glob import glob
import geopandas as gp
import json
import multiprocessing as mp
from p_tqdm import p_map
from functools import partial
import time
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.text_guide import cursor_print



# Load configs
f = open('simulation/configs.json')
data = json.load(f)
metadata = data['geo_metadata']  # geodata from spec library

spectra_files = data['spectra_file']  # spectral files
datasets = data['datasets']  # spectral datasets

# load asd wavelegnths
wavelengths_asd = spectra.load_asd_wavelenghts()
emit_wvls, emit_fwhm = spectra.load_wavelengths(sensor='emit')


def process_all_data(base_directory, output_directory):
    "This fucntion merges all raw data into one  csv file"
    # check if directory for all data exists:
    create_directory(os.path.join(output_directory, "all_data"))

    # loop for each dataset
    for i in datasets:
        ds_name = os.path.basename(i)
        lat_lon_keys = data['lat_lon_keys']
        print()
        cursor_print("\t loading... " + ds_name.lower())

        if ds_name == "MEYER-OKIN":
            spectral_table = os.path.join(base_directory, 'raw_data', i, spectra_files[i])
            df = pd.read_csv(spectral_table)
            df['dataset'] = df['dataset'].str.replace('_', '-')

        elif ds_name == 'PILOT-2':
            spectral_table = os.path.join(base_directory, 'raw_data', i, spectra_files[i])
            col_wls = ["X" + str(x) for x in range(350, 2501)]
            cols = ['Soil_type_USDA', 'ID', lat_lon_keys[ds_name][1], lat_lon_keys[ds_name][0]] + col_wls
            df = pd.read_csv(spectral_table, usecols=cols)

            # sort order of columns in df
            df = df[cols]

            # rename latitude/longitude
            df = df.rename(columns={
                lat_lon_keys[ds_name][0]: 'latitude',
                lat_lon_keys[ds_name][1]: 'longitude'})

            # add level 1 classification
            df.insert(0, 'level_1', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            col_pilot2 = ["dataset", "level_1", "level_2", "level_3", 'longitude', 'latitude'] + wavelengths_asd
            df.columns = col_pilot2
            df.insert(4, 'fname', df.level_3)

        elif ds_name == 'OSSL':
            col_wls = ["scan_visnir." + str(x) + '_pcnt' for x in range(350, 2501, 2)]
            meta_cols = ['id.layer_uuid_c', lat_lon_keys[ds_name][1], lat_lon_keys[ds_name][0]]
            spectra_cols = ['id.layer_uuid_c'] + col_wls

            metadata_table = pd.read_csv(os.path.join(base_directory, 'raw_data', i, metadata[ds_name]),
                                         usecols=meta_cols)
            spectral_table = pd.read_csv(os.path.join(base_directory, 'raw_data', i, spectra_files[i]),
                                         usecols=spectra_cols)

            # merge dataframes
            df = pd.merge(metadata_table, spectral_table, on='id.layer_uuid_c')
            df = df.rename(columns={
                lat_lon_keys[ds_name][0]: 'latitude',
                lat_lon_keys[ds_name][1]: 'longitude'})
            df.insert(0, 'level_1', 'soil')
            df.insert(1, 'level_2', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            ossl_wvls = [x for x in range(350, 2501, 2)]
            df.columns = ["dataset", "level_1", "level_2", "level_3", 'longitude', 'latitude'] + ossl_wvls
            df.insert(4, 'fname', df.index.astype(str).str.zfill(4) + "_" + df.level_3)

            # convert % to decimal
            for wvl in ossl_wvls:
                df[wvl] = df[wvl] / 100

            # fix splice @ 1000 nm
            df.iloc[:, 350:1000] *= df.iloc[:, 1000] / df.iloc[:, 1002]

        elif ds_name == 'NGSA':
            spectral_table = os.path.join(base_directory, 'raw_data', i, spectra_files[i])
            df = pd.read_csv(spectral_table, index_col='Wavelength_(nm)')
            df = df.T
            df = df.reset_index()
            df = df.rename(columns={
                'index': 'level_3'})
            df['level_3'] = df['level_3'].str.replace('.', '_')
            df['level_3'] = df['level_3'].str.replace(':', '_')
            df['level_3'] = df['level_3'].str.split('_').str[1]
            df['level_3'] = df['level_3'].str.replace('B', 'S')

            # load index
            df_appendix = pd.read_excel(
                os.path.join(base_directory, 'raw_data', i, 'Rec2010_018_Appendix_1.xls'),
                usecols=['TARGET SITEID', 'LATITUDE', 'LONGITUDE'], skiprows=7)
            df_appendix = df_appendix.rename(columns={
                'TARGET SITEID': 'level_3'})
            df_appendix = df_appendix.rename(columns={
                'LATITUDE': 'lat'})
            df_appendix = df_appendix.rename(columns={
                'LONGITUDE': 'long'})

            # merge the two dfs
            df = pd.merge(df, df_appendix, on='level_3')
            df.insert(0, 'level_1', 'soil')
            df.insert(1, 'level_2', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            df.insert(4, 'fname', df.level_3)
            df.insert(5, 'longitude', df.long)
            df.insert(6, 'latitude', df.lat)
            df = df.drop(columns=['long', 'lat'])

        elif ds_name == 'ASD':
            spectral_tables = sorted(glob(os.path.join(base_directory, 'raw_data', ds_name, '*.csv')))

            for table in spectral_tables:
                site_name = os.path.basename(table).split(".")[0].split('_')[0]
                df_asd = pd.read_csv(table, low_memory=False)
                df_asd = df_asd.drop(columns=['elevation', 'utc_time'])
                df_asd.insert(5, 'fname',
                              df_asd.site + "_" + df_asd.line_num.astype(str) + "_" + df_asd.file_num.astype(
                                  str).str.zfill(2) + '_' + df_asd.level_1)
                df_asd = df_asd.iloc[:, 3:]
                df_asd.insert(0, 'dataset', site_name.split("_")[0])
                df_asd.insert(3, 'level_3', df_asd.level_2)
                df_asd.to_csv(os.path.join(output_directory, "all_data", "all_data_" + site_name + ".csv"), index=False)

        elif ds_name == 'SSL-IR':
            spectral_table = os.path.join(base_directory, 'raw_data', i, spectra_files[i])
            df = pd.read_excel(spectral_table)
            df = df.drop(
                columns=['Sand (% wt.)', 'Silt (% wt.)', 'Clay (% wt.)', 'CaCO3 (%)', 'Organic Matter (%)'])
            df.insert(0, 'level_1', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            df.columns.values[2] = 'level_2'
            df.insert(3, 'level_3', df.level_2)
            df.insert(4, 'fname', df.dataset + "_" + df.level_3)
            df.insert(5, 'longitude', 35.217018)
            df.insert(6, 'latitude', 31.771959)

         # save data
        if ds_name == 'ASD':
            pass
        else:
            df.to_csv(os.path.join(output_directory, "all_data", "all_data_" + i + ".csv"), index=False)

    print()
    cursor_print("merging all data...")
    # merge all the outputs into one file
    csvs = sorted(glob(os.path.join(output_directory, "all_data", '*.csv')))
    list_dfs_all = []
    for csv in csvs:
        df_i = pd.read_csv(csv, low_memory=False)
        list_dfs_all.append(df_i)

    df_all = pd.concat(list_dfs_all)
    df_all.to_csv((os.path.join(output_directory, "all_data.csv")), index=False)
    time.sleep(3)
    print("\t\t done")


def geofilter_data(base_directory, output_directory):
    cursor_print("applying geo filter to global spectra...")

    # create output directory for geofilter
    create_directory(os.path.join(output_directory, 'geofilter'))
    tables = sorted(glob(os.path.join(output_directory, "all_data", '*.csv')))
    shp = gp.read_file(os.path.join(base_directory, 'gis', 'emit_mask.shp')).to_crs(4326)  # EMIT dust mask
    second_check_tables = sorted(glob(os.path.join(base_directory, 'raw_data', 'second_checks', '*.csv')))

    for i in tables:
        ds_name = os.path.basename(i).split(".")[0].split("_")[2]
        df = pd.read_csv(i, low_memory=False)
        df = df.drop_duplicates()
        df = df.drop(df[df.latitude == 'unk'].index)

        # implement secondary check
        selected_check = [x for x in second_check_tables if ds_name in x]
        df_check = pd.read_csv(selected_check[0], usecols=['fname', 'class', 'check(use y or n)'])

        # for NGSA, file names changed
        if ds_name == 'NGSA':
            df_check['fname'] = df_check['fname'].str.replace('.', '_')
            df_check['fname'] = df_check['fname'].str.replace(':', '_')
            df_check['fname'] = df_check['fname'].str.split('_').str[1]
            df_check['fname'] = df_check['fname'].str.replace('B', 'S')

        df_check = df_check.loc[df_check['check(use y or n)'] == 'y'].copy()
        df = pd.merge(df, df_check, how='left', indicator='Exist')
        df['Exist'] = np.where(df.Exist == 'both', True, False)
        df = df[df['Exist'] == True].drop(['Exist'], axis=1)
        df.drop(columns=df.columns[-2:], axis=1, inplace=True)

        if ds_name == 'SR' or ds_name == 'DP' or ds_name == 'SSL-IR': # these are the SHIFT domain box
            df.to_csv(os.path.join(output_directory, 'geofilter', 'geofilter_' + ds_name + '.csv'), index=False)
        else:
            points = gp.GeoDataFrame(df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
            within_points = gp.sjoin(points, shp, predicate='within')

            if within_points.empty:
                continue
            else:
                df = pd.DataFrame(within_points)
                df.drop(columns=df.columns[-14:], axis=1, inplace=True)

            df.to_csv(os.path.join(output_directory, 'geofilter', 'geofilter_' + ds_name + '.csv'), index=False)
    time.sleep(3)
    print("done")


def convolve_library(base_directory, output_directory, geo_filter: bool):
    # define radiometric resolution for spectra - 4 decimal places ??
    print()
    cursor_print("loading band convolution...")
    create_directory(os.path.join(output_directory, 'convolved'))

    if geo_filter:
        tables = sorted(glob(os.path.join(output_directory, "geofilter", '*.csv')))

    else:
        tables = sorted(glob(os.path.join(output_directory, "all_data", '*.csv')))

    # Begin parallel processing
    print()
    msg = f"Commencing parallel processing \nNumber of CPUs available: {mp.cpu_count()}"
    cursor_print(msg)
    print()

    all_results = []
    for i in tables:
        df = pd.read_csv(i, low_memory=False)
        ds_name = os.path.basename(i).split(".")[0].split("_")[1]

        if ds_name == 'OSSL':
           ossl_wvls = [x for x in range(350, 2501, 2)]
           results = p_map(partial(spectra.convolve, asd_wvl=ossl_wvls, wvl=emit_wvls, fwhm=emit_fwhm,
                                   spectra_starting_col=7), [row for row in df.iterrows()],
                           **{
                               "desc": "\t loading convolution... " + ds_name,
                               "ncols": 150})
        else:
            results = p_map(partial(spectra.convolve, asd_wvl=wavelengths_asd, wvl=emit_wvls, fwhm=emit_fwhm,
                                    spectra_starting_col=7), [row for row in df.iterrows()],
                            **{"desc": "\t loading convolution... " + ds_name, "ncols": 150})

        df = pd.concat([df.iloc[:, :7], pd.DataFrame(results)], axis=1)
        all_results.append(df)

    # create dataframe, remove duplicates, no data, and save
    output_cols = ["dataset", "level_1", "level_2", "level_3", 'fname', 'longitude',
                   'latitude'] + emit_wvls.tolist()

    df_merge = pd.concat(all_results)
    df_merge.columns = output_cols
    df_merge = df_merge.drop_duplicates()
    df_merge = df_merge[~df_merge[df_merge.columns[7]].isnull()] # this gets rid of no data
    df_merge = df_merge.drop_duplicates(subset=[df_merge.columns[7]], keep='first') # drops duplicates in first spectral row
    df_merge = df_merge.drop_duplicates(subset=['fname'], keep='first').sort_values("level_1") # drops duplicates using fname

    if geo_filter:
        df_merge.to_csv(os.path.join(output_directory, "convolved", "geofilter_convolved.csv"),
                        index=False)

        df_geo_data = df_merge.iloc[:, :7]
        df_shp = gp.GeoDataFrame(df_geo_data,geometry=gp.points_from_xy(df_geo_data.longitude, df_geo_data.latitude),
                                 crs="EPSG:4326")
        df_shp.to_file(os.path.join(base_directory, "gis", "emit_global_spectral_library.shp"),
                       driver='ESRI Shapefile')

    else:
        df_merge.to_csv(os.path.join(output_directory, "convolved", "all_data_convolved.csv"),
                        index=False)
    print('done')


def run_clean_workflow(base_directory, output_directory, geo_filter: bool):
    process_all_data(base_directory=base_directory, output_directory=output_directory)
    geofilter_data(base_directory=base_directory, output_directory=output_directory)
    convolve_library(geo_filter=geo_filter, output_directory=output_directory, base_directory=base_directory)
