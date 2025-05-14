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


# taken from CSRIO page 
def encodeIdentifier(identifier, **kwargs):

    baseURL = kwargs.get("baseURL")
    version = kwargs.get("version")

    if baseURL:
        if baseURL == "https://ws.data.csiro.au/":
            encodeChar = "~"
        elif baseURL == "https://data.csiro.au/dap/ws/v2/":
            encodeChar = "/"
        else:
            # default to v2 since v1 is being deprecated.
            encodeChar = "/"

    # specifying a version means baseURL effectively gets ignored.
    if version:
        if version == 1:
            encodeChar = "~"
        elif version == 2:
            encodeChar = "/"
        else:
            # default to v2 since v1 is being deprecated.
            encodeChar = "/"

    # default to v2 since v1 is being deprecated.
    if not baseURL and not version:
        version = 2
        encodeChar = "/"

    # Extract the ID.
    # This assumes that the URLs are from the persistent link on the DAP UI
    # collection landing pages
    # DOI URLs
    identifier = identifier.replace("https://doi.org/", "")
    identifier = identifier.replace("http://doi.org/", "")
    identifier = identifier.replace("https://dx.doi.org/", "")
    identifier = identifier.replace("http://dx.doi.org/", "")

    return identifier

# Load configs
f = open('simulation/config.json')
data = json.load(f)

spectra_files = data['spectra_file']  # spectral files
datasets = data['datasets_to_standardize']  # spectral datasets

# load asd wavelegnths




def download_data(base_directory, output_directory):
    create_directory(os.path.join(output_directory, "production"))

    import requests
    from io import BytesIO
    import gzip
    import io

   # this is for ossl
    col_wls = ["scan_visnir." + str(x) + '_pcnt' for x in range(350, 2501, 2)]
    meta_cols = ['id.layer_uuid_c', "longitude_wgs84_dd", "latitude_wgs84_dd"]
    spectra_cols = ['id.layer_uuid_c'] + col_wls

    metadata_table = pd.read_csv(os.path.join('objects', 'soilsite.data.csv'), usecols=meta_cols)
    spectral_table = pd.read_csv(os.path.join('objects', 'visnir.data.csv'), usecols=spectra_cols)

    df = pd.merge(metadata_table, spectral_table, on='id.layer_uuid_c')
    df = df.rename(columns={'latitude_wgs84_dd': 'latitude', 'longitude_wgs84_dd': 'longitude'})

    df.to_csv(os.path.join(output_directory, 'production', 'ossl.csv'), index=False)
    print(f"File downloaded to: {os.path.join(output_directory, 'production', 'ossl.csv')}")

    # this is the NGSA data from AUS
    metadata_url = f'https://d28rz98at9flks.cloudfront.net/70478/Rec2010_018_Appendix_1.xls'
    response = requests.get(metadata_url)
    df_ngsa_metadata = pd.read_excel(BytesIO(response.content), skiprows=7)


    baseURL = "https://data.csiro.au/dap/ws/v2/"
    endpoint = "collections/{id}"

    collectionURL = f"https://doi.org/10.25919/5cdba18939c29"
    encodedID = encodeIdentifier(collectionURL)
    url = baseURL + endpoint.format(id=encodedID)
    headers = {"Accept": "application/json"}
    r = requests.get(url, headers=headers)

    resultPage = r.json()  # A dict of the response

    # Get the data endpoint
    data_url = resultPage.get("data")
    r = requests.get(data_url, headers=headers)
    dataPage = r.json()

    # One thing you could do with the file list is create a dict matching
    files = dataPage.get("file")
    for file in files:
        filename = file["filename"]
        if filename == 'National Geochemical Survey of Australia FTIR spectra.csv':
            continue

        fileURL = file["link"]["href"]
        url_content = requests.get(fileURL)
        df_ngsa = pd.read_csv(BytesIO(url_content.content)).T.reset_index()
        df_ngsa.columns = df_ngsa.iloc[0]
        df_ngsa = df_ngsa[1:]
        df_ngsa.columns.values[0] = "SITEID"
        df_ngsa["SITEID"] = df_ngsa["SITEID"].str.split('.').str[0]
        df_ngsa["SITEID"] = df_ngsa["SITEID"].str.split(':').str[1]
        df_ngsa["SITEID"] = df_ngsa["SITEID"].astype(str)
        df_ngsa['SITEID'] = df_ngsa['SITEID'].str.replace('B', 'S')
        df_ngsa["SITEID"] = df_ngsa["SITEID"].str.strip()

        df_ngsa_metadata.columns.values[2] = "SITEID"
        df_ngsa_metadata["SITEID"] = df_ngsa_metadata["SITEID"].astype(str)
        df_ngsa_metadata["SITEID"] = df_ngsa_metadata["SITEID"].str.strip()

        df_ngsa_merged = pd.merge(df_ngsa_metadata, df_ngsa, on='SITEID', how="inner")
        df_ngsa_merged.to_csv(os.path.join(output_directory, 'production', f'{filename}'), index=False)
        print(f"File downloaded to: {os.path.join(output_directory, 'production', filename)}")

    # this is Meyer et al. 2022 and Ochoa et al. 2024
    def get_ecosis_url(ecosis_id):
        url = f"https://ecosis.org/api/package/{ecosis_id}"
        response = requests.get(url)
        ecosis_data = response.json()
        url = ecosis_data['ecosis']['resources'][0]['url']

        url_content = requests.get(url)
        ecosis_csv_name = ecosis_data['ecosis']['resources'][0]['name']

        with open(os.path.join(output_directory, "production", ecosis_csv_name), "wb") as file:
            file.write(url_content.content)

        print(f"File downloaded to: {os.path.join(output_directory, 'production', ecosis_csv_name)}")

        return url

    get_ecosis_url(ecosis_id='kalahari-ecosystem-endmember-set')
    get_ecosis_url(ecosis_id='drylands-spectral-libraries-in-support-of-emit')

    # this is pilot-2
    url_p2 = 'http://datahub.geocradle.eu/sites/default/files/SSL_GEOCRADLE_1.csv'
    url_content = requests.get(url_p2)
    with open(os.path.join(output_directory, "production", 'SSL_GEOCRADLE_1.csv'), "wb") as file:
        file.write(url_content.content)

    print(f"File downloaded to: {os.path.join(output_directory, 'production', 'SSL_GEOCRADLE_1.csv')}")



    #df_ssl_il = pd.read_excel(os.path.join('objects', 'SSL_IL.xlsx'))
    #df_ssl_il.to_csv(os.path.join(output_directory, 'production', 'ssl-il.csv'), index=False)
    #print(f"File downloaded to: {os.path.join(output_directory, 'production', 'ssl-il.csv')}")


def standardize_all_data(base_directory, output_directory):
    "This function merges all raw data into one csv file"
    # check if directory for all data exists:
    create_directory(os.path.join(output_directory, "all_data"))
    wavelengths_asd = spectra.load_asd_wavelenghts()

    df_global = spectra.load_global_library_metadata()

    # loop for each dataset
    for i in datasets:
        ds_name = os.path.basename(i)
        lat_lon_keys = data['lat_lon_keys']
        print()
        cursor_print("\t loading... " + ds_name.lower())
        print()

        if ds_name == "MEYER-OKIN":
            spectral_table = os.path.join(base_directory, 'output', 'production', spectra_files[i])
            df = pd.read_csv(spectral_table)
            df['dataset'] = df['dataset'].str.replace('_', '-')
            df = df.drop(columns=['notes', 'latin_genus', 'latin_species'])

        elif ds_name == 'PILOT-2':
            spectral_table = os.path.join(base_directory, 'output', 'production', spectra_files[i])
            col_wls = ["X" + str(x) for x in range(350, 2501)]
            cols = ['Soil_type_USDA', 'ID', lat_lon_keys[ds_name][1], lat_lon_keys[ds_name][0]] + col_wls
            df = pd.read_csv(spectral_table, usecols=cols)

            # sort order of columns in df
            df = df[cols]

            # rename latitude/longitude
            df = df.rename(columns={lat_lon_keys[ds_name][0]: 'latitude', lat_lon_keys[ds_name][1]: 'longitude'})

            # add level 1 classification
            df.insert(0, 'level_1', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            col_pilot2 = ["dataset", "level_1", "level_2", "level_3", 'longitude', 'latitude'] + wavelengths_asd
            df.columns = col_pilot2
            df.insert(4, 'fname', df.level_3)

            df_global_p2 = df_global.loc[df_global['dataset'] == 'pilot-2'].copy()
            df_global_p2['level_3'] = df_global_p2['level_3'].astype(str)
            df['level_3'] = df['level_3'].astype(str)
            df = df[df['level_3'].isin(df_global['level_3'])]

        elif ds_name == 'OSSL':
            col_wls = ["scan_visnir." + str(x) + '_pcnt' for x in range(350, 2501, 2)]
            spectra_cols = ['id.layer_uuid_c', 'longitude', 'latitude'] + col_wls

            df = pd.read_csv(os.path.join(base_directory, 'output', 'production', spectra_files[i]), usecols=spectra_cols)

            df.insert(0, 'level_1', 'soil')
            df.insert(1, 'level_2', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            ossl_wvls = [x for x in range(350, 2501, 2)]
            df.columns = ["dataset", "level_1", "level_2", "level_3", 'longitude', 'latitude'] + ossl_wvls
            df.insert(4, 'fname', df.index.astype(str).str.zfill(4) + "_" + df.level_3)

            df_global_ossl = df_global.loc[df_global['dataset'] == 'ossl'].copy()
            df_global_ossl['level_3'] = df_global_ossl['level_3'].astype(str)
            df['level_3'] = df['level_3'].astype(str)

            df = df[df['level_3'].isin(df_global['level_3'])]

            # convert % to decimal
            for wvl in ossl_wvls:
                df[wvl] = df[wvl] / 100

            # fix splice @ 1000 nm
            scaling_factor = df.loc[:, 1000] / df.loc[:, 1002]
            df.loc[:, 350:1000] = df.loc[:, 350:1000].mul(scaling_factor, axis=0)

            #df.iloc[:, index_350+7: index_1000 + 7] = (df.iloc[:, index_350 + 7: index_1000 + 7].values * (df.iloc[:, index_350 + 7].values[:, None] / df.iloc[:, index_1002 + 7].values[:, None]))

        elif ds_name == 'NGSA':
            spectral_table = os.path.join(base_directory, 'output', 'production', spectra_files[i])
            ngsa_cols = ['SITEID', 'LATITUDE', 'LONGITUDE'] + [str(float(x)) for x in range(350, 2501)]

            df = pd.read_csv(spectral_table, usecols=ngsa_cols)

            df = df.rename(columns={'SITEID': 'level_3', 'LATITUDE': 'lat', 'LONGITUDE': 'long'})

            df.insert(0, 'level_1', 'soil')
            df.insert(1, 'level_2', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            df.insert(4, 'fname', df.level_3)
            df.insert(5, 'longitude', df.long)
            df.insert(6, 'latitude', df.lat)
            df = df.drop(columns=['long', 'lat'])

            df_global_ngsa = df_global.loc[df_global['dataset'] == 'ngsa'].copy()
            df_global_ngsa['level_3'] = df_global_ngsa['level_3'].astype(str)
            df['level_3'] = df['level_3'].astype(str)

            df = df[df['level_3'].isin(df_global_ngsa['level_3'])]
            df = df.drop_duplicates(subset=['fname'], keep='first').sort_values("level_1")

        elif ds_name == 'OCHOA':
            spectral_table = os.path.join(base_directory, 'output', 'production', spectra_files[i])
            df = pd.read_csv(spectral_table)

            ochoa_ds_unique = sorted(list(df.dataset.unique()))

            for ochoa_ds in ochoa_ds_unique:
                df_selected = df.loc[df['dataset'] == ochoa_ds].copy()
                df_selected = df_selected.drop(columns=['latin_genus', 'latin_species'])
                df_selected.insert(3, 'level_3', df_selected.fname)
                df_selected.to_csv(os.path.join(output_directory, "all_data", f"all_data_{ochoa_ds}.csv"), index=False)

        elif ds_name == 'SSL-IR':
            spectral_table = os.path.join(base_directory, 'output', 'production', spectra_files[i])
            df = pd.read_csv(spectral_table)
            df = df.drop(columns=['Sand (% wt.)', 'Silt (% wt.)', 'Clay (% wt.)', 'CaCO3 (%)', 'Organic Matter (%)'])
            df.insert(0, 'level_1', 'soil')
            df.insert(0, 'dataset', ds_name.lower())
            df.columns.values[2] = 'level_2'
            df.insert(3, 'level_3', df.level_2)
            df.insert(4, 'fname', df.dataset + "_" + df.level_3)
            df.insert(5, 'longitude', 35.217018)
            df.insert(6, 'latitude', 31.771959)

            df_global_ssl_ir = df_global.loc[df_global['dataset'] == 'ssl-ir'].copy()
            df_global_ssl_ir['level_3'] = df_global_ssl_ir['level_3'].astype(str)
            df['level_3'] = df['level_3'].astype(str)

            df = df[df['level_3'].isin(df_global_ssl_ir['level_3'])]

         # save data
        if ds_name == 'OCHOA':
            pass
        else:
            df.to_csv(os.path.join(output_directory, "all_data", f"all_data_{i}.csv"), index=False)

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
    #second_check_tables = sorted(glob(os.path.join(base_directory, 'raw_data', 'second_checks', '*.csv')))

    for i in tables:
        ds_name = os.path.basename(i).split(".")[0].split("_")[2]
        df = pd.read_csv(i, low_memory=False)
        df = df.drop_duplicates()
        df = df.drop(df[df.latitude == 'unk'].index)

        # implement secondary check
        #selected_check = [x for x in second_check_tables if ds_name in x]
        #df_check = pd.read_csv(selected_check[0], usecols=['fname', 'class', 'check(use y or n)'])

        # for NGSA, file names changed
        # if ds_name == 'NGSA':
        #     df_check['fname'] = df_check['fname'].str.replace('.', '_')
        #     df_check['fname'] = df_check['fname'].str.replace(':', '_')
        #     df_check['fname'] = df_check['fname'].str.split('_').str[1]
        #     df_check['fname'] = df_check['fname'].str.replace('B', 'S')

        #df_check = df_check.loc[df_check['check(use y or n)'] == 'y'].copy()
        #df = pd.merge(df, df_check, how='left', indicator='Exist')
        #df['Exist'] = np.where(df.Exist == 'both', True, False)
        #df = df[df['Exist'] == True].drop(['Exist'], axis=1)
        #df.drop(columns=df.columns[-2:], axis=1, inplace=True)

        if ds_name == 'SR' or ds_name == 'DP' or ds_name == 'SSL-IR': # these are the SHIFT domain box
            df.to_csv(os.path.join(output_directory, 'geofilter', f'geofilter_{ds_name}.csv'), index=False)
        else:
            points = gp.GeoDataFrame(df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
            within_points = gp.sjoin(points, shp, predicate='within')

            if within_points.empty:
                continue
            else:
                df = pd.DataFrame(within_points)
                df.drop(columns=df.columns[-14:], axis=1, inplace=True)

            df.to_csv(os.path.join(output_directory, 'geofilter', f'geofilter_{ds_name}.csv'), index=False)
    time.sleep(3)
    print("done")


def convolve_library(base_directory, output_directory, sensor:str,geo_filter: bool):
    wavelengths_asd = spectra.load_asd_wavelenghts()

    emit_wvls, emit_fwhm = spectra.load_wavelengths(sensor=sensor)

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
                           **{"desc": "\t loading convolution... " + ds_name, "ncols": 150})
        else:
            results = p_map(partial(spectra.convolve, asd_wvl=wavelengths_asd, wvl=emit_wvls, fwhm=emit_fwhm,
                                    spectra_starting_col=7), [row for row in df.iterrows()],
                            **{"desc": "\t loading convolution... " + ds_name, "ncols": 150})

        df_data_merge = pd.concat([df.iloc[:, :7], pd.DataFrame(results)], axis=1)
        all_results.append(df_data_merge)

    # create dataframe, remove duplicates, no data, and save
    output_cols = ["dataset", "level_1", "level_2", "level_3", 'fname', 'longitude',
                   'latitude'] + emit_wvls.tolist()

    df_merge = pd.concat(all_results)
    df_merge.columns = output_cols
    df_merge = df_merge.reset_index().sort_values(["level_1"])
    df_merge['fname'] = df_merge['fname'].str.lower()

    if geo_filter:
        df_global = spectra.load_global_library_metadata()
        df_global['fname'] = df_global['fname'].str.lower()

        sorting_order = df_global['fname'].str.lower().values

        df_merge = df_merge.set_index('fname').reindex(sorting_order).reset_index()
        df_merge = df_merge.drop(columns=['index'])
        df_merge = df_merge[output_cols]
        df_merge.to_csv(os.path.join(output_directory, "convolved", "geofilter_convolved.csv"), index=False)

        df_geo_data = df_merge.iloc[:, :7]
        df_shp = gp.GeoDataFrame(df_geo_data, geometry=gp.points_from_xy(df_geo_data.longitude, df_geo_data.latitude),
                                 crs="EPSG:4326")
        df_shp.to_file(os.path.join(base_directory, "gis", "emit_global_spectral_library.shp"),
                       driver='ESRI Shapefile')

    else:
        df_merge.to_csv(os.path.join(output_directory, "convolved", "all_data_convolved.csv"),
                        index=False)
    print('done')


def run_clean_workflow(base_directory, output_directory, sensor, geo_filter: bool):
    download_data(base_directory=base_directory, output_directory=output_directory)
    standardize_all_data(base_directory=base_directory, output_directory=output_directory)
    geofilter_data(base_directory=base_directory, output_directory=output_directory)
    convolve_library(geo_filter=geo_filter, output_directory=output_directory, base_directory=base_directory, sensor=sensor)
