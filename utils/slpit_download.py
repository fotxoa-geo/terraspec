import os
import datetime
import subprocess
import time
from dateutil.relativedelta import relativedelta
from zerionPy import IFB
import pickle
import earthaccess
import pandas as pd
import geopandas as gp
from utils.create_tree import create_directory
from sys import platform
import json
from glob import glob

# create object folder to store the pickle objects
create_directory('objects')

def get_iform_records(server_name:str, client_key:str, secret_key:str, profile_id:int, page_id: int):
    api = IFB(server_name, 'us', client_key, secret_key, 6)
    results = api.getRecords(profile_id, page_id).response

    print("downloading... ", len(results), " records")
    records = []
    for i in results:
        data = api.getRecord(profile_id, page_id, i['id']).response
        records.append(dict(list(data.items())[14:]))

    return records


def save_pickle(object, filename):
    with open(os.path.join('objects', filename + '.pickle'), 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(os.path.join('objects', filename + '.pickle'), 'rb') as handle:
        b = pickle.load(handle)

        return b


data_product_key = {"emit": { 'reflectance': 'EMITL2ARFL',
                              'radiance': 'EMITL1BRAD',
                              'version': '001'}}


def download_emit(base_directory, sensor):
    auth = earthaccess.login(strategy="netrc")

    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data'))

    if sensor == 'emit':
        create_directory(os.path.join(base_directory, 'gis', f'emit-data', 'nc_files'))
        create_directory(os.path.join(base_directory, 'gis', f'emit-data', 'nc_files', 'l1b'))
        create_directory(os.path.join(base_directory, 'gis', f'emit-data', 'nc_files', 'l2a'))

    # get plot center points from ipad
    shapefile = os.path.join('gis', "Observation.json")

    df = pd.DataFrame(gp.read_file(shapefile))
    df = df.sort_values('Name')

    for index, row in df.iterrows():
        plot = row['Name']
        plot_num = int(plot.split('-')[1])
        if plot_num <= 1:
            lon = row['geometry'].x
            lat = row['geometry'].y
            emit_date = row['EMIT DATE']

            plot_date = datetime.datetime.strptime(emit_date, '%Y%m%dT%H%M%S')

            next_plot_months =  plot_date + relativedelta(months=3)
            next_plot_months = next_plot_months.strftime('%Y-%m')

            previous_plot_months = plot_date - relativedelta(months=3)
            previous_plot_months = previous_plot_months.strftime('%Y-%m')

            lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat = lon, lat, lon, lat

            print(f"downloading... {plot}")
            results = earthaccess.search_data(short_name=data_product_key[sensor]['reflectance'],
                                              version=data_product_key[sensor]['version'], cloud_hosted=True,
                                              bounding_box=(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat),
                                              temporal=(previous_plot_months, next_plot_months), count=-1)
            files = earthaccess.download(results, os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a'))
            results = earthaccess.search_data(short_name=data_product_key[sensor]['radiance'],
                                              version=data_product_key[sensor]['version'], cloud_hosted=True,
                                              bounding_box=(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat),
                                              temporal=(previous_plot_months, next_plot_months), count=-1)
            files = earthaccess.download(results, os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l1b'))

    # Create output directories
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs'))

    nc_files = (glob(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l1b', '*.nc'), recursive=True)
                + glob(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a', '*.nc'), recursive=True))

    out_base = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products')
    out_logs = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs')

    em_file = os.path.join('terraspec_output', 'simulation', 'output', 'endmember_libraries',
                           f'convex_hull__n_dims_4_sensor_{sensor}_geofilter_True_unmix_library.csv')

    for nc_file in nc_files:
        basename = os.path.basename(nc_file)
        base_call = f'sh {os.path.join("slpit", "emit_image_process.sh")} {nc_file} {em_file} {out_base}'
        outfile = os.path.join(f"{os.path.join(out_logs, basename)}.out")
        sbatch_cmd = f"sbatch --export=ALL -p patient -N 1 -c 40 --mem 50G --output {outfile} --job-name slpit --wrap='{base_call}'"
        subprocess.call(sbatch_cmd, shell=True)

def run_download_emit(base_directory, sensor):
    download_emit(base_directory=base_directory, sensor=sensor)


def sync_gdrive(base_directory, project):

    if "linux" in platform:
        output_directory = os.path.join(base_directory, 'data', 'spectral_transects')
        create_directory(output_directory)
        if project == 'emit':
            base_call = f"rclone copy gdrive:terraspec/slpit/data/spectral_transects {output_directory} -P"
        else:
            output_directory = os.path.join(base_directory, 'data')
            base_call = f"rclone copy gdrive:terraspec/shift/data/ {output_directory} -P"
        subprocess.call(base_call, shell=True)
    else:
        print("Cannot sync between local machine! Upload data from ASD computer to google drive.")


def sync_extracts(base_directory, project):
    if "linux" in platform:
        output_directory = os.path.join(base_directory, 'gis', f'{project}-data-clip')
        create_directory(output_directory)
        if project == 'emit':
            base_call = f"rclone copy {output_directory} gdrive:terraspec/slpit/gis/{project}-data-clip/ -P"
        else:
            base_call = f"rclone copy {output_directory} gdrive:terraspec/shift/gis/{project}-data-clip/ -P"
        subprocess.call(base_call, shell=True)
    else:
        print("Extracts are being done in cluster! Cannot sync between local machine.")


def get_ck_sk():
    f = open('slpit/config.json')
    data = json.load(f)
    metadata = data['keys']  # geodata from spec library
    ck = metadata['ck']
    sk = metadata['cs']

    return ck, sk

profile_id = 504019
spectral_endmembers_page_id = 3856841
emit_transects_page_id = 3856847
shift_transects_id = 3856837
server_name = 'tech-ate'


def run_dowloand_slpit():
    ck, sk = get_ck_sk()
    emit_slpit_recrods = get_iform_records(server_name=server_name, client_key=ck, secret_key=sk, profile_id=profile_id,
                                           page_id=emit_transects_page_id)
    save_pickle(emit_slpit_recrods, 'emit_slpit')

def download_shift_slpit():
    ck, sk = get_ck_sk()
    shift_slpit_recrods = get_iform_records(server_name=server_name, client_key=ck, secret_key=sk, profile_id=profile_id,
                                           page_id=shift_transects_id)
    save_pickle(shift_slpit_recrods, 'shift_slpit')


