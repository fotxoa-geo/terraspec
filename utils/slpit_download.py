import os
import datetime
import subprocess
from zerionPy import IFB
import pickle
import earthaccess
import pandas as pd
import geopandas as gp
from utils.create_tree import create_directory
from sys import platform
import json
from glob import glob
from datetime import datetime, timedelta

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


data_product_key = {"emit": {'reflectance': 'EMITL2ARFL',
                              'radiance': 'EMITL1BRAD',
                              'version': '001'},
                    "aviris_ng": {'reflectance': 'SHIFT_AVNG_L2A_RFL_V2_2431',
                                    'version': '2'}}

aviris_ng_scenes = [ '20220308t190523', '20220308t191151', '20220308t204043', '20220308t205512', '20220316t210303',
                     '20220322t204749', '20220412t205405', '20220511t190344', '20220511t212317', '20220914t184300',
                     '20220915t185652', '20220915t195816', '20220915t200714', '20220915t203517']


def download_scenes(base_directory, sensor, aoi):
    auth = earthaccess.login(strategy="netrc")

    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data'))

    if sensor == 'emit':
        create_directory(os.path.join(base_directory, 'gis', f'emit-data', 'nc_files'))
        create_directory(os.path.join(base_directory, 'gis', f'emit-data', 'nc_files', 'l1b'))
        create_directory(os.path.join(base_directory, 'gis', f'emit-data', 'nc_files', 'l2a'))

    df = gp.read_file(aoi)
    lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat = df.total_bounds

    # Create output directories
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs'))

    # create outlog directory
    out_base = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products')
    out_logs = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs')


    results = earthaccess.search_data(short_name=data_product_key[sensor]['reflectance'],
                                      version=data_product_key[sensor]['version'], cloud_hosted=True,
                                      bounding_box=(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat))

    em_file = os.path.join('terraspec_output', 'simulation', 'output', 'endmember_libraries',
                           f'convex_hull__n_dims_4_sensor_{sensor}_geofilter_True_unmix_library.csv')

    if results:
        print(f'found {len(results)} granules!')
        files = earthaccess.download(results, os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a'))

        print(f"\t download successful... {len(files)} granules downloaded")

        # run nc downloads
        for nc_file in files:
            basename = os.path.basename(nc_file)
            base_call = f'sh {os.path.join("fire", "emit_aoi_process.sh")} {nc_file} {em_file} {out_base} {aoi}'
            outfile = os.path.join(f"{os.path.join(out_logs, basename)}.out")
            sbatch_cmd = f"sbatch -p patient -N 1 -c 10 --mem 25G --output {outfile} --job-name lake-fire --wrap='{base_call}'"
            subprocess.call(sbatch_cmd, shell=True)


def enmap_process(base_directory, sensor, aoi):

    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data'))

    # Create output directories
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs'))

    # create outlog directory
    out_base = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products')
    out_logs = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs')

    em_file = os.path.join('terraspec_output', 'simulation', 'output', 'endmember_libraries',
                           f'convex_hull__n_dims_4_sensor_{sensor}_geofilter_True_unmix_library.csv')

    files = glob(os.path.join(base_directory, 'gis', f'{sensor}-data', 'tif_files', '*.TIF'))
    print(f'found {len(files)} TIF files!')

    # run tif downloads
    for tif_file in files:
        basename = os.path.basename(tif_file).split('.')[0]
        base_call = f'sh {os.path.join("fire", "emit_aoi_process.sh")} {tif_file} {em_file} {out_base} {aoi}'
        outfile = os.path.join(f"{os.path.join(out_logs, basename)}.out")
        sbatch_cmd = f"sbatch -p patient -N 1 -c 10 --mem 25G --output {outfile} --job-name lake-fire --wrap='{base_call}'"
        subprocess.call(sbatch_cmd, shell=True)

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

    if sensor == 'aviris_ng':
        create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files'))
        create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l1b'))
        create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a'))

        # get plot center points from ipad
        shapefile = os.path.join('gis', "shift_transects_centroid.shp")
        df = pd.DataFrame(gp.read_file(shapefile))

    # Create output directories
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs'))
    
    # create outlog directory
    out_base = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products')
    out_logs = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs')
    
    # em file for unmixing
    em_file = os.path.join('terraspec_output', 'simulation', 'output', 'endmember_libraries', f'convex_hull__n_dims_4_sensor_{sensor}_geofilter_True_unmix_library.csv')

    if sensor == 'emit':
        sensor_dates = sorted(list(df['EMIT DATE'].unique()))
    elif sensor == 'aviris_ng':
        sensor_dates = sorted(list(aviris_ng_scenes))
    
    for sensor_date in sensor_dates:
        if sensor_date == '':
            continue
        
        print(f"downloading... {sensor_date}")
        results = earthaccess.search_data(short_name=data_product_key[sensor]['reflectance'], version=data_product_key[sensor]['version'], cloud_hosted=True, granule_name=f'*{sensor_date}*')
        if results:
            print(f'found {len(results)} granules!')
            files = earthaccess.download(results, os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a'))
            
            print(f"\t download successful... {len(files)} granules downloaded")
            
            # run nc downloads
            for nc_file in files:
                basename = os.path.basename(nc_file)
                base_call = f'sh {os.path.join("slpit", "emit_image_process.sh")} {nc_file} {em_file} {out_base}'
                outfile = os.path.join(f"{os.path.join(out_logs, basename)}.out")
                sbatch_cmd = f"sbatch -p patient -N 1 -c 20 --mem 25G --output {outfile} --job-name slpit.em --wrap='{base_call}'"
                subprocess.call(sbatch_cmd, shell=True)
        
        else:
            print(f'no scenes found for {sensor_date}')

       
def download_shift_imagery(base_directory, sensor):
    auth = earthaccess.login(strategy="netrc")

    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l1b'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a'))

    # get plot center points from ipad
    shapefile = os.path.join('gis', "shift_plot_coordinates.csv")
    df = pd.read_csv(shapefile)

    # Create output directories
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products'))
    create_directory(os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs'))

    # create outlog directory
    out_base = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products')
    out_logs = os.path.join(base_directory, 'gis', f'{sensor}-data', 'products', 'logs')

    # em file for unmixing
    em_file = os.path.join('terraspec_output', 'simulation', 'output', 'endmember_libraries',
                           f'convex_hull__n_dims_4_sensor_{sensor}_geofilter_True_unmix_library.csv')
    for index, row in df.iterrows():
        lon = row['longitude']
        lat = row['latitude']
        plot_name = row['Plot Name']
        season = row['Season']
        date = row['Date']

        sample_date = datetime.strptime(date, "%m/%d/%Y")
        next_date = sample_date + timedelta(days=5) 
        previous_date = sample_date - timedelta(days=5)
        previous_date = previous_date.strftime("%Y/%m/%d")
        next_date = next_date.strftime("%Y/%m/%d")

        print(f"downloading... {plot_name} {season} {date}")
        results = earthaccess.search_data(short_name=data_product_key[sensor]['reflectance'],
                                          version=data_product_key[sensor]['version'], cloud_hosted=True,
                                          bounding_box=(lon, lat, lon, lat), temporal=(previous_date, next_date))
        if results:
            print(f'found {len(results)} granules!')
            files = earthaccess.download(results,
                                         os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a'))
            print(f"\t download successful... {len(files)} granules downloaded")

        else:
            print(f'no scenes found for {plot_name}')
    
    # run nc downloads
    files = list(sorted(glob(os.path.join(base_directory, 'gis', f'{sensor}-data', 'nc_files', 'l2a', '*.nc'))))
    files = [f for f in files if f.endswith('.nc')]
    for nc_file in files:
        basename = os.path.basename(nc_file)
        base_call = f'sh {os.path.join("shift", "aviris_image_process.sh")} {nc_file} {em_file} {out_base} {sensor}'
        outfile = os.path.join(f"{os.path.join(out_logs, basename)}.out")
        sbatch_cmd = f"sbatch -p patient -N 1 -c 20 --mem 40G --output {outfile} --job-name shift --wrap='{base_call}'"
        subprocess.call(sbatch_cmd, shell=True)


def run_download_emit(base_directory, sensor):
    download_emit(base_directory=base_directory, sensor=sensor)

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


