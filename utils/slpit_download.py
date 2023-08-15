import os
import datetime
import time
from dateutil.relativedelta import relativedelta
from zerionPy import IFB
import pickle
import earthaccess
import pandas as pd
import geopandas as gp
from utils.create_tree import create_directory


# create object folder to store the pickle objects
create_directory('objects')

ck = r'rnrLR3K9SHt3DopUdvqSXN7me4FGtjblRyuqSzL6.YquivC50dLbkqWfBZ7AH'
sk = r'7Og6inIanLbg0G6xwSvwrOGUTilBqm4htXQ1N1oX'
profile_id = 504019
spectral_endmembers_page_id = 3856841
emit_transects_page_id = 3856847
server_name = 'tech-ate'


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


def download_emit(base_directory):
    auth = earthaccess.login(strategy="interactive")

    create_directory(os.path.join(base_directory, 'gis', 'emit-data'))
    create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'nc_files'))

    # get plot center points from ipad
    shapefile = os.path.join(base_directory, 'gis', "Observation.shp")
    today = datetime.datetime.today().strftime('%Y-%m')
    today_date = datetime.datetime.strptime(today, '%Y-%m')
    next_month_date = today_date + relativedelta(months=1)
    next_month_str = next_month_date.strftime('%Y-%m')

    df = pd.DataFrame(gp.read_file(shapefile))
    df = df.sort_values('Name')

    for index, row in df.iterrows():
        plot = row['Name']
        lon = row['geometry'].x
        lat = row['geometry'].y

        lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat = lon, lat, lon, lat

        results = earthaccess.search_data(short_name="EMITL2ARFL", version="001", cloud_hosted=True,
                                              bounding_box=(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat),
                                              temporal=("2022-08", next_month_str), count=-1)
        
        if plot == 'SPEC - 028':
            print(results)
        files = earthaccess.download(results, os.path.join(base_directory, 'gis', 'emit-data', 'nc_files'))


def run_download_scripts(base_directory):
    #download_emit(base_directory=base_directory)
    emit_slpit_recrods = get_iform_records(server_name=server_name, client_key=ck, secret_key=sk, profile_id=profile_id,
                                           page_id=emit_transects_page_id)
    save_pickle(emit_slpit_recrods, 'emit_slpit')
