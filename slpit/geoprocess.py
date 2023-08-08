
from utils.create_tree import create_directory
from utils.text_guide import cursor_print, query_yes_no, input_date, execute_call
from utils.envi import envi_tiff_rgb
import os
import sys
from glob import glob
from p_tqdm import p_map
from functools import partial
import subprocess
import pandas as pd
import geopandas as gp
from datetime import datetime

# path fix for linux
path = os.environ['PATH']
path = path.replace('\Library\\bin;', ':')
os.environ['PATH'] = path


class emit:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.spectral_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')

        # create output directories
        create_directory(os.path.join(base_directory, 'gis', 'emit-data'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'nc_files'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'envi'))
        create_directory(os.path.join(base_directory, 'gis', 'rgb-quick-look'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data-clip'))

        # define gis directory
        self.gis_directory = os.path.join(base_directory, 'gis')

    def nc_to_envi(self):
        msg = f"\nTerraSpec has created the following directory: {os.path.join(self.base_directory, 'gis', 'emit-data', 'envi')}\n" \
              f"ENVI files along with corresponding HDR files will be stored here."
        cursor_print(msg)

        nc_files = glob(os.path.join(self.gis_directory, 'emit-data', 'nc_files', '*.nc'))
        for i in nc_files:
            basename = os.path.basename(i).split(".")[0]
            base_call = f'python ./slpit/emit_utils/reformat.py {i} {os.path.join(self.gis_directory, "emit-data", "envi")} --orthorectify'

            subprocess.call(['sbatch', '-N', '1', '-c', '40', '--mem', '180G', '--wrap', f'{base_call}'])

    def rgb_quick_look(self):
        msg = f"\nTerraSpec has created the following directory: {os.path.join(self.base_directory, 'gis', 'rgb-quick-look')}\n" \
              f"Please download and move the ENVI files from the LP DAAC to the previously listed directory."

        cursor_print(msg)
        user_input = query_yes_no('\nHave the ENVI reformat jobs finihsed processing?', default="yes")

        while True:
            if user_input:
                user_date = input_date(msg="Please provide EMIT acquisition date in YYYYMMDD: ",
                                       gis_directory=self.gis_directory)

                # # get rgb quick look image
                p_map(partial(envi_tiff_rgb, output_directory=os.path.join(self.gis_directory, 'rgb-quick-look')),
                      user_date[1],
                      **{
                          "desc": "\t\t processing envi files: " + user_date[0] + " ...",
                          "ncols": 150})

                another_date = query_yes_no(f'\nTerraSpec finished processing: {user_date[0]}\n'
                                            f'Do you wish to process another acquisition date?', default="yes")
                if another_date:
                    continue
                else:
                    break
            else:
                sys.stdout.write(
                    f"Please move envi files to: {os.path.join(self.gis_directory, 'emit-data')}\n")

    def clip_emit(self, window_size: int, pad: int, dry_run: bool):
        msg = f"\nTerraSpec has created the following directory: {os.path.join(self.gis_directory, 'emit-data-clip')}\n"\
              f"ENVI files clipped to plot extents along with corresponding HDR files will be stored here."
        cursor_print(msg)

        # get plot center points from ipad - these are the plot centers
        shapefile = os.path.join(self.gis_directory, "Observation.shp")

        # get emit reflectance files
        reflectance_files = glob(os.path.join(self.gis_directory, 'emit-data', 'envi', '*_reflectance'))

        for reflectance_file in reflectance_files:
            base_call = f'python ./slpit/window_extract.py -rfl_img {reflectance_file} -w_size {window_size} ' \
                        f'-shp {shapefile} -pad {pad} -out {os.path.join(self.gis_directory, "emit-data-clip")} '

            # make call to clipping file using os run
            execute_call(['sbatch', '-N', "1", '-c', '40', '--mem', "180G", '--wrap', f'{base_call}'], dry_run)

        # get emit reflectance uncertainties
        reflectance_uncer_files = glob(os.path.join(self.gis_directory, 'emit-data', 'envi', '*_reflectance_uncertainty'))

        for reflectance_uncer_file in reflectance_uncer_files:
            base_call = f'python ./slpit/window_extract.py -rfl_img {reflectance_uncer_file} -w_size {window_size} ' \
                        f'-shp {shapefile} -pad {pad} -out {os.path.join(self.gis_directory, "emit-data-clip")} '

            # make call to clipping file using os run
            execute_call(['sbatch', '-N', "1", '-c', '40', '--mem', "180G", '--wrap', f'{base_call}'], dry_run)


def run_geoprocess_utils(base_directory, nc_to_envi:bool):
    geo = emit(base_directory=base_directory)

    if nc_to_envi:
        geo.nc_to_envi()
    geo.rgb_quick_look()


def run_geoprocess_extract(base_directory, dry_run: bool):
    geo = emit(base_directory=base_directory)
    geo.clip_emit(window_size=3, pad=1, dry_run=dry_run)





