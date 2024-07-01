from utils.create_tree import create_directory
from utils.text_guide import cursor_print, query_yes_no, input_date, execute_call
from utils.envi import envi_tiff_rgb
import os
import sys
from glob import glob
from p_tqdm import p_map
from functools import partial
import subprocess
import time

# path fix for linux
#path = os.environ['PATH']
#path = path.replace('\Library\\bin;', ':')
#os.environ['PATH'] = path

class emit:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.spectral_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')

        # create output directories
        create_directory(os.path.join(base_directory, 'gis', 'emit-data'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'nc_files'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'envi'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'envi', 'l1b'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data', 'envi', 'l2a'))
        create_directory(os.path.join(base_directory, 'gis', 'rgb-quick-look'))
        create_directory(os.path.join(base_directory, 'gis', 'emit-data-clip'))
        create_directory(os.path.join(base_directory, 'gis', 'outlogs'))
        create_directory(os.path.join(base_directory, 'gis', 'rgb-envi'))

        # define gis directory
        self.gis_directory = os.path.join(base_directory, 'gis')

    def nc_to_envi(self, product, ortho=False):
        msg = f"\nTerraSpec has created the following directory: {os.path.join(self.base_directory, 'gis', 'emit-data', 'envi')}\n" \
              f"ENVI files along with corresponding HDR files will be stored here."
        cursor_print(msg)
        create_directory(os.path.join(self.gis_directory, 'outlogs', 'nc_processes'))
        nc_files = glob(os.path.join(self.gis_directory, 'emit-data', 'nc_files', product, '*.nc'))
        
        for i in nc_files:
            basename = os.path.basename(i).split(".")[0]
            nc_outfile = os.path.join(os.path.join(self.gis_directory, 'outlogs', 'nc_processes', f'{basename}.out'))
            
            if ortho == True:
                base_call = f'python ./emit_utils/reformat.py {i} {os.path.join(self.gis_directory, "emit-data", "envi", product)} --orthorectify'
            else:
                base_call = f'python ./emit_utils/reformat.py {i} {os.path.join(self.gis_directory, "emit-data", "envi", product)}'
    
            subprocess.call(['sbatch', '-N', '1', '-c', '1', '--mem', '80G', '--output', nc_outfile, '--job-name', f'emit.{product}', '--wrap', f'{base_call}'])

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
        shapefile = os.path.join('gis', "Observation.shp")

        # get emit reflectance files
        reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data', 'envi', 'l2a', '*_reflectance')))

        # get emit uncertainty files
        reflectance_uncertainty_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data', 'envi', 'l2a', '*_reflectance_uncertainty')))

        # get mask files
        mask_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data', 'envi', 'l2a', '*_mask')))
        mask_files = [file for file in mask_files if '_band_mask' not in file]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        window_extract_path = os.path.join(current_dir, 'window_extract.py')

        all_files = reflectance_files + reflectance_uncertainty_files + mask_files
        
        create_directory(os.path.join(self.gis_directory, 'outlogs', 'extracts'))
        
        for file in all_files:
            acquisition_date = os.path.basename(file).split("_")[4]
            acquisition_type = os.path.basename(file).split("_")[2]
            version = os.path.basename(file).split("_")[3]
            product = os.path.basename(file).split("_")[1]

            corresponding_nc_file = sorted(glob(os.path.join(self.gis_directory, 'emit-data', 'nc_files', product.lower(), f'*{product}_{acquisition_type}_{version}_{acquisition_date}*.nc')))
            nc_file = corresponding_nc_file[0]
            
            base_call = f'python {window_extract_path} -rfl_img {file} -nc_file {nc_file} -w_size {window_size} ' \
                        f'-shp {shapefile} -pad {pad} -out {os.path.join(self.gis_directory, "emit-data-clip")} '
            
            outfile = os.path.join(self.gis_directory, 'outlogs', 'extracts', acquisition_date + '_' + acquisition_type + '.out')
            sbatch_cmd = f"sbatch -N 1 -c 1 --mem 50G --output {outfile} --job-name emit.extract  --wrap='{base_call}'"
            
            subprocess.run(sbatch_cmd, shell=True, text=True)
    

    def envi_rgbs(self):
        reflectance_files = sorted(glob(os.path.join(self.gis_directory, 'emit-data', 'envi', 'l2a', '*_reflectance')))
        out_path = os.path.join(self.gis_directory, f'rgb-envi')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        rgb_path = os.path.join(current_dir, 'rgb_envi.py')
        
        for file in reflectance_files:
            acquisition_date = os.path.basename(file).split("_")[4]
            acquisition_type = os.path.basename(file).split("_")[2]
            version = os.path.basename(file).split("_")[3]
            product = os.path.basename(file).split("_")[1]
            
            corresponding_nc_file = sorted(glob(os.path.join(self.gis_directory, 'emit-data', 'nc_files', product.lower(), f'*{product}_{acquisition_type}_{version}_{acquisition_date}*.nc')))
            nc_file = corresponding_nc_file[0]

            base_call = f'python {rgb_path} -rfl_img {file} -nc_file {nc_file} -out {out_path} '

            outfile = os.path.join(self.gis_directory, 'outlogs', f"{acquisition_date}-rgb.out")
            sbatch_cmd = f"sbatch -N 1 -c 1 --mem 50G --output {outfile} --job-name emit.rgb  --wrap='{base_call}'"
            subprocess.run(sbatch_cmd, shell=True, text=True)

def geoprocess_menu():
    print("geoprocess mode for SLPIT...")
    print("A... Field rgbs - 8 bit")
    print("B... ENVI RGB - Float 32")
    print("C... NC to ENVI")
    print("D... Extract 3 x 3 pixels")
    print("E... Exit")

def run_geoprocess_utils(base_directory, dry_run):
    geo = emit(base_directory=base_directory)
    
    while True:
        
        geoprocess_menu()

        user_input = input('\nPlease indicate the desired command: ').upper()
        
        if user_input == 'A':
            geo.rgb_quick_look()
        elif user_input == 'B':
            geo.envi_rgbs()
        elif user_input == 'C':
            geo.nc_to_envi(product="l1b", ortho=True)
            geo.nc_to_envi(product="l2a", ortho=False)
        elif user_input == 'D':
            geo.clip_emit(window_size=3, pad=1, dry_run=dry_run)
        elif user_input == 'E':
            break





