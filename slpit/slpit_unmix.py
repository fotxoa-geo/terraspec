import os
import subprocess
from itertools import product
import pandas as pd
from glob import glob
from simulation.run_unmix import call_unmix
from utils.create_tree import create_directory
from utils.text_guide import cursor_print
from utils.spectra_utils import spectra
import geopandas as gp
from datetime import datetime
import time

class unmix_runs:
    def __init__(self, base_directory: str,  dry_run=False):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        create_directory(os.path.join(self.output_directory, 'outlogs'))
        create_directory(os.path.join(self.output_directory, 'scratch'))

        self.output_directory = os.path.join(base_directory, 'output')
        self.spectral_transect_directory = os.path.join(self.output_directory, 'spectral_transects', 'transect')
        self.spectral_em_directory = os.path.join(self.output_directory, 'spectral_transects', 'endmembers')
        self.gis_directory = os.path.join(base_directory, 'gis')

        # simulation parameters for spatial and hypertrace unmix
        self.optimal_parameters_sma = ['num_endmembers 20', '--n_mc 25', '--normalization brightness']
        self.optimal_parameters_mesma = ['max_combinations 100', '--n_mc 25', '--normalization brightness']
        self.non_opt_mesma = ['max_combinations 100', '--n_mc 1', '--normalization brightness']

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='emit')

        # load dry run parameter
        self.dry_run = dry_run

        # set scale to 1 reflectance
        self.scale = '0.5'

        # emit global library
        terraspec_base = os.path.join(base_directory, "..")
        em_sim_directory = os.path.join(terraspec_base, 'simulation', 'output', 'endmember_libraries')
        self.emit_global = os.path.join(em_sim_directory, 'convex_hull__n_dims_4_unmix_library.csv')
        self.spectra_starting_column_local = '11'
        self.spectra_starting_column_global = '8'

    def unmix_calls(self, mode:str):
        # Start unmixing process
        cursor_print(f"commencing... {mode}")

        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
        df = df.sort_values('Name')

        # create directory
        create_directory(os.path.join(self.output_directory, mode))
        
        count = 0

        for index, row in df.iterrows():
            #if count != 0:
            #    continue

            plot = row['Name']

            emit_filetime = row['EMIT Date']
            reflectance_img_emit = glob(os.path.join(self.gis_directory, 'emit-data-clip', f'*{plot.replace(" ", "")}_RFL_{emit_filetime}'))
            reflectance_uncer_img_emit = glob(os.path.join(self.gis_directory, 'emit-data-clip',f'*{plot.replace(" ", "")}_RFLUNCERT_{emit_filetime}'))
            em_local = os.path.join(self.spectral_em_directory, plot.replace("SPEC", 'Spectral').replace(" ", "") + '-emit.csv')
            asd_reflectance = glob(os.path.join(self.spectral_transect_directory, f'*{plot.replace("SPEC", "Spectral").replace(" ", "")}'))

            if os.path.isfile(em_local):
                if mode == 'mesma':
                    simulation_parameters = [self.optimal_parameters_mesma, self.non_opt_mesma] 
                else:
                    simulation_parameters = [self.optimal_parameters_sma]

                output_dest = os.path.join(self.output_directory, mode)

                for i in simulation_parameters:
                    # unmix asd with local library
                    out_param_string = " ".join(i)
                    out_param_name = plot.replace(" ", "") + "___" + out_param_string.replace('--','').replace('_', '-').replace(' ', '_')
                    
                    call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=asd_reflectance[0], em_file=em_local,
                           parameters=i, output_dest=os.path.join(output_dest, 'asd-local___' + out_param_name),
                           scale=self.scale,  spectra_starting_column=self.spectra_starting_column_local)

                    # unmix asd with global library
                    call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=asd_reflectance[0], em_file=self.emit_global,
                           parameters=i, output_dest=os.path.join(output_dest, 'asd-global___' + out_param_name),
                           scale=self.scale,  spectra_starting_column=self.spectra_starting_column_global)

                    # emit pixels unmixed with local em
                    call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_img_emit[0], em_file=em_local,
                           parameters=i, output_dest=os.path.join(output_dest, 'emit-local___' + out_param_name),
                           scale=self.scale, uncertainty_file=reflectance_uncer_img_emit[0],
                           spectra_starting_column=self.spectra_starting_column_local)

                    # emit pixels unmixed with global
                    call_unmix(mode=mode, dry_run=self.dry_run, reflectance_file=reflectance_img_emit[0], em_file=self.emit_global,
                           parameters=i, output_dest=os.path.join(output_dest, 'emit-global___' + out_param_name),
                           scale=self.scale, uncertainty_file=reflectance_uncer_img_emit[0],
                           spectra_starting_column=self.spectra_starting_column_global)
                    
                    count += 1
            else:
                print(em_local, " does not exist.")
            
                

def run_slipt_unmix(base_directory, dry_run):
    all_runs = unmix_runs(base_directory, False)
    all_runs.unmix_calls(mode='sma-best')
    all_runs.unmix_calls(mode='mesma')
