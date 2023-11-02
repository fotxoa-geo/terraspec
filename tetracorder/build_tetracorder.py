import os
import shutil
import time
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.envi import envi_to_array, get_meta, save_envi, load_band_names, augment_envi
from utils.text_guide import cursor_print
import numpy as np
import pandas as pd
from p_tqdm import p_map
from functools import partial
from glob import glob
import itertools
import geopandas as gp
from datetime import datetime
from utils.unmix_utils import call_unmix, call_hypertrace_unmix, hypertrace_meta, create_uncertainty
from simulation.run_hypertrace import hypertrace_workflow

class tetracorder:

    def __init__(self, base_directory: str, sensor:str):

        self.base_directory = os.path.join(base_directory, 'tetracorder')
        self.tetra_data_directory = os.path.join(self.base_directory, 'data')
        self.tetra_output_directory = os.path.join(self.base_directory, 'output')
        self.simulation_output_directory = os.path.join(base_directory, 'simulation', 'output')
        self.slpit_output_directory = os.path.join(base_directory, 'slpit', 'output')
        self.slpit_gis_directory = os.path.join(base_directory, 'slpit', 'gis')

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)
        self.sensor = sensor

        # create output directory for augmented files
        create_directory(os.path.join(self.tetra_output_directory, 'augmented'))
        create_directory(os.path.join(self.tetra_output_directory, 'spectral_abundance'))

        self.augmented_dir = os.path.join(os.path.join(self.tetra_output_directory, 'augmented'))

        #create_directory(os.path.join(self.output_directory, 'outlogs'))
        #create_directory(os.path.join(self.output_directory, 'scratch'))

    def generate_tetracorder_reflectance(self):
        cursor_print('generating reflectance')

        df_sim = pd.read_csv(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                          'convex_hull__n_dims_4_simulation_library.csv'))

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim,
                                      level='level_1', spectral_bundles=10000, increment_size=0.10,
                                      output_directory=self.augmented_dir, wvls=self.wvls,
                                      name='tetracorder', spectra_starting_col=8)

    def hypertrace_tetracorder(self):
        cursor_print('hypertrace: tetracorder')
        hypertrace_workflow(dry_run=False, clean=False,
                            configfile=os.path.join('simulation', 'hypertrace', 'tetracorder.json'))

    def unmix_tetracorder(self):
        cursor_print('unmixing tetracorder')

        em_file = os.path.join(self.simulation_output_directory, 'endmember_libraries',
                               'convex_hull__n_dims_4_unmix_library.csv')

        optimal_parameters = ['--num_endmembers 30', '--n_mc 25', '--normalization brightness']

        reflectance_file = os.path.join(self.augmented_dir, 'tetracorder_spectra')
        call_unmix(mode='sma-best', dry_run=False, reflectance_file=reflectance_file, em_file=em_file,
                   parameters=optimal_parameters, output_dest=self.augmented_dir, scale='1',
                   spectra_starting_column='8')

        print("loading hypertrace outputs...")
        estimated_reflectances = glob(os.path.join(self.augmented_dir, "hypertrace", '**', '*estimated-reflectance'), recursive=True)
        uncertainty_files = []
        for reflectance_file in estimated_reflectances:
            uncertainty_file = os.path.join(os.path.dirname(reflectance_file), 'posterior-uncertainty')
            uncertainty_files.append(uncertainty_file)

        p_map(partial(create_uncertainty, wvls=self.wvls), uncertainty_files, **{"desc": "\t\t saving new uncertainty files...", "ncols": 150})

        for reflectance_file in estimated_reflectances:
            basename = hypertrace_meta(reflectance_file)
            new_reflectance_file = os.path.join(self.augmented_dir, basename)
            shutil.copyfile(reflectance_file, new_reflectance_file)
            shutil.copyfile(reflectance_file + '.hdr', new_reflectance_file + '.hdr')

            uncertainty_file = os.path.join(os.path.dirname(reflectance_file), 'reflectance_uncertainty')
            new_uncertainty_file = os.path.join(self.augmented_dir, basename + '_uncer')
            shutil.copyfile(uncertainty_file, new_uncertainty_file)
            shutil.copyfile(uncertainty_file + '.hdr' , new_uncertainty_file + '.hdr')

            call_hypertrace_unmix(mode='sma-best', dry_run=False, reflectance_file=new_reflectance_file, em_file=em_file,
                                  parameters=optimal_parameters, output_dest=self.augmented_dir, scale='1',
                                  spectra_starting_column='8', uncertainty_file=new_uncertainty_file)

    def reconstruct_soil_simulation(self):
        cursor_print('reconstructing soil from simulation...')

        # reconstructed soil from simulated reflectance
        simulation_fractions_array = envi_to_array(os.path.join(self.augmented_dir, 'tetracorder_fractions'))
        simulation_index_array = envi_to_array(os.path.join(self.augmented_dir, 'tetracorder_index'))
        simulation_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                                         'convex_hull__n_dims_4_simulation_library'))

        spectra_grid = np.zeros((simulation_fractions_array.shape[0], simulation_fractions_array.shape[1], len(self.wvls)))

        for _row, row in enumerate(simulation_fractions_array):
            for _col, col in enumerate(row):
                picked_soil = int(simulation_index_array[_row, 0, 2])
                soil_spectra = simulation_library_array[picked_soil, 0, :]
                spectra_grid[_row, _col, :] = soil_spectra * simulation_fractions_array[_row, _col, 2]

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.augmented_dir, "simulated-soil.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)
        print("\t- done")

    def reconstruct_soil_sma(self):
        cursor_print('reconstructing soil from sma...')
        # reconstructed soil from fractions and unmix library
        complete_fractions_array = envi_to_array(os.path.join(self.augmented_dir, 'sma-best', 'tetracorder_spectra_complete_fractions'))

        df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv'))
        min_soil_index = np.min(df_unmix[df_unmix['level_1'] == 'soil'].index)

        unmix_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'endmember_libraries',
                                                   'convex_hull__n_dims_4_unmix_library'))

        spectra_grid = np.zeros((complete_fractions_array.shape[0], 1, len(self.wvls)))

        for _row, row in enumerate(complete_fractions_array):
            sma_soils = np.where(complete_fractions_array[_row, 0, : -1] != 0)[0]
            sma_soils = [x for x in sma_soils if x >= min_soil_index]

            if sma_soils:
                soils_grid = np.zeros((len(sma_soils), len(self.wvls)))

                for _soil, soil in enumerate(sma_soils):
                    soil_spectra = unmix_library_array[soil, 0, :] * complete_fractions_array[_row, 0, soil]
                    soils_grid[_soil, :] = soil_spectra

                spectra_grid[_row, :, :] = soils_grid.sum(axis=0)

            else:
                spectra_grid[_row, :, :] = -9999.

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.tetra_data_directory, "unmixing-soil.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)
        print("\t- done")

    def augment_slpit_pixels(self):
        cursor_print('augmenting slpit pixels...')
        transect_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'transect', '*[!.csv][!.hdr][!.aux][!.xml]'))
        em_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'endmembers', '*[!.csv][!.hdr][!.aux][!.xml]'))

        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
        print(df)
        df = df.sort_values('Name')

        for index, row in df.iterrows():
            plot = row['Name']
            emit_filetime = row['EMIT Date']
            reflectance_img_emit = glob(os.path.join(self.slpit_gis_directory, 'emit-data-clip',
                                                     f'*{plot.replace(" ", "")}_RFL_{emit_filetime}'))

            basename = os.path.basename(reflectance_img_emit[0])
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_pixels_augmented.hdr")
            augment_envi(file=reflectance_img_emit[0], wvls=self.wvls, out_raster=output_raster)

        for i in transect_files:
            basename = os.path.basename(i)
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_transect_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster)

        for i in em_files:
            basename = os.path.basename(i)
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_ems_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster)

        cursor_print("\t- done")

    def augment_simulation(self):
        cursor_print('augmenting data for tetracorder...')
        print()
        cursor_print('\t loading simulation data...')

        # load simulation library - 4 dimension; convex hull
        simulation_lib = os.path.join(self.simulation_output_directory, 'simulation_libraries', 'convex_hull__n_dims_4_simulation_library')

        # load unmix library - 4 dimensions; convex hull
        unmix_lib = os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library')

        files_to_augment = [simulation_lib, unmix_lib]

        for i in files_to_augment:
            basename = os.path.basename(i)
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_simulation_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster)

        cursor_print("\t- done")


def run_tetracorder_build(base_directory, sensor):
    tc = tetracorder(base_directory=base_directory, sensor=sensor)
    #tc.generate_tetracorder_reflectance()
    #tc.unmix_tetracorder()
    #tc.reconstruct_soil_simulation()
    #tc.reconstruct_soil_sma()
    tc.augment_slpit_pixels()
    tc.augment_simulation()

