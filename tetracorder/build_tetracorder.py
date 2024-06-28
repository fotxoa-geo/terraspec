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


def tetracorder_build_menu():
    msg = f"You have entered Tetracorder build mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to the Tetracorder build Mode....")
    print("A... Simulate Tetracorder reflectance")
    print("B... Hypertrace workflow")
    print("C... Unmix simulated reflectance")
    print("D... Reconstruct soil simulation")
    print("E... Augment pixels ")
    print("F... Exit")


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
        create_directory(os.path.join(self.tetra_output_directory, 'fractions'))
        create_directory(os.path.join(self.tetra_output_directory, 'simulated_spectra'))
        create_directory(os.path.join(self.tetra_output_directory, 'hypertrace'))
        create_directory(os.path.join(self.tetra_output_directory, 'veg-correction'))

        self.augmented_dir = os.path.join(os.path.join(self.tetra_output_directory, 'augmented'))
        self.fractions_dir = os.path.join(os.path.join(self.tetra_output_directory, 'fractions'))
        self.sim_spectra_dir = os.path.join(os.path.join(self.tetra_output_directory, 'simulated_spectra'))
        self.veg_correction_dir = os.path.join(self.tetra_output_directory, 'veg-correction')

        #create_directory(os.path.join(self.output_directory, 'outlogs'))
        #create_directory(os.path.join(self.output_directory, 'scratch'))

    def generate_tetracorder_reflectance(self):
        cursor_print('generating reflectance')

        df_sim = pd.read_csv(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                          'convex_hull__n_dims_4_simulation_library.csv'))

        df_sim_array = envi_to_array(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                                         'convex_hull__n_dims_4_simulation_library'))

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim,
                                      level='level_1', spectral_bundles=10000, increment_size=0.05,
                                      output_directory=self.sim_spectra_dir, wvls=self.wvls,
                                      name='tetracorder_soil', spectra_starting_col=8, endmember='soil', simulation_library_array=df_sim_array)

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim,
                                      level='level_1', spectral_bundles=10000, increment_size=0.05,
                                      output_directory=self.sim_spectra_dir, wvls=self.wvls,
                                      name='tetracorder_npv', spectra_starting_col=8, endmember='npv', simulation_library_array=df_sim_array)

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim,
                                      level='level_1', spectral_bundles=10000, increment_size=0.05,
                                      output_directory=self.sim_spectra_dir, wvls=self.wvls,
                                      name='tetracorder_pv', spectra_starting_col=8, endmember='pv', simulation_library_array=df_sim_array)

    def hypertrace_tetracorder(self):
        cursor_print('hypertrace: tetracorder')
        hypertrace_workflow(dry_run=False, clean=False,
                            configfile=os.path.join('simulation', 'hypertrace', 'tetracorder.json'))

    def unmix_tetracorder(self, dry_run:bool):
        cursor_print('unmixing tetracorder')

        em_file = os.path.join(self.simulation_output_directory, 'endmember_libraries',
                               'convex_hull__n_dims_4_unmix_library.csv')

        optimal_parameters = ['--num_endmembers 30', '--n_mc 25', '--normalization brightness']

        reflectance_files = glob(os.path.join(self.sim_spectra_dir, 'tetracorder_*_spectra*'))
        for i in reflectance_files:
            call_unmix(mode='sma', dry_run=dry_run, reflectance_file=i, em_file=em_file,
                       parameters=optimal_parameters, output_dest=self.fractions_dir, scale='1',
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
            augment_envi(file=new_reflectance_file, wvls=self.wvls, out_raster=new_reflectance_file + '.hdr')

            uncertainty_file = os.path.join(os.path.dirname(reflectance_file), 'reflectance_uncertainty')
            new_uncertainty_file = os.path.join(self.augmented_dir, basename + '_uncer')
            augment_envi(file=uncertainty_file, wvls=self.wvls, out_raster=new_uncertainty_file + '.hdr')

            call_hypertrace_unmix(mode='sma-best', dry_run=False, reflectance_file=new_reflectance_file, em_file=em_file,
                                  parameters=optimal_parameters, output_dest=self.augmented_dir, scale='1',
                                  spectra_starting_column='8', uncertainty_file=new_uncertainty_file)


    def reconstruct_soil_sma(self):
        cursor_print('reconstructing soil from sma...')

        # reconstructed soil from fractions and unmix library
        complete_fractions_array = envi_to_array(os.path.join(self.augmented_dir, 'sma-best', f'tetracorder_soil_spectra_complete_fractions'))

        df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv'))
        min_soil_index = np.min(df_unmix[df_unmix['level_1'] == 'soil'].index)

        unmix_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library'))

        spectra_grid = np.zeros((complete_fractions_array.shape[0], complete_fractions_array.shape[1], len(self.wvls)))

        for _row, row in enumerate(complete_fractions_array):
            for _col, col in enumerate(row):
                sma_soils = np.where(complete_fractions_array[_row, _col, : -1] != 0)[0]
                sma_soils = [x for x in sma_soils if x >= min_soil_index]

                if sma_soils:
                    soils_grid = np.zeros((len(sma_soils), len(self.wvls)))

                    for _soil, soil in enumerate(sma_soils):
                        soil_spectra = unmix_library_array[soil, 0, :] * complete_fractions_array[_row, _col, soil]
                        soils_grid[_soil, :] = soil_spectra

                    spectra_grid[_row, _col, :] = soils_grid.sum(axis=0)

                else:
                    spectra_grid[_row, _col, :] = -9999.

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.tetra_data_directory, "unmixing-soil.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)

        # augment the file
        aug_raster = os.path.join(self.augmented_dir, "sma-unmixing-soil.hdr")
        augment_envi(file=os.path.splitext(output_raster)[0], wvls=self.wvls, out_raster=aug_raster, vertical_average=False)

        print("\t- done")

    def augment_slpit_pixels(self):
        cursor_print('augmenting slpit pixels...')
        transect_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'transect', '*[!.csv][!.hdr][!.aux][!.xml]'))
        em_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'endmembers-raw', '*[!.csv][!.hdr][!.aux][!.xml]'))

        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
        df = df.sort_values('Name')

        for index, row in df.iterrows():
            plot = row['Name']
            emit_filetime = row['EMIT DATE']
            reflectance_img_emit = glob(os.path.join(self.slpit_gis_directory, 'emit-data-clip',
                                                     f'*{plot.replace(" ", "")}_RFL_{emit_filetime}'))

            basename = os.path.basename(reflectance_img_emit[0])
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_pixels_augmented.hdr")
            augment_envi(file=reflectance_img_emit[0],  vertical_average=True, wvls=self.wvls, out_raster=output_raster)

        for i in transect_files:
            basename = os.path.basename(i)
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_transect_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=True)

        for i in em_files:
            basename = os.path.basename(i)
            df_em = pd.read_csv(i + '.csv')
            soil_index = min(df_em.index[df_em['level_1'] == 'Soil'].tolist())
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_ems_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=True, em_index=soil_index)

        cursor_print("\t- done")

    def augment_simulation(self):
        cursor_print('augmenting data for tetracorder...')
        print()
        cursor_print('\t loading simulation data...')

        # load simulation library - 4 dimension; convex hull
        simulation_lib = os.path.join(self.simulation_output_directory, 'simulation_libraries', 'convex_hull__n_dims_4_simulation_library')

        # load unmix library - 4 dimensions; convex hull
        unmix_lib = os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library')

        # simulation spectra
        sim_soil_spectra = os.path.join(self.sim_spectra_dir, 'tetracorder_soil_spectra')
        sim_soil_em_spectra = os.path.join(self.sim_spectra_dir, 'tetracorder_soil_em_spectra')

        sim_npv_spectra = os.path.join(self.sim_spectra_dir, 'tetracorder_npv_spectra')
        sim_pv_spectra = os.path.join(self.sim_spectra_dir, 'tetracorder_pv_spectra')

        files_to_augment = [simulation_lib, unmix_lib, sim_soil_spectra, sim_soil_em_spectra]

        for i in files_to_augment:
            basename = os.path.basename(i)
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f"{basename}_simulation_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster)

        cursor_print("\t- done")

    def mineral_lib_refl_cont(self):
        # case 1 - em spectra - this is pure soil!
        em_tetracorder_index = envi_to_array(os.path.join(self.base_directory, 'output', 'spectral_abundance', 'tetracorder_soil_em_spectra_simulation_augmented_min'))
        em_spectra_array = envi_to_array(os.path.join(self.sim_spectra_dir, 'tetracorder_soil_em_spectra'))
        output_file = os.path.join(self.veg_correction_dir, 'tetracorder_soil_em_spectra_variables.hdr')
        spectra.mineral_components(index_array=em_tetracorder_index[:, :21, :], spectra_array=em_spectra_array, output_file=output_file)


def run_tetracorder_build(base_directory, sensor, dry_run):
    tc = tetracorder(base_directory=base_directory, sensor=sensor)
    while True:
        tetracorder_build_menu()

        user_input = input('\nPlease indicate the desired mode: ').upper()

        if user_input == 'A':
            tc.generate_tetracorder_reflectance()
            tc.augment_simulation()
        elif user_input == 'B':
            tc.hypertrace_tetracorder()
        elif user_input == 'C':
            tc.unmix_tetracorder(dry_run=dry_run)
        elif user_input == 'D':
            #tc.reconstruct_soil_sma()
            tc.mineral_lib_refl_cont()
        elif user_input == 'E':
            tc.augment_slpit_pixels()

        elif user_input == 'F':
            print("Returning to Tetracorder main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
