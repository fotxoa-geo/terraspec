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
    print("D... Reconstruct soil simulation and band depths")
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

        df_pv = df_sim.loc[df_sim['level_1'] == 'pv'].copy()
        df_pv = df_pv.sample(n=1, random_state=13).reset_index(drop=True)

        df_npv = df_sim.loc[df_sim['level_1'] == 'npv'].copy()
        df_npv = df_npv.sample(n=1, random_state=13).reset_index(drop=True)

        df_veg = pd.concat([df_npv, df_pv], axis=0, ignore_index=True)

        # this includes all values - we need two of these
        df_sim_array = envi_to_array(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                                  'convex_hull__n_dims_4_simulation_library'))

        # load spectral abundance of simulation library
        spectral_abundance_array = envi_to_array(os.path.join(self.tetra_output_directory, 'spectral_abundance',
                                                  'convex_hull__n_dims_4_simulation_library_augmented_min'))[:, 0, :]

        # these are the corresponding indices
        valid_rows_g1 = []
        indices_used_g1 = []
        valid_rows_g2 = []
        indices_used_g2 = []

        df_soil = df_sim.loc[df_sim['level_1'] == 'soil'].copy()

        for df_index, df_row in df_soil.iterrows():

            g1_index = spectral_abundance_array[df_index, 1]

            if g1_index not in [0, 1, 13, 15, 20, 21, 22, 25, 28, 29, 37, 38, 40, 41, 49, 56, 57, 60, 82, 83, 94]:
                valid_rows_g1.append(df_row)
                indices_used_g1.append(g1_index)

            g2_index = spectral_abundance_array[df_index, 3]

            if g2_index not in [0, 96, 97, 98, 99, 100, 105, 106, 135, 136, 182, 144, 148, 152, 194, 196, 228, 234, 238, 270, 271]:
                valid_rows_g2.append(df_row)
                indices_used_g2.append(g2_index)

        df_soil_g1 = pd.DataFrame(valid_rows_g1)
        df_sim_g1 = pd.concat([df_veg, df_soil_g1], axis=0, ignore_index=True)
        df_sim_g1 = df_sim_g1.sort_values('level_1')

        df_soil_g2 = pd.DataFrame(valid_rows_g2)
        df_sim_g2 = pd.concat([df_veg, df_soil_g2], axis=0, ignore_index=True)
        df_sim_g2 = df_sim_g2.sort_values('level_1')

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim_g1,
                                      level='level_1', spectral_bundles=10000, increment_size=0.05,
                                      output_directory=self.sim_spectra_dir, wvls=self.wvls,
                                      name='tetracorder_g1_simulation', spectra_starting_col=8, endmember='soil',
                                      simulation_library_array=df_sim_array, spectral_bundle_project='tetracorder_g1')

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim_g2,
                                      level='level_1', spectral_bundles=10000, increment_size=0.05,
                                      output_directory=self.sim_spectra_dir, wvls=self.wvls,
                                      name='tetracorder_g2_simulation', spectra_starting_col=8, endmember='soil',
                                      simulation_library_array=df_sim_array, spectral_bundle_project='tetracorder_g2')


    def hypertrace_tetracorder(self):
        cursor_print('hypertrace: tetracorder')
        hypertrace_workflow(dry_run=False, clean=False,
                            configfile=os.path.join('simulation', 'hypertrace', 'tetracorder.json'))


    def unmix_tetracorder(self, dry_run:bool):
        cursor_print('unmixing tetracorder')

        em_file = os.path.join(self.simulation_output_directory, 'endmember_libraries',
                               'convex_hull__n_dims_4_unmix_library.csv')

        optimal_parameters = ['--num_endmembers 30', '--n_mc 25', '--normalization none']

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

            call_hypertrace_unmix(mode='sma', dry_run=False, reflectance_file=new_reflectance_file, em_file=em_file,
                                  parameters=optimal_parameters, output_dest=self.augmented_dir, scale='1',
                                  spectra_starting_column='8', uncertainty_file=new_uncertainty_file)

    def reconstruct_veg_simulated_signal(self):
        cursor_print(f'reconstructing simulated vegetation...')

        fractions_array = envi_to_array(os.path.join(self.sim_spectra_dir, 'tetracorder_soil_fractions'))
        index_array = envi_to_array(os.path.join(self.sim_spectra_dir, 'tetracorder_soil_index'))

        simulated_array = envi_to_array(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                                         'convex_hull__n_dims_4_simulation_library'))

        spectra_grid = np.zeros((fractions_array.shape[0], fractions_array.shape[1], len(self.wvls)))

        for _row, row in enumerate(fractions_array):
            for _col, col in enumerate(row):

                gv_index = index_array[_row, _col, 1]
                npv_index = index_array[_row, _col, 0]

                gv_simulated_spectra = simulated_array[gv_index, 0, :] * fractions_array[_row, _col, 1]
                npv_simulated_spectra = simulated_array[npv_index, 0, :] * fractions_array[_row, _col, 0]

                spectra_grid[_row, _col, :] = gv_simulated_spectra + npv_simulated_spectra

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.sim_spectra_dir, f"vegetation_spectra_pure.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)

        print("\t- done")


    def reconstruct_em_sma(self, user_em):
        cursor_print(f'reconstructing {user_em} from sma...')

        for group in ['g1', 'g2']:
            # reconstructed soil from fractions and unmix library
            complete_fractions_array = envi_to_array(os.path.join(self.tetra_output_directory, 'fractions', 'sma', f'tetracorder_{group}_simulation_spectra_complete_fractions'))

            df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv'))

            min_em_index = np.min(df_unmix[df_unmix['level_1'] == 'soil'].index)

            unmix_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library'))

            spectra_grid = np.zeros((complete_fractions_array.shape[0], complete_fractions_array.shape[1], len(self.wvls)))

            for _row, row in enumerate(complete_fractions_array):
                for _col, col in enumerate(row):
                    sma_ems = np.where(complete_fractions_array[_row, _col, : -1] != 0)[0]
                    if user_em == 'soil':
                        sma_ems = [x for x in sma_ems if x >= min_em_index]
                    else:
                        sma_ems = [x for x in sma_ems if x < min_em_index]

                    if sma_ems:

                        em_grid = np.zeros((len(sma_ems), len(self.wvls)))

                        for _em, em in enumerate(sma_ems):
                            em_spectra = unmix_library_array[_em, 0, :] * complete_fractions_array[_row, _col, _em]
                            em_grid[_em, :] = em_spectra

                        spectra_grid[_row, _col, :] = em_grid.sum(axis=0)

                    else:
                        spectra_grid[_row, _col, :] = -9999.

            meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                    wvls=True)
            output_raster = os.path.join(self.sim_spectra_dir, f"unmixing-{group}-sma.hdr")
            save_envi(output_raster, meta_spectra, spectra_grid)

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
        simulation_lib = os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                      'convex_hull__n_dims_4_simulation_library')

        # load unmix library - 4 dimensions; convex hull
        unmix_lib = os.path.join(self.simulation_output_directory, 'endmember_libraries',
                                 'convex_hull__n_dims_4_unmix_library')

        # simulation spectra
        sim_spectra_files = glob(os.path.join(self.sim_spectra_dir, '*'))
        exclude = ['.hdr', '.xml', '.aux']

        files_to_augment = [simulation_lib, unmix_lib] + sim_spectra_files

        output_rasters = []
        output_files = []
        for i in files_to_augment:
            basename = os.path.basename(i)
            file_type = os.path.basename(i).split('_')[-1]

            if os.path.splitext(i)[1] not in exclude:
                if file_type in ['index', 'fractions']:
                    continue
                else:
                    output_raster = os.path.join(self.tetra_output_directory, 'augmented', f"{basename}_augmented.hdr")
                    output_rasters.append(output_raster)
                    output_files.append(i)

        p_map(partial(augment_envi, wvls=self.wvls), output_files, output_rasters,
              **{"desc": "\t\t augmenting envi files...", "ncols": 150})

        cursor_print("\t- done")

    def mineral_lib_refl_cont(self):

        for group in ['g1', 'g2']:
            # case 1 - simulated soil
            sim_fractions = envi_to_array(os.path.join(self.sim_spectra_dir,
                                                       f'tetracorder_{group}_simulation_fractions'))

            sim_soil_index = envi_to_array(os.path.join(self.base_directory, 'output', 'spectral_abundance',
                                                              f'tetracorder_{group}_simulation_soils_augmented_min'))[:, :21, :]

            sim_soil_spectra = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_soils'))
            output_file = os.path.join(self.veg_correction_dir, f'tetracorder_{group}_soil_only.hdr')
            spectra.mineral_components(index_array=sim_soil_index, spectra_array=sim_soil_spectra, output_file=output_file,
                                       fractions_array=sim_fractions, group=group)

            # case 2 - mixed simulated spectra - this is pure soil mixed with random GV + NPV
            sim_spectra_index = envi_to_array(os.path.join(self.base_directory, 'output', 'spectral_abundance',
                                                                 f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, :]

            sim_spectra_ = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_spectra'))
            vegetation_array = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_vegetation'))

            mix_output_file = os.path.join(self.veg_correction_dir, f'tetracorder_{group}_vegetation_correction.hdr')
            spectra.mineral_components(index_array=sim_spectra_index, spectra_array=sim_spectra_,
                                       output_file=mix_output_file, veg_correction=True, vegetation_array=vegetation_array,
                                       fractions_array=sim_fractions, group=group)

            # case 4 - mixed sim spectra - no correction
            mix_output_file = os.path.join(self.veg_correction_dir, f'tetracorder_{group}_no-vegetation_correction.hdr')
            spectra.mineral_components(index_array=sim_spectra_index, spectra_array=sim_spectra_,
                                       output_file=mix_output_file, fractions_array=sim_fractions, group=group)

            # case 3 - sma derived signal from mixed spectra
            sma_fractions = envi_to_array(os.path.join(self.tetra_output_directory, 'fractions', 'sma',
                                                       f'tetracorder_{group}_simulation_spectra_fractional_cover'))
            veg_rfl_sma = envi_to_array((os.path.join(self.sim_spectra_dir, f'unmixing-{group}-sma')))  # sma based veg rfl
            sma_output_file = os.path.join(self.veg_correction_dir, f'tetracorder_{group}_vegetation_correction_sma.hdr')
            spectra.mineral_components(index_array=sim_spectra_index, spectra_array=sim_spectra_,
                                       output_file=sma_output_file, veg_correction=True, vegetation_array=veg_rfl_sma,
                                       group=group, fractions_array=sma_fractions)

            # case 4 - removing reconstructed vegetation signal

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
            #tc.reconstruct_veg_simulated_signal()
            tc.reconstruct_em_sma(user_em='gv')
            tc.mineral_lib_refl_cont()
        elif user_input == 'E':
            tc.augment_slpit_pixels()

        elif user_input == 'F':
            print("Returning to Tetracorder main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
