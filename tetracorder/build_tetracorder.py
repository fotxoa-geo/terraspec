import os
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
from simulation.run_unmix import call_unmix


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

    def reconstruct_soil_simulation(self):
        cursor_print('reconstructing soil from simulation...')

        # reconstructed soil from simulated reflectance
        simulation_fractions_array = envi_to_array(os.path.join(self.simulation_output_directory, 'convex_hull__n_dims_4_fractions'))
        simulation_index_array = envi_to_array(os.path.join(self.simulation_output_directory, 'convex_hull__n_dims_4_index'))
        simulation_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                                         'convex_hull__n_dims_4_simulation_library'))

        spectra_grid = np.zeros((simulation_fractions_array.shape[0], 1, len(self.wvls)))

        for _row, row in enumerate(simulation_fractions_array):
            picked_soil = int(simulation_index_array[_row, :, 2])
            spectra_grid[_row, :, :] = simulation_library_array[picked_soil, :, :] * simulation_fractions_array[_row, :, 2]

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.tetra_data_directory, "simulated-soil.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)
        print("\t- done")

    def reconstruct_soil_sma(self):
        cursor_print('reconstructing soil from sma...')
        # reconstructed soil from fractions and unmix library
        complete_fractions_array = envi_to_array(os.path.join(self.simulation_output_directory, 'sma-best', 'convex_hull_n_dims_4_spectra_sma-best_normalization_brightness_num_endmembers_30_n_mc_25_complete_fractions'))

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

    def build_increment_instances(self, increment_size, mineral_index):
        sa_sim_library = os.path.join(self.base_directory, 'tetracorder', 'output', 'spectral_abundance', 'convex_hull__n_dims_4_simulation_library_sa_mineral')
        soil_sa_sim_pure = envi_to_array(sa_sim_library)[:, 0, :]
        minerals = load_band_names(sa_sim_library)

        df_sim = pd.read_csv(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                            'convex_hull__n_dims_4_simulation_library.csv'))

        df_not_soil = df_sim[df_sim['level_1'] != 'soil']
        df_soil = df_sim[df_sim['level_1'] == 'soil']
        min_soil_index = np.min(df_sim[df_sim['level_1'] == 'soil'].index)

        positive_detection_index = (soil_sa_sim_pure[min_soil_index:, mineral_index] != 0) & (soil_sa_sim_pure[min_soil_index:, mineral_index] == np.max(soil_sa_sim_pure[min_soil_index:, mineral_index]))

        df_positive_soil = df_soil[positive_detection_index].reset_index(drop=True)

        # merge dataframes
        df_merge = pd.concat([df_not_soil, df_positive_soil], ignore_index=True)
        spectra.df_to_envi(df=df_merge, spectral_starting_column=7, wvls=self.wvls,
                           output_raster=os.path.join(self.tetra_output_directory, minerals[mineral_index] + '_increment_sim_library.hdr'))

        df_merge.insert(0, 'index', df_merge.index)
        class_lists = []

        for em in sorted(list(df_merge.level_1.unique())):
            df_select = df_merge.loc[df_merge['level_1'] == em].copy()
            df_select = df_select.values.tolist()
            class_lists.append(df_select)

        all_combinations = list(itertools.product(*class_lists))
        np.random.seed(13)
        picked_spectral_bundles_index = np.random.choice(len(all_combinations), replace=False, size=1000)
        picked_spectra = [all_combinations[i] for i in picked_spectral_bundles_index]

        # create grids
        cols = int(1/increment_size) + 1
        fraction_grid = np.zeros((len(picked_spectral_bundles_index), cols, len(sorted(list(df_merge.level_1.unique())))))
        spectra_grid = np.zeros((len(picked_spectral_bundles_index), 100, len(self.wvls)))
        index_grid = np.zeros((len(picked_spectral_bundles_index), cols, len(sorted(list(df_merge.level_1.unique())))))

        # spectra array - row = combinations; cols=  # of classes (each col is an em) ; by  wavelengths
        spec_array = np.array(picked_spectra)

        for _col, col in enumerate(range(0, cols)):
            col_soil_frac = np.round(col * increment_size, 2)
            col_fractions, col_index, col_spectra = spectra.increment_synthetic_reflectance(data=spec_array, wvls=self.wvls,
                                                                                            em_fraction=col_soil_frac,
                                                                                            seed=_col, spectra_start=8)

            fraction_grid[:, _col, :] = col_fractions
            spectra_grid[:, _col, :] = col_spectra
            index_grid[:, _col, :] = col_index

        # save arrays
        refl_meta = get_meta(lines=len(picked_spectral_bundles_index), samples=100, bands=self.wvls, wvls=True)
        index_meta = get_meta(lines=len(picked_spectral_bundles_index), samples=cols, bands=sorted(list(df_merge.level_1.unique())), wvls=False)
        fraction_meta = get_meta(lines=len(picked_spectral_bundles_index), samples=cols, bands=sorted(list(df_merge.level_1.unique())), wvls=False)

        # save index, spectra, fraction grid
        output_files = [os.path.join(self.tetra_output_directory, minerals[mineral_index] + '_increment_' + str(increment_size)[-2] + '_index.hdr'),
                        os.path.join(self.tetra_output_directory, 'augmented', minerals[mineral_index] + '_increment_' + str(increment_size)[-2] + '_spectra.hdr'),
                        os.path.join(self.tetra_output_directory, minerals[mineral_index] + '_increment_' + str(increment_size)[-2] + '_fractions.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta]
        grids = [index_grid, spectra_grid, fraction_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})

    def reflectance_increment(self):

        df_sim = pd.read_csv(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                          'convex_hull__n_dims_4_simulation_library'))
        em_file = os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv')

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim,
                                      level='level_1', spectral_bundles=50000, increment_size=0.05,
                                      output_directory=self.simulation_output_directory, wvls=self.wvls,
                                      name='tetracorder_reflectance', spectra_starting_col=7)

        # unmix
        optimal_parameters = ['--num_endmembers 30', '--n_mc 25', '--normalization brightness']

        # reflectance_file = os.path.join(self.simulation_output_directory, 'tetracorder_reflectance')
        # call_unmix(mode='sma-best', dry_run=False, reflectance_file=reflectance_file, em_file=em_file,
        #            parameters=optimal_parameters, output_dest=self.tetra_output_directory, scale='1',
        #            spectra_starting_column='8')

    def augment_slpit_pixels(self):
        cursor_print('augmenting slpit pixels...')
        transect_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'transect', '*[!.csv][!.hdr][!.aux][!.xml]'))
        em_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'endmembers', '*[!.csv][!.hdr][!.aux][!.xml]'))

        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.shp")))
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

        # load simulated reflectance - 4 dimension; convex hull
        simulation_refl = os.path.join(self.simulation_output_directory, 'convex_hull__n_dims_4_spectra')

        # reconstructed soil from simulated reflectance
        simulation_soil = os.path.join(self.tetra_data_directory, "simulated-soil")

        # load unmix library - 4 dimensions; convex hull
        unmix_lib = os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library')

        # reconstructed soil from unmix library and fractions
        unmix_soil = os.path.join(self.tetra_data_directory, "unmixing-soil")

        files_to_augment = [simulation_lib, simulation_refl, simulation_soil, unmix_lib, unmix_soil]

        for i in files_to_augment:
            basename = os.path.basename(i)
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', basename + "_simulation_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster)

        cursor_print("\t- done")


def run_tetracorder_build(base_directory, sensor):
    tc = tetracorder(base_directory=base_directory, sensor=sensor)
    #tc.build_increment_instances(increment_size=0.05, mineral_index=0)
    #tc.reconstruct_soil_simulation()
    #tc.reconstruct_soil_sma()
    tc.augment_slpit_pixels()
    tc.augment_simulation()

