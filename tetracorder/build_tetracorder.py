import os
from utils.create_tree import create_directory
from utils.spectra_utils import spectra
from utils.envi import envi_to_array, get_meta, save_envi
from utils.text_guide import cursor_print
import numpy as np
import pandas as pd
from p_tqdm import p_map
from functools import partial
from glob import glob


def augment_envi(envi_file, wvls, directory):
    ds_array = envi_to_array(envi_file)

    # augment for tetracorder read
    spectra_grid = np.zeros((ds_array.shape[0], 100, len(wvls)))

    for _row, row in enumerate(ds_array):
        spectra_grid[_row, 0, :] = ds_array[_row, :, :]

    meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls,
                            wvls=True)

    output_raster = os.path.join(directory, os.path.basename(envi_file) + '.hdr')
    save_envi(output_raster, meta_spectra, spectra_grid)


class tetracorder:

    def __init__(self, base_directory: str, sensor:str):

        self.base_directory = base_directory
        self.tetra_data_directory = os.path.join(base_directory, 'tetracorder', 'data')
        self.tetra_output_directory = os.path.join(base_directory, 'tetracorder', 'output')
        self.simulation_output_directory = os.path.join(base_directory, 'simulation', 'output')
        self.slpit_output_directory = os.path.join(base_directory, 'slpit', 'output')

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

    def build_slpit_pixels(self):
        cursor_print('reconstructing slpit pixels...')
        emit_clip_files = glob(os.path.join(self.base_directory, 'slpit', 'gis', 'emit-data-clip', '*[!.hdr][!.aux][!.xml]'))

        # save emit clip files as envi files
        spectra_grid = np.zeros((len(emit_clip_files * 9), 1, len(self.wvls)))

        counter = 0
        meta_data = []
        for i in emit_clip_files:
            refl_array = envi_to_array(i)
            site = os.path.basename(i)

            for _row, row in enumerate(refl_array):
                for _col, col in enumerate(row):
                    spectra_grid[counter, 0, :] = refl_array[_row, _col, :]
                    counter += 1
                    meta_data.append([_row, _col, site])

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.slpit_output_directory, 'pixels' + '_' + self.sensor + ".hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)

        df_meta = pd.DataFrame(meta_data)
        df_meta.columns = ['row', 'column', 'site']
        df_meta.to_csv(os.path.join(self.slpit_output_directory, 'emit_pixel_meta.csv'), index=False)
        print("\t- done")

    def augment_reflectance(self):
        cursor_print('augmenting data for tetracorder...')
        print()
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

        # reconstructed soil atmosphere - the best case scenario ???

        # load reflectance atmosphere - the best case scenario ???

        # load endmember file - slipit
        slpit_em = os.path.join(self.slpit_output_directory, 'all-endmembers-emit')

        # load transect spectra
        slpit_transect = os.path.join(self.slpit_output_directory, 'all-transect-emit')

        # load emit pixels
        slpit_pixels = os.path.join(self.slpit_output_directory, 'pixels' + '_' + self.sensor)

        files_to_augment = [simulation_lib, simulation_refl, simulation_soil, unmix_lib, unmix_soil,
                            slpit_em, slpit_transect, slpit_pixels]

        p_map(partial(augment_envi, directory=os.path.join(self.tetra_output_directory, 'augmented'), wvls=self.wvls),
              files_to_augment, **{"desc": "\t\t augmenting simulation files: ", "ncols": 150})
        print("\t- done")


def run_tetracorder_build(base_directory, sensor):
    tc = tetracorder(base_directory=base_directory, sensor=sensor)
    tc.reconstruct_soil_simulation()
    tc.reconstruct_soil_sma()
    tc.build_slpit_pixels()
    tc.augment_reflectance()

