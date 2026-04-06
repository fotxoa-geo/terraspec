import argparse
import os
import time

import pandas as pd
import numpy as np
from p_tqdm import p_map
from functools import partial
from utils.envi import save_envi, get_meta, envi_to_array
from utils.spectra_utils import spectra


def process_complete_fractions_row(row, unmix_library_array, wvls):

    spectra_grid = np.ones((row.shape[0], len(wvls))) * -9999.

    for _col, col in enumerate(row):
        em_col = np.zeros((unmix_library_array.shape[0], len(wvls)))
        frac_weights = np.zeros((unmix_library_array.shape[0]))

        for _em, em in enumerate(unmix_library_array):
            fraction = row[_col, _em]
            em_col[_em, :] = unmix_library_array[_em, :]
            frac_weights[_em] = fraction

        if np.sum(frac_weights) == 0:
            continue
        else:
            spectra_grid[_col, :] = np.average(em_col, weights=frac_weights, axis=0)

    return spectra_grid


def main():
    parser = argparse.ArgumentParser(description='Run Vegetation Extraction')

    parser.add_argument('-out_dir', '--output_directory', type=str, help='Out directory')
    parser.add_argument('-sns', '--sensor', type=str, help='specify sensor to use')
    parser.add_argument('-veg_fracs', '--vegetation_complete_fractions', type=str, help='Vegetation complete fractions file from SpectralUnmixing.jl',
                        default=True)
    parser.add_argument('-rfl', '--reflectance_image', type=str, help='Reflectance image that was unmixed')
    parser.add_argument('-unmix_lib_csv', '--unmixing_library_csv', type=str, help='Unmixing library csv file')
    parser.add_argument('-unmix_lib_envi', '--unmixing_library_envi', type=str, help='Unmixing library envi file')
    parser.add_argument('-3_comp_frac', '--three_component_fractions', type=str, help='Unmixing library envi file')
    parser.add_argument('--tetracorder', action='store_true', help='Run tetracorder after extraction of veg signal')
    args = parser.parse_args()

    df_ems = pd.read_csv(args.unmixing_library_csv)
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)
    basename = os.path.basename(args.reflectance_image)

    complete_fractions_array = envi_to_array(args.vegetation_complete_fractions)
    unmix_library_array = envi_to_array(args.unmixing_library_envi)
    rfl_mixed_array = envi_to_array(args.reflectance_image)
    three_component_fractions_array = envi_to_array(args.three_component_fractions)

    df_unmix = pd.read_csv(args.unmixing_library_csv)

    for _em, em in enumerate(df_unmix.level_1.unique()):
        min_em_index = np.min(df_unmix[df_unmix['level_1'] == em].index)
        max_em_index = np.max(df_unmix[df_unmix['level_1'] == em].index)

        em_library_array = unmix_library_array[min_em_index:max_em_index + 1, 0, :]
        em_fractions_array = complete_fractions_array[:, :, min_em_index:max_em_index + 1]

        spectra_grid = np.zeros((complete_fractions_array.shape[0], complete_fractions_array.shape[1], len(wvls)))

        func = partial(process_complete_fractions_row, unmix_library_array=em_library_array, wvls=wvls)
        results = p_map(func,
                        [em_fractions_array[_row, :, :] for _row in range(em_fractions_array.shape[0])],
                        **{"desc": f"\t\t rebuilding spectra ...", "ncols": 150})

        for _row, row in enumerate(results):
            spectra_grid[_row, :, :] = row

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls, wvls=True)
        output_raster = os.path.join(args.output_directory, f"extracted_{basename}_{em}_signal.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)
        print(f'\t successfully saved: {output_raster}')

        if args.tetracorder and em in ['npv', 'pv']:
            print('Extracting vegetation signal from rfl img for Tetracorder run...')
            rfl_mixed_array -= spectra_grid * three_component_fractions_array[:,:, _em][:, :, np.newaxis]

    if args.tetracorder:
        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls, wvls=True)
        output_raster = os.path.join(args.output_directory, f"extracted_vegetation_{basename}.hdr")
        save_envi(output_raster, meta_spectra, rfl_mixed_array)

if __name__ == '__main__':
    main()
