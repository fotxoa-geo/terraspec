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
    """
    row shape: (num_cols, num_endmembers)
    unmix_library_array shape: (num_endmembers, num_bands)
    wvls shape: (num_bands,)
    """
    # 1. Calculate sum of fraction weights for each column/pixel: shape (num_cols,)
    weight_sums = np.sum(row, axis=1)
    
    # 2. Compute weighted linear combination via matrix multiplication: shape (num_cols, num_bands)
    #    (num_cols x num_endmembers) @ (num_endmembers x num_bands) -> (num_cols x num_bands)
    weighted_spectra = row @ unmix_library_array
    
    # 3. Create output grid initialized to fill value (-9999.0)
    spectra_grid = np.full((row.shape[0], len(wvls)), -9999.0)
    
    # 4. Identify pixels with valid non-zero weight sums
    valid_mask = weight_sums > 0
    
    # 5. Normalize weighted sum by total weight for valid pixels
    spectra_grid[valid_mask] = weighted_spectra[valid_mask] / weight_sums[valid_mask, np.newaxis]
    
    return spectra_grid


def main():
    parser = argparse.ArgumentParser(description='Run Vegetation Extraction')

    parser.add_argument('-out_dir', '--output_directory', type=str, help='Out directory')
    parser.add_argument('-sns', '--sensor', type=str, help='specify sensor to use')
    parser.add_argument('-veg_fracs', '--vegetation_complete_fractions', type=str,
                        help='Vegetation complete fractions file from SpectralUnmixing.jl', default=None) # Fixed default
    parser.add_argument('-rfl', '--reflectance_image', type=str, help='Reflectance image that was unmixed')
    parser.add_argument('-unmix_lib_csv', '--unmixing_library_csv', type=str, help='Unmixing library csv file')
    parser.add_argument('-unmix_lib_envi', '--unmixing_library_envi', type=str, help='Unmixing library envi file')
    parser.add_argument('-three_comp_frac', '--three_component_fractions', type=str, help='Three component fractions file') # Renamed flag
    parser.add_argument('--tetracorder', action='store_true', help='Run tetracorder after extraction of veg signal')
    args = parser.parse_args()

    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)
    basename = os.path.basename(args.reflectance_image)
    unmix_basename = os.path.basename(args.unmixing_library_envi)

    complete_fractions_array = envi_to_array(args.vegetation_complete_fractions)
    unmix_library_array = envi_to_array(args.unmixing_library_envi)
    rho = envi_to_array(args.reflectance_image)
    f_hat = envi_to_array(args.three_component_fractions)

    df_unmix = pd.read_csv(args.unmixing_library_csv)

    # Initialize rho_hat_vf to starting reflectance array
    rho_hat_vf = rho.copy()

    for _em, em in enumerate(df_unmix.level_1.unique()):
        min_em_index = np.min(df_unmix[df_unmix['level_1'] == em].index)
        max_em_index = np.max(df_unmix[df_unmix['level_1'] == em].index)

        em_library_array = unmix_library_array[min_em_index:max_em_index + 1, 0, :]
        em_fractions_array = complete_fractions_array[:, :, min_em_index:max_em_index + 1]

        rho_em_spectra_grid = np.zeros((complete_fractions_array.shape[0], complete_fractions_array.shape[1], len(wvls)))

        func = partial(process_complete_fractions_row, unmix_library_array=em_library_array, wvls=wvls)
        results = p_map(func, [em_fractions_array[_row, :, :] for _row in range(em_fractions_array.shape[0])],
                        **{"desc": f"\t\t rebuilding spectra ...", "ncols": 150})

        for _row, row in enumerate(results):
            rho_em_spectra_grid[_row, :, :] = row

        # Fixed variable reference: spectra_grid -> rho_em_spectra_grid
        meta_spectra = get_meta(lines=rho_em_spectra_grid.shape[0], samples=rho_em_spectra_grid.shape[1], bands=wvls, wvls=True)
        output_raster = os.path.join(args.output_directory, f"extracted_{basename}_{em}_{unmix_basename}_signal.hdr")
        save_envi(output_raster, meta_spectra, rho_em_spectra_grid)
        print(f'\t successfully saved: {output_raster}')

        if args.tetracorder and em in ['npv', 'pv']:
            rho_hat_vf -= (rho_em_spectra_grid * f_hat[:, :, _em][:, :, np.newaxis])

    # Normalize reflectance by fraction of soil (index 2) to retrieve rho_s
    rho_hat_vfs = rho_hat_vf / f_hat[:, :, 2][:, :, np.newaxis]

    meta_spectra = get_meta(lines=rho.shape[0], samples=rho.shape[1], bands=wvls, wvls=True)
    output_raster = os.path.join(args.output_directory, f"recon_rho_{basename}_{unmix_basename}.hdr")
    save_envi(output_raster, meta_spectra, rho_hat_vf)
    print(f'\t successfully saved: {output_raster}')

    if args.tetracorder:
        meta_spectra = get_meta(lines=rho.shape[0], samples=rho.shape[1], bands=wvls, wvls=True)
        output_raster = os.path.join(args.output_directory, f"ext_veg_{basename}_{unmix_basename}_tc.hdr")
        save_envi(output_raster, meta_spectra, rho_hat_vfs)
        print(f'\t successfully saved: {output_raster}')

if __name__ == '__main__':
    main()
