import argparse
import os
import platform
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from spectral.io import envi
import tetracorder.tetracorder_engine as tc

# --- Globals for Worker Processes ---
worker_matrix = None
worker_expert = None

bad_wv_regions = [[.350, .400], [1.310, 1.490], [1.770, 2.050], [2.450, 2.500]]


def get_good_bands_mask(wavelengths, wavelength_pairs=None):
    wavelengths = np.array(wavelengths)
    if wavelength_pairs is None:
        wavelength_pairs = bad_wv_regions  # Assuming this holds bad regions

    # Start with ALL bands marked as True (Good)
    good_bands = np.ones(len(wavelengths), dtype=bool)

    for wvp in wavelength_pairs:
        # Find index closest to the lower bound of the BAD region
        lower_index = np.nanargmin(np.abs(wavelengths - wvp[0]))
        # Find index closest to the upper bound of the BAD region
        upper_index = np.nanargmin(np.abs(wavelengths - wvp[1]))

        # Turn off ONLY this bad window
        good_bands[lower_index:upper_index + 1] = False

    return good_bands


def init_worker(matrix_file, expert_file):
    """
    Runs ONCE when each worker process starts. Loads the heavy configuration
    data into the worker's local global memory space.
    """
    global worker_matrix, worker_expert
    worker_matrix = pd.read_csv(matrix_file).fillna(-9999)
    worker_expert = tc.decode_expert_system(expert_file, log_file=None, log_level='INFO')


def process_line(args_tuple, spectral_reference_library, output_file_path, num_out_bands):
    """
    Worker task. Receives a single row from each of the input arrays.
    """
    global worker_matrix, worker_expert

    # Unpack the lines zipped together in the pool map
    line_idx, line, reflectance_line, fraction_line, rho_gv_line, rho_npv_line = args_tuple

    # OPTIMIZATION: Use NumPy to find indices of non-background pixels instantly.
    # This completely bypasses slow Python iterations over '0' pixels.
    active_cols = np.flatnonzero(line != 0)

    output_row_data = np.ones((line.shape[0], num_out_bands), dtype=np.float32) * -9999

    for _col in active_cols:
        col_val = int(line[_col])

        pixel_result = cont_removal(
            df_mineral_matrix=worker_matrix,
            spectral_reference_library=spectral_reference_library,
            tetracorder_expert_system=worker_expert,
            mineral_index=col_val,
            reflectance=reflectance_line[_col, :],
            fractions=fraction_line[_col, :],
            rho_gv=rho_gv_line[_col, :],
            rho_npv=rho_npv_line[_col, :]
        )

        output_row_data[_col, :] = pixel_result

    out_img = envi.open(output_file_path)
    out_memmap = out_img.open_memmap(writable=True)

    # Write directly to our unique row slice (no locks required!)
    out_memmap[line_idx, :, :] = output_row_data

    # Flush changes to disk
    del out_memmap

def cont_removal(df_mineral_matrix, spectral_reference_library, tetracorder_expert_system, mineral_index, reflectance,
                 fractions, rho_gv, rho_npv):

    record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral_index), 'Record'].iloc[0]
    ref_library = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Library'].iloc[0]
    filename = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Filename'].iloc[0]

    # # row index pertains specifically to df; not value from Tetracorder!
    row_index = df_mineral_matrix[df_mineral_matrix['Record'] == record].index[0]
    mineral_row = df_mineral_matrix.iloc[row_index, 7:]
    mineral_row = mineral_row.apply(pd.to_numeric, errors='coerce')

    # # load library
    item = spectral_reference_library[ref_library]

    library = envi.open(f'{item}.hdr')
    library_reflectance = library.spectra.copy()
    library_records = [int(q) for q in library.metadata['record']]

    library_reflectance = library_reflectance[library_records.index(record), :]

    hdr = envi.read_envi_header(f'{item}.hdr')
    wavelengths = np.array([float(q) for q in hdr['wavelength']])

    normalized_group_name = os.path.normpath(filename.split('.depth.gz')[0])
    expert_file_selection = tetracorder_expert_system[normalized_group_name]['features']

    # # thresholds from expert file system
    ct_thresholds = {'CTHRESH1': 0.01, 'CTHRESH2': 0.02, 'CTHRESH4': 0.04, 'CTHRESH5': 0.05}

    # # this holds the multiple values of bd if multiple features are passed by the expert file
    integrals_array = np.ones((len(expert_file_selection))) * -9999
    bd_array = np.ones((len(expert_file_selection))) * -9999
    bd_prime_array = np.ones((len(expert_file_selection))) * -9999
    bd_library_array = np.ones((len(expert_file_selection))) * -9999

    reflectance[reflectance == -9999] = np.nan
    reflectance[reflectance == -0.1] = np.nan

    good_sensor_bands = get_good_bands_mask(wavelengths, wavelength_pairs=bad_wv_regions)
    reflectance[~good_sensor_bands] = np.nan
    valid_wavelengths = ~np.isnan(reflectance)

    soil_fraction = fractions[2]
    npv_fraction = fractions[0]
    gv_fraction = fractions[1]

    # loop through features
    for _cont_feat, cont_feat in enumerate(expert_file_selection):

        feature = cont_feat['continuum']

        if soil_fraction == 0:
            continue

        left_inds = np.where(np.logical_and.reduce((wavelengths >= feature[0], wavelengths <= feature[1], valid_wavelengths)))[0]
        right_inds = np.where(np.logical_and.reduce((wavelengths >= feature[2], wavelengths <= feature[3], valid_wavelengths)))[-1]

        if len(left_inds) == 0 or len(right_inds) == 0: # if not present in valid wvls, we are skipping
            continue

        feature_inds = np.logical_and(wavelengths >= wavelengths[left_inds][0],
                                      wavelengths <= wavelengths[right_inds][-1])

        # x boundaries - used for all calculations
        x1, x2 = wavelengths[feature_inds][0], wavelengths[feature_inds][-1]  # λi, λj

        # calculate continuum for tetracorder library
        m_l = (library_reflectance[feature_inds][-1] - library_reflectance[feature_inds][0]) / (x2 - x1)
        b_l = library_reflectance[feature_inds][0] - m_l * x1
        lc = m_l * wavelengths + b_l
        bd_l = 1 - np.array(library_reflectance[feature_inds] / lc[feature_inds])
        bd_max_l = bd_l.argmax()
        bd_library_array[_cont_feat] = bd_l[bd_max_l]

        # calculate integral of tetracorder library reference
        h_x = lc[feature_inds] / lc[feature_inds] - lc[feature_inds]
        integral = np.trapz(h_x, wavelengths[feature_inds])
        integrals_array[_cont_feat] = integral

        # calculate vegetation correction
        Ci = x1 / (x2 - x1)
        Ri = reflectance[feature_inds][0]
        Rj = reflectance[feature_inds][-1]

        fnpv = npv_fraction
        fgv = gv_fraction
        pnpv_j = rho_npv[feature_inds][-1]
        pnpv_i = rho_npv[feature_inds][0]
        pgv_j = rho_gv[feature_inds][-1]
        pgv_i = rho_gv[feature_inds][0]

        # this is our soil spectrum
        psoil = (1 / soil_fraction) * (reflectance - (fnpv * rho_npv + fgv * rho_gv))

        # these are the continuum for Bd
        m = (Rj - Ri) / (x2 - x1)
        b = Ri - m * x1
        rc = m * wavelengths + b
        bd = 1 - np.array(reflectance[feature_inds] / rc[feature_inds])
        bd_max = np.nanargmax(bd)
        bd_array[_cont_feat] = bd[bd_max]

        # these are the continuum for Bd'
        b_prime = (1 / soil_fraction) * ((1 + Ci) * (Ri - fnpv * pnpv_i - fgv * pgv_i) - Ci * (Rj - fnpv * pnpv_j - fgv * pgv_j))
        m_prime = (Ri - fnpv * pnpv_i - fgv * pgv_i) / (x1 * soil_fraction) - (b_prime / x1)
        rc_prime = m_prime * wavelengths + b_prime
        bd_ρsoil = 1 - np.array(psoil[feature_inds] / rc_prime[feature_inds])
        bd_max_ρsoil = np.nanargmax(bd_ρsoil)
        bd_prime_array[_cont_feat] = bd_ρsoil[bd_max_ρsoil]

    # correct data for -9999.
    integrals_array[integrals_array == -9999] = np.nan
    bd_return_array = np.ones(4) * -9999

    # calculate weighted band depths
    for _i, i in enumerate([bd_library_array, bd_array, bd_prime_array]):
        i[i == -9999] = np.nan
        relative_area = integrals_array / np.nansum(integrals_array)
        band_depth_w = np.nansum(relative_area * i)

        if band_depth_w <= 1.:
            bd_return_array[_i] = band_depth_w
        else:
            pass

    return bd_return_array

def main():
    parser = argparse.ArgumentParser(description='Run RECLAIMER (post unmixing mode)')
    parser.add_argument('-out_dir', '--output_directory', type=str, help='Output directory')
    parser.add_argument('-tc_out', '--tetracorder_output_file', type=str, help='Tetracorder output file')
    parser.add_argument('-rfl', '--tetracorder_reflectance_file', type=str, help='Reflectance file')
    parser.add_argument('-um_out', '--unmixing_fraction_out_file', type=str, help='Unmixing fractions')
    parser.add_argument('-expert_file', '--expert_system_file', type=str, help='Tetracorder expert file', default=os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'))
    parser.add_argument('-mineral_csv', '--mineral_matrix_file', type=str, help='Mineral matrix file', default=os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
    parser.add_argument('-g_num', '--group_number', type=str, help='group number for Tetracorder expert system')
    parser.add_argument('-rho_gv', '--rho_endmember_gv', type=str, help='Rho of GV (reconstructed from spectral unmixing complete fraction file)')
    parser.add_argument('-rho_npv', '--rho_endmember_npv', type=str,help='Rho of NPV (reconstructed from spectral unmixing complete fraction file)')
    args = parser.parse_args()

    SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                  'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

    # --------- Load files --------------
    group_dict = {'g1': 1, 'g2': 3}
    img_tetracorder = envi.open(args.tetracorder_output_file)
    tetracorder_array = np.array(img_tetracorder.load()[:, :, group_dict[f'g{int(args.group_number)}']])

    refl_tetracorder = envi.open(args.tetracorder_reflectance_file)
    tetracorder_rfl_array = np.array(refl_tetracorder.load()[:, :, :])

    fractions_tetracorder = envi.open(args.unmixing_fraction_out_file)
    tetracorder_fraction_array = np.array(fractions_tetracorder.load()[:, :, :])

    rho_gv = envi.open(args.rho_endmember_gv)
    rho_gv_array = np.array(rho_gv.load()[:, :, :])

    rho_npv = envi.open(args.rho_endmember_npv)
    rho_npv_array = np.array(rho_npv.load()[:, :, :])

    # ---------- Create Output files to write directly to ---------
    num_lines = tetracorder_array.shape[0]
    num_cols = tetracorder_array.shape[1]
    num_out_bands = 4  # Change depending on how many spectral bands your output has

    output_hdr_path = os.path.join(args.output_directory, f'RECLAIMER_g{int(args.group_number)}_{os.path.basename(args.unmixing_fraction_out_file)}')

    # Inherit and adjust metadata from your input image
    out_metadata = img_tetracorder.metadata.copy()
    out_metadata['bands'] = num_out_bands
    out_metadata['lines'] = num_lines
    out_metadata['samples'] = num_cols
    out_metadata['data type'] = 4  # 4 = 32-bit floating point

    # Create the empty output file structure on disk (forces creation of both .hdr and image binary)
    if os.path.exists(output_hdr_path):
        os.remove(output_hdr_path)

    # This creates the physical blank file ready for raw writing
    envi.create_image(output_hdr_path, out_metadata, force=True, ext='')

    # ---------- Parallel processing ---------
    line_indices = np.arange(num_lines)

    zipped_data = list(zip(
        line_indices,
        tetracorder_array,
        tetracorder_rfl_array,
        tetracorder_fraction_array,
        rho_gv_array,
        rho_npv_array
    ))

    # Freeze constant argument (spectral library reference dictionary is lightweight and safe to partial)
    worker_task = partial(process_line, spectral_reference_library=SPECTRAL_REFERENCE_LIBRARY,
                          output_file_path=output_hdr_path, num_out_bands=num_out_bands)

    print(f"Processing file: {os.path.basename(args.tetracorder_output_file)}...")
    start_time = time.time()

    # Launch multiprocessing pool with process initializer
    with Pool(initializer=init_worker, initargs=(args.mineral_matrix_file, args.expert_system_file)) as pool:
        pool.map(worker_task, zipped_data)

    print(f"Finished processing in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    if platform.system() == "Darwin":
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    main()
