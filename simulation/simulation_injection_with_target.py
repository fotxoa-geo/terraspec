import numpy as np
import pandas as pd
import os
import argparse
from utils.spectra_utils import spectra
from utils.create_tree import create_directory
from functools import partial
from p_tqdm import p_map
from utils.envi import get_meta, save_envi, envi_to_array

bad_wv_regions = [[1310, 1490], [1770, 2050]]

def main():
    parser = argparse.ArgumentParser(description='Simulate spectra with target injection')

    parser.add_argument('-out_dir', '--output_directory', type=str, help='Out directory')
    parser.add_argument('-sns', '--sensor', type=str, help='specify sensor to use', default='emit')
    parser.add_argument('-em_lib_csv', '--endmember_library', type=str, help='Reflectance of endmembers to use')
    parser.add_argument('-target', '--target_spectra', type=str, help='Target spectra to inject in simulation')
    args = parser.parse_args()

    df_ems = pd.read_csv(args.endmember_library)
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)
    asd_wvls = spectra.load_asd_wavelenghts()

    create_directory(os.path.join(args.output_directory, f'target_{os.path.basename(args.target_spectra).split(".")[0]}'))
    out_dir = os.path.join(args.output_directory, f'target_{os.path.basename(args.target_spectra).split(".")[0]}')

    df_target = pd.read_csv(args.target_spectra)
    results = p_map(partial(spectra.convolve, asd_wvl=asd_wvls, wvl=wvls, fwhm=fwhm,
                                    spectra_starting_col=7), [row for row in df_target.iterrows()],
                            **{"desc": f"\t Wavelength Convolve {os.path.basename(args.target_spectra)}... ", "ncols": 150})
    df_convolve = pd.DataFrame(results)
    df_convolve.columns = list(wvls)
    df_convolve = pd.concat([df_target.iloc[:, :7].reset_index(drop=True), df_convolve], axis=1)
    df_convolve.to_csv(os.path.join(out_dir, f'target_spectra_{args.sensor}.csv'), index=False)

    spec_grid, frac_grid, idx_grid = spectra.increment_reflectance(class_names=sorted(list(df_ems.level_1.unique())),
                                                                        simulation_table=df_ems,
                                                                        level='level_1', spectral_bundles=1000,
                                                                        increment_size=0.02, output_directory=out_dir,
                                                                        wvls=wvls, name='target_injection_tarp_',
                                                                        spectra_starting_col=8, endmember='soil',
                                                                        spectral_bundle_project='target_injection',
                                                                        new_simulation_bundles=1000,
                                                                        target_injection=args.target_spectra)

    target_spectra = np.ones((1, spec_grid.shape[1], spec_grid.shape[2]))
    em_array = df_ems.iloc[:, 7:].to_numpy()
    target_array = df_convolve.iloc[:, 7:].to_numpy()
    target_array_spectrum = np.nanmean(target_array, axis=0)

    for _col, col in enumerate(range(0, target_spectra.shape[1])):
        increment_frac = np.round(col * 0.02, 2)
        np.random.seed(_col)

        target_frac = increment_frac
        remaining_fraction = 1 - target_frac
        pv_frac = np.random.uniform(0, remaining_fraction)
        remaining_fraction = remaining_fraction - pv_frac
        soil_frac = np.random.uniform(0, remaining_fraction)
        npv_frac = remaining_fraction - soil_frac

        target_spectra[0, _col, :] = (target_array_spectrum * target_frac) + \
                                     (em_array[idx_grid[49, _col, 0].astype(int), :].astype(float) * npv_frac) + \
                                     (em_array[idx_grid[49, _col, 1].astype(int), :].astype(float) * pv_frac) + \
                                     (em_array[idx_grid[49, _col, 2].astype(int), :].astype(float) * soil_frac)


    spec_grid[49, :, :] = target_spectra

    refl_meta = get_meta(lines=spec_grid.shape[0], samples=spec_grid.shape[1], bands=wvls, wvls=True)
    output_file = os.path.join(out_dir, f'target_{os.path.basename(args.target_spectra).split(".")[0]}.hdr')
    save_envi(output_file=output_file, meta=refl_meta, grid=spec_grid)

if __name__ == "__main__":
    main()