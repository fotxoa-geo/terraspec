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
import subprocess
from spectral.io import envi
import isofit.core.common as isc

def tetracorder_build_menu():
    msg = f"You have entered Tetracorder build mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to the Tetracorder build Mode....")
    print("A... Run Tetrecorder on endmember libraries")
    print("B... Generate synthetic reflectance")
    print("C... ")
    print("D... ")
    print("E... ")
    print("F... ")
    print("G... ")
    print("H... Exit")


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


class Tetracorder:

    def __init__(self, base_directory: str, sensor:str):

        self.base_directory = os.path.join(base_directory, 'tetracorder')
        self.tetra_data_directory = os.path.join(self.base_directory, 'data')
        self.tetra_output_directory = os.path.join(self.base_directory, 'output')
        self.simulation_output_directory = os.path.join('terraspec_output', 'simulation', 'output')

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)
        self.sensor = sensor

        # create output directory for augmented files
        create_directory(os.path.join(self.tetra_output_directory, 'synthethic_rfls'))
        self.synthetic_dir = os.path.join(os.path.join(self.tetra_output_directory, 'synthethic_rfls'))
        # create_directory(os.path.join(self.tetra_output_directory, 'spectral_abundance'))
        # create_directory(os.path.join(self.tetra_output_directory, 'fractions'))
        # create_directory(os.path.join(self.tetra_output_directory, 'simulated_spectra'))
        # create_directory(os.path.join(self.tetra_output_directory, 'hypertrace'))
        # create_directory(os.path.join(self.tetra_output_directory, 'veg-correction'))
        # create_directory(os.path.join(self.tetra_output_directory, 'outlogs'))
        #
        # self.augmented_dir = os.path.join(os.path.join(self.tetra_output_directory, 'augmented'))
        # self.fractions_dir = os.path.join(os.path.join(self.tetra_output_directory, 'fractions'))
        # self.sim_spectra_dir = os.path.join(os.path.join(self.tetra_output_directory, 'simulated_spectra'))
        # self.veg_correction_dir = os.path.join(self.tetra_output_directory, 'veg-correction')
        # self.spectral_abun_dir = os.path.join(self.tetra_output_directory, 'spectral_abundance')
        # self.outlogs_dir = os.path.join(self.tetra_output_directory, 'outlogs')

    def run_tc(self, augmented_file):
        exclude = ['.hdr', '.xml', '.aux']

        if os.path.splitext(augmented_file)[1] in exclude:
            pass
        else:
            basename = os.path.basename(augmented_file)
            output = os.path.join(self.spectral_abun_dir, f'{basename}_abun_min')

            if os.path.isfile(output):
                pass
            else:

                if os.name in ['posix']:
                    basecall = f'./tetracorder/tetracorder.sh {augmented_file} {self.spectral_abun_dir + "/"}'
                    sbatch_cmd = f'sbatch -N 1 -c 1 --output {os.path.join(self.outlogs_dir, basename + ".out")} --mem=40G {basecall}'
                    subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
                else:
                    print("Tetracorder not installed!")

    def generate_tetracorder_reflectance(self, spectral_bundles):
        cursor_print('generating reflectance...')

        df_sim = pd.read_csv(os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                          'convex_hull__n_dims_4_sensor_emit_geofilter_True_simulation_library.csv'))

        df_pv = df_sim.loc[df_sim['level_1'] == 'pv'].copy()
        df_pv = df_pv.sample(n=8, random_state=13).reset_index(drop=True)

        df_npv = df_sim.loc[df_sim['level_1'] == 'npv'].copy()
        df_npv = df_npv.sample(n=8, random_state=13).reset_index(drop=True)

        df_veg = pd.concat([df_npv, df_pv], axis=0, ignore_index=True)

        # load spectral abundance of simulation library
        spectral_abundance_array = envi_to_array(os.path.join(self.tetra_output_directory, 'libraries', 'sim_lib', 'tetracorder',
                                                              'sim_lib_augmented_min'))[:, 0, :]
        
        # load txt files outputs from tetracorder
        df_minerals_indentified_dict, df_minerals_indentified = spectra.get_mineral_reclassification(os.path.join(self.tetra_output_directory, 'libraries', 'sim_lib', 'tetracorder', 'sim_lib_augmented_minerals'))

        # these are the corresponding indices
        valid_rows_g1 = []
        indices_used_g1 = []
        valid_rows_g2 = []
        indices_used_g2 = []

        df_soil = df_sim.loc[df_sim['level_1'] == 'soil'].copy()

        df_minerals_indentified_g1 = df_minerals_indentified.loc[df_minerals_indentified['Group'] == 1].copy()
        df_minerals_indentified_g2 = df_minerals_indentified.loc[df_minerals_indentified['Group'] == 2].copy()

        valid_g1_indices = df_minerals_indentified_g1['Index'].values
        valid_g2_indices = df_minerals_indentified_g2['Index'].values
        valid_g2_indices = valid_g2_indices[valid_g2_indices != 228] # this removes organic dry grass

        valid_g1_indices = sorted(list(valid_g1_indices))
        valid_g2_indices = sorted(list(valid_g2_indices))

        valid_g1_indices.append(0)
        valid_g2_indices.append(0)

        for df_index, df_row in df_soil.iterrows():
            g1_index = spectral_abundance_array[df_index, 1]

            if g1_index in valid_g1_indices:
                valid_rows_g1.append(df_row)
                indices_used_g1.append(g1_index)

            g2_index = spectral_abundance_array[df_index, 3]
            if g2_index in valid_g2_indices:
                valid_rows_g2.append(df_row)
                indices_used_g2.append(g2_index)
        
        create_directory(os.path.join(self.synthetic_dir, 'tetracorder_g1'))
        df_soil_g1 = pd.DataFrame(valid_rows_g1)
        df_sim_g1 = pd.concat([df_veg, df_soil_g1], axis=0, ignore_index=True)
        df_sim_g1 = df_sim_g1.sort_values('level_1')
        
        df_sim_g1.to_csv(os.path.join(self.synthetic_dir, 'tetracorder_g1', 'df_sim_1.csv'))
        
        create_directory(os.path.join(self.synthetic_dir, 'tetracorder_g2'))
        df_soil_g2 = pd.DataFrame(valid_rows_g2)
        df_sim_g2 = pd.concat([df_veg, df_soil_g2], axis=0, ignore_index=True)
        df_sim_g2 = df_sim_g2.sort_values('level_1')
        df_sim_g2.to_csv(os.path.join(self.synthetic_dir, 'tetracorder_g2' ,'df_sim_2.csv'))
        

        create_directory(os.path.join(self.synthetic_dir, 'rem'))
        rem_array = envi_to_array(os.path.join(self.tetra_data_directory, 'Esfordi_emit'))
        rem_array = rem_array.reshape(rem_array.shape[0], rem_array.shape[2])
        df_rem = pd.DataFrame(rem_array, columns=df_sim_g2.columns[-285:])
        df_rem.insert(0, 'level_1', 'soil') 
        
        df_sim_rem = pd.concat([df_veg, df_rem], axis=0, ignore_index=True)
        df_sim_rem = df_sim_rem.sort_values('level_1')
        df_sim_rem.to_csv(os.path.join(self.synthetic_dir, 'rem' ,'df_sim_rem.csv'))

        print(f"Indices used G1: {sorted(list(set(indices_used_g1)))}")
        print(f"Indices used G2: {sorted(list(set(indices_used_g2)))}")

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim_g1,
                                      level='level_1', spectral_bundles=spectral_bundles, increment_size=0.05,
                                      output_directory=os.path.join(self.synthetic_dir, 'tetracorder_g1'), wvls=self.wvls,
                                      name='tetracorder_g1_simulation', spectra_starting_col=8, endmember='soil',
                                      spectral_bundle_project='tetracorder_g1', new_simulation_bundles=spectral_bundles)

        spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim_g2,
                                      level='level_1', spectral_bundles=spectral_bundles, increment_size=0.05,
                                      output_directory=os.path.join(self.synthetic_dir, 'tetracorder_g2'), wvls=self.wvls,
                                      name='tetracorder_g2_simulation', spectra_starting_col=8, endmember='soil',
                                      spectral_bundle_project='tetracorder_g2', new_simulation_bundles=spectral_bundles)
        
        spectra.increment_reflectance(class_names=sorted(list(df_sim_rem.level_1.unique())), simulation_table=df_sim_rem,
                                      level='level_1', spectral_bundles=spectral_bundles, increment_size=0.05,
                                      output_directory=os.path.join(self.synthetic_dir, 'rem'), wvls=self.wvls,
                                      name='tetracorder_rem_simulation', spectra_starting_col=8, endmember='soil',
                                      spectral_bundle_project='tetracorder_rem', new_simulation_bundles=spectral_bundles)
        
        
        create_directory(os.path.join(self.synthetic_dir, 'outlogs'))
        log_file_dir = os.path.join(self.synthetic_dir, 'outlogs')
        
        rfl_files = sorted(list(glob(os.path.join(self.synthetic_dir, '**',  '*_spectra'), recursive=True)))

        for _, rfl_img in enumerate(rfl_files):
            outfile = os.path.join(log_file_dir, f'{os.path.basename(rfl_img)}.out')
            lib_dir = os.path.dirname(rfl_img)
            
            base_call = f'sh {os.path.join("tetracorder", "libraries_tetracorder.sh")} {rfl_img} {lib_dir} --unmix '
            sbatch_cmd = f"sbatch --export=ALL -p patient -N 1 -c 20 --mem 40G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
            subprocess.run(sbatch_cmd, shell=True, text=True)

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

            call_hypertrace_unmix(mode='sma', dry_run=False, reflectance_file=new_reflectance_file, em_file=em_file,
                                  parameters=optimal_parameters, output_dest=self.augmented_dir, scale='1',
                                  spectra_starting_column='8', uncertainty_file=new_uncertainty_file)

    def reconstruct_em_sma(self, user_em):
        cursor_print(f'reconstructing {user_em} from sma...')

        for group in ['g1', 'g2']:
            # reconstructed soil from fractions and unmix library
            complete_fractions_array = envi_to_array(os.path.join(self.tetra_output_directory, 'fractions', f'tetracorder_{group}_simulation_spectra_complete_fractions'))
            df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library.csv'))

            min_em_index = np.min(df_unmix[df_unmix['level_1'] == user_em].index)
            max_em_index = np.max(df_unmix[df_unmix['level_1'] == user_em].index)
            unmix_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'endmember_libraries', 'convex_hull__n_dims_4_unmix_library'))
            unmix_library_array = unmix_library_array[min_em_index:max_em_index + 1, 0, :]

            complete_fractions_array = complete_fractions_array[:, :, min_em_index:max_em_index + 1]
            spectra_grid = np.zeros((complete_fractions_array.shape[0], complete_fractions_array.shape[1], len(self.wvls)))

            func = partial(process_complete_fractions_row, unmix_library_array=unmix_library_array, wvls=self.wvls)
            results = p_map(func,
                        [complete_fractions_array[_row, :, :] for _row in range(complete_fractions_array.shape[0])],
                        **{"desc": f"\t\t rebuilding spectra ...", "ncols": 150})

            for _row, row in enumerate(results):
                spectra_grid[_row, :, :] = row

            meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls, wvls=True)
            output_raster = os.path.join(self.sim_spectra_dir, f"unmixing_{group}_{user_em}_emc2.hdr")
            save_envi(output_raster, meta_spectra, spectra_grid)

        print("\t- done")

    def augment_field_data(self):
        cursor_print('augmenting field data...')
        transect_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'transect', '*[!.csv][!.hdr][!.aux][!.xml]'))
        em_files = glob(os.path.join(self.slpit_output_directory, 'spectral_transects', 'endmembers-raw', '*[!.csv][!.hdr][!.aux][!.xml]'))

        # load shapefile
        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.json")))
        df = df.sort_values('Name')

        for index, row in df.iterrows():
            plot = row['Name']
            plot_num = int(plot.split('-')[1])

            if int(plot_num) > 60:
                continue

            print(f"{plot}... augmenting pixels")
            emit_filetime = row['EMIT DATE']

            reflectance_img_emit = glob(os.path.join(self.slpit_gis_directory, 'emit-data-clip', f'*{plot.replace(" ", "")}_RFL_{emit_filetime}'))
            reflectance_array = envi_to_array(reflectance_img_emit[0])[0,0,:]
            bad_band_indices = np.where(reflectance_array == -9999.)[0] # these are used for various

            basename = os.path.basename(reflectance_img_emit[0])
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f'{basename}_pixels_augmented.hdr')
            augment_envi(file=reflectance_img_emit[0],  vertical_average=True, wvls=self.wvls, out_raster=output_raster, bad_bands=bad_band_indices)
            self.run_tc(output_raster[:-4])

            # run each spectrum -
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f'{basename}_pixels_augmented_all.hdr')
            augment_envi(file=reflectance_img_emit[0],  vertical_average=False, wvls=self.wvls, out_raster=output_raster, bad_bands=bad_band_indices)
            self.run_tc(output_raster[:-4])

        for i in sorted(transect_files):
            basename = os.path.basename(i)
            print(f"{basename}... augmenting transects")
            plot_num = int(basename.split('-')[1])
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f"{basename}_transect_augmented.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=True, bad_bands=bad_band_indices)
            self.run_tc(output_raster[:-4])

            # run each spectrum
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f"{basename}_transect_augmented_all.hdr")
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=False, bad_bands=bad_band_indices)
            self.run_tc(output_raster[:-4])

        for i in sorted(em_files):
            basename = os.path.basename(i)
            plot_num = int(basename.split('-')[1])
            df_em = pd.read_csv(f'{i}.csv')
            
            # run each spectrum file
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f'{basename}_ems_augmented_all.hdr')
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=False)
            self.run_tc(output_raster[:-4])
            
            if plot_num > 60:
                continue

            print(f"{basename}... augmenting endmembers")
            soil_index_min = min(df_em.index[df_em['level_1'] == 'Soil'].tolist())
            soil_index_max = max(df_em.index[df_em['level_1'] == 'Soil'].tolist())
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f'{basename}_ems_augmented.hdr')
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=True, em_index_min=soil_index_min,
                         em_index_max=soil_index_max,bad_bands=bad_band_indices)
            self.run_tc(output_raster[:-4])

            # run each spectrum file - do not filter for bad bands, these are all endmembers
            output_raster = os.path.join(self.tetra_output_directory, 'augmented', f'{basename}_ems_augmented_all.hdr')
            augment_envi(file=i, wvls=self.wvls, out_raster=output_raster, vertical_average=False)
            self.run_tc(output_raster[:-4])


        # submitting tetracorder on emit scenes


        cursor_print("\t- done")

    
    def mineral_lib_refl_cont(self):

        for group in ['g1', 'g2']:
            # case 1 - simulations
            sim_fractions = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_fractions'))
            sim_soil_spectra = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_spectra'))
            sim_npv_spectra = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_npv'))
            sim_gv_spectra = envi_to_array(os.path.join(self.sim_spectra_dir, f'tetracorder_{group}_simulation_gv'))
            sim_mineral_index = envi_to_array(os.path.join(self.base_directory, 'output', 'spectral_abundance', f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, :]
            output_file = os.path.join(self.veg_correction_dir, f'tetracorder_{group}_simulated_spectra.hdr')

            spectra.mineral_components(index_array=sim_mineral_index, spectra_array=sim_soil_spectra, output_file=output_file,
                                       fractions_array=sim_fractions, group=group,  npv_array=sim_npv_spectra, gv_array=sim_gv_spectra)

            # case 2 - emc2 fractions and vegetation fractions derived from mixed spectra
            emc2_gv = envi_to_array(os.path.join(self.sim_spectra_dir, f'unmixing_{group}_pv_emc2'))
            emc2_npv = envi_to_array(os.path.join(self.sim_spectra_dir, f'unmixing_{group}_npv_emc2'))
            emc2_fractions = envi_to_array(os.path.join(self.fractions_dir,  f'tetracorder_{group}_simulation_spectra_fractional_cover'))
            output_file = os.path.join(self.veg_correction_dir, f'tetracorder_{group}_emc2_spectra.hdr')

            spectra.mineral_components(index_array=sim_mineral_index, spectra_array=sim_soil_spectra,
                                       output_file=output_file, fractions_array=emc2_fractions, group=group,
                                       npv_array=emc2_npv, gv_array=emc2_gv)
    def unmix_slpit_fractions(self, dry_run):
        em_file = os.path.join(self.simulation_output_directory, 'endmember_libraries',
                               'convex_hull__n_dims_4_unmix_library.csv')

        optimal_parameters = ['--num_endmembers 30', '--n_mc 25', '--normalization none']

        emit_pixels = glob(os.path.join(self.tetra_output_directory, 'augmented', 'SPEC*_RFL*[!.csv][!.hdr][!.aux][!.xml]'))
        transect_files = glob(os.path.join(self.tetra_output_directory, 'augmented', 'Spectral-*_tra*[!.csv][!.hdr][!.aux][!.xml]'))
        reflectance_files = emit_pixels + transect_files

        for i in reflectance_files:
            call_unmix(mode='sma', dry_run=dry_run, reflectance_file=i, em_file=em_file,
                       parameters=optimal_parameters, output_dest=self.fractions_dir, scale='1',
                       spectra_starting_column='8')


    def libraries_tetracorder(self):
        cursor_print('augmenting data for tetracorder...')
        print()
        cursor_print('\t loading simulation data...')

        # load simulation library - 4 dimension; convex hull
        simulation_lib_original = os.path.join(self.simulation_output_directory, 'simulation_libraries',
                                      'convex_hull__n_dims_4_sensor_emit_geofilter_True_simulation_library')
        
        shutil.copy(simulation_lib_original, os.path.join(self.tetra_data_directory, 'sim_lib'))
        shutil.copy(f'{simulation_lib_original}.hdr', os.path.join(self.tetra_data_directory, 'sim_lib.hdr'))
        simulation_lib = os.path.join(self.tetra_data_directory, f'sim_lib')

        # load unmix library - 4 dimensions; convex hull
        unmix_lib_original = os.path.join(self.simulation_output_directory, 'endmember_libraries',
                                 'convex_hull__n_dims_4_sensor_emit_geofilter_True_unmix_library')
        shutil.copy(unmix_lib_original, os.path.join(self.tetra_data_directory, 'unmix_lib'))
        shutil.copy(f'{unmix_lib_original}.hdr', os.path.join(self.tetra_data_directory, 'unmix_lib.hdr'))
        unmix_lib = os.path.join(self.tetra_data_directory, f'unmix_lib')

        # create rare_earth lib
        rem_lib = os.path.join('utils', 'tetracorder', 'Esfordi_speclib.img')
        rem_hdr = envi.open(os.path.join('utils', 'tetracorder', 'Esfordi_speclib.hdr'))
        rem_array = envi_to_array(rem_lib).transpose(1,0,2)
        rem_wvls = rem_hdr.bands.centers

        # convolve to emit
        spectra_grid = np.ones((rem_array.shape[0], rem_array.shape[1], np.shape(self.wvls)[0])) * -9999

        for _row, row in enumerate(rem_array):
            spectra = row[0, :]
            convolved_spectra = isc.resample_spectrum(x=spectra, wl=rem_wvls, wl2=self.wvls, fwhm2=self.fwhm, fill=False)
            spectra_grid[_row, :, :] = convolved_spectra

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(self.tetra_data_directory, f'Esfordi_{self.sensor}.hdr')
        save_envi(output_raster, meta_spectra, spectra_grid)
        rem_lib = os.path.join(self.tetra_data_directory, f'Esfordi_{self.sensor}')

        # create output directories for libraries
        create_directory(os.path.join(self.tetra_output_directory, 'libraries'))
        lib_output_dir = os.path.join(self.tetra_output_directory, 'libraries')
        create_directory(os.path.join(self.tetra_output_directory, 'libraries', 'log_files'))
        log_file_dir = os.path.join(self.tetra_output_directory, 'libraries', 'log_files')

        for _, rfl_img in enumerate([simulation_lib, unmix_lib, rem_lib]):
            outfile = os.path.join(log_file_dir, f'{os.path.basename(rfl_img)}.out')
            create_directory(os.path.join(lib_output_dir, os.path.basename(rfl_img)))
            lib_dir = os.path.join(lib_output_dir, os.path.basename(rfl_img))

            base_call = f'sh {os.path.join("tetracorder", "libraries_tetracorder.sh")} {rfl_img} {lib_dir} '
            sbatch_cmd = f"sbatch --export=ALL -p patient -N 1 -c 1 --mem 10G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
            subprocess.run(sbatch_cmd, shell=True, text=True)

        cursor_print("\t- done")
    
    def run_on_scenes(self):
        self.run_tc('/store/fochoa/terraspec_output/terraspec/slpit/gis/emit-data/envi/EMIT_L2A_RFL_001_20230831T152735_2324310_009_reflectance')

    def reconstruct_em_scenes(self, user_em):
        cursor_print(f'reconstructing {user_em} from sma...')


        # reconstructed soil from fractions and unmix library
        complete_fractions_array = envi_to_array(os.path.join(self.slpit_output_directory, 'scenes', 'sma',
                                                              f'EMIT_L2A_RFL_001_20230831T152735_2324310_009_reflectance_complete_fractions'))

        df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries',
                                            'convex_hull__n_dims_4_unmix_library.csv'))

        min_em_index = np.min(df_unmix[df_unmix['level_1'] == user_em].index)
        max_em_index = np.max(df_unmix[df_unmix['level_1'] == user_em].index)
        unmix_library_array = envi_to_array(os.path.join(self.simulation_output_directory, 'endmember_libraries',
                                                             'convex_hull__n_dims_4_unmix_library'))
        unmix_library_array = unmix_library_array[min_em_index:max_em_index + 1, 0, :]

        complete_fractions_array = complete_fractions_array[:, :, min_em_index:max_em_index + 1]
        spectra_grid = np.zeros((complete_fractions_array.shape[0], complete_fractions_array.shape[1], len(self.wvls)))

        func = partial(process_complete_fractions_row, unmix_library_array=unmix_library_array, wvls=self.wvls)
        results = p_map(func,
                            [complete_fractions_array[_row, :, :] for _row in range(complete_fractions_array.shape[0])],
                            **{"desc": f"\t\t rebuilding spectra ...", "ncols": 150})

        for _row, row in enumerate(results):
            spectra_grid[_row, :, :] = row

        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=self.wvls,
                                wvls=True)
        output_raster = os.path.join(r'G:\My Drive\terraspec\tetracorder\gis', f"unmixing_EMIT_L2A_RFL_001_20230831T152735_{user_em}_emc2.hdr")
        save_envi(output_raster, meta_spectra, spectra_grid)

        print("\t- done")

    def mineral_veg_correction_scene(self):
        # case 1 - simulations
        scene_fractions = envi_to_array(os.path.join(self.slpit_output_directory, 'scenes', 'sma',
                                                              f'EMIT_L2A_RFL_001_20230831T152735_2324310_009_reflectance_fractional_cover'))

        scene_spectra = envi_to_array(os.path.join(self.slpit_gis_directory, 'emit-data', 'envi', f'EMIT_L2A_RFL_001_20230831T152735_2324310_009_reflectance'))
        scene_npv_spectra = envi_to_array(r"G:\My Drive\terraspec\tetracorder\gis\unmixing_EMIT_L2A_RFL_001_20230831T152735_npv_emc2")
        scene_gv_spectra = envi_to_array(r"G:\My Drive\terraspec\tetracorder\gis\unmixing_EMIT_L2A_RFL_001_20230831T152735_pv_emc2")

        scene_mineral_index = envi_to_array(os.path.join(self.base_directory, 'output', 'spectral_abundance',
                                                       f'EMIT_L2A_RFL_001_20230831T152735_2324310_009_reflectance_min'))

        for group in ['g1', 'g2']:
            output_file = os.path.join(self.veg_correction_dir, f'EMIT_L2A_RFL_001_20230831T152735_veg_correction_{group}.hdr')
            spectra.mineral_components(index_array=scene_mineral_index, spectra_array=scene_spectra,
                                       output_file=output_file,
                                       fractions_array=scene_fractions, group=group, npv_array=scene_npv_spectra,
                                       gv_array=scene_gv_spectra)

def run_tetracorder_build(base_directory, sensor, dry_run, spectral_bundles):
    tc = Tetracorder(base_directory=base_directory, sensor=sensor)
    while True:
        tetracorder_build_menu()

        user_input = input('\nPlease indicate the desired mode: ').upper()

        if user_input == 'A':
            tc.libraries_tetracorder()
        elif user_input == 'B':
            tc.generate_tetracorder_reflectance(spectral_bundles=spectral_bundles)
        # elif user_input == 'B':
        #     tc.hypertrace_tetracorder()
        # elif user_input == 'C':
        #     tc.unmix_tetracorder(dry_run=dry_run)
        # elif user_input == 'D':
        #     tc.reconstruct_em_sma(user_em='pv')
        #     tc.reconstruct_em_sma(user_em='npv')
        #     tc.reconstruct_em_sma(user_em='soil')
        #     tc.mineral_lib_refl_cont()
        # elif user_input == 'E':
        #     tc.augment_field_data()
        # elif user_input == 'F':
        #     tc.unmix_slpit_fractions(dry_run=dry_run)
        # elif user_input == 'G':
        #     #tc.reconstruct_em_scenes(user_em='pv')
        #     #tc.reconstruct_em_scenes(user_em='npv')
        #     #tc.reconstruct_em_scenes(user_em='soil')
            #tc.mineral_veg_correction_scene()
        elif user_input == 'H':
            print("Returning to Tetracorder main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
