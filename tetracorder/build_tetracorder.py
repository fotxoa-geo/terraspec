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
from collections import Counter

def tetracorder_build_menu():
    msg = f"You have entered Tetracorder build mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to the Tetracorder build Mode....")
    print("A... Run Tetrecorder on endmember libraries")
    print("B... Generate synthetic reflectance")
    print("C... Run RECLAIMER workflows")
    print('D... Run RECLAIMER field')
    print('E... Run RECLAIMER EMIT')
    print("F... Exit")


class Tetracorder:

    def __init__(self, base_directory: str, sensor:str, partition:str):

        self.base_directory = os.path.join(base_directory, 'tetracorder')
        self.tetra_data_directory = os.path.join(self.base_directory, 'data')
        self.tetra_output_directory = os.path.join(self.base_directory, 'output')
        self.simulation_output_directory = os.path.join('terraspec_output', 'simulation', 'output')
        self.slpit_output_directory = os.path.join('terraspec_output', 'slpit', 'output')

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor=sensor)
        self.sensor = sensor

        # create output directory for augmented files
        create_directory(os.path.join(self.tetra_output_directory, 'synthethic_rfls'))
        self.synthetic_dir = os.path.join(os.path.join(self.tetra_output_directory, 'synthethic_rfls'))

        self.partition = partition

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
        spectral_abundance_array = envi_to_array(os.path.join(self.tetra_output_directory, 'libraries', 'sim_lib', 'tetracorder_sim_lib',
                                                              'sim_lib_augmented_min'))[:, 0, :]
        
        # load txt files outputs from tetracorder
        df_minerals_indentified_dict, df_minerals_indentified = spectra.get_mineral_reclassification(os.path.join(self.tetra_output_directory, 'libraries', 'sim_lib', 'tetracorder_sim_lib', 'sim_lib_augmented_minerals'))

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
            g2_index = spectral_abundance_array[df_index, 3]
            
            if g1_index in valid_g1_indices and g2_index in valid_g2_indices:
                valid_rows_g1.append(df_row)
                indices_used_g1.append(g1_index)

            if g2_index in valid_g2_indices and g1_index in valid_g1_indices:
                valid_rows_g2.append(df_row)
                indices_used_g2.append(g2_index)
        
        create_directory(os.path.join(self.synthetic_dir, 'tetracorder_g1'), clear_existing=True)
        df_soil_g1 = pd.DataFrame(valid_rows_g1)
        df_sim_g1 = pd.concat([df_veg, df_soil_g1], axis=0, ignore_index=True)
        df_sim_g1 = df_sim_g1.sort_values('level_1')
        
        df_sim_g1.to_csv(os.path.join(self.synthetic_dir, 'tetracorder_g1', 'df_sim_1.csv'))
        
        create_directory(os.path.join(self.synthetic_dir, 'tetracorder_g2'), clear_existing=True)
        df_soil_g2 = pd.DataFrame(valid_rows_g2)
        df_sim_g2 = pd.concat([df_veg, df_soil_g2], axis=0, ignore_index=True)
        df_sim_g2 = df_sim_g2.sort_values('level_1')
        df_sim_g2.to_csv(os.path.join(self.synthetic_dir, 'tetracorder_g2' ,'df_sim_2.csv'))
        
        

        print(f"--- G1 Counts (Total: {len(indices_used_g1)}) ---")
        for index, count in sorted(Counter(indices_used_g1).items()):
            print(f"{index}: {count}")

        print(f"\n--- G2 Counts (Total: {len(indices_used_g2)}) ---")
        for index, count in sorted(Counter(indices_used_g2).items()):
            print(f"{index}: {count}")
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
        
        create_directory(os.path.join(self.synthetic_dir, 'outlogs'), clear_existing=True)
        log_file_dir = os.path.join(self.synthetic_dir, 'outlogs')
        
        rfl_files = sorted(list(glob(os.path.join(self.synthetic_dir, '**',  '*simulation_spectra'), recursive=True)))

        global_unmixing_library = os.path.join(self.tetra_data_directory, 'global_lib.csv')
        df_unmix = pd.read_csv(global_unmixing_library)
        df_rock = pd.read_csv(os.path.join('utils', 'tetracorder', 'rock.csv'))
        df_rock.insert(0, 'level_2', 'rock')
        df_rock['level_1'] = 'soil'
        asd_wvls = spectra.load_asd_wavelenghts()

        results = p_map(partial(spectra.convolve, asd_wvl=asd_wvls, wvl=self.wvls, fwhm=self.fwhm,
                                spectra_starting_col=2), [row for row in df_rock.iterrows()],
                        **{"desc": f"\t loading convolution... ", "ncols": 150})

        df_convolve = pd.DataFrame(results)
        df_convolve.columns = list(self.wvls)
        df_rock = pd.concat([df_rock.iloc[:, :2].reset_index(drop=True), df_convolve], axis=1)
        
        new_columns = list(df_unmix.columns)
        new_columns[7:] = list(self.wvls)
        df_unmix.columns = new_columns

        df_unmix_with_rock = pd.concat([df_unmix, df_rock], axis=0, ignore_index=True)
        df_unmix_with_rock = df_unmix_with_rock.fillna(-9999)
        output_with_rock = os.path.join(self.tetra_data_directory, 'global_rock.csv')
        df_unmix_with_rock.to_csv(output_with_rock, index=False)
        
        output_rock_envi = os.path.join(self.tetra_data_directory, 'global_rock.hdr')
        spectra.df_to_envi(df=df_unmix_with_rock, spectral_starting_column=7, wvls=self.wvls, output_raster=output_rock_envi)

        for __,unmixing_library in enumerate([global_unmixing_library, output_with_rock]):
            for _, rfl_img in enumerate(rfl_files):
                outfile = os.path.join(log_file_dir, f'{os.path.basename(rfl_img)}_{os.path.basename(unmixing_library).split(".")[0]}.out')
                lib_dir = os.path.dirname(rfl_img)
                
                if __ == 0:
                    base_call = f'sh {os.path.join("tetracorder", "libraries_tetracorder.sh")} {rfl_img} {lib_dir} {unmixing_library} --unmix --tetracorder '
                else:
                    base_call = f'sh {os.path.join("tetracorder", "libraries_tetracorder.sh")} {rfl_img} {lib_dir} {unmixing_library} --unmix '
                
                sbatch_cmd = f"sbatch --export=ALL -p {self.partition} -N 1 -c 20 --mem 40G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
                subprocess.run(sbatch_cmd, shell=True, text=True)

        soil_files = sorted(list(glob(os.path.join(self.synthetic_dir, '**',  '*simulation_soils'), recursive=True)))
        for _, soil_img in enumerate(soil_files):
            outfile = os.path.join(log_file_dir, f'{os.path.basename(soil_img)}.out')
            lib_dir = os.path.dirname(soil_img)

            base_call = f'sh {os.path.join("tetracorder", "libraries_tetracorder.sh")} {soil_img} {lib_dir} {unmixing_library}'
            sbatch_cmd = f"sbatch --export=ALL -p {self.partition} -N 1 -c 1 --mem 15G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
            subprocess.run(sbatch_cmd, shell=True, text=True)


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
        
        # create output directories for libraries
        create_directory(os.path.join(self.tetra_output_directory, 'libraries'))
        lib_output_dir = os.path.join(self.tetra_output_directory, 'libraries')
        create_directory(os.path.join(self.tetra_output_directory, 'libraries', 'log_files'))
        log_file_dir = os.path.join(self.tetra_output_directory, 'libraries', 'log_files')

        for _, rfl_img in enumerate([simulation_lib, unmix_lib, rem_lib]):
            outfile = os.path.join(log_file_dir, f'{os.path.basename(rfl_img)}.out')
            create_directory(os.path.join(lib_output_dir, os.path.basename(rfl_img)))
            lib_dir = os.path.join(lib_output_dir, os.path.basename(rfl_img))

            base_call = f'sh {os.path.join("tetracorder", "libraries_tetracorder.sh")} {rfl_img} {lib_dir} {self.sensor} '
            sbatch_cmd = f"sbatch --export=ALL -p {self.partition} -N 1 -c 1 --mem 10G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
            subprocess.run(sbatch_cmd, shell=True, text=True)

        cursor_print("\t- done")

    def run_reclaimer_workflow(self):
        rfl_files = sorted(list(glob(os.path.join(self.synthetic_dir, '*', '*_simulation_spectra.hdr'), recursive=True)))
        global_unmixing_library = os.path.join(self.tetra_data_directory, 'global_lib.csv')
        global_unmixing_library_with_rock = os.path.join(self.tetra_data_directory, 'global_rock.csv')

        for unmixing_library in [global_unmixing_library, global_unmixing_library_with_rock]:
            for _, rfl_img_hdr in enumerate(rfl_files):
                out_dir = os.path.dirname(rfl_img_hdr)
                rfl_basename = os.path.basename(rfl_img_hdr)
                complete_veg_fractions = os.path.join(self.synthetic_dir, f'tetracorder_{os.path.basename(rfl_basename).split("_")[1]}', 'emc2', f'{os.path.basename(unmixing_library).split(".")[0]}_{os.path.basename(rfl_basename.split(".")[0])}_normalization_brightness__complete_fractions')
                three_component_veg_fractions = os.path.join(self.synthetic_dir, f'tetracorder_{os.path.basename(rfl_basename).split("_")[1]}', 'emc2', f'{os.path.basename(unmixing_library).split(".")[0]}_{os.path.basename(rfl_basename.split(".")[0])}_normalization_brightness__fractional_cover')
                rfl_img = os.path.splitext(rfl_img_hdr)[0]
                unmixing_library_envi_file = os.path.splitext(unmixing_library)[0]
                files_to_check = {
                    "Output Directory": out_dir,
                    "Vegetation Fractions": complete_veg_fractions,
                    "Reflectance Image": rfl_img,
                    "Unmixing Library (CSV)": unmixing_library,
                    "Unmixing Library (ENVI)": unmixing_library_envi_file,
                    "Three Component Fractions": three_component_veg_fractions,
                }

                missing_files = []

                for label, path in files_to_check.items():
                    if not os.path.exists(path):
                        missing_files.append(f"{label}: {path}")

                if missing_files:
                    print("--- ERROR: Missing Required Files ---")
                    for msg in missing_files:
                        print(f"  [X] {msg}")
                    print("--------------------------------------")
                else:
                    # All files exist, proceed with the call
                    try:
                        create_directory(os.path.join(self.tetra_output_directory, 'synthethic_rfls', 'outlogs'))
                        log_file_dir = os.path.join(self.tetra_output_directory, 'synthethic_rfls', 'outlogs')
                        outfile = os.path.join(log_file_dir, f'RECLAIMER_{os.path.basename(rfl_img).split(".")[0]}_{os.path.basename(unmixing_library).split(".")[0]}.out')
                        
                        base_call = f'sh ./tetracorder/reclaimer_workflows.sh {out_dir} {self.sensor} {complete_veg_fractions} {rfl_img} {unmixing_library} {unmixing_library_envi_file} {three_component_veg_fractions}'
                        sbatch_cmd = f"sbatch --export=ALL -p {self.partition} -N 1 -c 1 --mem 10G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
                        subprocess.run(sbatch_cmd, shell=True, text=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Shell script failed for {rfl_img}")
    

    def run_reclaimer_field(self):
        slpit_files = sorted(list(glob(os.path.join(self.slpit_output_directory, 'spectral_transects', '**', '*_SLPIT_emit'), recursive=True)))
        emit_files = sorted(list(glob(os.path.join(self.slpit_output_directory, 'spectral_transects', '**', '*_RFL_*_EXT'), recursive=True)))
        
        global_unmixing_library = os.path.join(self.tetra_data_directory, 'global_lib.csv')
        global_unmixing_library_with_rock = os.path.join(self.tetra_data_directory, 'global_rock.csv')
          
        create_directory(os.path.join(self.tetra_output_directory, 'field'))
        create_directory(os.path.join(self.tetra_output_directory, 'field', 'outlogs'))
        
        # overpass table
        overpass_table = pd.read_csv(os.path.join('slpit', 'overpass_table.csv'))
        overpass_table = overpass_table[overpass_table['Sensor'] == 'EMIT']
        overpass_table['emit_format'] = (overpass_table['Overpass Date'].astype(str).str.upper().str.replace(r'[-\s:]', '', regex=True))
        
        # 
        for _rfl_list, rfl_list in enumerate([slpit_files, emit_files]):

            for _, rfl_img in enumerate(rfl_list):
                
                transect_num = os.path.basename(rfl_img).split("_")[0]
                plot_team = os.path.basename(rfl_img).split('_')[0].split('-')[0]

                if int(transect_num[-3:]) > 60:
                    continue
                
                if _rfl_list == 1:
                    emit_ext_date = os.path.basename(rfl_img).split('_')[2]
                    ext_overpass_table = overpass_table[overpass_table['Site ID1'] == f'SPEC - {transect_num[-3:]}']
                    valid_dates = set(ext_overpass_table['emit_format'])
                    
                    if emit_ext_date not in valid_dates:
                        continue

                create_directory(os.path.join(self.tetra_output_directory, 'field', transect_num))
                out_dir = os.path.join(self.tetra_output_directory, 'field', transect_num)
                
                
                unmixing_library_envi_file = os.path.splitext(global_unmixing_library)[0]
                
                unmixing_library_envi_rock_file = os.path.splitext(global_unmixing_library_with_rock)[0]

                local_em_csv = os.path.join(os.path.dirname(rfl_img), '..', 'EMS', f'{transect_num}_EMS_emit') 
                local_em_envi = os.path.join(os.path.dirname(rfl_img), '..', 'EMS', f'{transect_num}_EMS_emit.csv')

                files_to_check = {"Output Directory": out_dir, 
                                    "Reflectance Image": rfl_img,
                                    "Unmixing Library (CSV)": global_unmixing_library,
                                    "Unmixing Library (ENVI)": unmixing_library_envi_file,
                                    "Rock Unmixing (CSV)": global_unmixing_library_with_rock,
                                    "Rock Unmixing (ENVI)": unmixing_library_envi_rock_file,
                                    "Local Lib": local_em_envi, 
                                    "Local CSV": local_em_csv}

                missing_files = []

                for label, path in files_to_check.items():
                    if not os.path.exists(path):
                        missing_files.append(f"{label}: {path}")

                if missing_files:
                    print("--- ERROR: Missing Required Files ---")
                    for msg in missing_files:
                        print(f"  [X] {msg}")
                        print("--------------------------------------")
                else:
                    # All files exist, proceed with the call
                    try:
                        log_file_dir = os.path.join(self.tetra_output_directory, 'field', 'outlogs')
                        outfile = os.path.join(log_file_dir, f'RECLAIMER_{os.path.basename(rfl_img)}.out')
                        base_call = f'sh ./tetracorder/reclaimer_field.sh {out_dir} {self.sensor} {rfl_img} {global_unmixing_library} {unmixing_library_envi_file} {local_em_envi} {local_em_csv} {global_unmixing_library_with_rock} {unmixing_library_envi_rock_file}'
                        sbatch_cmd = f"sbatch --export=ALL -p {self.partition} -N 1 -c 1 --mem 10G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
                        subprocess.run(sbatch_cmd, shell=True, text=True)
                        #time.sleep(1000)
                    except subprocess.CalledProcessError as e:
                        print(f"Shell script failed for {rfl_img}")
   
    def reclaimer_emit(self):
        emit_files = sorted(list(glob(os.path.join(self.slpit_output_directory, '..', 'gis', 'emit-data', 'products', '**', '*EMIT_L2A_RFL_*_reflectance'), recursive=True)))
        
        global_unmixing_library = os.path.join(self.tetra_data_directory, 'global_lib.csv')
        global_unmixing_library_with_rock = os.path.join(self.tetra_data_directory, 'global_rock.csv')
        
        create_directory(os.path.join(self.tetra_output_directory, 'scenes'))
        create_directory(os.path.join(self.tetra_output_directory, 'scenes', 'outlogs'))

        for i in emit_files:
            file_type = os.path.basename(i).split('_')[0]
            fid = os.path.basename(i).split("_")[4]
            
            if file_type == 'RGB':
                continue

            create_directory(os.path.join(self.tetra_output_directory, 'scenes', fid))
            out_dir = os.path.join(self.tetra_output_directory, 'scenes', fid)
                
                
            unmixing_library_envi_file = os.path.splitext(global_unmixing_library)[0]    
            unmixing_library_envi_rock_file = os.path.splitext(global_unmixing_library_with_rock)[0]

            files_to_check = {"Output Directory": out_dir, 
                                "Reflectance Image": i,
                                    "Unmixing Library (CSV)": global_unmixing_library,
                                    "Unmixing Library (ENVI)": unmixing_library_envi_file,
                                    "Rock Unmixing (CSV)": global_unmixing_library_with_rock,
                                    "Rock Unmixing (ENVI)": unmixing_library_envi_rock_file}
                                    

            missing_files = []

            for label, path in files_to_check.items():
                if not os.path.exists(path):
                    missing_files.append(f"{label}: {path}")

            if missing_files:
                print("--- ERROR: Missing Required Files ---")
                for msg in missing_files:
                    print(f"  [X] {msg}")
                    print("--------------------------------------")
            else:
                # All files exist, proceed with the call
                try:
                    log_file_dir = os.path.join(self.tetra_output_directory, 'scenes', 'outlogs')
                    outfile = os.path.join(log_file_dir, f'RECLAIMER_{os.path.basename(i)}.out')
                    base_call = f'sh ./tetracorder/reclaimer_emit.sh {out_dir} {self.sensor} {i} {global_unmixing_library} {unmixing_library_envi_file} {global_unmixing_library_with_rock} {unmixing_library_envi_rock_file}'
                    sbatch_cmd = f"sbatch --export=ALL -p {self.partition} -N 1 -c 25 --mem 30G --output {outfile} --job-name reclaimr --wrap='{base_call}'"
                    subprocess.run(sbatch_cmd, shell=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Shell script failed for {i}")


def run_tetracorder_build(base_directory, sensor, dry_run, spectral_bundles, partition):
    tc = Tetracorder(base_directory=base_directory, sensor=sensor, partition=partition)
    while True:
        tetracorder_build_menu()

        user_input = input('\nPlease indicate the desired mode: ').upper()

        if user_input == 'A':
            tc.libraries_tetracorder()
        elif user_input == 'B':
            tc.generate_tetracorder_reflectance(spectral_bundles=spectral_bundles)
        elif user_input == 'C':
            tc.run_reclaimer_workflow()
        elif user_input == 'D':
            tc.run_reclaimer_field()
        elif user_input == 'E':
            tc.reclaimer_emit()

        elif user_input == 'F':
            print("Returning to Tetracorder main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
