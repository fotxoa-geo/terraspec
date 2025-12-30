import os
from utils.text_guide import cursor_print, query_slpit_mode, query_yes_no
from utils.slpit_download import run_download_emit, run_dowloand_slpit, sync_gdrive, sync_extracts
from slpit.geoprocess import run_geoprocess_utils
from slpit.build_slpit import run_build_workflow
from slpit.figures import run_figures
from slpit.slpit_tables import run_tables
import pandas as pd
import geopandas as gp
from glob import glob

def display_slpit_menu():
    msg = f"You have entered SLPIT mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to the SLPIT Mode....")
    print("A... Download Imagery")
    print("B... Download Transect Data")
    print("C... Build Reflectance Files from ASD")
    print("D... Geoprocess Data")
    print("F... Sync extracted 3x3 windows")
    print("G... Unmix Signals")
    print("H... Process Figures and Tables")
    print("I... Exit")

def display_result_menu():
    msg = f"You have entered SLPIT result mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("A... Figures")
    print("B... Tables")
    print("C... Exit")

def run_slpit_workflow(base_directory:str, dry_run, sensor):
    while True:
        display_slpit_menu()

        user_input = input('\nPlease indicate the desired mode: ').upper()

        # download EMIT NC images
        if user_input == 'A':
            run_download_emit(base_directory, sensor)

        # download SLPIT data
        elif user_input == 'B':
            run_dowloand_slpit()
            sync_gdrive(base_directory, project=sensor)

        # build and convolve the libraries to specified instruments
        elif user_input == 'C':
            run_build_workflow(base_directory=base_directory, sensor=sensor)

        #elif user_input =='F':
        #    sync_extracts(base_directory, project='emit')

        # run unmixing code
        #elif user_input == 'G':
            # load shapefile
        #    df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.json")))
        #    df = df.sort_values('Name')

            # create directory
           # count = 0

            #for index, row in df.iterrows():
             #   plot = row['Name']
#
 #               emit_filetime = row['EMIT DATE']
  #              reflectance_img_emit = glob(os.path.join(self.gis_directory, 'emit-data-clip', f'*{plot.replace(" ", "")}_RFL_{emit_filetime}'))
   #             reflectance_uncer_img_emit = glob(os.path.join(self.gis_directory, 'emit-data-clip',
      #                                                         f'*{plot.replace(" ", "")}_RFLUNCERT_{emit_filetime}'))
    #            em_local = os.path.join(self.spectral_em_directory,
     #                                   plot.replace("SPEC", 'Spectral').replace(" ", "") + '-emit.csv')
       #         asd_reflectance = glob(os.path.join(self.spectral_transect_directory,
        #                                            f'*{plot.replace("SPEC", "Spectral").replace(" ", "")}'))

         #       if os.path.isfile(em_local):

          #          output_dest = os.path.join(self.output_directory, mode)
                    # unmix asd with local library

           #     else:
            #        print(em_local, " does not exist.")


            #run_slipt_unmix(base_directory=base_directory, dry_run=dry_run)

        # run the figure set
        elif user_input == 'H':
            while True:
                display_result_menu()

                result_input = input('\nPlease indicate the desired mode: ').upper()
                if result_input == 'A':
                    run_figures(base_directory=base_directory)

                elif result_input == 'B':
                    run_tables(base_directory=base_directory)

                elif result_input == 'C':
                    print("Returning to SLPIT menu.")
                    break

        elif user_input == "I":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
