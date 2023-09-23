import os
from utils.text_guide import cursor_print, query_slpit_mode
from utils.slpit_download import run_download_emit, run_dowloand_slpit
from slpit.geoprocess import run_geoprocess_utils, run_geoprocess_extract
from slpit.build_slpit import run_build_workflow
from slpit.figures import run_figures
from slpit.slpit_unmix import run_slipt_unmix


def display_slpit_menu():
    msg = f"You have entered SLPIT mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to the SLPIT Mode....")
    print("A... Download EMIT Imagery")
    print("B... Download Transect Data")
    print("C... Build Reflectance Files from ASD")
    print("D... Geoprocess NC to ENVI data")
    print("E... Extract ENVI Data from SLPIT Points")
    print("F... Unmix Signals")
    print("G... Process Figures")
    print("H... Exit")


def run_slpit_workflow(base_directory:str, dry_run, sensor):
    while True:
        display_slpit_menu()

        user_input = input('\nPlease indicate the desired mode: ').upper()

        # download EMIT NC images
        if user_input == 'A':
            run_download_emit(base_directory)

        # download SLPIT data
        elif user_input == 'B':
            run_dowloand_slpit()

        # build and convolve the libraries to specified instruments
        elif user_input == 'C':
            run_build_workflow(base_directory=base_directory, sensor=sensor)

        # run the geoprocess on the emit imagery
        elif user_input == 'D':
            run_geoprocess_utils(base_directory=base_directory, nc_to_envi=dry_run)

        # extract the 3x3 windows
        elif user_input == 'E':
            run_geoprocess_extract(base_directory=base_directory, dry_run=dry_run)

        # run unmixing code
        elif user_input == 'F':
            run_slipt_unmix(base_directory=base_directory, dry_run=dry_run)

        # run the figure set
        elif user_input == 'G':
            run_figures(base_directory=base_directory)

        elif user_input == "H":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")