from utils.text_guide import cursor_print, query_slpit_mode, query_yes_no
from utils.slpit_download import run_download_emit, run_dowloand_slpit
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
    print("D... Process Figures and Tables")
    print("E... Exit")

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

        # build and convolve the libraries to specified instruments
        elif user_input == 'C':
            run_build_workflow(base_directory=base_directory, sensor=sensor)

        # run the figure set
        elif user_input == 'D':
            while True:
                display_result_menu()

                result_input = input('\nPlease indicate the desired mode: ').upper()
                if result_input == 'A':
                    run_figures(base_directory=base_directory, sensor=sensor)

                elif result_input == 'B':
                    run_tables(base_directory=base_directory)

                elif result_input == 'C':
                    print("Returning to SLPIT menu.")
                    break

        elif user_input == "E":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
