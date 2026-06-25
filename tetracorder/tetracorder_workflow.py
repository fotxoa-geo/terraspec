import os.path
import subprocess
from utils.text_guide import cursor_print
from tetracorder.build_tetracorder import run_tetracorder_build
from tetracorder.figures import run_figure_workflow
from glob import glob


def display_tetracorder_menu():
    msg = f"You have entered Tetracorder mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)
    print("Welcome to the Tetracorder Mode....")
    print("A... Simulation and augmentation of data")
    print("B... Figures")
    print("C... Exit")


def run_tetracorder_workflow(base_directory:str, sensor:str, dry_run:bool, spectral_bundles:int, partition:str):
    while True:
        display_tetracorder_menu()
        user_input = input('\nPlease indicate the desired mode: ').upper()

        # run build workflow
        if user_input == 'A':
            run_tetracorder_build(base_directory, sensor=sensor, dry_run=dry_run, spectral_bundles=spectral_bundles, partition=partition)

        # run figure workflow
        elif user_input == 'B':
            run_figure_workflow(base_directory)

        elif user_input == "C":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
