import argparse
import os
from utils.create_tree import create_directory
from utils.text_guide import cursor_print
import subprocess
from simulation.sim_workflow import run_sim_workflow
from slpit.slpit_workflow import run_slpit_workflow
from tetracorder.tetracorder_workflow import run_tetracorder_workflow
from shift.shift_workflow import run_shift_workflow
from utils.ecosis_format import run_ecosis
import time

def display_menu():
    print("Welcome to the Interactive Menu")
    print("A... Simulation")
    print("B... SLPIT")
    print("C... Tetracorder")
    print('D... SHIFT')
    print("E... ECOSIS tables")
    print("F... Exit")

def main():
    parser = argparse.ArgumentParser(description='Run Terraspec')
    parser.add_argument('-bd', '--base_root_directory', type=str, default='.', help='Specify project directory (e.g., where outputs will save')
    parser.add_argument('-dry', '--dry_run', type=bool, help=' Set the dry run parameter to True to print unmix call', default=False)
    parser.add_argument('-lvl', '--level', type=str, help='level of classification to use', default='level_1')
    parser.add_argument('-sns', '--sensor', type=str, help='specify sensor to use', default='emit',
                        choices=['emit', 'aviris_ng', 'modis', 'hlss30', 'neon', 'aviris-classic', 'mavic3m', 'av3'])
    parser.add_argument('-bundles', '--number_of_bundles', type=str, help='number of spectral bundles to use', default='100000')
    parser.add_argument('-dis_geo', '--disable_geo_filter', action='store_false', help='Apply EMIT mask and selection from Ochoa et al 2025', default=True)
    parser.add_argument('-norm', '--normalization', type=bool, help='Brightness normalization enabled for PC!', default=True)
    parser.add_argument('-spec_start_col', '--spectra_start_col', type=str, help='Spectra starting column', default='7')
    parser.add_argument('-n_cores', '--number_of_cores', type=str, help='Number of cores to use', default='40')

    args = parser.parse_args()
    
    base_directory = os.path.join(args.base_root_directory, 'terraspec_output')

    create_directory(base_directory)
    subprocess.call(['python', 'utils/create_tree.py', '-bd', base_directory])

    intro = f"Welcome to TerraSpec!\n" \
            f"Project directory has been set to: {base_directory}\n"
    cursor_print(intro)

    while True:
        display_menu()
        choice = input("Enter desired mode: ").upper()

        if choice == "A":
            run_sim_workflow(os.path.join(base_directory, 'simulation'), dry_run=args.dry_run, sensor=args.sensor,
                             level=args.level, geo_filter=args.disable_geo_filter,
                             spectral_bundles=int(args.number_of_bundles), normalization=args.normalization,
                             spectra_starting_column=int(args.spectra_start_col), n_cores=args.number_of_cores)

        elif choice == 'B':
            run_slpit_workflow(os.path.join(base_directory, 'slpit'), dry_run=args.dry_run, sensor=args.sensor)

        elif choice == 'C':
            run_tetracorder_workflow(base_directory, sensor=args.sensor, dry_run=args.dry_run,
                                     new_simulation_bundles=args.new_simulation_bundles, spectral_bundles=int(args.number_of_bundles))

        elif choice == 'D':
            run_shift_workflow(os.path.join(base_directory, 'shift'), sensor='aviris_ng', dry_run=args.dry_run)

        elif choice == 'E':
            run_ecosis(base_directory=base_directory)

        elif choice == "F":
            outro = "TerraSpec processes complete. Thank you for using Terraspec!"
            cursor_print(outro)
            time.sleep(1)
            print()
            f = open('utils/vault-boy.txt', 'r', encoding="utf8")

            print(''.join([line for line in f]))
            break
        else:
            print("Invalid choice. Please choose a valid option.")


if __name__ == '__main__':
    main()
