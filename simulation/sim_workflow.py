import os
from utils.text_guide import query_sim_mode, cursor_print
from simulation.clean_libraries import run_clean_workflow
from simulation.build_reflectance_files import run_build_reflectance
from simulation.run_unmix import run_unmix_workflow
from simulation.run_hypertrace import run_hypertrace_workflow
from simulation.build_tables import run_build_tables
from simulation.paper_figures import run_figures


def display_menu():
    print("Welcome to the simulation menu")
    print("A... Download and clean libraries")
    print("B... Build libraries")
    print("C... Hypertrace workflow")
    print('D... Unmix')
    print("E... Tables")
    print('F... Figures')
    print("G... Exit")


def run_sim_workflow(base_directory, dry_run, sensor, level, io_bug):
    msg = f"You have entered simulation mode! " \
          f"\nThere are various options to chose from: "

    cursor_print(msg)

    output_directory = os.path.join(base_directory, 'output')

    while True:
        display_menu()
        choice = input("Enter desired mode: ").upper()

        # run clean libraries workflow
        if choice == 'A':
            run_clean_workflow(base_directory=base_directory, output_directory=output_directory,
                               geo_filter=True, sensor=sensor)

        # build and convolve the libraries
        elif choice == 'B':
            run_build_reflectance(output_directory=output_directory, sensor=sensor, level=level)

        # run hypertrace
        elif choice == 'C':
            run_hypertrace_workflow()

        # run unmixing code
        elif choice == 'D':
            run_unmix_workflow(base_directory=base_directory, dry_run=dry_run, io_bug=io_bug)

        # build csv report tables and latex tables
        elif choice == 'E':
            run_build_tables(base_directory=base_directory)

        # run the figure set
        elif choice == 'F':
            run_figures(base_directory=base_directory, sensor='emit')

        elif choice == "G":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")



