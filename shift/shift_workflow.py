from utils.text_guide import cursor_print
from utils.slpit_download import download_shift_slpit, download_shift_imagery
from shift.build_shift import run_build_workflow
from shift.shift_unmix import run_shift_unmix
from shift.shift_tables import run_tables
from shift.figures import run_figures
from shift.field_methods_figures import run_figures as mt_fig

def display_shift_menu():
    msg = f"You have entered SHIFT mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to SHIFT Mode....")
    print("A... Download SHIFT Imagery")
    print("B... Download SHIFT Transect Data")
    print("C... Build Reflectance Files from ASD")
    #print("E... Extract Reflectance from SHIFT Centroids")
    #print("F... Sync extracted 3x3 windows")
    print("D... Unmix Signals")
    print("E... Process Figures and Tables")
    print("F... Exit")

def display_result_menu():
    msg = f"You have entered SHIFT mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("A... Tables")
    print("B... Figures")
    print("C... Exit")

def run_shift_workflow(base_directory:str, dry_run, sensor):
    while True:
        display_shift_menu()
        user_input = input('\nPlease indicate the desired mode: ').upper()

        # download EMIT NC images
        if user_input == 'A':
            download_shift_imagery(base_directory=base_directory, sensor=sensor)

        # download SLPIT data
        elif user_input == 'B':
            download_shift_slpit()

        # build and convolve the libraries to specified instruments
        elif user_input == 'C':
            run_build_workflow(base_directory=base_directory, sensor=sensor)

        # extract the 3x3 windows - can only be executed on EMIT cluster!
        elif user_input == 'D':
            run_geoprocess_extract(base_directory=base_directory, dry_run=dry_run)

        # sync extracts between google drive and ucla cluster
        elif user_input == 'E':
            sync_extracts(base_directory)

        # run unmixing code
        elif user_input == 'D':
            run_shift_unmix(base_directory=base_directory, dry_run=dry_run)

        # run the figure set
        elif user_input == 'E':
            while True:
                display_result_menu()

                result_input = input('\nPlease indicate the desired mode: ').upper()
                if result_input == 'B':
                    mt_fig(base_directory=base_directory)
                    run_figures(base_directory=base_directory)

                elif result_input == 'A':
                    run_tables(base_directory=base_directory)

                elif result_input == 'C':
                    print("Returning to SLPIT menu.")
                    break

        elif user_input == "F":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")