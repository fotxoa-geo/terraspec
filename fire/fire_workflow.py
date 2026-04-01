from utils.text_guide import cursor_print, query_slpit_mode, query_yes_no
from utils.slpit_download import download_scenes
from fire.time_series_build import run_build_workflow
from fire.figures import run_figures

def display_fire_menu():
    msg = f"You have entered Fire mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to Fire Mode....")
    print("A... Download Imagery")
    print("B... Build time series")
    print("C... Figures")
    print("D... Exit")

def display_result_menu():
    msg = f"You have entered SLPIT result mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("A... Figures")
    print("B... Tables")
    print("C... Exit")

def run_fire_workflow(base_directory:str, dry_run, sensor, aoi):
    while True:
        display_fire_menu()

        user_input = input('\nPlease indicate the desired mode: ').upper()

        # download EMIT NC images
        if user_input == 'A':
            download_scenes(base_directory, sensor, aoi=aoi)

        if user_input == 'B':
            run_build_workflow(base_directory, sensor, aoi=aoi)

        if user_input == 'C':
            run_figures(base_directory=base_directory, sensor=sensor, aoi=aoi)

        elif user_input == "C":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
