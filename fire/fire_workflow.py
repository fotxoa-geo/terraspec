from utils.text_guide import cursor_print, query_slpit_mode, query_yes_no
from utils.slpit_download import download_scenes, enmap_process
from fire.time_series_build import run_build_workflow
from fire.figures import run_figures

def display_fire_menu():
    msg = f"You have entered Fire mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)

    print("Welcome to Fire Mode....")
    print("A... Download Imagery")
    print("B... Process EnMAP")
    print("C... Build time series")
    print("D... Figures")
    print("E... Exit")

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
            enmap_process(base_directory=base_directory, sensor='enmap', aoi=aoi)

        if user_input == 'C':
            run_build_workflow(base_directory, sensor, aoi=aoi)

        if user_input == 'D':
            run_figures(base_directory=base_directory, sensor=sensor, aoi=aoi)

        elif user_input == "E":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")
