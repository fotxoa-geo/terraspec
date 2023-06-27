import os
from utils.text_guide import cursor_print, query_slpit_mode
from utils.slpit_download import run_download_scripts
from slpit.geoprocess import run_geoprocess_utils, run_geoprocess_extract
from slpit.build_slpit import run_build_workflow


def run_slpit_workflow(base_directory:str, dry_run, sensor):
    msg = f"You have entered SLPIT mode! " \
          f"\nThere are various options to chose from: " \
          f"\n\tdownload, build, geoprocess, extract ,unmix, figures, expenses"

    cursor_print(msg)
    user_input = query_slpit_mode('\nPlease indicate the desired mode: ', default="yes")

    # run clean libraries workflow
    if user_input == 'download':
        run_download_scripts(base_directory)

    # build and convolve the libraries to specified instruments
    if user_input == 'build':
        run_build_workflow(base_directory=base_directory, sensor=sensor)

    # run the geoprocess on the emit imagery
    if user_input == 'geoprocess':
        run_geoprocess_utils(base_directory=base_directory, nc_to_envi=dry_run)

    # extract the 3x3 windows
    if user_input == 'extract':
        run_geoprocess_extract(base_directory=base_directory, dry_run=dry_run)


    # # run unmixing code
    # if user_input == 'unmix':
    #     run_unmix_workflow(base_directory=base_directory, dry_run=dry_run)
    #
    # # run the figure set
    # if user_input == 'figures':
    #     print('figuresss')

