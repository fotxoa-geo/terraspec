import os
from utils.text_guide import cursor_print, query_slpit_mode
from utils.slpit_download import run_download_scripts


def run_slpit_workflow(base_directory:str, dry_run):
    msg = f"You have entered SLPIT mode! " \
          f"\nThere are six options to chose from: " \
          f"\n\tdownload, build, geoprocess, unmix, figures, expenses"

    cursor_print(msg)
    user_input = query_slpit_mode('\nPlease indicate the desired mode: ', default="yes")

    # run clean libraries workflow
    if user_input == 'download':
        run_download_scripts(base_directory)

    # build and convolve the libraries
    # if user_input == 'build':
    #     run_build_reflectance(output_directory=output_directory)
    #
    # # run the geoprocess on the emit imagery
    # if user_input == 'geoprocess':
    #     run_hypertrace_workflow()
    #
    # # run unmixing code
    # if user_input == 'unmix':
    #     run_unmix_workflow(base_directory=base_directory, dry_run=dry_run)
    #
    # # run the figure set
    # if user_input == 'figures':
    #     print('figuresss')

