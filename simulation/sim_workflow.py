import os
from utils.text_guide import query_sim_mode, cursor_print
from simulation.clean_libraries import run_clean_workflow
from simulation.build_reflectance_files import run_build_reflectance


def run_sim_workflow(base_directory):
    msg = f"You have entered simulation mode! " \
          f"\nThere are four options to chose from: " \
          f"\n\tclean, build, hypertrace, unmix, figures"

    cursor_print(msg)
    user_input = query_sim_mode('\nPlease indicate the desired mode: ', default="yes")

    output_directory = os.path.join(base_directory, 'output')

    # run clean libraries workflow
    if user_input == 'clean':
        run_clean_workflow(base_directory=base_directory, output_directory=output_directory,
                           geo_filter=True)

    # build and convolve the libraries
    if user_input == 'build':
        run_build_reflectance(output_directory=output_directory)

    # run hypertrace
    if user_input == 'hypertrace':
        print("hellooo")

    # run unmixing code
    if user_input == 'unmix':
        print('hellooo')

    # run the figure set
    if user_input == 'figures':
        print('figuresss')


