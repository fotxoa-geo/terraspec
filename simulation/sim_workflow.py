import os
from utils.text_guide import query_sim_mode, cursor_print
from simulation.clean_libraries import run_clean_workflow
from simulation.build_reflectance_files import run_build_reflectance
from simulation.run_unmix import run_unmix_workflow
from simulation.run_hypertrace import run_hypertrace_workflow
from simulation.build_tables import run_build_tables
from simulation.paper_figures import run_figures
from simulation.paper_figures_mesma import run_figures as mesf


def run_sim_workflow(base_directory, dry_run):
    msg = f"You have entered simulation mode! " \
          f"\nThere are various options to chose from: " \
          f"\n\tclean, build, hypertrace, unmix, figures, tables"

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
        run_hypertrace_workflow()

    # run unmixing code
    if user_input == 'unmix':
        run_unmix_workflow(base_directory=base_directory, dry_run=dry_run)

    # build csv report tables and latex tables
    if user_input == 'tables':
        run_build_tables(base_directory=base_directory)

    # run the figure set
    if user_input == 'figures':
        run_figures(base_directory=base_directory, sensor='emit')
        mesf(base_directory=base_directory, sensor='emit')



