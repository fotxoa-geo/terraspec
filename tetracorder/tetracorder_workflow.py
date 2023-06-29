from utils.text_guide import cursor_print, query_tetracorder_mode
from tetracorder.build_tetracorder import run_tetracorder_build
from tetracorder.figures import run_figure_workflow


def run_tetracorder_workflow(base_directory:str, sensor:str):
    msg = f"You have entered Tetracorder mode! " \
          f"\nTetracorder is not part of this package." \
          f"\nThere are two options to chose from: " \
          f"\n\t build, figures"

    cursor_print(msg)
    user_input = query_tetracorder_mode('\nPlease indicate the desired mode: ', default="yes")

    # run build workflow
    if user_input == 'build':
        run_tetracorder_build(base_directory, sensor=sensor)

    # run figure workflow
    if user_input == 'figures':
        run_figure_workflow(base_directory)