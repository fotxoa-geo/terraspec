import os.path
import subprocess
from utils.text_guide import cursor_print, query_tetracorder_mode
from tetracorder.build_tetracorder import run_tetracorder_build
from tetracorder.figures import run_figure_workflow
from glob import glob

def display_tetracorder_menu():
    msg = f"You have entered Tetracorder mode! " \
          f"\nThere are various options to chose from: "
    cursor_print(msg)
    print("Welcome to the Tetracorder Mode....")
    print("A... Simulation and augmentation of data")
    print("B... Run Tetracorder")
    print("C... Figures")
    print("D... Exit")


def run_tetracorder_workflow(base_directory:str, sensor:str, dry_run:bool):
    while True:
        display_tetracorder_menu()
        user_input = input('\nPlease indicate the desired mode: ').upper()

        # run build workflow
        if user_input == 'A':
            run_tetracorder_build(base_directory, sensor=sensor, dry_run=dry_run)

        # run Tetracorder workflow
        elif user_input == 'B':
            augmented_files = glob(os.path.join(base_directory, 'output', 'augmented', '*'))
            output = os.path.join(base_directory, 'output', 'spectral_abundance')
            exclude = ['.hdr', '.aux', '.xml']

            for i in augmented_files:
                if os.path.splitext(i)[1] in exclude:
                    pass
                else:
                    base_call = f'sh tetracorder/tetracorder.sh {i} {output}'
                    sbatch_cmd = f'srun -N 1 -C 1 --mem=40G {base_call}'
                    subprocess.call(sbatch_cmd, shell=True)

        # run figure workflow
        elif user_input == 'C':
            run_figure_workflow(base_directory)

        elif user_input == "D":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose a valid option.")