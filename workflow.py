import argparse
import os
from utils.create_tree import create_directory
from utils.text_guide import cursor_print
import subprocess
from simulation.sim_workflow import run_sim_workflow
from slpit.slpit_workflow import run_slpit_workflow


def main():
    parser = argparse.ArgumentParser(description='Run Terraspec')
    parser.add_argument('-bd', '--base_root_directory', type=str, help='Specify project directory (e.g., where outputs will save')
    parser.add_argument("-mode", type=str, help="set the run mode", default="convolve",
                        choices=['simulation', 'slpit', 'tetracorder'])
    parser.add_argument('-dry', '--dry_run', type=bool, help=' Set the dry run parameter to True to print unmix call',
                        default=False)
    parser.add_argument('-lvl', '--level', type=str, help='level of classification to use', default='level_1')

    args = parser.parse_args()

    base_directory = os.path.join(args.base_root_directory, 'terraspec')

    create_directory(base_directory)
    subprocess.call(['python', 'utils/create_tree.py', '-bd', base_directory])

    intro = f"Welcome to TerraSpec!\n" \
            f"Project directory has been set to: {base_directory}\n"
    cursor_print(intro)

    if args.mode in ['simulation']:
        run_sim_workflow(os.path.join(base_directory, 'simulation'), dry_run=args.dry_run)

    if args.mode in ['slpit']:
        run_slpit_workflow(os.path.join(base_directory, 'slpit'), dry_run=args.dry_run)

    if args.mode in ['tetracorder']:
        print('tc coming soon!')


if __name__ == '__main__':
    main()