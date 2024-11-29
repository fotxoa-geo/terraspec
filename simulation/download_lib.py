import argparse
import os
import requests


def main():
    parser = argparse.ArgumentParser(description='Run download script for data...')
    parser.add_argument('-bd', '--base_root_directory', type=str, default='/data1/geog/gregokin/',
                        help='Specify project directory (e.g., where outputs will save')
    parser.add_argument('-dry', '--dry_run', type=bool, help=' Set the dry run parameter to True to print unmix call',
                        default=False)
    parser.add_argument('-lvl', '--level', type=str, help='level of classification to use', default='level_1')
    parser.add_argument('-sns', '--sensor', type=str, help='specify sensor to use', default='emit',
                        choices=['emit', 'aviris-ng'])
    parser.add_argument('-io', '--io_bug', action='store_false', help='IO Bug found in simulation', default=True)

    args = parser.parse_args()

    base_directory = os.path.join(args.base_root_directory, 'terraspec')

    create_directory(base_directory)
    subprocess.call(['python', 'utils/create_tree.py', '-bd', base_directory])

    intro = f"Welcome to TerraSpec!\n" \
            f"Project directory has been set to: {base_directory}\n"
    cursor_print(intro)


if __name__ == '__main__':
    main()
