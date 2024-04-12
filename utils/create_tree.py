import os
import argparse

def create_directory(directory: str):
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)

def main():
    parser = argparse.ArgumentParser(description='Run spectra clean workflow')
    parser.add_argument('-bd', '--base_directory', type=str, help='Specify base directory')
    args = parser.parse_args()

    directories = ['simulation', 'slpit', 'tetracorder', 'shift']
    for directory in directories:
        create_directory(os.path.join(args.base_directory, directory))
        create_directory(os.path.join(args.base_directory, directory, 'figures'))
        create_directory(os.path.join(args.base_directory, directory, 'output'))
        create_directory(os.path.join(args.base_directory, directory, 'data'))
        create_directory(os.path.join(args.base_directory, directory, 'gis'))

        if directory == 'slpit' or directory == 'shift':
            create_directory(os.path.join(args.base_directory, directory, 'field'))

        if directory == 'simulation':
            create_directory(os.path.join(args.base_directory, directory, 'raw_data'))


if __name__ == '__main__':
    main()
