import os
import argparse
import shutil

def create_directory(directory: str, clear_existing: bool = False):
    if os.path.isdir(directory):
        if clear_existing:
            # Remove the directory and everything inside it
            shutil.rmtree(directory)
            os.mkdir(directory)
        else:
            # Do nothing if it exists and we aren't clearing it
            pass
    else:
        # Create it if it doesn't exist at all
        os.mkdir(directory)



def main():
    parser = argparse.ArgumentParser(description='Run spectra clean workflow')
    parser.add_argument('-bd', '--base_directory', type=str, help='Specify base directory')
    args = parser.parse_args()

    directories = ['simulation', 'slpit', 'tetracorder', 'shift', 'fire']
    for directory in directories:
        create_directory(os.path.join(args.base_directory, directory))
        create_directory(os.path.join(args.base_directory, directory, 'figures'))
        create_directory(os.path.join(args.base_directory, directory, 'output'))
        create_directory(os.path.join(args.base_directory, directory, 'data'))
        create_directory(os.path.join(args.base_directory, directory, 'gis'))

        if directory == 'slpit' or directory == 'shift' or directory == 'fire':
            create_directory(os.path.join(args.base_directory, directory, 'field'))

        if directory == 'simulation':
            create_directory(os.path.join(args.base_directory, directory, 'raw_data'))


if __name__ == '__main__':
    main()
