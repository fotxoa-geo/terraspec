import sys
import time
from glob import glob
import os
import subprocess


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        for c in question+prompt:
            sys.stdout.write(c)
            sys.stdout.flush()
            time.sleep(0.02)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def query_sim_mode(question, default=""):
    cursor_print(question)
    valid_responses = ["clean", "build", "unmix", "tables", "figures"]

    choice = input().lower()

    while choice not in valid_responses:
        choice = input("Please enter your selection: ")

        if choice not in choice:
            print("Invalid selection. Please try again.")

    return choice


def query_slpit_mode(question, default=""):
    cursor_print(question)
    valid_responses = ["download", 'build', 'geoprocess', 'extract', 'unmix', 'figures', 'expenses']

    choice = input().lower()

    while choice not in valid_responses:
        choice = input("Please enter your selection: ")

        if choice not in choice:
            print("Invalid selection. Please try again.")

    return choice


def cursor_print(string):
    for c in string:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(0.02)


def input_date(msg, gis_directory):
    while True:
        for c in msg:
            sys.stdout.write(c)
            sys.stdout.flush()
            time.sleep(0.02)
        date = input()

        envi = glob(os.path.join(gis_directory, 'emit-data', 'envi', '*' + date + '*_radiance'))

        if envi and date != "":
            break
        else:
            sys.stdout.write("TerraSpec could not find ENVI files for the requested date.\n"
                             "Please verify your date and your saved path.\n")

    return date, envi


def execute_call(cmd_list, dry_run=False):
    if dry_run:
        print(cmd_list)
    else:
        subprocess.call(cmd_list)