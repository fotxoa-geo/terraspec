import sys
import time

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
    valid_responses = ["download", 'build', 'geoprocess', 'unmix', 'figures', 'expenses']

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
