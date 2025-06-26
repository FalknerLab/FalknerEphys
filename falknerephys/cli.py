import os
import argparse


def main():
    """
    Main function to parse arguments and execute the appropriate action.

    Returns
    -------
    None
    """
    flags = ['-brainreg']
    defaults = [None]
    parser = argparse.ArgumentParser(prog='FalknerEphys',
                                     description='Falkner Lab codebase to process ephys data',
                                     epilog='See documentation at github.com/FalknerLab/FalknerEphys')
    parser.add_argument(flags[0], help='Run brainreg and register probe to Allen CCF')

    args = vars(parser.parse_args())
    num_args = 0
    for k, d in zip(args.keys(), defaults):
        if args[k] != d:
            num_args += 1

    if num_args == 0:
        print_info()

    if args['brainreg'] is not None:
        print(args['brainreg'])


def print_info():
    """
    Prints the version information from the version file.

    Returns
    -------
    None
    """
    v_file = open(os.path.abspath('resources/version.txt'), 'r')
    print(v_file.read())


if __name__ == '__main__':
    main()
