import os
import argparse

from falknerephys.io.register import register_probes
from falknerephys.io.spikesort import run_ks


def main():
    """
    Main function to parse arguments and execute the appropriate action.

    Returns
    -------
    None
    """
    flags = ['-brainreg', '-kilosort']
    defaults = [None, None]
    parser = argparse.ArgumentParser(prog='FalknerEphys',
                                     description='Falkner Lab codebase to process ephys data',
                                     epilog='See documentation at github.com/FalknerLab/FalknerEphys')
    parser.add_argument(flags[0],
                        nargs=2,
                        default=defaults[0],
                        metavar=('tiffpath', 'probepath'),
                        help='Run brainreg and register probe to Allen CCF')
    parser.add_argument(flags[1],
                        nargs='*',
                        default=defaults[1],
                        metavar=('binarypath', 'probepath'),
                        help='Run kilosort on raw, imec data')

    args = vars(parser.parse_args())
    num_args = 0
    for k, d in zip(args.keys(), defaults):
        if args[k] != d:
            num_args += 1

    if num_args == 0:
        print_info()

    if args['brainreg'] is not None:
        register_probes(args['brainreg'][0], args['brainreg'][1])

    if args['kilosort'] is not None:
        if len(args['kilosort']) == 2:
            run_ks(args['kilosort'][0], args['kilosort'][1])
        elif len(args['kilosort']) == 0:
            run_ks()
        else:
            print('Wrong number of arguments for -kilosort. Requires 0 or 2')


def print_info():
    """
    Prints the version information from the version file.

    Returns
    -------
    None
    """
    v_file = open(os.path.abspath('falknerephys/resources/version.txt'), 'r')
    print(v_file.read())


if __name__ == '__main__':
    main()
