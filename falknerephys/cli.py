import os
import argparse

from falknerephys.io.spikesort import run_ks, run_bombcell


def main():
    """
    Main function to parse arguments and execute the appropriate action.

    Returns
    -------
    None
    """
    flags = ['-brainreg', '-kilosort', '-bombcell']
    defaults = [None, None, None]
    mv_list = [('tiffpath', 'probepath'), ('binarypath', 'probepath'), ('binarypath', 'metapath', 'kilosortpath')]
    star_args = ['-kilosort']
    desc_list = ['Run Brainreg and register probe to Allen CCF',
                 'Run Kilosort on raw data with default settings',
                 'Run Bombcell on Kilosort output data, with default settings']
    parser = argparse.ArgumentParser(prog='FalknerEphys',
                                     description='Falkner Lab codebase to process ephys data',
                                     epilog='See documentation at github.com/FalknerLab/FalknerEphys')

    for f, d, mv, msg in zip(flags, defaults, mv_list, desc_list):
        if f in star_args:
            n_args = '*'
        else:
            n_args = len(mv)
        parser.add_argument(f, nargs=n_args, default=d, metavar=mv, help=msg)

    args = vars(parser.parse_args())
    num_args = 0
    for k, d in zip(args.keys(), defaults):
        if args[k] != d:
            num_args += 1

    if num_args == 0:
        print_info()

    if args['brainreg'] is not None:
        from falknerephys.io.register import register_probes
        register_probes(args['brainreg'][0], args['brainreg'][1])

    if args['kilosort'] is not None:
        if len(args['kilosort']) == 2:
            run_ks(args['kilosort'][0], args['kilosort'][1])
        elif len(args['kilosort']) == 0:
            run_ks()
        else:
            print('Wrong number of arguments for -kilosort. Requires 0 or 2')

    if args['bombcell'] is not None:
        run_bombcell(args['bombcell'][0], args['bombcell'][1], args['bombcell'][2])


def print_info():
    """
    Prints the version information from the version file.

    Returns
    -------
    None
    """
    this_path = os.path.dirname(os.path.abspath(__file__))
    v_file = open(os.path.join(this_path, 'resources', 'version.txt'), 'r')
    print(v_file.read())


if __name__ == '__main__':
    main()
