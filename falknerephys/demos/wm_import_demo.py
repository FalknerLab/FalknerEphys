import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from falknerephys.preprocess import bin_fr, square_fr
from falknerephys.io import whitematter as wm


def run_demo(data_path):
    f = plt.figure()
    phy_path = data_path + '/phy'
    print('Loading Phy data...')
    unit_dict = wm.load_phy(phy_path)
    h5_file = data_path + '/wm_demo_DAQ.h5'
    slp_h5 = data_path + '/wm_demo_SLEAP.h5'
    gs = GridSpec(4, 5)
    h5_ax = f.add_subplot(gs[0, :2])
    print('Plotting DAQ data...')
    wm.show_h5(h5_file, ax=h5_ax)


if __name__ == '__main__':
    run_demo('wm')
    plt.show()
