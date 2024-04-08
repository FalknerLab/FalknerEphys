import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from falknerephys.preprocess import get_time_win, bin_fr, square_fr, gaus_fr, spikes_to_timeseries
from falknerephys.io import whitematter as wm
from falknerephys.plotting import fr_heatmap


def run_demo(data_path):
    f = plt.figure()
    h5_file = data_path + '/wm_demo_DAQ.h5'
    slp_h5 = data_path + '/wm_demo_SLEAP.h5'
    gs = GridSpec(4, 5)
    daq_ax = f.add_subplot(gs[0, :2])
    print('Plotting DAQ data...')
    wm.show_h5(h5_file, ax=daq_ax)

    print('Loading Phy data...')
    phy_path = data_path + '/phy'
    unit_dict = wm.load_phy(phy_path)
    hm_hz = 40
    wm_start, vid_start = wm.get_starts(h5_file)
    start_t = vid_start - wm_start
    filt_dict = get_time_win(unit_dict, start_time=start_t, end_time=start_t + 60)
    ts, hm = spikes_to_timeseries(filt_dict, out_hz=hm_hz, ts_len_s=60)
    hm_ax = f.add_subplot(gs[1:3, :2])
    fr_heatmap(ts[-1], hm, ax=hm_ax)


if __name__ == '__main__':
    run_demo('wm')
    plt.show()
