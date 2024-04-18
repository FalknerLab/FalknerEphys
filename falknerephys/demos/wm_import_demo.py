import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from falknerephys.preprocess import get_time_win, bin_fr, square_fr, gaus_fr, spikes_to_timeseries
from falknerephys.io import whitematter as wm
from falknerephys.plotting import fr_heatmap, set_labels, jitter_plot, fr_per_xy
from falknerephys.behavior import get_sleap_data
import os


def run_demo(path=None, f=None, example_id=41):
    if f is None:
        f = plt.figure()

    # Load and plot DAQ Data
    daq_h5, phy_path, slp_h5 = get_demo_data_paths(path)
    gs = GridSpec(4, 5)
    daq_ax = f.add_subplot(gs[0, :2])
    wm.show_daq(daq_h5, ax=daq_ax)

    daq_data, unit_dict, t, x, y, v = get_demo_data()

    # Make Heatmap
    hm_hz = 100
    _, hm = spikes_to_timeseries(unit_dict, out_hz=hm_hz, ts_len_s=60, smooth_func=gaus_fr)
    hm_ax = f.add_subplot(gs[1:3, :2])
    unit_keys = np.array(list(unit_dict.keys()))
    unit_labs = unit_keys.copy()
    sel_vec = (np.arange(len(unit_labs)) % 5).astype(bool)
    unit_labs[sel_vec] = ''
    fr_heatmap(hm, ax=hm_ax, hz=hm_hz, unit_ids=unit_labs)
    hm_ax.set_title('Firing rate first 60s after video start')


    vel_ax = f.add_subplot(gs[3, :2])
    vel_ax.plot(t, v, 'k')
    set_labels('Mouse velocity', 'Time (s)', 'cm/s')
    vel_ax.set_xlim(0, 60)

    # Do sample unit processing plot
    ex_u = unit_keys[example_id]
    ex_spks = unit_dict[ex_u]
    jit_ax = f.add_subplot(gs[0, 2:4])
    jitter_plot(ex_spks, ax=jit_ax)
    jit_ax.set_xlim(0, 60)
    jit_ax.set_title('Unit ' + ex_u + ' Spikes')

    # Preprocessing example plots
    smooth_bin_ms = 100
    proc_funcs = [bin_fr, square_fr, gaus_fr]
    method = ['binned', 'moving average', 'gaussian']
    ylabs = ['# Spikes', 'Hz', 'Hz']
    for ind, p in enumerate(proc_funcs):
        ts, s_spks = p(ex_spks, hm_hz, smooth_bin_ms, 60)
        this_ax = f.add_subplot(gs[ind+1, 2:4])
        this_ax.plot(ts, s_spks)
        this_ax.set_xlim(0, 60)
        this_ax.set_title('Example unit smoothed: ' + method[ind])
        this_ax.set_ylabel(ylabs[ind])

    # FR heatmap over XY position
    fr_hm_ax = f.add_subplot(gs[:, 4:])
    hm_im = fr_per_xy(fr_hm_ax, ex_spks, x, y)
    set_labels('Mean firing rate across homecage', 'X Pos. (cm)', 'Y Pos. (cm)')
    fr_hm_ax.set_xlim(-25, 25)
    fr_hm_ax.set_ylim(-25, 25)
    plt.colorbar(hm_im, label='Hz', location='bottom')
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()


def get_demo_data_paths(path=None):
    if path is not None:
        data_path = path
    else:
        this_dir = os.path.split(os.path.abspath(__file__))[0]
        data_path = os.path.join(this_dir, 'wm')
    daq_h5 = os.path.join(data_path, 'wm_demo_DAQ.h5')
    slp_h5 = os.path.join(data_path, 'wm_demo_SLEAP.h5')
    phy_path = os.path.join(data_path, 'phy')
    return daq_h5, phy_path, slp_h5


def get_demo_data():
    daq_h5, phy_path, slp_h5 = get_demo_data_paths()
    daq_data = wm.get_daq_data(daq_h5)
    ephys_data = wm.load_phy(phy_path)

    # Load ephys data for first 60s after video start
    wm_start, vid_start = wm.get_starts(daq_h5)
    start_t = vid_start - wm_start
    filt_dict = get_time_win(ephys_data, start_time=start_t, end_time=start_t + 60)

    # Get SLEAP first 60s
    t, x, y, vel = get_sleap_data(slp_h5, (638, 518), 7.382)
    f60s = t < 60
    t = t[f60s]
    v = vel[f60s]
    x = x[f60s]
    y = y[f60s]
    return daq_data, filt_dict, t, x, y, v


if __name__ == '__main__':
    run_demo()
