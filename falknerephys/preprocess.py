import os

import numpy as np


def get_time_win(ephys_dict, start_time=0, end_time=0):
    filt_dict = dict()
    for k in ephys_dict.keys():
        spikes = ephys_dict[k]
        filt_spikes = spikes[np.logical_and(spikes > start_time, spikes < end_time)]
        if len(filt_spikes) > 0:
            filt_spikes = filt_spikes - filt_spikes[0]
        filt_dict[k] = filt_spikes
    return filt_dict


def resample_spikes(spikes, in_hz, out_hz):
    resamp_spikes = float(out_hz)*(spikes/float(in_hz))
    resamp_spikes = np.round(resamp_spikes).astype(int)
    return resamp_spikes


def bin_spikes(spikes, fs, bin_ms, length_s):
    time_vec, bin_out = square_fr(spikes, fs, bin_ms, length_s, as_fr=False)
    return time_vec, bin_out


def square_fr(spks, fs, time_width_ms, out_len_s, as_fr=True):
    win_samps = int(fs*(time_width_ms/1000))
    spk_inds = np.floor(spks * fs).astype(int)
    spks_t = np.zeros(int(np.ceil(fs*out_len_s)))
    u_ts = np.unique(spk_inds)
    sum_spk = [np.sum(spk_inds == i) for i in u_ts]
    spks_t[u_ts] = sum_spk
    kern = np.ones(win_samps)
    if as_fr:
        kern = kern / win_samps
    conv_spks = np.convolve(spks_t, kern, mode='same')
    fr = conv_spks
    if as_fr:
        fr = fr * fs
    fr = fr.astype(np.float32)
    t = np.linspace(0, out_len_s, len(fr), dtype=np.float32)
    return t, fr


def gaus_fr(spks, fs, time_width_ms, out_len_s):
    win_samps = fs*(time_width_ms/1000)/2
    spk_inds = np.floor(spks * fs).astype(int)
    spks_t = np.zeros(int(np.ceil(fs*out_len_s)))
    u_ts, u_cnts = np.unique(spk_inds, return_counts=True)
    spks_t[u_ts] = u_cnts
    sig = win_samps/3
    x = np.linspace(-fs/2, fs/2, fs)
    kern = np.exp(-(x / sig) ** 2 / 2)
    kern_norm = kern / sum(kern)
    conv_spks = np.convolve(spks_t, kern_norm, mode='same')
    t = np.linspace(0, out_len_s, len(spks_t))
    fr = conv_spks * fs
    return t, fr


def spikes_to_timeseries(unit_dict, smooth_func=square_fr, ephys_hz=30000, out_hz=40, ts_len_s=60, time_win_ms=250, save_path=''):
    units = []
    t_vec = []
    if os.path.isfile(save_path):
        spike_dict = np.load(save_path)
        t_vec = spike_dict['t_vec']
        spk_data = spike_dict['spk_data']
        unit_ids = spike_dict['unit_ids']
    else:
        for u in unit_dict.keys():
            t, fr = smooth_func(unit_dict[u], out_hz, time_win_ms, ts_len_s)
            units.append(fr)
            t_vec = t
        spk_data = np.array(units).T
        unit_ids = list(unit_dict.keys())
        if save_path is not None:
            np.savez(save_path, spk_data=spk_data, t_vec=t_vec, smooth_func=str(smooth_func), out_hz=out_hz, time_win_ms=time_win_ms, unit_ids=unit_ids)
    return t_vec, spk_data, unit_ids


def split_trials_from_daq(spike_ts, daq_data, spike_len, daq_len, spike_hz=40, daq_hz=30303, do_split=False):
    if spike_len < daq_len:
        daq_data = daq_data[:int(np.floor(spike_len*daq_hz))]
    else:
        spike_ts = spike_ts[:int(np.floor(daq_len*spike_hz)), :]
    ds_chan0 = daq_data[np.floor(np.linspace(0, len(daq_data)-1, len(spike_ts))).astype(int)]
    trial_starts = np.where(np.logical_and(ds_chan0[:-1] < 2.5, ds_chan0[1:] > 2.5))[0]
    trial_ends = np.where(np.logical_and(ds_chan0[:-1] > 2.5, ds_chan0[1:] < 2.5))[0]
    trial_vec = np.zeros(len(spike_ts))
    for ti, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        trial_vec[ts:te] = ti + 1
    run_data = spike_ts[trial_vec > 0, :]
    trial_vec = trial_vec[trial_vec > 0] - 1
    if do_split:
        trial_list = []
        for i in range(max(trial_vec)):
            trial_list.append(spike_ts[trial_vec == i, :])
        return trial_list
    else:
        return run_data, trial_vec
