import numpy as np
import matplotlib.pyplot as plt


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


def bin_fr(spikes, fs, bin_ms, length_s):
    bin_out = np.zeros(fs*length_s)
    bin_sz_ind = round(fs * bin_ms / 1000)
    spike_inds = spikes * fs
    num_i = np.floor(len(bin_out) / bin_sz_ind).astype(int)
    for i in range(num_i):
        bin_spks = np.logical_and(spike_inds > i * bin_sz_ind, spike_inds < (i + 1) * bin_sz_ind)
        end_ind = i*bin_sz_ind + bin_sz_ind
        if end_ind > len(bin_out):
            end_ind = len(bin_out)
        spk_cnt = sum(bin_spks)
        bin_out[i*bin_sz_ind:end_ind] = spk_cnt
    time_vec = np.linspace(0, length_s, len(bin_out))
    return time_vec, bin_out


def square_fr(spks, fs, time_width_ms, out_len_s):
    win_samps = int(fs*(time_width_ms/1000))
    spk_inds = np.round(spks * fs).astype(int)
    spks_t = np.zeros(round(fs*out_len_s))
    u_ts = np.unique(spk_inds)
    sum_spk = [np.sum(spk_inds == i) for i in u_ts]
    spks_t[u_ts] = sum_spk
    kern = np.ones(win_samps) / win_samps
    conv_spks = np.convolve(spks_t, kern, mode='same')
    fr = conv_spks * fs
    t = np.linspace(0, out_len_s, len(fr))
    return t, fr


def gaus_fr(spks, fs, time_width_ms, out_len_s):
    win_samps = fs*(time_width_ms/1000)/2
    spk_inds = np.round(spks * fs).astype(int)
    spks_t = np.zeros(round(fs*out_len_s))
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


def spikes_to_timeseries(unit_dict, smooth_func=square_fr, ephys_hz=25000, out_hz=40, ts_len_s=60, time_win_ms=50):
    units = []
    t_vec = []
    for u in unit_dict.keys():
        t, fr = smooth_func(unit_dict[u], out_hz, time_win_ms, ts_len_s)
        units.append(fr)
        t_vec = t
    spk_data = np.array(units).T
    return t_vec, spk_data
