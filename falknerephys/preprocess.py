import numpy as np


def resample_spikes(spikes, in_hz, out_hz):
    resamp_spikes = float(out_hz)*(spikes/float(in_hz))
    resamp_spikes = np.round(resamp_spikes).astype(int)
    return resamp_spikes


def bin_fr(spikes, fs, bin_ms, length_s, same_shape=True):
    bin_s = bin_ms/1000
    bin_samps = int(bin_s*fs)
    num_bins = np.round(length_s/bin_s).astype(int)
    binned = np.zeros(num_bins)
    for i in range(num_bins):
        bin_spks = np.logical_and(spikes > i*bin_samps, spikes < (i+1)*bin_samps)
        binned[i] = np.sum(bin_spks)
    if same_shape:
        ups_binned = []
        for b in binned:
            [ups_binned.append(b) for n in range(bin_samps)]
        binned = np.array(ups_binned)
    time_s = np.linspace(0, length_s, num_bins)
    return time_s, binned/bin_s


def square_fr(spks, fs, time_width_ms, out_len_s):
    win_samps = int(fs*(time_width_ms/1000))
    spks_t = np.zeros(round(fs*out_len_s))
    u_ts = np.unique(spks)
    sum_spk = [np.sum(spks == i) for i in u_ts]
    spks_t[u_ts] = sum_spk
    kern = np.ones(win_samps)
    conv_spks = np.convolve(spks_t, kern, mode='same')
    fr = conv_spks/(time_width_ms/1000)
    t = np.linspace(0, out_len_s, len(fr))
    return t, fr


def gaus_fr(spks, fs, time_width_ms, out_len_s):
    win_samps = fs*(time_width_ms/1000)
    spks_t = np.zeros(fs*out_len_s)
    u_ts = np.unique(spks)
    sum_spk = [np.sum(spks == i) for i in u_ts]
    spks_t[u_ts] = sum_spk
    sig = 1
    # r = range(-int(win_samps / 2), int(win_samps / 2) + 1)
    # kern = [1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-float(x) ** 2 / (2 * sig ** 2)) for x in r]
    x = np.arange(-3*sig, 3*sig, 1/len(spks_t))
    kern = np.exp(-(x / sig) ** 2 / 2)
    conv_spks = np.convolve(spks_t, kern, mode='same')
    return spks_t, conv_spks


def spikes_to_timeseries(unit_dict, smooth_func=square_fr, ephys_hz=25000, out_hz=40, ts_len_s=60, time_win_ms=50):
    units = []
    t_vec = []
    for ind, u in enumerate(unit_dict.keys()):
        spks = resample_spikes(unit_dict[u], ephys_hz, out_hz)
        t, fr = smooth_func(spks, out_hz, time_win_ms, ts_len_s)
        units.append(fr)
        t_vec = t
    spk_data = np.array(units).T
    return t_vec, spk_data