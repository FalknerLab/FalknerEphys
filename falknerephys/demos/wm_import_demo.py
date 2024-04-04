import matplotlib.pyplot as plt
import numpy as np
from falknerephys.preprocess import resample_spikes, square_fr


def demo_process(spks, fs, time_win=1000):
    spk_s = spks/fs
    plt.scatter(spk_s, 2*np.ones_like(spk_s) + np.random.uniform(low=-1, size=len(spks)), s=0.5)
    ds_spk = resample_spikes(spks, 25000, 1000)
    fr_vec_len = max(spk_s)+1
    t, mov_avg = square_fr(ds_spk, 1000, time_win, fr_vec_len)
    norm_ma = mov_avg/max(mov_avg)
    plt.plot(t, norm_ma)
    plt.show()
