import numpy as np
import os


def load_phy(phy_path, offset=0, ephys_hz=25000):
    """

    Parameters
    ----------
    phy_path : str
        Directory containing all Phy output files
    offset : int, optional
        Number of samples (based on ephys sampling rate) to be subtracted from spike times
    ephys_hz: int, optional
        Recording frequency of ephys data

    Returns
    -------
    A directory containing all the 'good' units where each key is the ID of the unit and the value is its spikes

    """

    #Load spike data from Phy folder
    spks = np.load(phy_path + '/spike_times.npy')
    spk_ids = np.load(phy_path + '/spike_clusters.npy')
    good_units = np.loadtxt(phy_path + '/cluster_group.tsv', delimiter='\t', skiprows=1, dtype=str)

    #Only keep the ones labeled good from the tsv
    keep_clus = []
    for i in range(np.shape(good_units)[0]):
        if good_units[i, 1] == 'good':
            keep_clus.append(good_units[i, 0].astype(int))

    #Make dictionary from good units
    ephys_data = dict()
    for c in keep_clus:
        inds = np.where(spk_ids == c)
        g_spks = spks[inds]
        # compute relative spike times based on offset
        rel_spk_ts = g_spks.squeeze().astype(int) - offset
        # ignore spikes before offset and convert to seconds
        ephys_data[str(c)] = rel_spk_ts[rel_spk_ts > 0] / ephys_hz
    return ephys_data
