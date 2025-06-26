import numpy as np
import os


def load_phy(phy_path, offset_s=0, ephys_hz=30000, return_table=False):
    """

    Parameters
    ----------
    phy_path : str
        Directory containing all Phy output files
    offset_s : float, optional
        Number of seconds to be subtracted from spike times
    ephys_hz: int, optional
        Recording frequency of ephys data

    Returns
    -------
    A directory containing all the 'good' units where each key is the ID of the unit and the value is its spikes

    """

    #Load spike data from Phy folder
    spks = np.load(os.path.join(phy_path, 'spike_times.npy'))
    spk_ids = np.load(os.path.join(phy_path,  'spike_clusters.npy'))
    good_units = np.loadtxt(os.path.join(phy_path, 'cluster_group.tsv'), delimiter='\t', skiprows=1, dtype=str)
    clus_info = np.loadtxt(os.path.join(phy_path, 'cluster_info.tsv'), delimiter='\t', skiprows=1, dtype=str)

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
        rel_spk_ts = g_spks.squeeze().astype(int)
        # ignore spikes before offset and convert to seconds
        ephys_data[str(c)] = (rel_spk_ts[rel_spk_ts > 0] / ephys_hz) - offset_s
    good_info = None
    if np.all(clus_info[keep_clus, 0].astype(int) == keep_clus):
        good_info = clus_info[keep_clus, :]

    if return_table:
        return ephys_data, good_info
    else:
        depths = good_info[:, 6].astype(float)
        shanks = good_info[:, 10].astype(float).astype(int)
        amps = good_info[:, 1].astype(float)
        return ephys_data, amps, depths, shanks
