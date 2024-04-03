import yaml
import numpy as np
from nptdms import TdmsFile


def import_ephys_data(prefix, yaml_file=None, start_frame=0):
    ephys_hz = 25000
    offset = 0
    if yaml_file is not None:
        yaml_file = open(yaml_file)
        start_times = yaml.safe_load(yaml_file)
        offset_s = start_times['motif_start'] - start_times['wm_start']
        offset = round(offset_s*ephys_hz)
    spks = np.load(prefix + '/spike_times.npy')
    spk_ids = np.load(prefix + '/spike_clusters.npy')
    good_units = np.loadtxt(prefix + '/cluster_group.tsv', delimiter='\t', skiprows=1, dtype=str)
    keep_clus = []
    for i in range(np.shape(good_units)[0]):
        if good_units[i, 1] == 'good':
            keep_clus.append(good_units[i, 0].astype(int))
    ephys_data = dict()
    for c in keep_clus:
        inds = np.where(spk_ids == c)
        g_spks = spks[inds]
        rel_spk_ts = g_spks.squeeze().astype(int) - (offset + start_frame)
        ephys_data[str(c)] = rel_spk_ts[rel_spk_ts > 0]
    return ephys_data


def tdms_to_yaml(tdms_file):
    file_prt = tdms_file.split('_')
    t_file = TdmsFile(tdms_file)
    fs = t_file['Analog'].properties['ScanRate']
    m_start = np.where(t_file['Analog']['AI1'].data > 3)[0][0] / fs
    wm_start = np.where(t_file['Analog']['AI4'].data > 3)[0][0] / fs
    out_dict = {'wm_start': float(wm_start),
                'motif_start': float(m_start)}
    out_file = '_'.join(file_prt[:4]) + '.yaml'
    file = open(out_file, "w")
    yaml.dump(out_dict, file)
    file.close()
    return out_file
