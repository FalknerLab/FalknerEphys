import yaml
import numpy as np
from nptdms import TdmsFile
import h5py
from falknerephys.io.import_generic import load_phy
import matplotlib.pyplot as plt


def import_wm_data(phy_path, yaml_file=None, ephys_start=0, ephys_hz=25000):
    offset = 0
    if yaml_file is not None:
        yaml_file = open(yaml_file)
        start_times = yaml.safe_load(yaml_file)
        offset_s = start_times['motif_start'] - start_times['wm_start']
        offset = round(offset_s * ephys_hz)
    ephys_data = load_phy(phy_path, offset=offset + ephys_start)
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


def tdms_to_h5(tdms_path, h5_name, out_len_s=None, v_thresh=3):
    print('Converting tdms to h5...')
    tdms_file = TdmsFile(tdms_path)
    fs = tdms_file['Analog'].properties['ScanRate']
    m_data = tdms_file['Analog']['AI1'].data
    wm_start = tdms_file['Analog']['AI4'].data
    wm_sync = tdms_file['Analog']['AI2'].data
    stim = tdms_file['Analog']['AI5'].data
    num_inds = int(out_len_s * fs)
    h5_file = h5py.File(h5_name, 'w')
    names = ['wm_sync', 'wm_start', 'video_start', 'stim']
    for n, data in zip(names, [wm_sync, wm_start, m_data, stim]):
        bin_data = data > v_thresh
        if out_len_s is not None:
            h5_file[n] = bin_data[:num_inds]
        else:
            h5_file[n] = bin_data
    h5_file['ScanRate'] = [fs]
    h5_file.close()


def show_tdms(tdms_path, ax=None):
    if ax is None:
        ax = plt.gca()
    print('Loading and plotting TDMS file...')
    tdms_file = TdmsFile(tdms_path)
    fs = tdms_file['Analog'].properties['ScanRate']
    m_data = tdms_file['Analog']['AI1'].data
    wm_start = tdms_file['Analog']['AI4'].data
    wm_sync = tdms_file['Analog']['AI2'].data
    stim_start = tdms_file['Analog']['AI5'].data
    times = np.linspace(0, len(m_data) / fs, len(m_data))
    labs = ['Ephys Sync', 'Ephys Start', 'Video Start', 'Stimuli']
    for i, t in enumerate([wm_sync, wm_start, m_data, stim_start]):
        norm_t = (t - min(t)) / (max(t) - min(t))
        ax.plot(times, i + norm_t, label=labs[i])


def show_h5(h5_path, ax=None):
    if ax is None:
        ax = plt.gca()
    h5_file = h5py.File(h5_path, 'r')
    fs = h5_file['ScanRate'][0]
    names = ['wm_sync', 'wm_start', 'video_start', 'stim']
    labs = ['Ephys Sync', 'Ephys Start', 'Video Start', 'Stimuli']
    for i, k in enumerate(names):
        t = h5_file[k][:].astype(int)
        times = np.linspace(0, len(t) / fs, len(t))
        ax.plot(times, i + (0.1 * i) + t, label=labs[i])
    ax.set_title('DAQ TTLs')
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right')

