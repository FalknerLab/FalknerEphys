import yaml
import numpy as np
from nptdms import TdmsFile
from import_generic import load_phy


def import_wm_data(phy_path, yaml_file=None, ephys_start=0, ephys_hz=25000):
    offset = 0
    if yaml_file is not None:
        yaml_file = open(yaml_file)
        start_times = yaml.safe_load(yaml_file)
        offset_s = start_times['motif_start'] - start_times['wm_start']
        offset = round(offset_s*ephys_hz)
    ephys_data = load_phy(phy_path, offset=offset+ephys_start)
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
