import os

import numpy as np
from kilosort.io import save_probe

from falknerephys.io.spikesort import run_ks


def read_onebox_bin(bin_file, num_chans=3, bytes_per_samp=2):
    print(f'Loading OneBox data from: {bin_file}')

    file_parts = bin_file.split('.')
    out_file = file_parts[0] + '_obx.npy'
    meta_file = ''.join(('.'.join(file_parts[:-1]), '.meta'))
    meta_data = np.loadtxt(meta_file, delimiter='=', dtype=str)
    ob_samp_rate = int(meta_data[meta_data[:, 0] == 'obSampRate', 1][0])
    max_int = int(meta_data[meta_data[:, 0] == 'obMaxInt', 1][0])
    max_v = float(meta_data[meta_data[:, 0] == 'obAiRangeMax', 1][0])
    min_v = float(meta_data[meta_data[:, 0] == 'obAiRangeMin', 1][0])

    if os.path.isfile(out_file):
        print('Found Obx Data...')
        out_data = np.load(out_file)
        time_vec = np.linspace(0, len(out_data)/ob_samp_rate, len(out_data))
        return time_vec, out_data

    v_range = (max_v - min_v)
    v_cent = min_v + v_range // 2
    conv_fac = v_range / max_int

    these_reads = []
    with open(bin_file, mode="rb") as f:
        chunk = f.read()
        for i in range(len(chunk)//6):
            for c in range(num_chans):
                this_samp = int.from_bytes(chunk[(6*i+2*c):((6*i+2*c)+2)], byteorder='little', signed=True)
                these_reads.append(conv_fac*this_samp + v_cent)
    chan_data = []
    for i in range(num_chans):
        chan_inds = np.arange(i, len(these_reads), num_chans)
        read_ar = np.array(these_reads)[chan_inds]
        chan_data.append(read_ar)
    out_data = np.array(chan_data).T
    time_vec = np.linspace(0, len(out_data)/ob_samp_rate, len(out_data))

    np.save(out_file, out_data)
    return time_vec, out_data


def find_imec_files(root_fold, target_suf):
    files = os.listdir(root_fold)
    found_file = None
    for f in files:
        this_dir = os.path.join(root_fold, f)
        if os.path.isdir(this_dir):
            recur_has = find_imec_files(this_dir, target_suf)
            if recur_has is not None:
                found_file = recur_has
        elif f.split('.')[-2] == target_suf and f.split('.')[-1] == 'bin':
            found_file = this_dir
    return found_file


def process_imec_data(root_dir):
    ap_bin = find_imec_files(root_dir, 'ap')
    ob_bin = find_imec_files(root_dir, 'obx')
    time_vec, ob_data = read_onebox_bin(ob_bin)
    phy_path = run_ks(ap_bin)
    return phy_path, ob_data


def make_probe_from_imro(imro_file, tl_micron=175):
    imro_data = np.loadtxt(imro_file, delimiter=')', dtype=str)
    imro_data = [d[1:] for d in imro_data]
    probe_id, num_chans = imro_data[0].split(',')
    n_chan = int(num_chans)
    imro_data = imro_data[1:-1]
    out_table = np.empty((0, 5))
    for x in imro_data:
        new_x = np.fromstring(x, dtype=int, sep=' ')[None, :]
        out_table = np.vstack((out_table, new_x))

    ss_dist = 250
    ee_dist = 15 # micron distance between sites
    chanMap = []
    xc = []
    yc = []
    kcoords = []
    for c in out_table:
        c_num, s_num, s_elec = c[0], c[1], c[4]
        x_pos = ss_dist*s_num
        if s_elec % 2 == 1:
            x_pos += 2*ee_dist
        xc.append(x_pos)
        y_pos = tl_micron + ee_dist*(s_elec//2)
        yc.append(y_pos)
        kcoords.append(s_num)
        chanMap.append(c_num)

    chanMap = np.array(chanMap)
    xc = np.array(xc)
    yc = np.array(yc)
    kcoords = np.array(kcoords)
    probe = {
        'chanMap': chanMap,
        'xc': xc,
        'yc': yc,
        'kcoords': kcoords,
        'n_chan': n_chan
    }
    new_file = imro_file.replace('imro', 'json')
    save_probe(probe, new_file)
    return new_file


def get_imec_metadata(md_path, md_target):
    md = np.loadtxt(md_path, delimiter='=', dtype=str)
    t_ind = np.where(md[:, 0] == md_target)[0][0]
    return md[t_ind, 1]
