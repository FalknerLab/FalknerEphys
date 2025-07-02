import os

import numpy as np


def find_file_recur(root_fold, target_suf):
    """
    Recursively finds a file with the given suffix in the root folder.

    Parameters
    ----------
    root_fold : str
        Root folder to search.
    target_suf : str
        Suffix of the target file.

    Returns
    -------
    str or None
        Path to the found file, or None if not found.
    """
    files = os.listdir(root_fold)
    found_file = None
    for f in files:
        this_dir = os.path.join(root_fold, f)
        if os.path.isdir(this_dir):
            recur_has = find_file_recur(this_dir, target_suf)
            if recur_has is not None:
                found_file = recur_has
        elif f.split('_')[-1] == target_suf:
            found_file = this_dir
    return found_file

def find_files(root_dir: str, targets=None):
    """
    Finds all territory files in the given root directory.

    Parameters
    ----------
    root_dir : str
        Root directory to search.

    Returns
    -------
    dict
        Dictionary containing paths to the found files.
    """
    if targets is None:
        targets = ['spike_clusters.npy', 'spike_templates.npy', 'spike_times.npy', 'metadata.yml']
    out_paths = {s: None for s in targets}
    out_paths['root'] = root_dir
    for t in targets:
        found_file = find_file_recur(root_dir, t)
        if found_file is not None:
            out_paths[t] = found_file
    return out_paths


def compute_over_2d_bins(x_data, y_data, data, func, bins=20, range=None):
    _, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, range=range)
    out_hist = np.zeros((len(xedges) - 1, len(yedges) - 1))
    for i, xe in enumerate(xedges[:-1]):
        in_x = np.logical_and(x_data > xe, x_data < xedges[i + 1])
        for j, ye in enumerate(yedges[:-1]):
            in_y = np.logical_and(y_data > ye, y_data < yedges[j + 1])
            in_bin = np.logical_and(in_x, in_y)
            out_hist[i, j] = func(data[in_bin])
    return out_hist, xedges, yedges


def compute_over_1d_bins(x_data, data, func, bins=20, range=None):
    _, xedges = np.histogram(x_data, bins=bins, range=range)
    out_hist = np.zeros(len(xedges) - 1)
    for i, xe in enumerate(xedges[:-1]):
        in_x = np.logical_and(x_data > xe, x_data < xedges[i + 1])
        out_hist[i] = func(data[in_x])
    return out_hist, xedges
