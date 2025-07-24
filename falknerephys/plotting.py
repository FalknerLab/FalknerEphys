import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from debugpy.common.log import warning
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


def jitter_plot(spk_s, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(spk_s, 2*np.ones_like(spk_s) + np.random.uniform(low=-1, size=len(spk_s)), s=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])


def fr_heatmap(unit_fr, ax=None, unit_ids=None, hz=None, x_tick_s=10, fr_min=0, fr_max=25, **kwargs):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(unit_fr.T, aspect='auto', interpolation='none', vmin=fr_min, vmax=fr_max, **kwargs)
    num_samps, num_us = np.shape(unit_fr)
    if hz is not None:
        x_tick = np.round(np.arange(0, num_samps, x_tick_s*hz))
        x_tick_labels = x_tick / hz
        ax.set_xticks(x_tick)
        ax.set_xticklabels(x_tick_labels)
    if unit_ids is not None:
        ax.set_yticks(np.arange(num_us))
        ax.set_yticklabels(unit_ids)
    ax.set_ylabel('Unit ID')
    ax.set_xlabel('Time (s)')
    return ax, im


def fr_per_xy(ax, spk_s, x, y, num_bins=30, xy_range=None, xy_hz=40, fr_min=0, fr_max=25):
    ## Count indices in spks based on binned 2D locations using x and y
    spks = np.round(spk_s * xy_hz).astype(int)
    if xy_range is None:
        xy_range = np.array([[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]])

    total_xy, _, _ = np.histogram2d(x, y, bins=num_bins, range=xy_range)
    spks_xy, _, _ = np.histogram2d(x[spks], y[spks], bins=num_bins, range=xy_range)
    total_xy[total_xy == 0] = np.nan
    norm_fr = spks_xy / (total_xy/40)
    im = ax.imshow(norm_fr.T, extent=(xy_range[0, 0], xy_range[0, 1], xy_range[1, 0], xy_range[1, 1]),
              origin='lower', aspect='auto', interpolation='none', vmin=fr_min, vmax=fr_max)
    return im


def psth_per_unit(data, behav):
    units = data['units']
    num_u = np.shape(units)[1]
    f, axs = plt.subplots(num_u, 1)
    psth_acc = []
    xs = []
    for u in range(num_u):
        x, psth = make_psth(units[:, u], data[behav])
        psth_acc.append(psth)
        xs = x
    psth_npy = np.array(psth_acc)
    dists = pdist(psth_npy[:, 1:])
    Z = hierarchy.complete(dists)
    reorder = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, dists))
    reorder = np.arange(np.shape(psth_npy)[0])
    for ind, p in enumerate(reorder):
        axs[ind].bar(xs, psth_acc[p], color='k')
        axs[ind].set_title('Unit: ' + data['unit_ids'][p], x=1.05, y=0.5)
    axs[num_u - 1].set_xlabel(behav)
    axs[num_u - 1].set_ylabel('Mean FR (Hz)')


def plot_units_behav(units, behav):
    num_u = np.shape(units)[1]
    f, axs = plt.subplots(num_u)
    for u in range(num_u):
        axs[u].plot(units[:, u])
        ax2 = axs[u].twinx()
        ax2.plot(behav, color='r')


def plot_units(us, num_plot_per_fig=10):
    num_u = np.shape(us)[1]
    num_f = np.ceil(num_u/num_plot_per_fig)
    fig, axs = plt.subplots(num_plot_per_fig, 1)
    c = 0
    for u in us.T:
        if c >= num_plot_per_fig:
            _, axs = plt.subplots(num_plot_per_fig, 1)
            c = 0
        else:
            axs[c].plot(u)
            c += 1


def spikes_vs_speed(spikes, vel):
    spike_height = 1.1*max(vel)
    max_ind = len(vel)
    max_ind = 25000
    good_inds = np.where(np.logical_and(spikes > 0, spikes < max_ind))
    good_spikes = spikes[good_inds]
    plt.plot(vel[:max_ind], zorder=0)
    plt.scatter(good_spikes, spike_height * np.ones(len(good_spikes)), c='r', s=1)


def make_psth(unit_fr, x_data, bin_sz=80):
    non_nan = np.where(np.array(x_data) >= 0)[0]
    fr_no_nan = unit_fr[non_nan]
    x_data_no_nan = x_data[non_nan]
    _, bin_edges = np.histogram(x_data_no_nan, bins=bin_sz)
    bin_id = np.digitize(x_data_no_nan, bins=bin_edges)
    psth = []
    for b in range(max(bin_id)):
        psth.append(np.mean(fr_no_nan[bin_id == b]))
    return bin_edges, psth


def set_labels(title, xlabel, ylabel, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def density_3d(data_3d, bin_n=15, thresh=0, ax=None):

    if data_3d.ndim == 3:
        hist_3d = data_3d
        x_edges = np.linspace(0, data_3d.shape[0], data_3d.shape[0])
        y_edges = np.linspace(0, data_3d.shape[1], data_3d.shape[1])
        z_edges = np.linspace(0, data_3d.shape[2], data_3d.shape[2])
        bin_x, bin_y, bin_z = data_3d.shape
    elif data_3d.ndim == 2:
        hist_3d, all_edges = np.histogramdd(data_3d, bins=bin_n, density=True)
        bin_x, bin_y, bin_z = bin_n, bin_n, bin_n
        x_edges, y_edges, z_edges = all_edges[:]
    else:
        warning(f'Wrong number of dimensions for desnity plot. Got {data_3d.ndim}, need 2 or 3')
        return None

    filled = np.argwhere(hist_3d)

    c_map = np.zeros(len(filled))
    for i, p in enumerate(filled):
        c_map[i] = hist_3d[p[0], p[1], p[2]]

    high_dens = c_map > thresh

    filled = filled.astype(np.float32)

    filled = filled[high_dens, :]
    c_map = c_map[high_dens]

    half_x = (x_edges[1] - x_edges[0]) / 2
    half_y = (y_edges[1] - y_edges[0]) / 2
    half_z = (z_edges[1] - z_edges[0]) / 2
    filled[:, 0] = filled[:, 0] * (x_edges[-1] - x_edges[0]) / bin_x + x_edges[0] + half_x
    filled[:, 1] = filled[:, 1] * (y_edges[-1] - y_edges[0]) / bin_y + y_edges[0] + half_y
    filled[:, 2] = filled[:, 2] * (z_edges[-1] - z_edges[0]) / bin_z + z_edges[0] + half_z

    if ax is not None:
        ax.scatter(filled[:, 0], filled[:, 1], filled[:, 2], c=c_map, marker='s', s=3, alpha=0.005)

    return filled, c_map


def venn2(Ab, aB, AB, col0=(1, 0.3, 0.2, 0.5), col1=(0.2, 0.7, 0.5, 0.5), ax=None, labels=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    circ0_r = (Ab + AB) / 2
    circ1_r = (aB + AB) / 2
    circ1_x = circ0_r + circ1_r - AB

    ax.add_patch(Circle((0, 0), circ0_r, facecolor=col0))
    ax.add_patch(Circle((circ1_x, 0), circ1_r, facecolor=col1))

    ax.set_xlim(-1.1*circ0_r, 1.1*(circ1_x+circ1_r))
    ax.set_ylim(-1.1 * max(circ0_r, circ1_r), 1.1 * max(circ0_r, circ1_r))

    txt_x_A = 0 - circ0_r
    txt_x_B = circ1_x + circ1_r
    txt_x_AB = np.mean([circ1_x - circ1_r, circ0_r])
    labels_n = [Ab + AB, aB + AB, AB]
    ha_align = ['left', 'right', 'center']
    y_pos = [0, 0, -np.min([circ0_r, circ1_r]) / 2]
    for x, n, h, y in zip((txt_x_A, txt_x_B, txt_x_AB), labels_n, ha_align, y_pos):
        ax.text(x, y, f'{n}', ha=h)

    if labels is not None:
        ax.text(0-circ0_r, circ0_r, labels[0], ha='center', color=col0)
        ax.text(circ1_x + circ1_r, circ0_r, labels[1], ha='center', color=col1)

    # if labels is not None:
    #     l0 = ax.scatter(0, 10*max(circ0_r, circ1_r), 10, [col0])
    #     l1 = ax.scatter(0, 10*max(circ0_r, circ1_r), 10, [col1])
    #     l2 = ax.scatter(0, 10 * max(circ0_r, circ1_r), 10, [np.mean(np.vstack((col0, col1)), axis=0)])
    #     labels_n = (f'{labels[0]} n={Ab + AB}', f'{labels[1]} n={aB + AB}', f'Overlap n={AB}')
    #     plt.legend([l0, l1, l2], labels_n, loc='upper right')

    ax.set_axis_off()