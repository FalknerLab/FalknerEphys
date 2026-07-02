import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from debugpy.common.log import warning
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def fempl_style():
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['grid.linestyle'] = ':'
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['axes3d.xaxis.panecolor'] = (1.0, 1.0, 1.0, 0.0)
    plt.rcParams['axes3d.yaxis.panecolor'] = (1.0, 1.0, 1.0, 0.0)
    plt.rcParams['axes3d.zaxis.panecolor'] = (1.0, 1.0, 1.0, 0.0)
    plt.rcParams['axes.grid'] = False


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


def fr_per_xy(ax, spk_s, x, y, num_bins=30, xy_range=None, xy_hz=40, fr_min=0, fr_max=0.1):
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


def ternary(vals, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    vals = np.array(vals)
    if vals.ndim == 1:
        vals = vals[np.newaxis, :]

    for v in vals:
        print(vals)


def plot_waveforms(wf_mat, chan_pos, ax=None, col='k'):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    norm_wf = wf_mat / np.max(np.max(np.abs(wf_mat)))
    x = np.linspace(chan_pos[:, 0], chan_pos[:, 0]+75, len(wf_mat))
    y = chan_pos[:, 1] + 10*norm_wf
    good_ts = np.where(np.min(norm_wf, axis=0) < -0.35)[0]
    ax.plot(x[:, good_ts], y[:, good_ts], color=col)


def plot_glm(feature_weights, model_r2s, labels=None, num_gmm_comp=-1, r2_thresh=0.0, max_c=15, sort_method='gmm'):
    feature_weights = feature_weights[model_r2s > r2_thresh, :]
    model_r2s = model_r2s[model_r2s > r2_thresh]
    bics = []
    clus_num = np.zeros_like(model_r2s)
    if sort_method == 'gmm':
        if num_gmm_comp < 0:
            for c in range(1, max_c):
                test_gmm = GaussianMixture(n_components=c)
                test_gmm.fit(feature_weights)
                bics.append(test_gmm.bic(feature_weights))
            num_gmm_comp = np.argmin(np.array(bics)) + 1
        clus_num = GaussianMixture(n_components=num_gmm_comp).fit_predict(feature_weights)
    if sort_method == 'max':
        clus_num = np.argmax(feature_weights, axis=1)
    if sort_method == 'agg':
        clus_num = AgglomerativeClustering(num_gmm_comp).fit_predict(feature_weights)
    sort_ord = np.argsort(clus_num)
    f, ax = plt.subplots(1, 2)
    ax[0].pcolor(feature_weights[sort_ord, :])
    if labels is not None:
        ax[0].set_xticks(np.arange(len(labels)) + 0.5, labels)
        ax[0].tick_params(axis='x', labelrotation=90)
    ax[0].set_title(f'Feature Weights n_clusters = {num_gmm_comp}')
    ax[1].stem(model_r2s[sort_ord], orientation='horizontal')
    ax[1].set_ylim(-0.5, len(model_r2s) - 0.5)

    if sort_method == 'gmm':
        plt.figure()
        plt.plot(np.arange(1, max_c), bics, 'ko--')

    return clus_num, sort_ord


def plot_confusion_matrix(conf_mat, class_names=None, axs=None, cmap='OrRd', vmin=0, vmax=1):
    if axs is None:
        f, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]})


    cm_im = axs[0].imshow(conf_mat, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    if class_names is not None:
        axs[0].set_yticks(np.arange(len(class_names)), class_names)
        axs[0].set_xticks(np.arange(len(class_names)), class_names)

    plt.colorbar(mappable=cm_im, cax=axs[1], label='Recall')

    return cm_im