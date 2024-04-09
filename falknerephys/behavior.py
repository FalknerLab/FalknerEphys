import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import h5py


def xy_to_cm(xy, center_pt, px_per_cm):
    rel_xy = xy - center_pt
    rel_xy[:, 1] = -rel_xy[:, 1]
    cm_x = rel_xy[:, 0] / px_per_cm
    cm_y = rel_xy[:, 1] / px_per_cm
    return cm_x, cm_y


def get_sleap_data(slp_h5, origin_px, px_per_cm, hz=40):
    slp_data = h5py.File(slp_h5, 'r')
    mouse_data = slp_data['tracks'][0]
    mouse_centroid = np.nanmean(mouse_data, axis=1).T
    cm_x, cm_y = xy_to_cm(mouse_centroid, origin_px, px_per_cm)
    rel_xy = np.vstack((cm_x, cm_y)).T
    dist_vec = np.linalg.norm(rel_xy[1:, :] - rel_xy[:-1, :], axis=1) / px_per_cm
    vel_vec = np.hstack(([0], dist_vec))*hz
    time_vec = np.arange(len(cm_x)) / hz
    return time_vec, cm_x, cm_y, vel_vec


def behavioral_raster(rasters, ax=plt.gca, fs=1):
    cols = ['tab:green', 'tab:orange', 'tab:gray', 'c', 'm', 'r', 'k']
    len_behav, num_rasts = np.shape(rasters)
    rast_im = 255*np.ones((1, len_behav, 3))
    for i in range(num_rasts):
        col = to_rgb(cols[i])
        inds = np.where(rasters[:, i])[0]
        rast_im[:, inds, :] = col
    ax.imshow(rast_im, aspect='auto', interpolation=None)
    ax.set_xlim([0, len_behav])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    x_ticks = ax.get_xticks()
    x_ticks = x_ticks[x_ticks < len_behav]
    ax.set_xticks(x_ticks, labels=x_ticks/fs)
    ax.get_yaxis().set_visible(False)