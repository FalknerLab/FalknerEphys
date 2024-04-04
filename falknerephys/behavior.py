import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


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