import json
import os
import subprocess
import re

import numpy as np
import tifffile as tiff
from sklearn.cluster import AgglomerativeClustering
from brainrender import Scene
from brainrender.actors import Line, Points

from falknerephys.plotting import density_3d


def make_tiff_stack(tiff_prefix, out_fold, data_folder, chans=None):
    if chans is None:
        chans = [0]
    all_ims = os.listdir(data_folder)
    for c in chans:
        chan_ims = []
        for im in all_ims:
            chan_num = re.findall("Filter\d\d\d\d", im)[0]
            chan_num = int(chan_num.replace("Filter", ""))
            if chan_num == c:
                chan_ims.append(im)

        im0 = tiff.imread(os.path.join(data_folder, chan_ims[0]), is_ome=False)
        out_tiff = np.zeros((len(chan_ims), 1, im0.shape[0], im0.shape[1]), dtype=im0.dtype)
        for i, cim in enumerate(chan_ims):
            print(f'Chan {c}, Image {cim}')
            base_name = os.path.join(data_folder, cim)
            ch0 = tiff.imread(base_name, is_ome=False)
            out_tiff[i, :, :, :] = ch0[None, None, :, :]
        out_name = os.path.join(out_fold, tiff_prefix + f"_chan{c}.tiff")
        tiff.imwrite(out_name, out_tiff, dtype=im0.dtype, imagej=True)


def register_brain(tiff_stack, out_dir, vox_dims=None, orientation='sal', atlas='auto'):
    if vox_dims is None:
        vox_dims = [5.91, 5.91, 10]
    if atlas == 'auto':
        atlas = 'allen_mouse_25um'
    br_command = f'brainreg {tiff_stack} {out_dir} -v {vox_dims[0]} {vox_dims[1]} {vox_dims[2]} --orientation {orientation} --atlas {atlas}'
    print(br_command)
    subprocess.run(br_command)


def segment_tracks(atlas_reg_tiff, sig_thresh=350, poly_deg=3, find_roi=True, td_thresh=25, roi_pad=0, vox_sz=25, save_path=None):

    tiff_vol = tiff.imread(atlas_reg_tiff)

    vol = density_3d(tiff_vol, thresh=sig_thresh)[0]

    if find_roi:
        top_down = np.sum(tiff_vol > sig_thresh, axis=1)
        shank_mask = top_down > td_thresh
        x_min = np.min(np.where(np.sum(shank_mask, axis=0))) - roi_pad
        x_max = np.max(np.where(np.sum(shank_mask, axis=0))) + roi_pad
        y_min = np.min(np.where(np.sum(shank_mask, axis=1))) - roi_pad
        y_max = np.max(np.where(np.sum(shank_mask, axis=1))) + roi_pad
        x_inds = np.logical_and(vol[:, 2] > x_min, vol[:, 2] < x_max)
        y_inds = np.logical_and(vol[:, 0] > y_min, vol[:, 0] < y_max)
        keep_inds = np.logical_and(x_inds, y_inds)
        vol = vol[keep_inds, :]

    clus = AgglomerativeClustering(distance_threshold=2, n_clusters=None, linkage='single').fit_predict(vol)
    c_id, counts = np.unique(clus, return_counts=True)
    shank_clus = c_id[np.argsort(counts)[-4:]]

    brain = Scene(atlas_name="allen_mouse_25um", title="Reconstructed Implant Locations")

    cols = ['#44AA99', '#88CCEE', '#D0C590', '#CC6677']
    shank_tips = []
    shank_coefs = []
    for ci, c in enumerate(shank_clus):
        x, y, z = vox_sz*vol[clus == c, 0], vox_sz*vol[clus == c, 1], vox_sz*vol[clus == c, 2]
        t = np.linspace(np.max(y), np.min(y), 100) #fit line across DV
        x_poly = np.polyfit(y, x, poly_deg)
        z_poly = np.polyfit(y, z, poly_deg)
        fitx = np.polyval(x_poly, t)
        fitz = np.polyval(z_poly, t)
        shk_pts = Points(vox_sz*vol[clus == c, :], colors=cols[ci], alpha=0.2)
        brain.add(shk_pts)
        brain.add(Line(np.vstack((fitx, t, fitz)).T, color='k'))
        shank_tips.append(np.max(y))
        shank_coefs.append(np.vstack((x_poly, z_poly)))

    if save_path is not None:
        np.savez(os.path.join(save_path, 'shank_locations.npz'), shank_tips, shank_coefs)

    brain.render()


def map_channels_brain(npx_chan_file, shank_data):
    shank_tips = np.load(shank_data)['arr_0']
    shank_coefs = np.load(shank_data)['arr_1']
    chan_data = json.load(open(npx_chan_file))
    shank_ids = np.array(chan_data['kcoords']).astype(int)
    depths = np.array(chan_data['yc'])
    brain = Scene(atlas_name="allen_mouse_25um", title="Reconstructed Channel Locations")
    for s in range(4):
        chan_depths =  shank_tips[s] - depths[shank_ids == s]
        chan_xs = np.polyval(shank_coefs[s, 0, :], chan_depths)
        chan_zs = np.polyval(shank_coefs[s, 1, :], chan_depths)
        brain.add(Points(np.vstack((chan_xs, chan_depths, chan_zs)).T))

    regions = ['LSc', 'LSr', 'LSv']
    for r in regions:
        brain.add_brain_region(r, alpha=0.2)

    brain.render()
