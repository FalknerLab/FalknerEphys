import json
import os
import subprocess
import re

import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_dilation
from sklearn.cluster import AgglomerativeClustering
from brainrender import Scene
from brainrender.actors import Line, Points, Volume, Point
import matplotlib
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from falknerephys.plotting import density_3d


def make_tiff_stack(tiff_prefix, out_fold, data_folder, chans=None):
    if chans is None:
        chans = [0]
    all_ims = os.listdir(data_folder)
    all_ims = sorted(all_ims)
    for c in chans:
        chan_ims = []
        for im in all_ims:
            chan_num = re.findall("Filter\d\d\d\d", im)
            chan_name = re.findall("C00", im)
            if len(chan_num) == 1 and len(chan_name) == 1:
                chan_num = chan_num[0]
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
        vox_dims = [10, 5.91, 5.91]
    if atlas == 'auto':
        atlas = 'allen_mouse_25um'
    reg_tiff = os.path.join(out_dir, 'downsampled_standard.tiff')
    if not os.path.isfile(reg_tiff):
        br_command = f'brainreg {tiff_stack} {out_dir} -v {vox_dims[0]} {vox_dims[1]} {vox_dims[2]} --orientation {orientation} --atlas {atlas}'
        subprocess.run(br_command)
    else:
        print('Registered data found. Skipping Brainreg...')
    return reg_tiff


def segment_tracks(atlas_reg_tiff, shanks_thresh=500, poly_deg=4, shank_ord='PostAnt',
                   td_thresh=25, roi_pad=(10, 5), vox_sz=25, save_path=None, npx_chan_file=None):
    if type(roi_pad) == int:
        roi_pad = (roi_pad, roi_pad)

    tiff_vol = tiff.imread(atlas_reg_tiff)
    thresh_tiff = tiff_vol > shanks_thresh
    vol = density_3d(tiff_vol, thresh=shanks_thresh)[0]

    top_down = np.sum(tiff_vol > shanks_thresh, axis=1)
    shank_mask = top_down > td_thresh
    x_min = np.min(np.where(np.sum(shank_mask, axis=1))) - roi_pad[0]
    x_max = np.max(np.where(np.sum(shank_mask, axis=1))) + roi_pad[0]
    z_min = np.min(np.where(np.sum(shank_mask, axis=0))) - roi_pad[1]
    z_max = np.max(np.where(np.sum(shank_mask, axis=0))) + roi_pad[1]
    x_inds = np.logical_and(vol[:, 0] > x_min, vol[:, 0] < x_max)
    z_inds = np.logical_and(vol[:, 2] > z_min, vol[:, 2] < z_max)
    keep_inds = np.logical_and(x_inds, z_inds)
    vol = vol[keep_inds, :]
    most_vent = np.max(vol, axis=0)[1]

    box_pts = np.array([[x_min, 0, z_min],
                       [x_min, 0, z_max],
                       [x_max, 0, z_min],
                       [x_max, 0, z_max],
                        [x_min, most_vent, z_min],
                        [x_min, most_vent, z_max],
                        [x_max, most_vent, z_min],
                        [x_max, most_vent, z_max]])

    def dist_metric(*args):
        print(args)

    clus = AgglomerativeClustering(distance_threshold=2, n_clusters=None, linkage='single', metric=dist_metric).fit_predict(vol)
    c_id, counts = np.unique(clus, return_counts=True)
    shank_clus = c_id[np.argsort(counts)[-4:]]


    shank_tips = []
    shank_coefs = []
    tip_xs = []
    side_view = np.sum(thresh_tiff, axis=2) > 0
    side_view = binary_dilation(side_view, iterations=2)
    for ci, c in enumerate(shank_clus):
        x, y, z = vol[clus == c, 0], vol[clus == c, 1], vol[clus == c, 2]
        t = np.linspace(np.max(y)+20, np.min(y), 100) #fit line across DV
        x_poly = np.polyfit(y, x, poly_deg)
        z_poly = np.polyfit(y, z, poly_deg)
        fitx = np.polyval(x_poly, t)
        in_shank = side_view[fitx.astype(int), t.astype(int)]
        find_tip = np.where(in_shank)[0][0]
        shank_tips.append(t[find_tip])
        shank_coefs.append(np.vstack((x_poly, z_poly)))
        tip_xs.append(fitx[find_tip])

    shank_tips = np.array(shank_tips)
    shank_coefs = np.array(shank_coefs)
    shank_o = np.argsort(tip_xs).astype(int)
    if shank_ord == 'PostAnt':
        shank_o = shank_o[::-1]

    shank_tips = shank_tips[shank_o]
    shank_coefs = shank_coefs[shank_o]

    chan_xyz = np.array([])
    if npx_chan_file is not None:
        chan_data = json.load(open(npx_chan_file))
        chan_num = np.array(chan_data['chanMap'])
        shank_ids = np.array(chan_data['kcoords']).astype(int)
        depths = np.array(chan_data['yc'])
        chan_xyz = []
        for s in range(4):
            chan_depths = vox_sz*shank_tips[s] - depths[shank_ids == s]
            chan_xs = vox_sz*np.polyval(shank_coefs[s, 0, :], chan_depths/vox_sz)
            chan_zs = vox_sz*np.polyval(shank_coefs[s, 1, :], chan_depths/vox_sz)
            chan_xyz.append(np.vstack((chan_num[shank_ids == s], chan_xs, chan_depths, chan_zs)).T)
        chan_xyz = np.vstack(chan_xyz)

    if save_path is not None:
        save_path = os.path.join(save_path, 'shank_locations.npz')
        np.savez(save_path, tip_dvs=shank_tips, shank_coefs=shank_coefs, shank_volume=box_pts, sig_thresh=shanks_thresh, tiff_path=atlas_reg_tiff, vox_size=vox_sz, chan_ccf=chan_xyz)

    return save_path


def show_shank_tracks(shank_data_file, return_brain=False, brain=None, tiff_path=None, show_sig=True, show_lines=True,
                      show_label=True, chan_col='k', show_bounds=True):
    if brain is None:
        brain = Scene(atlas_name="allen_mouse_25um", title="Reconstructed Implant Locations")

    file_dict = np.load(shank_data_file)
    vox_sz = file_dict['vox_size']

    if show_bounds:
        brain.add(Points(vox_sz*file_dict['shank_volume'], colors='k'))

    cols = ['#44AA99', '#88CCEE', '#D0C590', '#CC6677']

    if show_lines:
        for dv, poly_cs, col in zip(file_dict['tip_dvs'], file_dict['shank_coefs'], cols):
            t = np.linspace(dv, 0, 100)
            fitx = np.polyval(poly_cs[0, :], t)
            fitz = np.polyval(poly_cs[1, :], t)
            tip = Point(vox_sz * np.array([fitx[0], t[0], fitz[0]]), color=col)
            brain.add(tip)
            brain.add(Line(vox_sz * np.vstack((fitx, t, fitz)).T, color=col))

    if show_sig:
        if tiff_path is None:
            tiff_path = file_dict['tiff_path']
        tiff_vol = tiff.imread(tiff_path)
        raw_vol = Volume(tiff_vol, 25, min_value=file_dict['sig_thresh'], cmap='gray')
        raw_vol.mesh.alpha(0.1)
        brain.add(raw_vol)

    if len(file_dict['chan_ccf']) > 0:
        chan_pts = Points(file_dict['chan_ccf'][:, 1:], colors=chan_col)
        brain.add(chan_pts)
        if show_label:
            brain.add_label(chan_pts, 'Channel Locations', radius=0, size=128, xoffset=-500, yoffset=500)

    if return_brain:
        return brain
    else:
        brain.render()


def register_probes(tiff_path, probe_json, out_path=None):
    if out_path is None:
        out_path = os.path.join(os.path.split(tiff_path)[0], 'brainreg')

    registered_tiff = register_brain(tiff_path, out_path)
    shank_data_file = segment_tracks(registered_tiff, save_path=out_path, npx_chan_file=probe_json)
    show_shank_tracks(shank_data_file)


def add_regions(*args, brain=None, colored=False, alpha=0.25):
    if brain is None:
        brain = Scene()

    if colored:
        cmap = matplotlib.colormaps['Accent']
        colors = cmap(np.linspace(0, 1, len(args)))
        cols_hex = [matplotlib.colors.to_hex(c) for c in colors]
    else:
        cols_hex = ['w' for i in range(len(args))]
        alpha = 0.1
    for region, col in zip(args, cols_hex):
        brain.add_brain_region(region, alpha=alpha, color=col)


def add_allen_data(allen_exp_id, brain=None, vox_sz=25, min_density=0.25):
    if brain is None:
        brain = Scene()

    # tell the cache class what resolution (in microns) of data you want to download
    mcc = MouseConnectivityCache(resolution=vox_sz)

    # download the projection density volume for one of the experiments
    pd = mcc.get_projection_density(allen_exp_id)

    brain.add(Volume(pd[0], voxel_size=vox_sz, min_value=min_density, cmap='viridis'))
    return brain
