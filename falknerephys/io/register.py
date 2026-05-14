import json
import os
import subprocess
import re

import cv2
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_dilation
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from brainrender import Scene
from brainrender.actors import Line, Points, Volume, Point
import matplotlib
import matplotlib.pyplot as plt
# from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import networkx as nx
from networkx.algorithms.approximation.traveling_salesman import traveling_salesman_problem
import requests
import gzip


def make_tiff_stack(root_fold, out_fold=None, tiff_prefix='braintiff', chans=None, scope='lv'):
    if out_fold is None:
        out_fold = root_fold
    match scope:
        case 'lv':
            make_tiff_lv(tiff_prefix, out_fold, root_fold, chans)
        case 'ss':
            make_tiff_ss(root_fold)


def make_tiff_ss(tiff_prefix, root_fold, out_fold, scale_fac=0.5):
    w, h = 0, 0
    im_files = os.listdir(root_fold)
    num_ims = len(im_files)
    out_tiff = None
    this_im = None
    for i, f in enumerate(im_files):
        print(i)
        this_im = tiff.imread(os.path.join(root_fold, f), is_ome=False)
        if i == 0:
            w, h = this_im.shape
            if scale_fac != 1:
                w, h = int(w * scale_fac), int(h * scale_fac)
            out_tiff = np.zeros((num_ims, 1, int(w * scale_fac), int(h * scale_fac)), dtype=this_im.dtype)
        if scale_fac != 1:
            rs_im = this_im.copy()
            rs_im = cv2.resize(rs_im, (int(h * scale_fac), int(w * scale_fac)))
            this_im = rs_im
        out_tiff[i, :, :, :] = this_im[None, None, :, :]
    out_name = os.path.join(out_fold, tiff_prefix + ".tiff")
    tiff.imwrite(out_name, out_tiff, dtype=this_im.dtype, imagej=True)


def make_tiff_lv(tiff_prefix, data_folder, out_fold, chans=None):
    if chans is None:
        chans = [0]
    all_ims = os.listdir(data_folder)
    all_ims = sorted(all_ims)
    for c in chans:
        chan_ims = []
        for im in all_ims:
            chan_num = re.findall(r"Filter\d\d\d\d", im)
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


def register_brain(tiff_stack, out_dir, vox_dims=None, orientation='sal', atlas='auto', bg_stack=None):
    if vox_dims is None:
        vox_dims = [10, 5.91, 5.91]
    if atlas == 'auto':
        atlas = 'allen_mouse_25um'
    reg_tiff = os.path.join(out_dir, 'downsampled_standard.tiff')
    br_command = f'brainreg {tiff_stack} {out_dir} -v {vox_dims[0]} {vox_dims[1]} {vox_dims[2]} --orientation {orientation} --atlas {atlas}'
    if os.path.isfile(reg_tiff):
        print('Registered data found. Skipping Brainreg...')
    else:
        if bg_stack is not None:
            br_command += f' --pre-processing skip --freeform-n-steps 2 --freeform-use-n-steps 1'
        subprocess.run(br_command)
    return reg_tiff


def auto_segment_tracks(atlas_reg_tiff, samp_pt_scale=10, shanks_thresh=350, poly_deg=1, shank_ord='PostAnt', vox_sz=25,
                    save_path=None, npx_chan_file=None, shank_box=25):


    tiff_vol = tiff.imread(atlas_reg_tiff)
    thresh_tiff = tiff_vol > shanks_thresh
    topdown = np.sum(thresh_tiff, axis=1)
    my = np.arange(topdown.shape[0]) @ np.mean(topdown, axis=1) / np.sum(np.mean(topdown, axis=1))
    mx = np.arange(topdown.shape[1]) @ np.mean(topdown, axis=0) / np.sum(np.mean(topdown, axis=0))
    thresh_tiff[int(my + shank_box):, :, :] = 0
    thresh_tiff[:int(my - shank_box), :, :] = 0
    thresh_tiff[:, :, int(mx + shank_box):] = 0
    thresh_tiff[:, :, :int(mx - shank_box)] = 0
    kx, ky, kz = np.where(thresh_tiff)
    filt_vol = np.zeros_like(tiff_vol)
    filt_vol[kx, ky, kz] = tiff_vol[kx, ky, kz]
    norm_vol = filt_vol / np.max(filt_vol)
    xyz_pts = []
    for x, y, z in zip(kx, ky, kz):
        n_pts = int(np.round(norm_vol[x, y, z] * samp_pt_scale))
        gen_pts = np.tile(np.array([x, y, z]), (n_pts, 1))
        xyz_pts.append(gen_pts)
    xyz_pts = np.vstack(xyz_pts)

    clus = AgglomerativeClustering(distance_threshold=2, n_clusters=None, linkage='single').fit_predict(xyz_pts)
    c_id, counts = np.unique(clus, return_counts=True)
    shank_clus = c_id[np.argsort(counts)[-4:]]

    shank_tips = []
    shank_coefs = []
    tip_xs = []
    side_view = np.sum(filt_vol, axis=2) > 0
    side_view = binary_dilation(side_view, iterations=2)
    for ci, c in enumerate(shank_clus):
        x, y, z = xyz_pts[clus == c, 0], xyz_pts[clus == c, 1], xyz_pts[clus == c, 2]
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
            chan_depths = vox_sz * shank_tips[s] - depths[shank_ids == s]
            chan_xs = vox_sz * np.polyval(shank_coefs[s, 0, :], chan_depths / vox_sz)
            chan_zs = vox_sz * np.polyval(shank_coefs[s, 1, :], chan_depths / vox_sz)
            chan_xyz.append(np.vstack((chan_num[shank_ids == s], chan_xs, chan_depths, chan_zs)).T)
        chan_xyz = np.vstack(chan_xyz)

    if save_path is not None:
        save_path = os.path.join(save_path, 'shank_locations.npz')
        np.savez(save_path, tip_dvs=shank_tips, shank_coefs=shank_coefs, sig_thresh=shanks_thresh,
                 tiff_path=atlas_reg_tiff, vox_size=vox_sz, chan_ccf=chan_xyz)

    return save_path


def manual_segment_tracks(atlas_reg_tiff, num_shanks=4, shank_ord='PostAnt', save_path=None, npx_chan_file=None, shanks_thresh=350, vox_sz=25):

    tiff_vol = tiff.imread(atlas_reg_tiff) > shanks_thresh
    topdown = np.sum(tiff_vol, axis=1)
    sideview = np.sum(tiff_vol, axis=2)
    plt.figure()
    plt.imshow(topdown)
    ap_ml = plt.ginput(2*num_shanks, timeout=0)
    plt.figure()
    plt.imshow(sideview)
    tip_ap_dv = plt.ginput(2*num_shanks, timeout=0)

    tops_ap = np.array(ap_ml)[:4, 1]
    ml_top = np.array(ap_ml)[:4, 0]
    ml_tip = np.array(ap_ml)[4:, 0]
    tips_ap = np.array(tip_ap_dv)[4:, 1]
    tips_dv = np.array(tip_ap_dv)[4:, 0]
    tops_dv = np.array(tip_ap_dv)[:4, 0]

    shank_tips = tips_dv
    slope_ap = (tips_ap - tops_ap)/(tips_dv - tops_dv)
    slope_ml = (ml_tip - ml_top)/(tips_dv - tops_dv)
    y_ints_ap = tips_ap - tips_dv*slope_ap
    y_ints_ml = ml_tip - tips_dv * slope_ml

    chan_xyz = np.array([])
    if npx_chan_file is not None:
        chan_data = json.load(open(npx_chan_file))
        chan_num = np.array(chan_data['chanMap'])
        shank_ids = np.array(chan_data['kcoords']).astype(int)
        depths = np.array(chan_data['yc'])
        chan_xyz = []
        for s in range(4):
            chan_depths = vox_sz * shank_tips[s] - depths[shank_ids == s]
            # chan_zs = np.repeat(vox_sz * ml[s], len(chan_depths))
            chan_xs = vox_sz*(slope_ap[s]* chan_depths/vox_sz + y_ints_ap[s])
            chan_zs = vox_sz * (slope_ml[s] * chan_depths / vox_sz + y_ints_ml[s])
            chan_xyz.append(np.vstack((chan_num[shank_ids == s], chan_xs, chan_depths, chan_zs)).T)
        chan_xyz = np.vstack(chan_xyz)

    if save_path is not None:
        save_path = os.path.join(save_path, 'shank_locations.npz')
        np.savez(save_path, tip_dvs=shank_tips, shank_coefs=slope_ap, sig_thresh=shanks_thresh,
                 tiff_path=atlas_reg_tiff, vox_size=vox_sz, chan_ccf=chan_xyz, method='manual')

    return save_path


def show_shank_tracks(shank_data_file, return_brain=False, brain=None, tiff_path=None, show_sig=True, show_lines=False,
                      show_label=True, chan_col='k', show_bounds=False):
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
            tiff_path = str(file_dict['tiff_path'])
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


def show_signal(registered_tiff, return_brain=False, brain=None, vox_sz=25, min_sig=350):
    if brain is None:
        brain = Scene(atlas_name="allen_mouse_25um", title="Reconstructed Implant Locations")

    tiff_vol = tiff.imread(registered_tiff)
    raw_vol = Volume(tiff_vol, vox_sz, min_value=min_sig, cmap='gray')
    raw_vol.mesh.alpha(0.1)
    brain.add(raw_vol)

    if return_brain:
        return brain
    else:
        brain.render()


def register_probes(tiff_path, probe_json=None, out_path=None, notrace=False, min_sig=350, auto=False):
    if out_path is None:
        out_path = os.path.join(os.path.split(tiff_path)[0], 'brainreg')

    registered_tiff = register_brain(tiff_path, out_path)
    if notrace:
        show_signal(registered_tiff, min_sig=min_sig)
    else:
        if auto:
            shank_data_file = auto_segment_tracks(registered_tiff, save_path=out_path, npx_chan_file=probe_json, shanks_thresh=min_sig)
        else:
            shank_data_file = manual_segment_tracks(registered_tiff, save_path=out_path, npx_chan_file=probe_json, shanks_thresh=min_sig)
        show_shank_tracks(shank_data_file)


def add_regions(region_list, brain=None, colors=None, alpha=0.25):
    if brain is None:
        brain = Scene()

    if colors is None:
        cols_hex = ['w' for i in range(len(region_list))]
        alpha = 0.1
    elif type(colors) == str:
        cols_hex = [colors for i in range(len(region_list))]
    elif len(colors) == len(region_list):
        cols_hex = colors
    else:
        cmap = matplotlib.colormaps['Accent']
        colors = cmap(np.linspace(0, 1, len(region_list)))
        cols_hex = [matplotlib.colors.to_hex(c) for c in colors]
    for region, col in zip(region_list, cols_hex):
        brain.add_brain_region(region, alpha=alpha, color=col)

    return brain


# def add_allen_data(allen_exp_id, brain=None, vox_sz=25, min_density=0.25):
#     if brain is None:
#         brain = Scene()
#
#     # tell the cache class what resolution (in microns) of data you want to download
#     mcc = MouseConnectivityCache(resolution=vox_sz)
#
#     # download the projection density volume for one of the experiments
#     pd = mcc.get_projection_density(allen_exp_id)
#
#     brain.add(Volume(pd[0], voxel_size=vox_sz, min_value=min_density, cmap='viridis'))
#     return brain


def project_svg(brain=None, use_silhouette=False, slice_pos=5000, slice_axis='sag', save_file=None, ax=None,
                show_brain=True, alph=0.7):
    if brain is None:
        brain = Scene()
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax_ind = 0
    match slice_axis:
        case 'sag':
            ax_ind = 2
        case 'cor':
            ax_ind = 0
        case 'hor':
            ax_ind = 1
    norm = [0, 0, 0]
    org = [0, 0, 0]
    norm[ax_ind] = 1
    org[ax_ind] = slice_pos
    sil_ax = ['x', 'y', 'z']
    pt_inds = [(1, 2), (0, 2), (0, 1)]
    for a in brain.actors:
        if a.name != 'root' or show_brain:
            if use_silhouette:
                splane = a.mesh.clone().project_on_plane(sil_ax[ax_ind])
            else:
                try:
                    splane = a.mesh.clone().slice(origin=org, normal=norm)
                except:
                    splane = None
            if splane is not None:
                if a.br_class == 'Volume':
                    pts = splane.vertices[:, [pt_inds[ax_ind][0], pt_inds[ax_ind][1]]]
                    pt_cols = splane.pointcolors / 255
                    ax.scatter(pts[:, 0], pts[:, 1], c=pt_cols)
                elif a.br_class == 'Point':
                    pts = splane.coordinates[:, [pt_inds[ax_ind][0], pt_inds[ax_ind][1]]]
                    clean_pts = make_shapes(pts)
                    ax.fill(clean_pts[:, 0], clean_pts[:, 1], c=a.color(), alpha=alph, linewidth=0)
                else:
                    splane = splane.boundaries()
                    pts = splane.coordinates[:, [pt_inds[ax_ind][0], pt_inds[ax_ind][1]]]
                    clean_pts = make_shapes(pts)
                    ax.plot(clean_pts[:, 0], clean_pts[:, 1], 'k')
    if save_file is not None:
        plt.savefig(save_file)
    return ax


def make_shapes(points, max_dist=250):
    dist_mat = distance_matrix(points, points)
    G = nx.Graph(dist_mat)
    sort_ord = traveling_salesman_problem(G)
    sort_pts = points[sort_ord, :]
    dists = np.linalg.norm(sort_pts[1:] - sort_pts[:-1], axis=1)
    shapes = np.where(dists > max_dist)[0]
    sort_pts[shapes, :] = [np.nan, np.nan]
    return sort_pts
