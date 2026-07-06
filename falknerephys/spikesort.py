import glob
import os
from pathlib import Path

import numpy as np
from kilosort import run_kilosort
from kilosort.io import load_probe, save_to_phy
import matplotlib.pyplot as plt
from tkfilebrowser import askopenfilename, askopendirnames
import falknerephys.UnitMatchPy.extract_raw_data as erd
import falknerephys.UnitMatchPy.bayes_functions as bf
import falknerephys.UnitMatchPy.utils as util
import falknerephys.UnitMatchPy.overlord as ov
import falknerephys.UnitMatchPy.default_params as default_params
from falknerephys.UnitMatchPy.DeepUnitMatch.utils import param_fun, helpers
from falknerephys.UnitMatchPy.DeepUnitMatch.testing import test
import falknerephys.UnitMatchPy.metric_functions as mf
from joblib import Parallel, delayed

from falknerephys.plotting import venn2


def run_ks4(imec_data_paths=None, npx_probe=None, probe_name=None, bad_channels=None, n_chans=None,
           batch_size=60000, num_blocks=5):

    if type(imec_data_paths) is str:
        file_type = imec_data_paths.split('.')[-1]
        if file_type == 'txt':
            imec_data_paths = np.loadtxt(imec_data_paths, delimiter='\n')
        else:
            imec_data_paths = [imec_data_paths]

    if imec_data_paths is None:
        fold_paths = askopendirnames(title='Select folders to process')
        imec_data_paths = find_bins_no_ks(fold_paths)

    auto_chan_n = 385
    match npx_probe:
        case '3A':
            probe = load_probe('io/probe_chan_maps/neuropixPhase3A_kilosortChanMap.mat')
            auto_chan_n = 385
        case '3B1':
            probe = load_probe('io/probe_chan_maps/neuropixPhase3B1_kilosortChanMap.mat')
            auto_chan_n = 385
        case '3B2':
            probe = load_probe('io/probe_chan_maps/neuropixPhase3B2_kilosortChanMap.mat')
            auto_chan_n = 385
        case 'NP2':
            probe = load_probe('io/probe_chan_maps/NP2_kilosortChanMap.mat')
            auto_chan_n = 385
        case None:
            probe_file = askopenfilename(
                title="Select channel map file",
                filetypes=(("JSON", "*.json"), ("Matlab", "*.mat"), ("All files", "*.*")))
            probe = load_probe(probe_file)
        case _:
            probe = load_probe(npx_probe)

    if n_chans is None:
        n_chans = auto_chan_n

    settings = {'n_chan_bin': n_chans,
                'batch_size': batch_size,
                'nblocks': num_blocks}

    phy_results = []
    for f in imec_data_paths:
        f_split = os.path.split(f)
        out_dir = os.path.join(f_split[0], f_split[1].split('.')[0] + '_kilosort4')
        print(f"Run ks on {f} --> save to {out_dir}")
        ks_out = run_kilosort(settings, probe=probe, probe_name=probe_name, filename=f, results_dir=out_dir,
                              do_CAR=True, save_extra_vars=True, save_preprocessed_copy=False, bad_channels=bad_channels,
                              verbose_console=True, clear_cache=True)
        ops, st, clu, tF, Wall, similat_templates, is_ref, est_contam_rate, kept_spikes = ks_out[:]
        phy_res = save_to_phy(st, clu, tF, Wall, probe, ops, 0, results_dir=out_dir)
        phy_results.append(phy_res)
    return phy_results


def find_bins_no_ks(root_dirs):
    bins_to_run = []
    for f in root_dirs:
        bin_paths = glob.glob(os.path.join(f, '**/*.ap.bin'), recursive=True)
        for b in bin_paths:
            ap_root = os.path.split(b)[0]
            ks_path = glob.glob(os.path.join(ap_root, '*kilosort4'))
            if len(ks_path) == 0:
                bins_to_run.append(b)
    return bins_to_run


def load_phy(phy_path, offset_s=0, ephys_hz=30000, return_table=False, use_bombcell=False):
    """

    Parameters
    ----------
    phy_path : str
        Directory containing all Phy output files
    offset_s : float, optional
        Number of seconds to be subtracted from spike times
    ephys_hz: int, optional
        Recording frequency of ephys data

    Returns
    -------
    A directory containing all the 'good' units where each key is the ID of the unit and the value is its spikes

    """

    #Load spike data from Phy folder
    spks = np.load(os.path.join(phy_path, 'spike_times.npy'))
    spk_ids = np.load(os.path.join(phy_path,  'spike_clusters.npy'))
    good_units = np.loadtxt(os.path.join(phy_path, 'cluster_group.tsv'), delimiter='\t', skiprows=1, dtype=str)
    clus_info = np.loadtxt(os.path.join(phy_path, 'cluster_info.tsv'), delimiter='\t', skiprows=1, dtype=str)

    keep_clus = []
    bc_path = os.path.join(phy_path, 'bombcell')
    if use_bombcell and os.path.exists(bc_path):
        from bombcell import load_bc_results
        from bombcell.quality_metrics import get_quality_unit_type

        param, quality_metrics, _ = load_bc_results(bc_path)
        unit_type, unit_type_string = get_quality_unit_type(param, quality_metrics)
        _, phy_info = load_phy(phy_path, return_table=True)
        keep_clus = np.where(unit_type_string == 'GOOD')[0]
    else:
        if use_bombcell:
            print('Did not find the bombcell folder -> reverting to manual labels...')
        #Only keep the ones labeled good from the tsv
        for i in range(np.shape(good_units)[0]):
            if good_units[i, 1] == 'good':
                keep_clus.append(good_units[i, 0].astype(int))

    #Make dictionary from good units
    ephys_data = dict()
    for c in keep_clus:
        inds = np.where(spk_ids == c)
        g_spks = spks[inds]
        # compute relative spike times based on offset
        rel_spk_ts = g_spks.squeeze().astype(int)
        # ignore spikes before offset and convert to seconds
        ephys_data[str(c)] = (rel_spk_ts[rel_spk_ts > 0] / ephys_hz) - offset_s

    keep_info = np.isin(clus_info[:, 0].astype(int), keep_clus)
    good_info = clus_info[keep_info, :]

    if return_table:
        return ephys_data, good_info
    else:
        depths = good_info[:, 6].astype(float)
        shanks = good_info[:, 10].astype(float).astype(int)
        amps = good_info[:, 1].astype(float)
        chans = good_info[:, 5].astype(float).astype(int)
        return ephys_data, amps, depths, shanks, chans


def run_bombcell(raw_path, meta_path, phy_path, ks_version=4, do_plots=True):
    from bombcell import run_bombcell, get_default_parameters
    bc_path = os.path.join(phy_path, 'bombcell')
    param = get_default_parameters(phy_path,
                                      raw_file=raw_path,
                                      meta_file=meta_path,
                                      kilosort_version=ks_version)
    param['plotGlobal'] = do_plots
    quality_metrics, param, unit_type, unit_type_string = run_bombcell(phy_path, bc_path, param)
    return quality_metrics, param, unit_type, unit_type_string


def compare_bombcell_manual(phy_path, ax=None):
    from bombcell import load_bc_results
    from bombcell.quality_metrics import get_quality_unit_type
    if ax is None:
        ax = plt.gca()
    bc_path = os.path.join(phy_path, 'bombcell')
    param, quality_metrics, _ = load_bc_results(bc_path)
    unit_type, unit_type_string = get_quality_unit_type(param, quality_metrics)
    _, phy_info = load_phy(phy_path, return_table=True)
    man_good = phy_info[:, 0].astype(int)
    bc_good = np.where(unit_type_string == 'GOOD')[0]
    man_bc = set(man_good).intersection(set(bc_good))
    n_man, n_bc, n_both = len(man_good) - len(man_bc), len(bc_good) - len(man_bc), len(man_bc)
    venn2(n_man, n_bc, n_both, labels=('Manual', 'Bombcell'), ax=ax)
    return len(man_good), len(bc_good), n_both


def prep_raw_unitmatch(folds, only_good = True):
    # List of paths to a KS directory, can pass paths
    KS_dirs = []
    data_paths = []
    meta_paths = []
    for f in folds:
        fold_name = os.path.split(f)[-1]
        bin_name = os.path.join(f, fold_name + '_g0_t0.imec0.ap.bin')
        meta_name = os.path.join(f, fold_name + '_g0_t0.imec0.ap.meta')
        data_paths.append(bin_name)
        meta_paths.append(meta_name)
        KS_dirs.append(os.path.join(f, 'kilosort4'))

    # Set Up Parameters
    sample_amount = 1000  # for both CV, at least 500 per CV
    spike_width = 82  # assuming 30khz sampling, 82 and 61 are common choices, covers the AP and space around needed for processing
    samples_before = 20
    samples_after = spike_width - samples_before
    half_width = np.floor(spike_width / 2).astype(int)
    max_width = np.floor(spike_width / 2).astype(
        int)  # Size of area at start and end of recording to ignore to get only full spikes
    n_channels = 384  # neuropixels default

    n_sessions = len(KS_dirs)  # How many session are being extracted
    spike_ids, spike_times, good_units, all_ids = erd.extract_KS_data(KS_dirs, extract_good_units_only=only_good)

    if only_good:
        for sid in range(n_sessions):
            # Load metadata
            meta_data = erd.read_meta(Path(meta_paths[sid]))
            n_elements = int(meta_data['fileSizeBytes']) / 2
            n_channels_tot = int(meta_data['nSavedChans'])

            # Create memmap to raw data, for that session
            data = np.memmap(data_paths[sid], dtype='int16', shape=(int(n_elements / n_channels_tot), n_channels_tot))

            # Remove spikes which won't have a full waveform recorded
            spike_ids_tmp = np.delete(spike_ids[sid], np.logical_or((spike_times[sid] < max_width),
                                                                    (spike_times[sid] > (data.shape[0] - max_width))))
            spike_times_tmp = np.delete(spike_times[sid], np.logical_or((spike_times[sid] < max_width), (
                        spike_times[sid] > (data.shape[0] - max_width))))

            # Might be slow extracting sample for good units only?
            sample_idx = erd.get_sample_idx(spike_times_tmp, spike_ids_tmp, sample_amount, units=good_units[sid])

            avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode='r', max_nbytes=None)(
                delayed(erd.extract_a_unit_KS4)(sample_idx[uid], data, samples_before, samples_after, spike_width,
                                                n_channels, sample_amount)
                for uid in range(good_units[sid].shape[0])
            )
            avg_waveforms = np.asarray(avg_waveforms)

            # Save in file named 'RawWaveforms' in the KS Directory
            print(good_units[sid])
            erd.save_avg_waveforms(avg_waveforms, KS_dirs[sid], all_ids, good_units[sid], extract_good_units_only=True)

    else:
        for sid in range(n_sessions):
            # Extracting ALL the Units

            n_units = len(np.unique(spike_ids[sid]))

            # Load metadata
            this_path = meta_paths[sid]
            meta_data = erd.read_meta(Path(this_path))

            n_elements = int(meta_data['fileSizeBytes']) / 2
            n_channels_tot = int(meta_data['nSavedChans'])

            # Create memmap to raw data, for that session
            data = np.memmap(data_paths[sid], dtype='int16', shape=(int(n_elements / n_channels_tot), n_channels_tot))

            # Remove spikes which won't have a full waveform recorded
            spike_ids_tmp = np.delete(spike_ids[sid], np.logical_or((spike_times[sid] < max_width),
                                                                    (spike_times[sid] > (data.shape[0] - max_width))))
            spike_times_tmp = np.delete(spike_times[sid], np.logical_or((spike_times[sid] < max_width), (
                        spike_times[sid] > (data.shape[0] - max_width))))

            # Extract sample indices for all units
            sample_idx = erd.get_sample_idx(spike_times_tmp, spike_ids_tmp, sample_amount,
                                            units=np.unique(spike_ids[sid]))

            avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode='r', max_nbytes=None)(
                delayed(erd.extract_a_unit_KS4)(sample_idx[uid], data, samples_before, samples_after, spike_width,
                                                n_channels, sample_amount)
                for uid in range(n_units)
            )
            avg_waveforms = np.asarray(avg_waveforms)

            # Save in file named 'RawWaveforms' in the KS Directory
            erd.save_avg_waveforms(avg_waveforms, KS_dirs[sid], all_ids, good_units[sid])


def load_unitmatch(fold0, fold1, only_good=True, use_bombcell=False):
    # Get default parameters, can add your own before or after!
    param = default_params.get_default_param()
    param['waveidx'] = np.arange(9, 32).astype(int)
    param['peak_loc'] = 21

    # Give the paths to the KS directories for each session
    # If you don't have a dir with channel_positions.npy etc look at the detailed example for supplying paths separately
    KS_dirs = [fold0 + '/kilosort4', fold1 + '/kilosort4']

    param['KS_dirs'] = KS_dirs
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)

    if not use_bombcell:
        unit_label_paths[0] = os.path.join(KS_dirs[0], 'cluster_group.tsv')
        unit_label_paths[1] = os.path.join(KS_dirs[1], 'cluster_group.tsv')

    param = util.get_probe_geometry(channel_pos[0], param)
    # STEP 0 -- data preparation
    # Read in data and select the good units and exact metadata
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths,
                                                                                                       unit_label_paths,
                                                                                                       param,
                                                                                                       good_units_only=only_good)
    param['good_units'] = good_units
    return channel_pos, waveform, session_id, session_switch, within_session, good_units, param


def run_unitmatch(fold0, fold1, only_good=True, thresh=0.75, use_bombcell=False, use_dum=True):

    if use_dum:
        return run_deepunitmatch(fold0, fold1, only_good=only_good, thresh=thresh, use_bombcell=use_bombcell)
    else:
        channel_pos, waveform, session_id, session_switch, within_session, good_units, param = load_unitmatch(fold0, fold1,
                                                                                                              only_good,
                                                                                                              use_bombcell)


        # Create clus_info, contains all unit id/session related info
        clus_info = {'good_units': good_units, 'session_switch': session_switch, 'session_id': session_id,
                     'original_ids': np.concatenate(good_units)}

        # STEP 1
        # Extract parameters from waveform

        extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)


        # STEP 2, 3, 4
        # Extract metric scores
        total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(extracted_wave_properties,
                                                                                               session_switch,
                                                                                               within_session, param,
                                                                                               niter=2)

        # STEP 5
        # Probability analysis
        # Get prior probability of being a match
        prior_match = 1 - (param['n_expected_matches'] / param['n_units'] ** 2)  # freedom of choose in prior prob
        priors = np.array((prior_match, 1 - prior_match))

        # Construct distributions (kernels) for Naive Bayes Classifier
        labels = candidate_pairs.astype(int)
        cond = np.unique(labels)
        score_vector = param['score_vector']
        parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

        parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one=1)

        # Get probability of each pair of being a match
        probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

        output_prob_matrix = probability[:, 1].reshape(param['n_units'], param['n_units'])

        util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold=thresh)

        output_threshold = np.zeros_like(output_prob_matrix)
        output_threshold[output_prob_matrix > thresh] = 1

        matches = np.argwhere(output_threshold == 1)

        num_units_l = len(good_units[0])

        between_matches = []
        for m in matches:
            if m[0] < num_units_l < m[1]:
                between_matches.append([m[0], m[1]])
        between_matches = np.array(between_matches)
        good_unit_ids = np.vstack(good_units)
        left_inds, right_inds, left_labs, right_labs, out_right_inds = [], [], [], [], []
        if len(between_matches) > 0:
            left_inds = between_matches[:, 0].astype(int)
            right_inds = between_matches[:, 1].astype(int)
            left_labs = good_unit_ids[left_inds]
            right_labs = good_unit_ids[right_inds]
            out_right_inds = right_inds - num_units_l

        return left_inds, out_right_inds, left_labs, right_labs, waveform[left_inds, :, :, 0], waveform[right_inds, :, :, 0]


def run_deepunitmatch(fold0, fold1, only_good=True, thresh=0.5, use_bombcell=False, device="cpu"):
    channel_pos, waveform, session_id, session_switch, within_session, good_units, param = load_unitmatch(fold0, fold1,
                                                                                                          only_good,
                                                                                                          use_bombcell)

    # Where to write/read DeepUnitMatch preprocessed HDF5s (creates `processed_waveforms/`)
    save_path = os.path.join(fold0, 'TMP')  # Note, this folder will be removed between runs.

    # make sure RawWaveforms are of appropriate size when using our trained DeepUnitMatch model (spike_width 82, samples_before 20, samples_after 61)

    # Preprocess the DeepUnitMatch way and save as HDF5 files for each session in 'processed_waveforms'.

    # save_path is defined in the "User inputs" cell above
    unit_ids = np.concatenate(param["good_units"]).squeeze()  # cluster IDs in the same order as `waveform`

    snippets, positions = param_fun.get_snippets(waveform, channel_pos, session_id, save_path=save_path, unit_ids=unit_ids)

    # Load the neural net
    model = test.load_trained_model(device=device)

    # We have stored the preprocessed data here (from the get_snippets function)
    data_dir = os.path.join(save_path, 'processed_waveforms')

    # Pass the preprocessed data through the neural net
    sim_matrix = test.inference(model, data_dir) # n_sessions prevents more data to be loaded in from other runs

    # Use the same Naive Bayes as in UnitMatchPy

    clus_info = {'good_units': param['good_units'], 'session_switch': session_switch, 'session_id': session_id,
                 'original_ids': np.concatenate(param['good_units'])}
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info,
                                                      param)  # contains spatial locations
    within_session = 1 - (session_id[:, None] == session_id).astype(int)
    sessions = np.unique(session_id)
    match_dfs = []
    probs = np.zeros(sim_matrix.shape)
    distance_matrix = np.zeros(sim_matrix.shape)

    for r1 in sessions:
        for r2 in sessions:
            if r1 >= r2:
                continue

            mask = np.isin(session_id, [r1, r2])
            sim_mat = sim_matrix[mask][:, mask]
            n = np.sum(mask)
            n_units_r1 = session_switch[r1 + 1] - session_switch[r1]
            n_units_r2 = session_switch[r2 + 1] - session_switch[r2]
            session_switch_pair = np.array([0, n_units_r1, n_units_r1 + n_units_r2])

            indices = np.where(np.isin(session_id, [r1, r2]))[0]
            df = helpers.create_dataframe([param['good_units'][r1], param['good_units'][r2]], sim_mat,
                                          session_list=[r1, r2])
            matches = test.get_matches(df, sim_mat, session_id[indices], data_dir, dist_thresh=50)

            labels = np.eye(sim_mat.shape[0])
            subsessionid = np.array([r1] * len(param['good_units'][r1]) + [r2] * len(param['good_units'][r2]))
            for (recses1, recses2), group in matches.groupby(by=['RecSes1', 'RecSes2']):
                asmatrix = group['match'].values.reshape(len(param['good_units'][recses1]),
                                                         len(param['good_units'][recses2])).astype(int)
                labels[np.ix_(subsessionid == recses1, subsessionid == recses2)] = asmatrix


            avg_centroid, avg_waveform_per_tp = extracted_wave_properties['avg_centroid'][:, mask, :], \
            extracted_wave_properties['avg_waveform_per_tp'][:, mask, :, :]
            avg_waveform_per_tp = mf.drift_correct_session_pair(labels.astype(bool), session_switch_pair, avg_centroid,
                                                                avg_waveform_per_tp, 0, param)
            avg_waveform_per_tp_flip = mf.flip_dim(avg_waveform_per_tp, param, n)
            euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip, param, n)
            centroid_dist, _ = mf.centroid_metrics(euclid_dist, param)

            scores_to_incl = {
                'similarity': sim_mat,
                'distance': centroid_dist,
            }

            n_units = int(np.sqrt(len(df)))
            priors = np.array([1 - 2 / n_units, 2 / n_units])
            parameter_kernels = bf.get_parameter_kernels(scores_to_incl, labels, np.unique(labels), param)
            predictors = np.stack([scores for scores in scores_to_incl.values()], axis=2)
            probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, np.unique(labels))
            prob_matrix = probability[:, 1].reshape(n_units, n_units)
            # Debug: verify shapes match before assignment
            target_shape = np.ix_(mask, mask)
            target_rows = np.where(mask)[0]
            if prob_matrix.shape != centroid_dist.shape:
                print(
                    f"  WARNING: Shape mismatch! prob_matrix={prob_matrix.shape} vs centroid_dist={centroid_dist.shape}")
            probs[np.ix_(mask, mask)] = prob_matrix
            distance_matrix[np.ix_(mask, mask)] = centroid_dist

    util.evaluate_output(probs, param, within_session, session_switch, match_threshold=thresh)

    # Process the output probability matrix to get final set of matches (across sessions)
    # thresh is defined in the "User inputs" cell above
    final_matches = test.directional_filter(probs, session_id, thresh)


    # Divide final number of matches by 2 to account for double counting in the matrix
    num_m = np.sum(final_matches) // 2
    print(f" Found {num_m} matches in these sessions using the threshold of {thresh}.")

    left_inds, right_inds, left_labs, right_labs, out_right_inds = [], [], [], [], []

    num_units_l = len(good_units[0])
    matches = np.argwhere(final_matches)

    between_matches = []
    for m in matches:
        if m[0] < num_units_l < m[1]:
            between_matches.append([m[0], m[1]])

    between_matches = np.array(between_matches)
    good_unit_ids = np.vstack(good_units)
    if len(between_matches) > 0:
        left_inds = between_matches[:, 0].astype(int)
        right_inds = between_matches[:, 1].astype(int)
        left_labs = good_unit_ids[left_inds]
        right_labs = good_unit_ids[right_inds]

        # Now we can check performance using the AUC. This tests the agreement between DeepUnitMatch matches and functional scores (in this case, ISI histogram correlations).
        isicorr = test.ISI_correlations(param)
        auc = test.AUC(final_matches, isicorr, session_id)
        print(f"AUC for DeepUnitMatch matches: {auc:.3f}")

        out_right_inds = right_inds - num_units_l

    return left_inds, out_right_inds, left_labs, right_labs, waveform[left_inds, :, :, 0], waveform[right_inds, :, :, 0]


def get_time_win(ephys_dict, start_time=0, end_time=0):
    filt_dict = dict()
    for k in ephys_dict.keys():
        spikes = ephys_dict[k]
        filt_spikes = spikes[np.logical_and(spikes > start_time, spikes < end_time)]
        if len(filt_spikes) > 0:
            filt_spikes = filt_spikes - filt_spikes[0]
        filt_dict[k] = filt_spikes
    return filt_dict


def resample_spikes(spikes, in_hz, out_hz):
    resamp_spikes = float(out_hz)*(spikes/float(in_hz))
    resamp_spikes = np.round(resamp_spikes).astype(int)
    return resamp_spikes


def bin_spikes(spks, fs, bin_ms, out_len_s):
    # time_vec, bin_out = square_fr(spikes, fs, bin_ms, length_s, as_fr=False)
    spk_inds = np.floor(spks * fs).astype(int)
    spks_t = np.zeros(int(np.ceil(fs*out_len_s)))
    u_ts = np.unique(spk_inds)
    sum_spk = [np.sum(spk_inds == i) for i in u_ts]
    spks_t[u_ts] = sum_spk
    bin_starts = np.floor(fs*np.arange(0, out_len_s, bin_ms/1000)).astype(int)
    for b0, b1 in zip(bin_starts[:-1], bin_starts[1:]):
        spks_t[b0:b1] = np.sum(spks_t[b0:b1])
    spks_t[bin_starts[-1]:] = np.sum(spks_t[bin_starts[-1]:])
    t = np.linspace(0, out_len_s, len(spks_t), dtype=np.float32)
    return t, spks_t


def square_fr(spks, fs, time_width_ms, out_len_s, as_fr=True):
    win_samps = int(fs*(time_width_ms/1000))
    spk_inds = np.floor(spks * fs).astype(int)
    spks_t = np.zeros(int(np.ceil(fs*out_len_s)))
    u_ts = np.unique(spk_inds)
    sum_spk = [np.sum(spk_inds == i) for i in u_ts]
    spks_t[u_ts] = sum_spk
    kern = np.ones(win_samps)
    if as_fr:
        kern = kern / win_samps
    conv_spks = np.convolve(spks_t, kern, mode='same')
    fr = conv_spks
    if as_fr:
        fr = fr * fs
    fr = fr.astype(np.float32)
    t = np.linspace(0, out_len_s, len(fr), dtype=np.float32)
    return t, fr


def gaus_fr(spks, fs, time_width_ms, out_len_s):
    win_samps = fs*(time_width_ms/1000)/2
    spk_inds = np.floor(spks * fs).astype(int)
    spks_t = np.zeros(int(np.ceil(fs*out_len_s)))
    u_ts, u_cnts = np.unique(spk_inds, return_counts=True)
    spks_t[u_ts] = u_cnts
    sig = win_samps/3
    x = np.linspace(-fs/2, fs/2, fs)
    kern = np.exp(-(x / sig) ** 2 / 2)
    kern_norm = kern / sum(kern)
    conv_spks = np.convolve(spks_t, kern_norm, mode='same')
    t = np.linspace(0, out_len_s, len(spks_t))
    fr = conv_spks * fs
    return t, fr


def spikes_to_timeseries(unit_dict, smooth_func=square_fr, ephys_hz=30000, out_hz=40, ts_len_s=60, time_win_ms=250, save_path='', overwrite=False):
    units = []
    t_vec = []
    if os.path.isfile(save_path) and not overwrite:
        spike_dict = np.load(save_path)
        t_vec = spike_dict['t_vec']
        spk_data = spike_dict['spk_data']
        unit_ids = spike_dict['unit_ids']
    else:
        for u in unit_dict.keys():
            t, fr = smooth_func(unit_dict[u], out_hz, time_win_ms, ts_len_s)
            units.append(fr)
            t_vec = t
        spk_data = np.array(units).T
        unit_ids = list(unit_dict.keys())
        if save_path is not None:
            np.savez(save_path, spk_data=spk_data, t_vec=t_vec, smooth_func=str(smooth_func), out_hz=out_hz, time_win_ms=time_win_ms, unit_ids=unit_ids)
    return t_vec, spk_data, unit_ids


def split_trials_from_daq(spike_ts, daq_data, spike_len, daq_len, spike_hz=40, daq_hz=30303, do_split=False):
    if spike_len < daq_len:
        daq_data = daq_data[:int(np.floor(spike_len*daq_hz))]
    else:
        spike_ts = spike_ts[:int(np.floor(daq_len*spike_hz)), :]
    ds_chan0 = daq_data[np.floor(np.linspace(0, len(daq_data)-1, len(spike_ts))).astype(int)]
    trial_starts = np.where(np.logical_and(ds_chan0[:-1] < 2.5, ds_chan0[1:] > 2.5))[0]
    trial_ends = np.where(np.logical_and(ds_chan0[:-1] > 2.5, ds_chan0[1:] < 2.5))[0]
    trial_vec = np.zeros(len(spike_ts))
    for ti, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        trial_vec[ts:te] = ti + 1
    run_data = spike_ts[trial_vec > 0, :]
    trial_vec = trial_vec[trial_vec > 0] - 1
    if do_split:
        trial_list = []
        for i in range(max(trial_vec)):
            trial_list.append(spike_ts[trial_vec == i, :])
        return trial_list
    else:
        return run_data, trial_vec
