import os
from pathlib import Path

import numpy as np
from kilosort import run_kilosort
from kilosort.io import load_probe, save_to_phy
import matplotlib.pyplot as plt
from tkinter import filedialog
import UnitMatchPy.extract_raw_data as erd
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import UnitMatchPy.default_params as default_params
from joblib import Parallel, delayed

from falknerephys.plotting import venn2


def run_ks(imec_data_paths=None, npx_probe=None, probe_name=None, bad_channels=None, n_chans=None,
           batch_size=60000, num_blocks=5):

    if type(imec_data_paths) is str:
        imec_data_paths = [imec_data_paths]

    if imec_data_paths is None:
        imec_data_paths = filedialog.askopenfilenames(
            title="Select raw data file(s)",
            filetypes=(("Binary file", "*.bin"), ("All files", "*.*")))

    auto_chan_n = 385
    match npx_probe:
        case '3A':
            probe = load_probe('./probe_chan_maps/neuropixPhase3A_kilosortChanMap.mat')
            auto_chan_n = 385
        case '3B1':
            probe = load_probe('./probe_chan_maps/neuropixPhase3B1_kilosortChanMap.mat')
            auto_chan_n = 385
        case '3B2':
            probe = load_probe('./probe_chan_maps/neuropixPhase3B2_kilosortChanMap.mat')
            auto_chan_n = 385
        case 'NP2':
            probe = load_probe('./probe_chan_maps/NP2_kilosortChanMap.mat')
            auto_chan_n = 385
        case None:
            probe_file = filedialog.askopenfilename(
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
        out_dir = os.path.join(f_split[0], f_split[1].split('.')[0] + '_kilosort')
        print(f"Run ks on {f} --> save to {out_dir}")
        ks_out = run_kilosort(settings, probe=probe, probe_name=probe_name, filename=f, results_dir=out_dir,
                              do_CAR=True, save_extra_vars=True, save_preprocessed_copy=False, bad_channels=bad_channels,
                              verbose_console=True, clear_cache=True)
        ops, st, clu, tF, Wall, similat_templates, is_ref, est_contam_rate, kept_spikes = ks_out[:]
        phy_res = save_to_phy(st, clu, tF, Wall, probe, ops, 0, results_dir=out_dir)
        phy_results.append(phy_res)
    return phy_results


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
    good_info = None
    if np.all(clus_info[keep_clus, 0].astype(int) == keep_clus):
        good_info = clus_info[keep_clus, :]

    if return_table:
        return ephys_data, good_info
    else:
        depths = good_info[:, 6].astype(float)
        shanks = good_info[:, 10].astype(float).astype(int)
        amps = good_info[:, 1].astype(float)
        return ephys_data, amps, depths, shanks


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


def prep_raw_unitmatch(folds, only_good = False):
    # List of paths to a KS directory, can pass paths
    KS_dirs = []
    data_paths = []
    meta_paths = []
    for f in folds:
        fold_name = os.path.split(f)[-1]
        bin_name = os.path.join(f, '_'.join(fold_name.split('_')[:-1]) + '_t0.imec0.ap.bin')
        meta_name = os.path.join(f, '_'.join(fold_name.split('_')[:-1]) + '_t0.imec0.ap.meta')
        data_paths.append(bin_name)
        meta_paths.append(meta_name)
        KS_dirs.append(os.path.join(f, 'kilosort4'))

    # Set Up Parameters
    sample_amount = 1000  # for both CV, at least 500 per CV
    spike_width = 82  # assuming 30khz sampling, 82 and 61 are common choices, covers the AP and space around needed for processing
    half_width = np.floor(spike_width / 2).astype(int)
    max_width = np.floor(spike_width / 2).astype(
        int)  # Size of area at start and end of recording to ignore to get only full spikes
    n_channels = 384  # neuropixels default

    KS4_data = True  # bool, set to true if using Kilosort
    if KS4_data:
        spike_width = 61
        samples_before = 20
        samples_after = spike_width - samples_before

    n_sessions = len(KS_dirs)  # How many session are being extracted
    spike_ids, spike_times, good_units = erd.extract_KS_data(KS_dirs, extract_good_units_only=only_good)

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

            if KS4_data:
                avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode='r', max_nbytes=None)(
                    delayed(erd.extract_a_unit_KS4)(sample_idx[uid], data, samples_before, samples_after, spike_width,
                                                    n_channels, sample_amount)
                    for uid in range(good_units[sid].shape[0])
                )
                avg_waveforms = np.asarray(avg_waveforms)
            else:
                avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode='r', max_nbytes=None)(
                    delayed(erd.extract_a_unit)(sample_idx[uid], data, half_width, spike_width, n_channels,
                                                sample_amount)
                    for uid in range(good_units[sid].shape[0])
                )
                avg_waveforms = np.asarray(avg_waveforms)

            # Save in file named 'RawWaveforms' in the KS Directory
            erd.save_avg_waveforms(avg_waveforms, KS_dirs[sid], good_units[sid])

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

            if KS4_data:
                avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode='r', max_nbytes=None)(
                    delayed(erd.extract_a_unit_KS4)(sample_idx[uid], data, samples_before, samples_after, spike_width,
                                                    n_channels, sample_amount)
                    for uid in range(n_units)
                )
                avg_waveforms = np.asarray(avg_waveforms)
            else:
                avg_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode='r', max_nbytes=None)(
                    delayed(erd.extract_a_unit)(sample_idx[uid], data, half_width, spike_width, n_channels,
                                                sample_amount)
                    for uid in range(n_units)
                )
                avg_waveforms = np.asarray(avg_waveforms)

            # Save in file named 'RawWaveforms' in the KS Directory
            erd.save_avg_waveforms(avg_waveforms, KS_dirs[sid], good_units[sid])


def run_unitmatch(fold0, fold1):
    # Get default parameters, can add your own before or after!
    param = default_params.get_default_param()

    # Give the paths to the KS directories for each session
    # If you don't have a dir with channel_positions.npy etc look at the detailed example for supplying paths separately
    KS_dirs = [fold0 + '/kilosort4', fold1 + '/kilosort4']

    param['KS_dirs'] = KS_dirs
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)
    param = util.get_probe_geometry(channel_pos[0], param)
    # STEP 0 -- data preparation
    # Read in data and select the good units and exact metadata
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths,
                                                                                                       unit_label_paths,
                                                                                                       param,
                                                                                                       good_units_only=True)

    # param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

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
    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold=0.75)

    match_threshold = param['match_threshold']
    # match_threshold = try different values here!

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    num_units_l = len(good_units[0])
    num_units_r = len(good_units[1])

    plt.imshow(output_threshold, cmap='Greys')
    matches = np.argwhere(output_threshold == 1)

    between_matches = []
    for m in matches:
        if m[0] < num_units_l < m[1]:
            between_matches.append([m[0], m[1]])
    between_matches = np.array(between_matches)
    good_unit_ids = np.vstack(good_units)
    # left_ids = good_unit_ids[between_matches[:, 0].astype(int)]
    # right_ids = good_unit_ids[between_matches[:, 1].astype(int)]
    left_inds = between_matches[:, 0].astype(int)
    right_inds = between_matches[:, 1].astype(int)
    left_labs = good_unit_ids[left_inds]
    right_labs = good_unit_ids[right_inds]
    plt.show()
    return left_inds, right_inds-num_units_l, left_labs, right_labs, waveform[left_inds, :, :, 0], waveform[right_inds, :, :, 1]


if __name__ == '__main__':
    run_ks()