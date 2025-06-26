import os
from pathlib import Path
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np

import UnitMatchPy.extract_raw_data as erd
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import UnitMatchPy.default_params as default_params

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
