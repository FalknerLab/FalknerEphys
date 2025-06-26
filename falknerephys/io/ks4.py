from kilosort import run_kilosort
from kilosort.io import load_probe, save_to_phy


def run_ks(imec_data_path, npx_probe='NP2', probe_name=None, out_dir=None, bad_channels=None, n_chans=None,
           batch_size=60000, num_blocks=5):

    auto_chan_n = None
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
        case _:
            probe = load_probe(npx_probe)

    if n_chans is None:
        n_chans = auto_chan_n

    settings = {'n_chan_bin': n_chans,
                'batch_size': batch_size,
                'nblocks': num_blocks}

    ks_out = run_kilosort(settings, probe=probe, probe_name=probe_name, filename=imec_data_path, results_dir=out_dir,
                          do_CAR=True, save_extra_vars=True, save_preprocessed_copy=False, bad_channels=bad_channels,
                          verbose_console=True)
    ops, st, clu, tF, Wall, similat_templates, is_ref, est_contam_rate, kept_spikes = ks_out[:]
    phy_res = save_to_phy(st, clu, tF, Wall, probe, ops, 0, results_dir=out_dir)
    return phy_res
