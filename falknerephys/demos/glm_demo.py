import matplotlib.pyplot as plt
from falknerephys.spikesort import gaus_fr, spikes_to_timeseries
from falknerephys.demos import wm_import_demo as wmd
from falknerephys.modeling import run_rc_glm


def run_demo(ephys_data=None, behavior_data=None, b_hz=40):
    ts, fr_hm = spikes_to_timeseries(ephys_data, smooth_func=gaus_fr, time_win_ms=50, ts_len_s=int(len(behavior_data)/b_hz))
    for u in fr_hm.T:
        f, axs = plt.subplots(2, 1)
        y_test, pred, full_pred, r2, model = run_rc_glm(u, behavior_data)
        axs[0].plot(ts, behavior_data)
        axs[0].plot(ts, full_pred)
        axs[1].plot(ts, u)
        axs[1].set_title('Unit FR R2=' + str(r2))
        plt.show()
    return None


if __name__ == '__main__':
    _, unit_dict, t, x, y, v = wmd.get_demo_data()
    run_demo(ephys_data=unit_dict, behavior_data=y)
