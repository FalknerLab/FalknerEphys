import random
import time
from copy import deepcopy

import neo.core
import numpy as np
from numpy.random import Generator, PCG64

import matplotlib.pyplot as plt
import sklearn
from matplotlib.colors import to_rgb
from sklearn import svm
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import TweedieRegressor, BayesianRidge, LogisticRegression, LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP
from scipy.special import gamma
from scipy.stats import binned_statistic_dd, zscore
from sklearn.model_selection import RepeatedKFold
import tqdm
from statsmodels.tsa.stattools import grangercausalitytests
# import jax.numpy as jnp
# import jax.random as jr
import matplotlib.pyplot as plt
# from dynamax.hidden_markov_model import GaussianHMM
# from elephant.gpfa import GPFA
import matplotlib.animation as animation

from falknerephys.preprocess import gaus_fr, spikes_to_timeseries, bin_spikes


def uniform_sample(cat_data, max_samps=0, method='random'):
    u_cats, c_cnts = np.unique(cat_data, return_counts=True)
    if max_samps == 0:
        max_samps = np.min(c_cnts)
    cat_inds = []
    for c in u_cats:
        all_inds = np.where(cat_data == c)[0]
        num_samps = min(len(all_inds), max_samps)
        match method:
            case 'first':
                cat_inds.append(all_inds[:num_samps])
            case 'random':
                rand_inds = np.random.choice(all_inds, len(all_inds), replace=False)
                cat_inds.append(rand_inds[:num_samps])
    return cat_inds


def run_rc_glm(behav_pred, u_fr, refit=True, glm_model=TweedieRegressor(power=1, alpha=0.01, link='log', max_iter=1000), cv=0):
    nans = np.any(np.isnan(behav_pred), axis=1)
    behav_pred = behav_pred[~nans]
    u_fr = u_fr[~nans]
    num_f = np.shape(behav_pred)[1]
    X_train, X_test, y_train, y_test = train_test_split(behav_pred, u_fr,
                                                        test_size=0.2, random_state=42)
    full_glm = clone(glm_model)
    full_glm.fit(X_train, y_train)
    full_pred = full_glm.predict(X_test)
    r2_full = r2_score(y_test, full_pred)
    rel_cont = []
    for f in range(num_f):
        if refit:
            part_X_train = np.delete(X_train, f, axis=1)
            part_X_test = np.delete(X_test, f, axis=1)
            part_glm = sklearn.clone(glm_model)
            part_glm.fit(part_X_train, y_train)
            part_pred = part_glm.predict(part_X_test)
            r2_part = r2_score(y_test, part_pred)
        else:
            full_copy = deepcopy(full_glm)
            full_copy.coef_[f] = 0
            part_pred = full_copy.predict(X_test)
            r2_part = r2_score(y_test, part_pred)
        rc = (1 - (r2_part / r2_full))
        rc = max(rc, 0)
        rel_cont.append(rc)
    rel_cont = np.array(rel_cont)
    coefs = rel_cont/np.sum(rel_cont)
    return coefs, r2_full


def make_design_matrix(behav_mat, pred_type=None, fs=30, time_width_ms=250):
    design_mat = []
    if pred_type is None:
        design_mat = behav_mat
    else:
        for b, t in zip(behav_mat.T, pred_type):
            match t:
                case 'event':
                    win_samps = fs * (time_width_ms / 1000) / 2
                    sig = win_samps / 3
                    x = np.linspace(-fs / 2, fs / 2, fs)
                    kern = np.exp(-(x / sig) ** 2 / 2)
                    kern_norm = kern / sum(kern)
                    conv_b = np.convolve(b, kern_norm, mode='same')
                    design_mat.append(conv_b)
                case 'continuous':
                    design_mat.append(b)
                case 'categorical':
                    one_hot = make_one_hot(b)
                    design_mat.append(one_hot.T)
        design_mat = np.vstack(design_mat).T
    return design_mat


def run_reg_decoder(x_data, target_vars, model='glm', k=5, categorical=False, test_inds=None, stratify=False,
                    test_stat='mse', max_iter=1000):
    if test_inds is None:
        if stratify:
            train_input, test_input, train_output, test_output = train_test_split(x_data, target_vars, test_size=0.2,
                                                                                  random_state=42, stratify=target_vars)
        else:
            train_input, test_input, train_output, test_output = train_test_split(x_data, target_vars, test_size=0.2,
                                                                                  random_state=42)
    else:
        test_input = x_data[test_inds]
        test_output = target_vars[test_inds]
        train_input = np.delete(x_data, test_inds, axis=0)
        train_output = np.delete(target_vars, test_inds, axis=0)
    model_obj = None
    if type(model) == str:
        if model == 'svm':
            model_obj = svm.SVR(kernel='linear')
            if categorical:
                model_obj = svm.SVC(kernel='linear')
        elif model == 'glm':
            model_obj = TweedieRegressor(power=1, alpha=0.05, link='log', max_iter=max_iter)
        elif model == 'bayes':
            model_obj = BayesianRidge(max_iter=max_iter)
        elif model == 'knn':
            model_obj = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
        elif model == 'logistic':
            model_obj = LogisticRegression(max_iter=max_iter)
        elif model == 'ols':
            model_obj = LinearRegression()
    else:
        model_obj = model
    if target_vars.ndim > 1:
        model_obj = MultiOutputRegressor(model_obj)
    model_obj.fit(train_input, train_output)
    test_pred = model_obj.predict(test_input)
    pred_all = model_obj.predict(x_data)
    test_val = None
    if test_stat == 'mse':
        test_val = mean_squared_error(test_output, test_pred)
    elif test_stat == 'r2':
        test_val = r2_score(test_output, test_pred)
    f1 = None
    acc = None
    if categorical:
        f1 = f1_score(test_output, test_pred, average='weighted')
        test_c = np.argmax(test_output, axis=1)
        pred_c = np.argmax(test_pred, axis=1)
        acc = [100*np.sum(test_c[pred_c == i] == i)/len(test_c[pred_c == i]) for i in range(4)]

    return model_obj, pred_all, test_val, f1, test_output, test_pred, acc, train_input, test_input


def run_pred_randshuf(u_data, cat_data, k=3, chk_sz=300, rand_state=42, test_ratio=0.2, model=None, num_shufs=20, categorical=False):
    if model is None:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
    len_data = np.shape(cat_data)[0]
    num_perm = int(np.floor(len_data / chk_sz))
    u_data_trim = u_data[:num_perm * chk_sz, :]
    shuf_ord = np.arange(num_perm)
    met_id = 'mse'
    if categorical:
        met_id = 'f1'
    metrics = []
    for i in range(num_shufs):
        random.shuffle(shuf_ord)
        res_arr = [np.arange(start, stop) for start, stop in zip(shuf_ord*chk_sz, shuf_ord*chk_sz + chk_sz)]
        ind_ar = np.reshape(res_arr, len(u_data_trim))
        shuf_data = cat_data[ind_ar, :]
        X_train, X_test, y_train, y_test = train_test_split(u_data_trim, shuf_data,
                                                            test_size=test_ratio, random_state=rand_state)
        this_model = clone(model)
        this_model.fit(X_train, y_train)
        test = this_model.predict(X_test)
        met = mean_squared_error(y_test, test)
        if categorical:
            met = f1_score(y_test, test, average='weighted')
        metrics.append(met)
        print('Shuffle: ', i, ' of ', num_shufs)
    out_mets = metrics[num_perm // 2:] + metrics[:num_perm // 2]
    num_bins = len(out_mets)
    x = np.arange(-num_bins / 2, num_bins / 2) * 10
    return x, out_mets, met_id


def act_embed(spk_data, behavior=None, catergorical=False, method='pca', n_comp='units', do_plots=False, plot_n_comp=3, umap_nn=15, umap_mind=0.1):
    # matplotlib.use('Qt5Agg')
    if n_comp == 'units':
        n_comp = np.shape(spk_data)[1]
    decomp = None
    if method == 'pca':
        decomp = PCA(n_components=n_comp)
    elif method == 'tsne':
        decomp = TSNE(n_components=3)
    elif method == 'umap':
        decomp = UMAP(n_neighbors=umap_nn, n_components=n_comp, min_dist=umap_mind, random_state=42)
    test = decomp.fit_transform(spk_data)

    if do_plots:
        # cat_data must be a 1xn vector of integers representing category identity for each time point
        cols = ['tab:green', 'tab:orange', 'tab:gray', 'c', 'm', 'r', 'k']
        c_data = behavior
        if catergorical:
            col_vec = []
            for c in behavior:
                if c >= 0:
                    col_vec.append(to_rgb(cols[int(c)]))
                else:
                    col_vec.append([1, 1, 1])
            c_data = col_vec
        f, axs = plt.subplots(plot_n_comp, plot_n_comp)
        for i in range(plot_n_comp):
            for j in range(plot_n_comp):
                if i != j:
                    axs[i, j].scatter(test[:, j], test[:, i], c=c_data, alpha=0.3, s=1, vmin=0, vmax=2)
                    axs[i, j].set_xlabel(f'{method} {j}')
                    axs[i, j].set_ylabel(f'{method} {i}')
    return test, decomp


def make_one_hot(cats_1d):
    num_cats = int(max(cats_1d))+1
    one_hot = np.zeros((len(cats_1d), num_cats))
    for i in range(num_cats):
        one_hot[:, i] = cats_1d == i
    return one_hot


def circ_random_walk(n_steps=20000, bias=None, circ_rad=30, x_pos=0, y_pos=0, vel=1, ang=0):
    pdf_size = 180
    if bias is None:
        bias = np.ones(pdf_size)
    else:
        pdf_size = len(bias)
    norm_bias = bias / np.sum(bias)
    vels = np.random.poisson(lam=0.005, size=n_steps)
    t_vals = np.linspace(-np.pi, np.pi, pdf_size)
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    rand_gen = Generator(PCG64())
    for i in range(n_steps):
        temp_bias = norm_bias.copy()
        vel = vels[i]
        ang_diffs = (((t_vals - ang) + np.pi) % (2 * np.pi) - np.pi)
        new_angs = 0.25 * ang_diffs + ang
        print(min(new_angs), max(new_angs))
        samp_xs = x_pos + np.sin(new_angs) * vel
        samp_ys = y_pos + np.cos(new_angs) * vel
        samp_rs = np.array([np.linalg.norm([tx, ty]) for tx, ty in zip(samp_xs, samp_ys)])
        # print(np.min(samp_rs))
        temp_bias[samp_rs > circ_rad] = 0
        temp_bias = temp_bias / np.nansum(temp_bias)
        choice_ang = rand_gen.choice(t_vals, size=1, p=temp_bias)
        diff_ang = (((choice_ang - ang) + np.pi) % (2 * np.pi) - np.pi)
        new_ang = 0.25 * diff_ang + ang
        x_pos += np.sin(new_ang) * vel
        y_pos += np.cos(new_ang) * vel
        xs[i] = x_pos
        ys[i] = y_pos
    return xs, ys


def generate_spikes_from_behavior(behavior, exite_inhibit='excite', noise=0.1):
    norm_b = (behavior - np.min(behavior)) / (np.max(behavior) - np.min(behavior))
    if exite_inhibit == 'inhibit':
        norm_b = 1-norm_b
    norm_noise = ((1 - noise/2) - noise/2) * norm_b + noise/2
    sim_spikes = np.random.binomial(1, norm_noise, len(behavior))
    sim_spikes = np.where(sim_spikes == 1)[0]
    return sim_spikes


def simulate_decomp(vec_len=52000, num_u=150, method='umap', n_comp=3, umap_nn=45, umap_mind=0.4):
    x, y, v, a = circ_random_walk(n_steps=vec_len)
    unit_dict = {}
    behavs = [x, y, v, a]
    b_num = 0
    for i in range(num_u//2):
        b = behavs[b_num % len(behavs)]
        spks = generate_spikes_from_behavior(b, noise=np.random.uniform(0.3, 0.5)) / 30
        unit_dict[str(i)] = spks
        b_num += 1
    fr = spikes_to_timeseries(unit_dict, gaus_fr, 30, 30, vec_len / 30)[1]
    test, umod = act_embed(fr, method=method, n_comp=n_comp, umap_nn=umap_nn, umap_mind=umap_mind)
    return test


def bayesian_decode(cm_x, cm_y, fr, spat_bin_size=5, cv_folds=2, cv_repeats=1, dt=0.025):

    X = np.vstack((cm_x, cm_y)).T
    X += np.abs(np.nanmin(X, axis=0))

    y = zscore(fr, axis=0)

    Px, bin_edges, bin_numbers = compute_Px(X, spat_bin_size)

    crossval = RepeatedKFold(n_splits=cv_folds, n_repeats=cv_repeats)

    errors = []
    cv_pred = []
    for i, (train, test) in enumerate(crossval.split(y)):
        Y_train, Y_test = y[train], y[test]

        Pyx = compute_Pyx(Y_train, bin_numbers[train], Px.shape)

        lls = compute_lls(Y_test, Pyx, Px, dt)

        pred_X = np.array([np.unravel_index(time_bin.argmax(), time_bin.shape) for time_bin in lls])
        true_X = bin_numbers[test]

        errors.append(np.linalg.norm(true_X - pred_X, axis=1) * spat_bin_size)
        cv_pred.append((pred_X * spat_bin_size, true_X * spat_bin_size))

    errors = np.concatenate(errors)

    return errors, cv_pred


def compute_Px(x, spat_bin_size, bin_range=None):
    if bin_range is None:
        bins = (np.amax(x) - np.amin(x)) // spat_bin_size
        Px, bin_edges, bin_numbers = binned_statistic_dd(x, np.ones(x.shape[0]), statistic='count', bins=bins,
                                                         expand_binnumbers=True)
    else:
        bins = np.round((np.max(bin_range) - np.min(bin_range)) / spat_bin_size)
        Px, bin_edges, bin_numbers = binned_statistic_dd(x, np.ones(x.shape[0]), statistic='count', bins=bins,
                                                         expand_binnumbers=True, range=bin_range)
    bin_numbers -= 1
    Px = Px / np.sum(Px)
    return Px, bin_edges, bin_numbers.T


def compute_Pyx(y, bin_numbers, Px_shape):
    Pyx = np.zeros(Px_shape + (y.shape[1],))
    for bin_id in np.unique(bin_numbers, axis=0):
        Pyx[tuple(bin_id)] = np.mean(y[np.all(bin_numbers == bin_id, axis=1), :], axis=0)
    return Pyx


def compute_lls(y, Pyx, Px, dt):
    Pxy = np.zeros((y.shape[0],) + Px.shape)
    for time_bin in range(y.shape[0]):
        posterior = (np.power((dt * Pyx), y[time_bin, :])) * np.exp(dt * -1 * Pyx) / gamma(y[time_bin, :] + 1)
        posterior = posterior * np.expand_dims(Px, axis=-1)
        lls = np.log(posterior)
        lls = normalize_lls(lls)
        Pxy[time_bin] = lls

    return Pxy


def normalize_lls(lls):
    # lls[np.isneginf(lls)] = np.nan
    lls = np.nansum(lls, axis=-1)
    lls = np.exp(lls)

    # lls = lls - np.nanmax(lls)
    lls /= np.sum(lls)
    # lls = 1 - lls
    lls[np.isnan(lls)] = 0

    return lls


def granger(fr_mat, max_lag=10, time_len=40*60):
    if time_len is None:
        time_len = fr_mat.shape[0]
    num_us = fr_mat.shape[1]
    outps = np.zeros((num_us, num_us))
    for i in range(num_us):
        print(i, num_us)
        for j in range(num_us):
            if i != j:
                gres = grangercausalitytests(np.diff(fr_mat[:time_len, [i, j]], axis=0)[1:], max_lag, verbose=False)
                gpvals = [gres[g][0]['ssr_ftest'][1] for g in gres.keys()]
                minp, lagn = np.min(gpvals), np.argmin(gpvals)
                outps[i, j] = minp
    plt.imshow(outps)
    plt.show()


def functional_connectivity(spk_dict, ephys_hz=2500):
    fr_len_s = 60 * 10
    out_dict = {}
    for k, i in spk_dict.items():
        out_dict[k] = i[i < fr_len_s]
    fr_data = spikes_to_timeseries(out_dict, smooth_func=bin_spikes, time_win_ms=0.4, out_hz=ephys_hz, ts_len_s=fr_len_s + 1)[1]
    spk_len, n_us = np.shape(fr_data)
    cc_mat = np.nan * np.ones((n_us, n_us))
    lag_mat = np.nan * np.ones((n_us, n_us))
    num_lags = 50
    it_time = 0
    for u0 in range(50):
        for u1 in range(50):
            t0 = time.time()
            print(u0, u1, it_time)
            if u0 != u1 and u0 < u1:
                rs = []
                shifted = fr_data[:, u0]
                for lag in range(-num_lags, num_lags + 1):
                    end_val = shifted[0]
                    shifted[0:-1] = shifted[1:]
                    shifted[-1] = end_val
                    r = shifted @ fr_data[:, u1]
                    rs.append(r)
                cch = np.array(rs)
                c_ccg = hollowed_gaussian_kernel(cch)
                norm_ccg = zscore(c_ccg)
                max_lag = np.argmax(np.abs(norm_ccg))
                if norm_ccg[max_lag] > 4:
                    lag_mat[u0, u1] = (max_lag - num_lags) * 0.4
                    cc_mat[u0, u1] = norm_ccg[max_lag]
            t1 = time.time()
            it_time = t1 - t0
    f, axs = plt.subplots(1, 2)
    a0 = axs[0].imshow(cc_mat, aspect='auto', interpolation='none')
    plt.colorbar(a0)
    a1 = axs[1].imshow(lag_mat, aspect='auto', interpolation='none')
    plt.colorbar(a1)
    plt.show()


def hollowed_gaussian_kernel(cch, sigma=1, fraction_hollowed=.6):
    """
    Description
    ----------
    This function takes a cross-correlation histogram and convolves it with a large window
    "partially-hollowed" Gaussian.

    Detailed: To generate the low frequency baseline CCH, the observed CCG was convolved
    with a “partially hollow” Gaussian kernel (Stark and Abeles, JoNM, 2009), with a standard
    deviation of 10 ms, with a hollow fraction of 60% (i.e. 60% off the center bin).
    ----------

    Parameters
    ----------
    cch : np.ndarray
        The CCH array that should be smoothed.
    sigma : int
        The sigma for smoothing (in bins); defaults to 1.
    fraction_hollowed : float
        Proportion-wise, the amount of window hollowed; defaults to .6.
    ----------

    Returns
    ----------
    smoothed_cch : np.ndarray
        The hollow-Gaussian convolved CCH.
    ----------
    """

    smoothed_cch = np.zeros(cch.shape[0] * 3)
    input_array_reflected = np.concatenate((cch[::-1], cch, cch[::-1]))
    x_v = np.arange(smoothed_cch.shape[0])
    for idx in x_v:
        kernel_idx = np.exp(-(x_v - idx) ** 2 / (2 * sigma ** 2))
        kernel_idx[int(np.floor(kernel_idx.shape[0] / 2))] = kernel_idx[int(np.floor(kernel_idx.shape[0] / 2))] * (1 - fraction_hollowed)
        kernel_idx = kernel_idx / kernel_idx.sum()
        smoothed_cch[idx] = np.dot(kernel_idx, input_array_reflected)
    return smoothed_cch[cch.shape[0]:cch.shape[0] * 2]


def calculate_mi(x, y, num_states_x=5, num_states_y=11, x_lims=None, y_lims=None, as_states=True):

    if as_states:
        states_x, states_y = x, y
    else:
        states_x, num_states_x = get_states(x, x_lims, num_states_x)
        states_y, num_states_y = get_states(y, y_lims, num_states_y)

    Pxy, xedges, yedges = np.histogram2d(states_x, states_y, bins=(np.linspace(0, num_states_x, num_states_x+1), np.linspace(0, num_states_y, num_states_y+1)))
    Prs = Pxy / np.sum(Pxy)
    P_R = np.sum(Prs, axis=0)
    P_S = np.sum(Prs, axis=1)

    I_S_R = np.nansum(Prs.T * np.log2(Prs.T / (P_R[:, None] @ P_S[None, :])))
    exp_mat = np.linspace(0, 1, len(Prs))[:, None] * np.ones((len(Prs), Prs.shape[1]))
    exp_x = np.sum(exp_mat * Prs, axis=0)
    return I_S_R, Prs, exp_x


def mutual_info_change(x, y, num_states_x=5, num_states_y=11, x_lims=None, y_lims=None, num_bootstraps=10,
                       num_shuffles=10, as_states=False):
    if as_states:
        states_x, states_y = x, y
    else:
        states_x, num_states_x = get_states(x, x_lims, num_states_x)
        states_y, num_states_y = get_states(y, y_lims, num_states_y)

    sem_mi = None
    if num_bootstraps > 1:
        kfolds = RepeatedKFold(n_splits=2, n_repeats=num_bootstraps//2)
        bs_mi = []
        exp_acc = []
        for b, (train_inds, test_inds) in enumerate(kfolds.split(x)):
            this_x = states_x[train_inds]
            this_y = states_y[train_inds]
            I_S_R, Prs, exp_x = calculate_mi(this_x, this_y, num_states_x=num_states_x, num_states_y=num_states_y,
                                             as_states=as_states)

            if np.isnan(I_S_R):
                I_S_R = 0

            bs_mi.append(I_S_R)
            exp_acc.append(exp_x)
        mean_mi = np.nanmean(bs_mi)
        sem_mi = np.nanstd(bs_mi) / np.sqrt(len(bs_mi))
        mean_Es = np.nanmean(np.array(exp_acc), axis=0)
    else:
        mean_mi, Prs, mean_Es = calculate_mi(states_x, states_y)

    null_mi = []
    roll_n = np.random.randint(len(states_y), size=num_shuffles)
    for s in range(num_shuffles):
        this_y = np.roll(states_y, roll_n[s], axis=0)
        n_mi = calculate_mi(states_x, this_y, num_states_x=num_states_x, num_states_y=num_states_y, as_states=as_states)[0]
        null_mi.append(n_mi)

    p_val = 1 - (np.sum(mean_mi > null_mi) / len(null_mi))
    mean_null = np.nanmean(null_mi)
    std_null = np.nanstd(null_mi)
    sig_change = mean_mi > (mean_null + 2*std_null)

    if num_bootstraps > 1:
        print(f'Mutual information I(S,R) = {mean_mi:.4f} SEM = {sem_mi:.4f} p = {p_val:.4f} ΔMI = {(mean_mi - mean_null):.4f} Sig? {sig_change}')
    else:
        print(f'Mutual information I(S,R) = {mean_mi:.4f} p = {p_val:.4f} ΔMI = {(mean_mi - mean_null):.4f} Sig? {sig_change}')

    return mean_mi, sem_mi, p_val, mean_null, sig_change


def get_states(x, x_lims=None, num_states=10):
    if x_lims is None:
        x_min = np.min(x)
        x_max = np.max(x)
    else:
        x_min, x_max = x_lims[0], x_lims[1]

    if x.ndim == 1:
        x = x[:, None]

    states_x = np.digitize(x, np.linspace(x_min, x_max, num_states)) - 1
    total_bins = num_states

    if states_x.shape[1] == 2:
        states_x = states_x[:, 0] * num_states + states_x[:, 1]
        total_bins = num_states ** 2
    else:
        states_x = states_x[:, 0]
    return states_x, total_bins


# def fit_hmm_to_fr(spk_dict, num_states=None):
#     gpfa = GPFA(bin_size=0.1*s, em_max_iters=10)
#     spike_trains = [[neo.core.SpikeTrain(spk_dict[u], 1800, units=s) for u in spk_dict.keys()]]
#     latent_space = gpfa.fit_transform(spike_trains)
#     # latent_space, _ = act_embed(fr_mat, n_comp=10, method='umap')
#     key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
#     num_states = 3
#     emission_dim = 10
#     hmm = GaussianHMM(num_states, emission_dim)
#     params, props = hmm.initialize(key3, method="kmeans", emissions=latent_space)
#     params, lls = hmm.fit_em(params, props, latent_space, num_iters=20)
#     smooth_res = hmm.smoother(params, latent_space)
#     state_probs = smooth_res.smoothed_probs
#     f, axs = plt.subplots(num_states, 1)
#     [axs[i].plot(state_probs[:, i]) for i in range(num_states)]
#     plt.show()
