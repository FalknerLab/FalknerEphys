import random

import numpy as np

import matplotlib.pyplot as plt
import sklearn
from matplotlib.colors import to_rgb
from sklearn import svm
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import TweedieRegressor, BayesianRidge, LogisticRegression
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

from falknerephys.preprocess import gaus_fr, spikes_to_timeseries


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


def run_rc_glm(behav_pred, unit_data, glm_model=TweedieRegressor(power=1, alpha=0.5, link='log')):
    nans = np.any(np.isnan(behav_pred), axis=1)
    behav_pred = behav_pred[~nans]
    unit_data = unit_data[~nans]
    num_u = np.shape(unit_data)[1]
    num_f = np.shape(behav_pred)[1]
    coef_per_unit = []
    r2_per_unit = []
    for u in range(num_u):
        u_fr = unit_data[:, u]
        X_train, X_test, y_train, y_test = train_test_split(behav_pred, u_fr,
                                                            test_size=0.2, random_state=42)
        full_glm = clone(glm_model)
        full_glm.fit(X_train, y_train)
        full_pred = full_glm.predict(X_test)
        r2_full = r2_score(y_test, full_pred)
        rel_cont = []
        for f in range(num_f):
            part_X_train = np.delete(X_train, f, axis=1)
            part_X_test = np.delete(X_test, f, axis=1)
            part_glm = sklearn.clone(glm_model)
            part_glm.fit(part_X_train, y_train)
            part_pred = part_glm.predict(part_X_test)
            r2_part = r2_score(y_test, part_pred)
            rel_cont.append((r2_full - r2_part) / r2_full)
        coef_per_unit.append(rel_cont/np.sum(rel_cont))
        r2_per_unit.append(r2_full)
    rc_per_unit = np.array(coef_per_unit)
    r2_per_unit = np.array(r2_per_unit)
    return rc_per_unit, r2_per_unit


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


def run_reg_decoder(x_data, target_vars, model='glm', k=0, categorical=False, test_inds=None):
    if test_inds is None:
        train_input, test_input, train_output, test_output = train_test_split(x_data, target_vars, test_size=0.2,
                                                                              random_state=42)
    else:
        test_input = x_data[test_inds]
        test_output = target_vars[test_inds]
        train_input = x_data[~test_inds]
        train_output = target_vars[~test_inds]
    model_obj = None
    if type(model) == str:
        if model == 'svm':
            model_obj = svm.SVR(kernel='linear')
            if categorical:
                model_obj = svm.SVC(kernel='linear')
        elif model == 'glm':
            model_obj = TweedieRegressor(power=1, alpha=0.5, link='log')
        elif model == 'bayes':
            model_obj = BayesianRidge()
        elif model == 'knn':
            model_obj = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
        elif model == 'logistic':
            model_obj = LogisticRegression()
    else:
        model_obj = model
    if target_vars.ndim > 1:
        model_obj = MultiOutputRegressor(model_obj)
    model_obj.fit(train_input, train_output)
    test_pred = model_obj.predict(test_input)
    mse = mean_squared_error(test_output, test_pred)
    f1 = None
    if categorical:
        f1 = f1_score(test_output, test_pred, average='weighted')
    pred_all = model_obj.predict(x_data)
    return model_obj, pred_all, mse, f1, test_output, test_pred


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


def simulate_movement(len_output=30000):
    vel_range = (0, 1)
    acc_range = (-0.05, 0.05)
    turn_vel_range = (-0.5, 0.5)
    x_pos = 0
    y_pos = 0
    ang = np.random.uniform(-np.pi, np.pi, 1)
    vel = np.random.uniform(vel_range[0], vel_range[1], 1)
    x_out = np.zeros(len_output)
    y_out = np.zeros(len_output)
    vel_out = np.zeros(len_output)
    acc_out = np.zeros(len_output)
    for i in range(len_output):
        x_pos += np.sin(ang) * vel
        y_pos += np.cos(ang) * vel
        r_acc = np.random.uniform(acc_range[0], acc_range[1], 1)
        vel += r_acc
        vel = max(vel, vel_range[0])
        vel = min(vel, vel_range[1])
        d_ang = np.random.uniform(turn_vel_range[0], turn_vel_range[1], 1)
        ang += d_ang
        x_out[i] = x_pos
        y_out[i] = y_pos
        vel_out[i] = vel
        acc_out[i] = r_acc
    return x_out, y_out, vel_out, acc_out


def generate_spikes_from_behavior(behavior, exite_inhibit='excite', noise=0.1):
    norm_b = (behavior - np.min(behavior)) / (np.max(behavior) - np.min(behavior))
    if exite_inhibit == 'inhibit':
        norm_b = 1-norm_b
    norm_noise = ((1 - noise/2) - noise/2) * norm_b + noise/2
    sim_spikes = np.random.binomial(1, norm_noise, len(behavior))
    sim_spikes = np.where(sim_spikes == 1)[0]
    return sim_spikes


def simulate_decomp(vec_len=52000, num_u=150, method='umap', n_comp=3, umap_nn=45, umap_mind=0.4):
    x, y, v, a = simulate_movement(len_output=vec_len)
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


def compute_Px(x, spat_bin_size):
    num_bins = (np.amax(x) - np.amin(x)) // spat_bin_size
    Px, bin_edges, bin_numbers = binned_statistic_dd(x, np.ones(x.shape[0]), statistic='count', bins=num_bins,
                                                     expand_binnumbers=True)

    # num_bins = np.floor(np.max(x, axis=0) / spat_bin_size)
    # bins0 = np.linspace(0, num_bins[0] * spat_bin_size, int(num_bins[0]))
    # bins1 = np.linspace(0, num_bins[1] * spat_bin_size, int(num_bins[1]))
    # Px, bin_edges, bin_numbers = binned_statistic_dd(x, np.ones(x.shape[0]), statistic='count', bins=(bins0, bins1),
    #                                                  expand_binnumbers=True)
    bin_numbers -= 1
    Px = Px / np.sum(Px)

    return Px, bin_edges, bin_numbers.T


def compute_Pyx(y, bin_numbers, Px_shape):
    Pyx = np.zeros(Px_shape + (y.shape[1],))
    for bin_id in np.unique(bin_numbers, axis=0):
        # bin_id = np.array([bin_id,])
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
