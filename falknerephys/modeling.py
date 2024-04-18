from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from matplotlib.colors import to_rgb
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn import svm
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import TweedieRegressor, BayesianRidge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from falknerephys.behavior import behavioral_raster


def fit_glm(design_mat: np.ndarray, output: np.ndarray,
            model: sklearn.linear_model = TweedieRegressor(power=1, alpha=0.5, link='log')):
    orig_len = np.shape(output)[0]
    if design_mat.ndim < 2:
        design_mat = np.expand_dims(design_mat, axis=1)
    # if output.ndim < 2:
    #     output = np.expand_dims(output, axis=1)
    nans_x = np.any(np.isnan(design_mat), axis=1)
    nans_y = np.isnan(output)
    keep_inds = ~np.logical_or(nans_x, nans_y)
    design_mat = design_mat[keep_inds, :]
    output = output[keep_inds]
    X_train, X_test, y_train, y_test = train_test_split(design_mat, output, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    full_pred = np.nan * np.ones(orig_len)
    pred_vals = model.predict(design_mat)
    full_pred[keep_inds] = pred_vals
    return y_test, pred, full_pred, r2, model


def run_rc_glm(behav_pred, unit_data, glm_model=TweedieRegressor(power=1, alpha=0.5, link='log')):
    nans = np.any(np.isnan(behav_pred), axis=1)
    behav_pred = behav_pred[~nans]
    unit_data = unit_data[~nans]
    num_u = np.shape(unit_data)[1]
    num_f = np.shape(behav_pred)[1]
    coef_per_unit = []
    for u in range(num_u):
        u_fr = unit_data[:, u]
        X_train, X_test, y_train, y_test = train_test_split(behav_pred, u_fr,
                                                            test_size=0.2, random_state=42)
        glm_model.fit(X_train, y_train)
        full_pred = glm_model.predict(X_test)
        r2_full = r2_score(y_test, full_pred)
        rel_cont = []
        for f in range(num_f):
            part_X_train = np.delete(X_train, f, axis=1)
            part_X_test = np.delete(X_test, f, axis=1)
            part_glm = sklearn.clone(glm_model)
            part_glm.fit(part_X_train, y_train)
            part_pred = part_glm.predict(part_X_test)
            r2_part = r2_score(y_test, part_pred)
            rel_cont.append(1 - (r2_part / r2_full))
        coef_per_unit.append(rel_cont)
    rc_per_unit = np.array(coef_per_unit)
    for i in range(np.shape(rc_per_unit)[0]):
        best_behav = np.argmax(rc_per_unit[i, :])


def run_knn_class(u_data, class_data, max_n=10, rand_state=42, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(u_data, class_data, test_size=test_ratio,
                                                        random_state=rand_state)
    f1s = []
    for n in range(1, max_n + 1):
        knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1, weights='distance')
        knn.fit(X_train, y_train)
        test = knn.predict(X_test)
        f1 = f1_score(y_test, test, average=None)
        f1s.append(f1)
        print('k=', n, 'F1=', f1)
    f1s = np.array(f1s)
    best_n = np.argmax(np.mean(f1s, axis=1)) + 1
    print('Best model: k=', best_n)
    knn = KNeighborsClassifier(n_neighbors=best_n, weights='distance')
    knn.fit(X_train, y_train)
    all_data = knn.predict(u_data)
    test = knn.predict(X_test)
    f = plt.figure()
    gs = plt.GridSpec(2, 6)
    op_ax = f.add_subplot(gs[:, :1])
    op_ax.plot(np.arange(1, max_n + 1), f1s, 'k')
    op_ax.set_xlabel('k')
    op_ax.set_ylabel('F1')
    orig_data_ax = f.add_subplot(gs[0, 1:5])
    behavioral_raster(class_data, ax=orig_data_ax, fs=40)
    pred_data_ax = f.add_subplot(gs[1, 1:5])
    behavioral_raster(all_data, ax=pred_data_ax, fs=40)
    cm_test_ax = f.add_subplot(gs[:, 5])
    cm_cat = y_test.argmax(axis=1)
    cm_cat_pred = test.argmax(axis=1)
    not_pred = ~np.any(test, axis=1)
    cm_cat_pred[not_pred] = 3
    cm = confusion_matrix(cm_cat[~not_pred], cm_cat_pred[~not_pred], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Familiar', 'Intruder', 'Neutral'])
    disp.plot(ax=cm_test_ax)
    plt.show()


def run_reg_decoder(x_data, target_vars, model='glm', k=0):
    train_input, test_input, train_output, test_output = train_test_split(x_data, target_vars, test_size=0.2,
                                                                          random_state=42)
    model_obj = None
    if model == 'svm':
        model_obj = svm.SVR()
    elif model == 'glm':
        model_obj = TweedieRegressor(power=1, alpha=0.5, link='log')
    elif model == 'bayes':
        model_obj = BayesianRidge()
    elif model == 'knn':
        model_obj = KNeighborsRegressor(n_neighbors=k, n_jobs=-1, weights='distance')
    if target_vars.ndim > 1:
        num_targets = len(target_vars)
        model_obj = MultiOutputRegressor(model_obj)
    model_obj.fit(train_input, train_output)
    test_pred = model_obj.predict(test_input)
    mse = mean_squared_error(test_output, test_pred)
    pred_all = model_obj.predict(x_data)
    return mse, pred_all


def run_pred_randshuf(u_data, ter_id, k=3, chk_sz=40, rand_state=42, test_ratio=0.2, ax=plt.gca, model=None):
    if model is None:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
    len_data = np.shape(ter_id)[0]
    num_perm = int(np.floor(len_data / chk_sz))
    u_data_trim = u_data[:num_perm * chk_sz, :]
    perms = permutations(range(num_perm))
    f1s = []
    for i in range(num_perm):
        res_arr = [np.arange(start, stop + 1) for start, stop in zip(perms[i], perms[i] + chk_sz)]
        ind_ar = np.reshape(res_arr, len(u_data_trim))
        shuf_data = ter_id[ind_ar, :]
        X_train, X_test, y_train, y_test = train_test_split(u_data, shuf_data,
                                                            test_size=test_ratio, random_state=rand_state)
        this_model = clone(model)
        model.fit(X_train, y_train)
        test = model.predict(X_test)
        f1 = f1_score(y_test, test, average='weighted')
        f1s.append(f1)
        print('Shuffle: ', i, ' of ', num_perm)
    out_f1s = f1s[num_perm // 2:] + f1s[:num_perm // 2]
    num_bins = len(out_f1s)
    x = np.arange(-num_bins / 2, num_bins / 2) * 10
    ax.plot(x, out_f1s, 'k')
    ax.plot([0, 0], [0, 1], 'r--')
    ax.show()


def run_pred_circshuf(u_data, ter_id, k=3, step_sz=1, rand_state=42, test_ratio=0.2, ax=plt.gca, model=None):
    if model is None:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
    len_data = np.shape(ter_id)[0]
    num_shifts = int(np.floor(len_data / step_sz))
    f1s = []
    for i in range(num_shifts):
        shift_data = np.roll(ter_id, i * step_sz, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(u_data, shift_data,
                                                            test_size=test_ratio, random_state=rand_state)
        this_model = clone(model)
        model.fit(X_train, y_train)
        test = model.predict(X_test)
        f1 = f1_score(y_test, test, average='weighted')
        f1s.append(f1)
        print('Shuffle: ', i, ' of ', num_shifts)
    out_f1s = f1s[num_shifts // 2:] + f1s[:num_shifts // 2]
    num_bins = len(out_f1s)
    x = np.arange(-num_bins / 2, num_bins / 2) * 10
    ax.plot(x, out_f1s, 'k')
    ax.plot([0, 0], [0, 1], 'r--')
    ax.show()


def act_embed(spk_data, cat_data, method='tsne'):
    decomp = None
    if method == 'pca':
        decomp = PCA(n_components=3)
    elif method == 'tsne':
        decomp = TSNE(n_components=3)
    test = decomp.fit_transform(spk_data)
    # cat_data must be a 1xn vector of integers representing category identity for each time point
    cols = ['tab:green', 'tab:orange', 'tab:gray', 'c', 'm', 'r', 'k']
    num_cats = max(cat_data)
    col_vec = []
    for c in cat_data:
        if c >= 0:
            col_vec.append(to_rgb(cols[int(c)]))
        else:
            col_vec.append([1, 1, 1])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(test[:, 0], test[:, 1], test[:, 2], c=col_vec)
    plt.show()

# def run_glm_across_units(behav_pred, unit_data, ax=plt.gca):
#     nans = np.any(np.isnan(behav_pred), axis=1)
#     behav_pred = behav_pred[~nans]
#     unit_data = unit_data[~nans]
#     num_u = np.shape(unit_data)[1]
#     num_f = np.shape(behav_pred)[1]
#     f, axs = plt.subplots(num_u, 1)
#     coef_per_unit = []
#     for u in range(num_u):
#         u_fr = unit_data[:, u]
#         X_train, X_test, y_train, y_test = train_test_split(behav_pred, u_fr,
#                                                             test_size=0.2, random_state=42)
#         full_glm = TweedieRegressor(power=1, alpha=0.5, link='log')  # poisson
#         full_glm.fit(X_train, y_train)
#         full_pred = full_glm.predict(X_test)
#         r2_full = r2_score(y_test, full_pred)
#         rel_cont = []
#         for f in range(num_f):
#             part_behav = np.delete(behav_pred, f, axis=1)
#             part_X_train = np.delete(X_train, f, axis=1)
#             part_X_test = np.delete(X_test, f, axis=1)
#             part_glm = TweedieRegressor(power=1, alpha=0.5, link='log')  # poisson
#             part_glm.fit(part_X_train, y_train)
#             part_pred = part_glm.predict(part_X_test)
#             r2_part = r2_score(y_test, part_pred)
#             rel_cont.append(1-(r2_part/r2_full))
#         coef_per_unit.append(rel_cont)
#     rc_per_unit = np.array(coef_per_unit)
#     dists = pdist(rc_per_unit)
#     Z = hierarchy.complete(dists)
#     reorder = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, dists))
#     for ind, i in enumerate(reorder):
#         best_behav = np.argmax(rc_per_unit[i, :])
#         axs[ind].plot(unit_data[:, i])
#         axs[ind].plot(behav_pred[:, best_behav])
#         axs[ind].set_title('Unit ' + str(i) + ': ' + str(rc_per_unit[i, :]))
