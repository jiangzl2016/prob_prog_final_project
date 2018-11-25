"""
The script stores utility functions.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import theano
import pymc3 as pm
from pymc3.stats import _log_post_trace
from pymc3.distributions.dist_math import rho2sd
from scipy.special import logsumexp as sp_logsumexp
import scipy.stats as st

TRAIN_DIR = "../data/BSR/BSDS500/data/images/train/"


def load_image_matrix(img_no, train_dir):
    full_train_path = train_dir + '{}.jpg'.format(img_no)
    img = plt.imread(full_train_path)
    reshape_img = img.reshape(-1, img.shape[-1])
    return img, reshape_img.astype(int)


def visualize_clustered_plot(img, clusters, posterior_mu, img_no, current_time,
                             d, k, t, save_img=False):
    nrows, ncols = img.shape[0], img.shape[1]
    segmented_img = np.zeros((nrows, ncols, d), dtype='int')
    cluster_reshape = clusters.reshape(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            cluster_number = cluster_reshape[i, j]
            segmented_img[i, j] = posterior_mu[cluster_number].astype(int)
    plt.figure()
    plt.imshow(segmented_img)
    if save_img:
        plt.savefig('../tmp/img_result/{}/\
        fitted_img={}_K={}_T={}_Time={}.png'.format(
            current_time, img_no, k, t, current_time))
    plt.figure()
    plt.imshow(img)
    if save_img:
        plt.savefig('../tmp/img_result/{}/\
        original_img={}_K={}_T={}_Time={}.png'.format(
            current_time, img_no, k, t, current_time))


def plot_and_save_image(img, metric_array, title, save_dir):
    plt.figure()
    nrows, ncols = img.shape[0], img.shape[1]
    metric_matrix = np.reshape(metric_array, (nrows, ncols))
    ax = sns.heatmap(metric_matrix)
    ax.set_title(title)
    plt.imshow(metric_matrix)
    plt.savefig(save_dir)


def log_likelihood_result(log_dir, img_no, k, t,
                          total_total_log_liks_value,
                          current_time, elapsed_time,
                          expected_log_liks_value):
    with open(log_dir + 'log_likelihood.txt', 'a') as fp:
        if os.stat(log_dir + 'log_likelihood.txt').st_size == 0:
            fp.write("img,K,T,log_lik,datetime,runtime\n")
        fp.write("{},{},{},{},{},{}\n".format(
            img_no, k, t, total_total_log_liks_value, current_time,
            elapsed_time))
    print('The data log likeilhood is: {}'.format(total_total_log_liks_value))
    print('The data expected log likelihood\
     is: {}'.format(expected_log_liks_value))


def log_important_ratio(approx, nsample):
    logp_func = approx.model.logp

    # in ADVI there are only 1 group approximation
    approx_group = approx.groups[0]
    if approx.short_name == "mean_field":
        mu_q = approx_group.params[0].eval()
        std_q = rho2sd(approx_group.params[1]).eval()
        logq_func = st.norm(mu_q, std_q)
    elif approx.short_name == "full_rank":
        packed_chol_q = approx_group.params[0]
        mu_q = approx_group.params[1].eval()
        dim = mu_q.shape[0]
        chol_q = pm.expand_packed_triangular(
            dim, packed_chol_q, lower=True).eval()
        cov_q = np.dot(chol_q, chol_q.T)
        logq_func = st.multivariate_normal(mu_q, cov_q)

    dict_to_array = approx_group.bij.map

    p_theta_y = []
    q_theta = []
    samples = approx.sample_dict_fn(nsample)  # type: dict
    points = ({name: records[i] for name, records in samples.items()}
              for i in range(nsample))

    for point in points:
        p_theta_y.append(logp_func(point))
        q_theta.append(np.sum(logq_func.logpdf(dict_to_array(point))))
    p_theta_y = np.asarray(p_theta_y)
    q_theta = np.asarray(q_theta)
    return p_theta_y, q_theta, p_theta_y - q_theta


def PSIS(approx, nsample):
    logp, logq, lw = log_important_ratio(approx, nsample)
    new_lw, k = pm.stats._psislw(lw[:, None], 1)
    return new_lw, k


def PDI(trace, model):
    log_px = _log_post_trace(trace, model)  # shape (nsamples, N_datapoints)

    # log posterior predictive density of data point n
    #  = E_{q(\theta)} p(x_n|\theta)
    lppd_n = sp_logsumexp(log_px, axis=0, b=1.0 / log_px.shape[0])

    mu_n = np.exp(lppd_n)

    var_log_n = np.var(log_px, axis=0)

    mu_log_n = np.mean(log_px, axis=0)

    var_n = np.var(np.exp(log_px), axis=0)

    pdi = np.divide(var_n, mu_n)
    pdi_log = np.divide(var_log_n, mu_log_n)

    wapdi = np.divide(var_log_n, np.log(mu_n))

    return pdi, pdi_log, wapdi


def predict_cluster(approx, nsample, X, model, K, cov="full"):
    xobs = model.x_obs
    complogp = xobs.distribution._comp_logp(theano.shared(X))
    f_complogp = model.model.fastfn(complogp)
    trace = approx.sample(nsample)

    point = model.test_point

    for i in np.arange(K):
        # take average over samples
        point['mu%i' % i] = np.mean(trace['mu%i' % i], axis=0)

        if cov == "full":
            label = 'chol_cov_%i_cholesky-cov-packed__' % i

        elif cov == "precision_diagonal":
            label = 'tau%i_log__' % i

        elif cov == "cov_diagonal":
            label = 'sigma_sq%i_log__' % i

        point[label] = np.mean(trace[label], axis=0)

    y = np.argmax(f_complogp(point), axis=1)
    return y, point


def get_segment_img(y, img, point, mcmc=False):
    nrows, ncols = img.shape[0], img.shape[1]
    D = img.shape[2]
    segmented_img = np.zeros((nrows, ncols, D), dtype='int')
    cluster_reshape = y.reshape(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            cluster_number = cluster_reshape[i, j]
            if mcmc:
                # point = posterior_mu
                segmented_img[i, j] = point[cluster_number].astype(int)
            else:
                segmented_img[i, j] = \
                    point['mu{0:d}'.format(cluster_number)].astype(int)
    return segmented_img


def test_import():
    print("successfully imported!")