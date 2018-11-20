import theano

import pymc3 as pm

from pymc3.math import logsumexp
from pymc3.stats import _log_post_trace
from pymc3.distributions.dist_math import rho2sd

from scipy.special import logsumexp as sp_logsumexp
import scipy.stats as st

import numpy as np



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
        chol_q = pm.expand_packed_triangular(dim, packed_chol_q, lower=True).eval()
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
    return p_theta_y - q_theta


def PSIS(approx, nsample):
    ratio = log_important_ratio(approx, nsample)
    lw, k = pm.stats._psislw(ratio[:, None], 1)
    return k


def PDI(trace, model):
    log_px = _log_post_trace(trace, model)  # shape (nsamples, N_datapoints)

    # log posterior predictive density of data point n = E_{q(\theta)} p(x_n|\theta)
    lppd_n = sp_logsumexp(log_px, axis=0, b=1.0 / log_px.shape[0])

    mu_n = np.exp(lppd_n)

    var_log_n = np.var(log_px, axis=0)

    mu_log_n = np.mean(log_px, axis=0)

    var_n = np.var(np.exp(log_px), axis=0)

    pdi = np.divide(var_n, mu_n)
    pdi_log = np.divide(var_log_n, mu_log_n)

    wapdi = np.divide(var_log_n, np.log(mu_n))

    return pdi, pdi_log, wapdi


def predict_cluster(approx, nsample, X, model, xobs, K):
    complogp = xobs.distribution._comp_logp(theano.shared(X))
    f_complogp = model.model.fastfn(complogp)
    trace = approx.sample(nsample)

    point = model.test_point
    for i in np.arange(K):
        point['mu%i' % i] = np.mean(trace['mu%i' % i], axis=0)  # take average over samples
        chollabel = 'chol_cov_%i_cholesky-cov-packed__' % i
        point[chollabel] = np.mean(trace[chollabel], axis=0)  # take average over samples

    y = np.argmax(f_complogp(point), axis=1)
    return y, point


def test_import_fun():
    print("hello, world!")