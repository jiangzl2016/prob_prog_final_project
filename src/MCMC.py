"""
The script runs MCMC on GMM.
"""
import tensorflow as tf
import numpy as np
import edward as ed
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
import seaborn as sns
from edward.models import Categorical, Dirichlet, InverseGamma, Normal, \
    MultivariateNormalDiag, Mixture, Empirical, ParamMixture
from Utils import load_image_matrix, visualize_clustered_plot, \
    plot_and_save_image, log_likelihood_result

img_no = 2092
TRAIN_DIR = "../data/BSR/BSDS500/data/images/train/"
img, train_img = load_image_matrix(img_no, TRAIN_DIR)

# Hyperparameters
N = train_img.shape[0]
K = 9
D = train_img.shape[1]
T = 300  # number of MCMC samples
M = 100  # number of posterior samples sampled

ed.set_seed(1234)

with tf.name_scope("model"):
    pi = Dirichlet(concentration=tf.constant([1.0] * K, name="pi/weights"),
                   name="pi")
    mu = Normal(loc=tf.zeros(D, name="centroids/loc"),
                scale=tf.ones(D, name="centroids/scale"),
                sample_shape=K, name="centroids")
    sigma = InverseGamma(concentration=tf.ones(D,
                                               name="variability/concentration"),
                         rate=tf.ones(D, name="variability/rate"),
                         sample_shape=K, name="variability")

    x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigma)},
                     MultivariateNormalDiag,
                     sample_shape=N, name="mixture")
    z = x.cat

with tf.name_scope("posterior"):
    qpi = Empirical(tf.get_variable(
        "qpi/params", [T, K],
        initializer=tf.constant_initializer(1.0/K)))
    qmu = Empirical(tf.get_variable(
        "qmu/params", [T, K, D],
        initializer=tf.zeros_initializer()))
    qsigma = Empirical(tf.get_variable(
        "qsigma/params", [T, K, D],
        initializer=tf.ones_initializer()))
    qz = Empirical(tf.get_variable(
        "qz/params", [T, N],
        initializer=tf.zeros_initializer(),
        dtype=tf.int32))

print("Running Gibbs Sampling...")
Gibbs_inference_startTime = time.time()
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
inference = ed.Gibbs({pi: qpi, mu: qmu, sigma: qsigma, z: qz},
                     data={x: train_img})
print("Sampling Done.")

inference.initialize(n_print=200, logdir='log/IMG={}_K={}_T={}'.format(
    img_no, K, T))
sess = ed.get_session()
tf.global_variables_initializer().run()
t_ph = tf.placeholder(tf.int32, [])
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)

learning_curve = []
for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']
    if t % inference.n_print == 0:
        print("\nInferred cluster means:")
        last_mu = sess.run(running_cluster_means, {t_ph: t - 1})
inference.finalize()

print("Inference Done.")
Gibbs_inference_elapsedTime = time.time() - Gibbs_inference_startTime
posterior_mu = qmu.params.eval().mean(axis=0)

# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
print("Sampling from Posterior...")
mu_sample = qmu.sample(M)
sigmasq_sample = qsigma.sample(M)
pi_sample = qpi.sample(M)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(train_img, [N, 1, 1, D]), [1, M, K, 1])
x_broadcasted = tf.cast(x_broadcasted, dtype=tf.float32)
# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = tf.reduce_mean(tf.reduce_sum(x_post.log_prob(x_broadcasted), 3), 1)

print("Calculating Cluster Assignment...")
clusters = tf.argmax(log_liks, 1).eval()

result_img_dirs = '../tmp/img_result/{}'.format(current_time)
os.makedirs(result_img_dirs)
plt.hist(clusters)
plt.savefig('../tmp/img_result/{}/cluster_dist_img={}_K={}_T={}_Time={}.png'.format(
    current_time, img_no, K, T, current_time))
result_cluster_assign_dirs = '../tmp/log/cluster_assign_matrix'
if not os.path.isdir(result_cluster_assign_dirs):
    os.makedirs(result_cluster_assign_dirs)
np.save(result_cluster_assign_dirs +
        "/cluster_assign={}_K={}_T={}_Time={}.npy".format(
            img_no, K, T, current_time),
        clusters.reshape(img.shape[0], img.shape[1]))


# Draw post-clustering image
visualize_clustered_plot(img, clusters, posterior_mu, img_no, current_time,
                         D, K, T, save_img=True)

# Criticism
# log posterior density
log_liks_3 = x_post.log_prob(x_broadcasted)
log_liks_3 = tf.reduce_sum(log_liks_3, 3)
pi_samples_tiled_3 = tf.tile(tf.expand_dims(tf.log(pi_sample), 0), [N, 1, 1])
sum_log_pi_normal_3 = tf.add(pi_samples_tiled_3, log_liks_3)
x_max_3 = tf.reduce_max(sum_log_pi_normal_3, axis=2)
edited_log_sum_3 = tf.add(x_max_3,
                          tf.log(tf.reduce_sum(tf.exp(sum_log_pi_normal_3 -
                                                      tf.expand_dims(x_max_3, 2)),
                                               axis=2)))
averaged_total_log_liks_over_pi_samples_3 = tf.reduce_mean(tf.reduce_sum(
    edited_log_sum_3, 0), 0)
averaged_expected_log_liks_over_pi_samples_3 = tf.reduce_mean(
    tf.reduce_mean(edited_log_sum_3, 0), 0)
averaged_log_liks_pointwise = tf.reduce_mean(edited_log_sum_3, 1)

total_total_log_liks_value = sess.run(averaged_total_log_liks_over_pi_samples_3)
expected_log_liks_value = sess.run(averaged_expected_log_liks_over_pi_samples_3)
pointwise_log_liks_value = sess.run(averaged_log_liks_pointwise)

# Record the log likelihood_result
LOG_RESULT_DIR = "../tmp/log/"
log_likelihood_result(LOG_RESULT_DIR, img_no, K, T, total_total_log_liks_value,
                      current_time, Gibbs_inference_elapsedTime,
                      expected_log_liks_value)

# WAPDI and PDI
pdi_mean, pdi_variance = tf.nn.moments(tf.exp(edited_log_sum_3), axes=[1])
pdi = tf.divide(pdi_variance, pdi_mean)
pdi_values = sess.run(pdi)

LOGPDI_DIR = '../tmp/img_result/{}/logpdi_img={}_K={}_T={}_Time={}.png'.format(
    current_time, img_no, K, T, current_time)
plot_and_save_image(img, np.log(pdi_values), "log pdi", LOGPDI_DIR)

# WAPDI
_, wapdi_variance = tf.nn.moments(edited_log_sum_3, axes=[1])
wapdi_log_mean = tf.log(pdi_mean)
wapdi = tf.divide(wapdi_variance, wapdi_log_mean)
wapdi_values = sess.run(wapdi)

WAPDI_DIR = '../tmp/img_result/{}/wapdi_img={}_K={}_T={}_Time={}.png'.format(
    current_time, img_no, K, T, current_time)
plot_and_save_image(img, wapdi_values, "wapdi", WAPDI_DIR)

POINT_LIKS_DIR = '../tmp/img_result/{}/logliks_img={}_K={}_T={}_Time={}.png'.format(
    current_time, img_no, K, T, current_time)
plot_and_save_image(img, pointwise_log_liks_value,
                    "pointwise log likeihood", POINT_LIKS_DIR)
