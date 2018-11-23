"""
The script runs MCMC on GMM.
"""
import tensorflow as tf
#import imageio
import numpy as np
import edward as ed
#import pystan
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from datetime import datetime
import pickle
import json
import seaborn as sns
from edward.models import Categorical, Dirichlet, InverseGamma, Normal, MultivariateNormalDiag, Mixture, Empirical, ParamMixture
from Utils import load_image_matrix

img_no = 2092
TRAIN_DIR = "../data/BSR/BSDS500/data/images/train/"
train_img = load_image_matrix(img_no, TRAIN_DIR, reshape=True)

# Hyperparameters
N = train_img.shape[0]
K = 6
D = train_img.shape[1]
T = 100  # number of MCMC samples
M = 200  # number of posterior samples sampled

ed.set_seed(1234)

with tf.name_scope("model"):
    pi = Dirichlet(concentration=tf.constant([1.0] * K, name="pi/weights"), name= "pi")
    mu = Normal(loc= tf.zeros(D, name="centroids/loc"),
                scale= tf.ones(D, name="centroids/scale"),
                sample_shape=K, name= "centroids")
    sigma = InverseGamma(concentration=tf.ones(D, name="variability/concentration"),
                         rate=tf.ones(D, name="variability/rate"), sample_shape=K, name= "variability")
    #cat = Categorical(probs= pi, sample_shape= N)

    x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigma)},
                     MultivariateNormalDiag,
                     sample_shape=N, name= "mixture")
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
print("Sampling Done!")

inference.initialize(n_print=50, logdir='log/IMG={}_K={}_T={}'.format(img_no, K, T))
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
        posterior_mu = sess.run(running_cluster_means, {t_ph: t - 1})
        print(posterior_mu)
inference.finalize()

print("Inference Done!")
Gibbs_inference_elapsedTime = time.time() - Gibbs_inference_startTime


# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(M)
sigmasq_sample = qsigma.sample(M)
pi_sample = qpi.sample(M)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(train_img, [N, 1, 1, D]), [1, M, K, 1])
x_broadcasted = tf.cast(x_broadcasted, dtype= tf.float32)

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

clusters = tf.argmax(log_liks, 1).eval()

nrows, ncols = img.shape[0], img.shape[1]
segmented_img = np.zeros((nrows, ncols, D),dtype='int')
cluster_reshape = clusters.reshape(nrows, ncols)
for i in range(nrows):
    for j in range(ncols):
        cluster_number = cluster_reshape[i, j]
        segmented_img[i, j] = posterior_mu[cluster_number].astype(int)
plt.imshow(segmented_img)
plt.savefig('./tmp/img={}_K={}_T={}.png'.format(img_no, K, T))
