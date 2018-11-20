"""
The script stores utility functions.
"""
import tensorflow as tf
#import imageio
import numpy as np
import edward as ed
#import pystan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from edward.models import Categorical, Dirichlet, InverseGamma, Normal, MultivariateNormalDiag, Mixture, Empirical, ParamMixture

TRAIN_DIR = "../data/BSR/BSDS500/data/images/train/"


def load_image_matrix(img_no, reshape=True):
    full_train_path = TRAIN_DIR + '{}.jpg'.format(img_no)
    img = plt.imread(full_train_path)
    if reshape:
        img = img.reshape(-1, img.shape[-1])
    return img.astype(int)


def visualize_clustered_plot(img, clusters, posterior_mu, img_no, D, K, T):
    nrows, ncols = img.shape[0], img.shape[1]
    segmented_img = np.zeros((nrows, ncols, D), dtype='int')
    cluster_reshape = clusters.reshape(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            cluster_number = cluster_reshape[i, j]
            segmented_img[i, j] = posterior_mu[cluster_number].astype(int)
    fig = plt.figure()
    plt.imshow(segmented_img)
    plt.savefig('../tmp/img={}_K={}_T={}.png'.format(img_no, K, T))


def calculate_pdi():
    pass


def plot_pdi_heatmap():
    pass

