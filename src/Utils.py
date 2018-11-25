"""
The script stores utility functions.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
    fig = plt.figure()
    plt.imshow(segmented_img)
    if save_img:
        plt.savefig('../tmp/img_result/{}/fitted_img={}_K={}_T={}_Time={}.png'.format(
            current_time, img_no, k, t, current_time))
    fig = plt.figure()
    plt.imshow(img)
    if save_img:
        plt.savefig('../tmp/img_result/{}/original_img={}_K={}_T={}_Time={}.png'.format(
            current_time, img_no, k, t, current_time))


def plot_and_save_image(img, metric_array, title, save_dir):
    fig = plt.figure()
    nrows, ncols = img.shape[0], img.shape[1]
    metric_matrix = np.reshape(metric_array, (nrows, ncols))
    ax = sns.heatmap(metric_matrix)
    ax.set_title(title)
    plt.imshow(metric_matrix)
    plt.savefig(save_dir)


def log_likelihood_result(log_dir, img_no, k, t, total_total_log_liks_value,
                          current_time, elapsed_time, expected_log_liks_value):
    with open(log_dir + 'log_likelihood.txt', 'a') as fp:
        if os.stat(log_dir + 'log_likelihood.txt').st_size == 0:
            fp.write("img,K,T,log_lik,datetime,runtime\n")
        fp.write("{},{},{},{},{},{}\n".format(
            img_no, k, t, total_total_log_liks_value, current_time,
            elapsed_time))
    print('The data log likeilhood is: {}'.format(total_total_log_liks_value))
    print('The data expected log likelihood is: {}'.format(expected_log_liks_value))
