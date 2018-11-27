import theano
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import theano.tensor as tt
from utils import PDI, predict_cluster, plot_pdi_wapdi

plt.style.use('ggplot')
# plt.rcParams["axes.grid"] = False


test_idx = 1

# Number of cluster setting
K = 20

# Read image
img_no = 299091
print("reading image {0:d} ...".format(img_no))

train_path = "../BSR/BSDS500/data/images/train/"
img = plt.imread(train_path + "{0:d}.jpg".format(img_no))

X = img.reshape(-1, 3).astype(int)
N, D = X.shape

# Out-path
out_path = "../out/ADVI_invgamma/" + \
           "{0:d}_{1:d}".format(img_no, test_idx)
os.mkdir(out_path)

plt.imshow(img)
plt.grid(None)
plt.savefig(out_path + "/{0:d}.jpg".format(img_no))

print("defining model...")
# Define model
X_shared = theano.shared(X)
minibatch_size = 500
X_minibatch = pm.Minibatch(X, minibatch_size)

# set up model
with pm.Model() as model:
    pi = pm.Dirichlet('pi', np.ones(K))

    comp_dist = []
    mu = []
    sigma_sq = []
    cov = []
    for i in range(K):
        temp_mean = np.random.randint(low=20, high=230, size=D)
        mu.append(pm.Normal('mu%i' % i, temp_mean, 20, shape=D))
        sigma_sq.append(
            pm.InverseGamma('sigma_sq%i' % i, 1, 1, shape=D))

        cov.append(tt.nlinalg.alloc_diag(sigma_sq[i]))
        comp_dist.append(pm.MvNormal.dist(mu=mu[i], cov=cov[i]))

    xobs = pm.Mixture('x_obs', pi, comp_dist,
                      observed=X_shared)

print("making inference...")
# Inference
with model:
    advi_mf = pm.ADVI()
    advi_mf.fit(10000, more_replacements={X_shared: X_minibatch},
                obj_optimizer=pm.adagrad(learning_rate=1e-2))

fig = plt.figure()
plt.plot(advi_mf.hist)
plt.title("loss function")
plt.savefig(out_path + "/lossPlot.jpg")

print("making prediction...")
# Prediction
y, point = predict_cluster(approx=advi_mf.approx, nsample=1000,
                           X=X, model=model, K=K,
                           cov="cov_diagonal")
nrows, ncols = img.shape[0], img.shape[1]
segmented_img = np.zeros((nrows, ncols, D), dtype='int')
cluster_reshape = y.reshape(nrows, ncols)
for i in range(nrows):
    for j in range(ncols):
        cluster_number = cluster_reshape[i, j]
        segmented_img[i, j] = \
            point['mu{0:d}'.format(cluster_number)].astype(int)

fig = plt.figure()
plt.imshow(segmented_img)
plt.grid(None)
plt.title("Segmented image using {0:d} clusters".format(K))
plt.savefig(out_path + "/seg_img.jpg")

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)
axs[0].grid(None)
axs[0].set_title("original image")
axs[1].imshow(segmented_img)
axs[1].grid(None)
axs[1].set_title("segmented image, {0:d} clusters".format(K))
plt.savefig(out_path + "/" + "comparison.jpg", pdi=400)

fig = plt.figure()
plt.hist(y, bins=K)
plt.title("histogram of cluster assignments")
plt.xlabel("cluster index")
plt.ylabel("number of pixels")
plt.savefig(out_path + "/cluster_histogram.jpg")

print("doing diagnosis...")
# Diagnosis
post_samples = advi_mf.approx.sample(100)
# average over 100 samples
pdi, pdi_log, wapdi = PDI(post_samples, model)

print("plotting post samples...")
pm.traceplot(post_samples)
plt.savefig(out_path + "/post_distribution.jpg")

print("ploting PDIs")
# plot PDI heatmap
log_pdi = np.log(pdi)
plot_pdi_wapdi(pdi, log_pdi, pdi_log, wapdi,
               k=0.5, plot_type="dist")
plt.savefig(out_path + "/pdi_dist.jpg")

plot_pdi_wapdi(pdi, log_pdi, pdi_log, wapdi,
               img=img, seg_img=segmented_img,
               name="ADVI", k=0.5, plot_type="heatmap")
plt.savefig(out_path + "/pdi_heat.jpg")
