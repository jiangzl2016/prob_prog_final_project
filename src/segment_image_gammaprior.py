import theano
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import theano.tensor as tt
from utils import PDI, predict_cluster


plt.style.use('ggplot')
# plt.rcParams["axes.grid"] = False

test_idx = 3

# Number of cluster setting
K = 20


# Read image
img_name = "61060"
print("reading image" + img_name + "...")

train_path = "/Users/leah/Columbia/courses/ml_prob_programming/" \
             "ENV/BSR/BSDS500/data/images/train/"
img = plt.imread(train_path + img_name + ".jpg")

X = img.reshape(-1, 3).astype(int)
N, D = X.shape

# Out-path
out_path = "out_gammaprior/" + img_name + \
           "_{0:d}".format(test_idx)
os.mkdir(out_path)

plt.imshow(img)
plt.grid(None)
plt.savefig(out_path + "/" + img_name + ".jpg")


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
    tau = []
    precision = []
    for i in range(K):
        temp_mean = np.random.randint(low=50, high=200, size=D)
        mu.append(pm.Normal('mu%i' % i, temp_mean, 20, shape=D))
        tau.append(pm.Gamma('tau%i' % i, 1, 1, shape=D))

        precision.append(tt.nlinalg.alloc_diag(tau[i]))
        comp_dist.append(pm.MvNormal.dist(mu=mu[i], tau=precision[i]))

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
                           X=X, model=model, K=K, cov="diagonal")
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
pdi, pdi_log, wapdi = PDI(post_samples, model)
# average over 100 samples

print("plotting post samples...")
pm.traceplot(post_samples)
plt.savefig(out_path + "/post_distribution.jpg")

print("ploting PDIs")
# plot PDI
pdi_reshape = pdi.reshape(nrows, ncols)
pdi_log_reshape = pdi_log.reshape(nrows, ncols)
wapdi_reshape = wapdi.reshape(nrows, ncols)


fig = plt.figure()
fig.subplots_adjust(hspace=0.5, wspace=0.5)

plt.subplot(1, 2, 1)
sns.set()
ax = sns.heatmap(pdi_reshape, cbar_kws={"shrink": 0.4})
ax.set_title("pdi")
plt.imshow(pdi_reshape)

plt.subplot(1, 2, 2)
sns.set()
ax = sns.heatmap(np.log(pdi_reshape), cbar_kws={"shrink": 0.4})
ax.set_title("log(pdi)")
plt.imshow(np.log(pdi_reshape))

plt.savefig(out_path + "/PDI_plot1.jpg", dpi=400)

fig = plt.figure()
fig.subplots_adjust(hspace=0.5, wspace=0.5)

plt.subplot(1, 2, 1)
sns.set()
ax = sns.heatmap(pdi_log_reshape, cbar_kws={"shrink": 0.4})
ax.set_title("pdi_log")
plt.imshow(pdi_log_reshape)

plt.subplot(1, 2, 2)
sns.set()
ax = sns.heatmap(wapdi_reshape, cbar_kws={"shrink": 0.4})
ax.set_title("wapdi")
plt.imshow(wapdi_reshape)

plt.savefig(out_path + "/PDI_plot2.jpg", dpi=400)
