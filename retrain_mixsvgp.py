import numpy as np
import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
from MixtureSVGP import MixtureSVGP, generate_updates
import pickle
from utils import load_data
import sys
import os

minibatch_size = 10000
grad_iters = 50

model_path = sys.argv[1]
params, assignments, elbos = pickle.load(open('output/models/mixsvgp_K2_L100_28369515', 'rb'))

pi, psi, rho = assignments['pi'], assignments['psi'], assignments['rho']
Phi, Lambda, Gamma = assignments['Phi'], assignments['Lambda'], assignments['Gamma']

N, K = Phi.shape
G, L = Lambda.shape

n_iters = 10

normalized_data_df, x, data_dict = load_data(
    'data/quantile_normalized_no_projection.txt')
n_lines, n_samples, n_genes = x.shape
y = x.transpose(0, 2, 1)
T = n_samples

y = y[:N, :G, :]
x = np.tile(np.arange(T).astype(np.float64), (N, G, 1))

# create update functions
compute_weights, update_assignments = generate_updates(N, G, K, L, T)

mask = ~np.isnan(y.reshape(-1, 1)).squeeze()
num_data = mask.sum()
num_clusters = K * L + L
minibatch_size = np.minimum(num_data, minibatch_size)

X = x.reshape(-1, 1)[mask]
Y = y.reshape(-1, 1)[mask]

weights = compute_weights(Phi, Lambda, Gamma)
_, weight_idx, = np.unique(
    np.tile(np.arange(N * G).reshape(
        (N, G))[:, :, None], T).reshape(-1, 1)[mask], return_inverse=True)

# create model
kernel = mk.SharedIndependentMok(gpflow.kernels.RBF(1), num_clusters)
feature = mf.SharedIndependentMof(
    gpflow.features.InducingPoints(np.arange(T).astype(
        np.float64).reshape(-1, 1)))

m = MixtureSVGP(X, Y, weight_idx,
                kern=kernel,
                num_clusters=num_clusters, num_data=num_data,
                likelihood=gpflow.likelihoods.Gaussian(),
                feat=feature, minibatch_size=minibatch_size)

m.feature.feat.Z.trainable = False
m.assign(params)


# optimize model parameters
opt = gpflow.train.AdamOptimizer()
for _ in range(n_iters):
    opt.minimize(m, maxiter=grad_iters, feed_dict={m.weights: weights})

    out_path = 'model'
    elbos.append(m.compute_log_likelihood(feed_dict={m.weights: weights}))

    # save model
    with open(model_path, 'wb') as f:
        pickle.dump([m.read_trainables(), params, elbos], f)


for _ in range(n_iters):
    # update assignments and mixture weights
    Phi, Lambda, Gamma = update_assignments(
        m, x, y, pi, psi, rho, Phi, Lambda, Gamma)

    pi = Phi.sum(axis=0) / Phi.sum()
    psi = Lambda.sum(axis=0) / Lambda.sum()

    params = {'pi': pi, 'psi': psi, 'rho': rho,
              'Phi': Phi, 'Lambda': Lambda, 'Gamma': Gamma}

    # recompute weights
    weights = compute_weights(Phi, Lambda, Gamma)

    # reassign model data
    elbos.append(m.compute_log_likelihood(feed_dict={m.weights: weights}))
    print(elbos[-1])

    # optimize gp parameters
    opt = gpflow.train.AdamOptimizer()
    opt.minimize(m, maxiter=grad_iters, feed_dict={m.weights: weights})
    elbos.append(m.compute_log_likelihood(feed_dict={m.weights: weights}))
    print(elbos[-1])

    # save model
    with open(model_path, 'wb') as f:
        pickle.dump([m.read_trainables(), params, elbos], f)
