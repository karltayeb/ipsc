import numpy as np
import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
from MixtureSVGP import MixtureSVGP, generate_updates
import pickle
from utils import load_data

import sys, os

out_dir = sys.argv[1]
model_id = sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

n_iters = 1000
normalized_data_df, X, data_dict = load_data(
    'data/quantile_normalized_no_projection.txt')
n_lines, n_samples, n_genes = X.shape
y = X.transpose(0, 2, 1)

N = n_lines
G = n_genes
K = 3
L = 100
T = n_samples


model_path = out_dir + 'mixsvgp_K' + str(K) + '_L' + str(L) + '_' + model_id
print(model_path)
y = y[:N, :G, :]
X = np.tile(np.arange(T).astype(np.float64), (N, G, 1))


# initialize mixture weights
pi = np.ones(K) / K
psi = np.ones(L) / L
rho = np.array([0.1, 0.9])

# initialize assignments
Phi = np.zeros((N, K))
assignments = np.random.choice(K, N)
for k in range(K):
    Phi[assignments == k, k] = 1.0

Lambda = np.zeros((G, L))
assignments = np.random.choice(L, G)
for l in range(L):
    Lambda[assignments == l, l] = 1.0

Gamma = np.tile(np.array([0.3, 0.7]), (L, 1))

num_clusters = (K + 1) * L
mask = ~np.isnan(y.reshape(-1, 1)).squeeze()

# create update functions
compute_weights, update_assignments = generate_updates(N, G, K, L, T)

# create model
kernel = mk.SharedIndependentMok(gpflow.kernels.RBF(1), num_clusters)
feature = mf.SharedIndependentMof(
    gpflow.features.InducingPoints(np.arange(T).astype(
        np.float64).reshape(-1, 1)))

m = MixtureSVGP(X=X.reshape(-1, 1), Y=y.reshape(-1, 1),
                kern=kernel, num_clusters=(K+1)*L,
                likelihood=gpflow.likelihoods.Gaussian(),
                feat=feature)
m.feature.feat.Z.trainable = False

out_path = 'model'
for _ in range(n_iters):
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, disp=True, maxiter=50,
                 feed_dict={m.weights: compute_weights(
                    Phi, Lambda, Gamma)[mask.squeeze()]})
    Phi, Lambda, Gamma = update_assignments(
        m, X, y, pi, psi, rho, Phi, Lambda, Gamma)

    pi = Phi.sum(axis=0) / Phi.sum()
    psi = Lambda.sum(axis=0) / Lambda.sum()

    params = {'pi': pi, 'psi': psi, 'rho': rho,
              'Phi': Phi, 'Lambda': Lambda, 'Gamma': Gamma}
    with open(model_path, 'wb') as f:
        pickle.dump([m.read_trainables(), params], f)
