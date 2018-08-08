import numpy as np
import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
from MixtureSVGP import MixtureSVGP, generate_updates
from utils import load_data
import sys
from actions import train_mixsvgp, Assignments


K = int(sys.argv[1])
L = int(sys.argv[2])
global_trajectories = bool(int(sys.argv[3]))
save_path = sys.argv[4]

n_iter = 100
minibatch_size = 1000


#############
# load data #
#############

normalized_data_df, y, data_dict = load_data(
    'data/quantile_normalized_no_projection.txt')
y = y.transpose(0, 2, 1).astype(np.float64)

N, G, T = y.shape

G = 1000

y = y[:N, :G, :T].transpose(2, 0, 1)
x = np.tile(np.arange(T).astype(np.float64)[:, None, None], (1, N, G))

mask = ~np.isnan(y.flatten())
X = x.reshape(-1, 1)[mask]
Y = y.reshape(-1, 1)[mask]
_, weight_idx = np.unique(
    np.tile(np.arange((N*G)).reshape((N, G))[None, :, :],
            (T, 1, 1)).flatten()[mask], return_inverse=True)

###############
# Make models #
###############

# gp objects
if global_trajectories:
    num_clusters = K * L + L
else:
    num_clusters = K * L

kernel = mk.SharedIndependentMok(gpflow.kernels.RBF(1), num_clusters)
feature = mf.SharedIndependentMof(
    gpflow.features.InducingPoints(np.arange(T).astype(
        np.float64).reshape(-1, 1)))
likelihood = gpflow.likelihoods.Gaussian()

# model -- for hyperparameter learning
with gpflow.defer_build():
    m = MixtureSVGP(X, Y, weight_idx,
                    kern=kernel,
                    num_clusters=num_clusters, num_data=X.shape[0],
                    likelihood=likelihood,
                    feat=feature, minibatch_size=minibatch_size)

    # only train hyperparams
    for param in m.parameters:
        param.trainable = False

# mean model -- for trajectory learning
# data are dummies, they'll be swapped out, just make correct shape
XB = np.arange(np.unique(X).size)[:, None].astype(np.float64)
YB = np.random.random([XB.shape[0], num_clusters]).astype(np.float64)

# put in model, weight_idx just arange because each observation unique
with gpflow.defer_build():
    m_bar = MixtureSVGP(XB, YB, np.arange(XB.size),
                        kern=kernel,
                        num_clusters=num_clusters, num_data=XB.shape[0],
                        likelihood=likelihood,
                        feat=feature, minibatch_size=None)

    # fix all parameters except inducing outputs
    for param in m_bar.parameters:
        param.trainable = False

    m_bar.q_mu.trainable = True
    m_bar.q_sqrt.trainable = True

m.q_mu = m_bar.q_mu
m.q_sqrt = m_bar.q_sqrt

# compile models
m.compile()
m_bar.compile()

compute_weights, update_assignments = generate_updates(
    N, G, K, L, T, global_trajectories)

pi = np.ones(K) / K
psi = np.ones(L) / L

if global_trajectories:
    rho = np.ones(2) / 2
else:
    rho = None

Phi = np.random.random((N, K))
Phi = Phi / Phi.sum(1)[:, None]
Lambda = np.random.random((G, L))
Lambda = Lambda / Lambda.sum(1)[:, None]

if global_trajectories:
    Gamma = np.tile(np.array([0.5, 0.5])[None, :], (L, 1))
else:
    Gamma = None

assignments = Assignments(pi, psi, rho, Phi, Lambda, Gamma)
m.weights = compute_weights(Phi, Lambda, Gamma)

logger = train_mixsvgp(m, m_bar, assignments, x, y,
                       compute_weights, update_assignments,
                       n_iter, save_path)