from utils import load_data
import autograd.numpy as np
from gp import rbf_covariance
from gp_gmm import gen_updates
from autograd import grad
from scipy.optimize import minimize
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
import pickle
import sys, os

outdir = sys.argv[1]
model_id = sys.argv[2]


if not os.path.exists(outdir):
    os.makedirs(outdir)

model_path = outdir + 'model_' + model_id

normalized_data_df, X, data_dict = load_data(
    'data/quantile_normalized_no_projection.txt')

n_lines, n_samples, n_genes = X.shape
X = X.transpose(0, 2, 1)[:, :200, :]

N, G, T = X.shape
K = 2
L = 5

fs = np.random.normal(size=(K, L, T))
sigma2s = np.ones((K, L)) * 1
sigma2 = sigma2s[0, 0]

kernel_params = np.zeros((K, L, 2))
Kers = np.array([[rbf_covariance(params, np.arange(T), np.arange(T))
                for params in sample_group] for sample_group in kernel_params])

pi = np.ones(K) / K
psi = np.ones(L) / L

Phi = np.array([np.random.multinomial(1, pi) for _ in range(N)])
Lambda = np.array([np.random.multinomial(1, psi) for _ in range(G)])

y = X

mixture_weights, update_means, update_sigma2s, update_phi, update_lambda, elbo = gen_updates(N, G, K, L, T, np.arange(T), rbf_covariance)
mus, Sigmas = update_means(y, sigma2s, Phi, Lambda, pi, psi, kernel_params)
elbos = [elbo(y, Phi, Lambda, pi, psi, sigma2s, mus, Sigmas, kernel_params)]

print(0, elbos[-1])
for i in range(1000):
    j = 0
    while True:
        Phi_old = Phi.copy()
        Phi = update_phi(y, Lambda, pi, sigma2s, mus, Sigmas)
        Lambda = update_lambda(y, Phi, psi, sigma2s, mus, Sigmas)
        j += 1
        if np.allclose(Phi - Phi_old, 0) or j > 10:
            break
    mus, Sigmas = update_means(y, sigma2s, Phi, Lambda, pi, psi, kernel_params)
    sigma2s = update_sigma2s(y, Phi, Lambda, mus, Sigmas)
    pi = Phi.sum(axis=0) / Phi.sum()
    psi = Lambda.sum(axis=0) / Lambda.sum()

    def obj(k, l, params):
        Ker = rbf_covariance(params, np.arange(T), np.arange(T))
        likelihood = mvn.logpdf(mus[k, l], np.zeros(T), Ker) \
            - 0.5 * np.trace(solve(Ker, Sigmas[k, l]))
        return -1 * likelihood

    for k in range(K):
        for l in range(L):
            init_params = kernel_params[k, l]
            try:
                learned_params = minimize(lambda p: obj(k, l, p), init_params,
                                          jac=False, method='Nelder-Mead',
                                          callback=None)
                kernel_params[k, l] = learned_params.x
            except:
                print('#')
                kernel_params[k, l] = init_params

    if i % 5 == 0:
        elbos.append(elbo(
            y, Phi, Lambda, pi, psi, sigma2s, mus, Sigmas, kernel_params))

    model_dict = {'N': N, 'G': G, 'K': K, 'L':  L, 'T': T,
                  'mus': mus, 'Sigmas': Sigmas, 'sigma2s': sigma2s,
                  'pi': pi, 'psi': psi, 'Phi': Phi, 'Lambda': Lambda,
                  'kernel_params': kernel_params, 'elbos': elbos}

    pickle.dump(model_dict, open('model', 'wb'))
    print(i, j, elbos[-1])
    if(np.abs(elbos[-1] - elbos[-2]) < 1e-8):
        break
