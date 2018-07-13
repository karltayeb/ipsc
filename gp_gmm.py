import autograd.numpy as np
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
import autograd.scipy.stats.norm as norm
import pdb


def diag_inv(y):
    """
    returns diagonal matrix with reciprocal
    of diagonal of y
    """
    return np.diag(1 / np.diagonal(y))


def multinomial_entropy(p):
    return -1 * np.nansum(p * np.log(p))


def gen_updates(N, G, K, L, T, inputs, cov_func):
    def update_mean(y, sigma2, weights, kernel_params):
        """
        y[N, G, T] data
        mu [T] means
        sigma2 = global noise variance
        priors = [T, T]
        weights = [N, G]
        """
        prior = cov_func(kernel_params, inputs, inputs) + np.eye(T) * 1e-6
        weights = weights[:, :, np.newaxis] * np.logical_not(np.isnan(y))
        B = np.diag(weights.reshape(-1, T).sum(axis=0) / sigma2)

        yB = np.nansum((y * weights / sigma2).reshape(-1, T), axis=0)
        Sigma = np.linalg.inv(B + np.linalg.inv(prior))
        mu = np.dot(Sigma, yB)
        # import pdb; pdb.set_trace()
        return mu, Sigma

    def update_means(y, sigma2s, Phi, Lambda, pi, psi, kernel_params):
        """
        y[N, G, T] data
        mu [T] means
        sigma2 = global noise variance
        priors = [K, L, T, T]
        weights = [K, L, N, G]
        """
        """
        phi = Phi.copy()
        lam = Lambda.copy()
        for k in range(K):
            if np.allclose(phi[:, k], 0):
                phi[:, k] = np.random.binomial(1, pi[k], N)
        phi = np.exp(phi - logsumexp(phi)[:, np.newaxis])

        for l in range(L):
            if np.allclose(lam[:, l], 0):
                lam[:, l] = np.random.binomial(1, psi[l], G)
        lam = np.exp(lam - logsumexp(lam)[:, np.newaxis])
        """
        weights = np.einsum('nk,gl->klng', Phi, Lambda)

        mus = np.zeros((K, L, T))
        Sigmas = np.zeros((K, L, T, T))
        for l in range(L):
            for k in range(K):
                mus[k, l], Sigmas[k, l] = update_mean(
                    y, sigma2s[k, l], weights[k, l], kernel_params[k, l])

        return mus, Sigmas

    def update_sigma2(y, weights, mu, Sigma):
        """
        y[N, G, T] data
        weights = [N, G, T]
        mu = T
        """
        diffs = y - mu
        total_weight = np.sum(
            weights[:, :, np.newaxis] * np.logical_not(np.isnan(y)))

        sigma2 = np.nansum(weights[:, :, np.newaxis] * (diffs ** 2))
        total_weight = np.maximum(1e-10, total_weight)
        sigma2 = (sigma2 / total_weight) + np.trace(Sigma)
        return sigma2, total_weight

    def update_sigma2s(y, Phi, Lambda, mus, Sigmas, tied=True):
        """
        y[N, G, T] data
        weights = [K, L, N, G]
        mu = [K, L, T]
        """
        weights = np.einsum('nk,gl->klng', Phi, Lambda)

        sigma2s = np.zeros((K, L))
        total_weights = np.zeros((K, L))
        for k in range(K):
            for l in range(L):
                sigma2s[k, l], total_weights[k, l] = update_sigma2(
                        y, weights[k, l], mus[k, l], Sigmas[k, l]
                    )

        if tied:
            sigma2 = np.sum(sigma2s * total_weights) / total_weights.sum()
            sigma2s[:, :] = sigma2

        return sigma2s

    def mixture_weights(Phi, Lambda, bias=0):
        mix = np.kron(Phi, Lambda).sum(axis=0) + bias
        mix = mix / mix.sum()
        mix = mix.reshape(K, L)
        return mix

    def update_phi(y, lam, pi, sigma2s, mus, Sigmas):
        """
        y = [N, G, T]
        lam = [G, L]
        pi = [K]
        sigma2 = noise variance
        mus = [K, L, T]
        Sigmas = [K, L, T, T]
        """
        phi = np.zeros((N, K))
        for k in range(K):
            for l in range(L):
                ll = np.nansum(norm.logpdf(
                    y, mus[k, l], np.sqrt(sigma2s[k, l])), axis=-1)
                ll = (ll - (0.5 * np.trace(Sigmas[k, l]) / (sigma2s[k, l])))
                ll = ll * lam[:, l]
                phi[:, k] = phi[:, k] + ll.sum(axis=1)

        phi = phi + np.log(pi)
        phi = np.exp(phi - logsumexp(phi)[:, np.newaxis])
        phi = surgery(phi)
        return phi

    def update_lambda(y, phi, psi, sigma2s, mus, Sigmas):
        """
        y = [N, G, T]
        phi = [N, K]
        pi = [K]
        sigma2 = noise variance
        mus = [K, L, T]
        Sigmas = [K, L, T, T]
        """
        lam = np.zeros((G, L))
        for l in range(L):
            for k in range(K):
                ll = np.nansum(norm.logpdf(
                    y, mus[k, l], np.sqrt(sigma2s[k, l])), axis=-1)  # N, G
                ll = (ll - 0.5 * np.trace(Sigmas[k, l] / (sigma2s[k, l])))
                ll = ll * phi[:, k][:, np.newaxis]
                lam[:, l] = lam[:, l] + ll.sum(axis=0)

        lam = lam + np.log(psi)
        lam = np.exp(lam - logsumexp(lam)[:, np.newaxis])
        lam = surgery(lam)
        return lam

    def elbo(y, phi, lam, pi, psi, sigma2s, mus, Sigmas, kernel_params):
        """
        phi [N, K] sample membership (cell line cluster)
        lam [G, L] feature membership (expression cluster)
        pi [K] sample mixture weight
        psi [L] feature mixture weights
        y[N, G, T] data
        mus [K, L, T] means
        """
        """
        conditional = np.array([list(map(
            lambda f, s: norm.logpdf(y, f, s).sum(axis=-1), Q[:, :-1], Q[:, -1]))
            for Q in np.concatenate([mus, sigma2s[:, :, np.newaxis]], 2)])

        conditional = conditional + np.log(mix)[:, :, np.newaxis, np.newaxis]
        assignments = np.einsum('nk, gl->klng', phi, lam)
        likelihood = np.sum(conditional * assignments)
        """

        likelihood = 0
        # data likelihood
        for l in range(L):
            for k in range(K):
                ll = np.sum(np.nan_to_num(norm.logpdf(
                    y, mus[k, l], np.sqrt(sigma2s[k, l]))), axis=-1)
                ll = ll - 0.5 * (np.trace(Sigmas[k, l] / sigma2s[k, l]))
                ll = ll * phi[:, k][:, np.newaxis]
                ll = ll * lam[:, l]
                likelihood = likelihood + np.sum(ll)

        # assignment likelihood
        likelihood = likelihood + np.sum(np.log(pi) * phi)
        likelihood = likelihood + np.sum(np.log(psi) * lam)

        # function liklihood
        for k in range(K):
            for l in range(L):
                Ker = cov_func(kernel_params[k, l], inputs, inputs)
                likelihood = likelihood \
                    + mvn.logpdf(mus[k, l], np.zeros(T), Ker) \
                    - 0.5 * np.trace(solve(Ker, Sigmas[k, l]))

        entropy = np.sum(list(map(multinomial_entropy, phi)) +
                         list(map(multinomial_entropy, lam)))
        for k in range(K):
            for l in range(L):
                entropy = entropy + mvn.entropy(mus[k, l], Sigmas[k, l])

        return likelihood + entropy

    return mixture_weights, update_means, update_sigma2s, update_phi, update_lambda, elbo


def logpdf(x, mu, sigma2):
    """
    not really logpdf. we need to use the weights
    to keep track of normalizing factors that differ
    across clusters
    """
    mask = np.where(np.logical_not(np.isnan(x)))
    x = np.atleast_1d(x[mask])
    mu = np.atleast_1d(mu[mask])
    D = x.size

    if D == 0:
        return 0
    sigma2 = sigma2 * np.ones(D)
    return np.sum([norm.logpdf(x[d], mu[d], np.sqrt(sigma2[d])) for d in range(D)])


def mvnlogpdf(x, mu, L):
    """
    not really logpdf. we need to use the weights
    to keep track of normalizing factors that differ
    across clusters

    L cholesky decomposition of covariance matrix
    """
    D = L.shape[0]
    logdet = 2 * np.sum(np.log(np.diagonal(L)))
    quad = np.inner(x - mu, solve(L.T, solve(L, (x - mu))))
    return -0.5 *(D * np.log(2 * np.pi) + logdet + quad)


def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x, axis=1)
    return max_x + np.log(np.sum(np.exp(x - max_x[:, np.newaxis]), axis=1))


def compress_observations(y, weights):
    """
    y [N, G, T]
    weights [N, G]

    takes convex combination of observations by weights
    returns mean and relative precision for each dimension of T
    """
    t = y.shape[-1]
    compressed_weights = (weights[:, :, np.newaxis] * np.logical_not(
        np.isnan(y))).sum(axis=0).sum(axis=0)

    compressed_y = np.nansum((y * weights[:, :, np.newaxis] / compressed_weights).reshape(
        -1, t), axis=0)

    if np.any(np.isnan(compressed_y)):
        pdb.set_trace()
    return compressed_y, compressed_weights


def surgery(Z):
    empties = np.isclose(Z.sum(axis=0), 0)
    Q, R = Z.shape
    if np.any(empties):
        print('!')
    while np.any(empties):
        for r, empty in enumerate(empties):
            if empty:
                # select a nonempty cluster and split it
                c = np.random.choice(np.where(np.logical_not(empties))[0])
                for q in range(Q):
                    if np.random.binomial(1, 0.5):
                        Z[q, r] = Z[q, c]
                        Z[q, c] = 0
        empties = np.isclose(Z.sum(axis=0), 0)
    return Z


def surgery2(Z):
    empties = np.isclose(Z.sum(axis=0), 0)
    Q, R = Z.shape
    if np.any(empties):
        print('surgery')

    while np.any(empties):
        for r, empty in enumerate(empties):
            if empty:
                # select a nonempty cluster and split it
                c = np.random.choice(np.where(np.logical_not(empties))[0])
                for q in range(Q):
                    if np.random.binomial(1, 0.5):
                        Z[q, r] = Z[q, c]
                        Z[q, c] = 0
        empties = np.isclose(Z.sum(axis=0), 0)
    return Z
