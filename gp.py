import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy import integrate


def make_gp_funs(cov_func, num_cov_params=1):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_kernel_params(params):
        mean = params[0]
        cov_params = params[2:]
        noise_variance = np.exp(params[1])
        return mean, cov_params, noise_variance

    def predict_full(params, x, y, xstar, weights):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_variance = unpack_kernel_params(params)
        cov_f_f = rbf_covariance(cov_params, xstar, xstar)
        cov_y_f = rbf_covariance(cov_params, x, xstar)
        cov_y_y = rbf_covariance(cov_params, x, x) + \
            np.diag(noise_variance / weights)

        z = solve(cov_y_y, cov_y_f).T
        pred_mean = mean + np.dot(z, (y - mean))
        pred_cov = cov_f_f - np.dot(z, cov_y_f)
        return pred_mean, pred_cov

    def predict(params, x, y, xstar, weights=None, condense=True, prediction_noise=True):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""

        n, t = y.shape

        if weights is None:
            weights = np.ones(n)

        if not condense:
            return predict_full(
                params,
                np.tile(x, n),
                y.flatten(),
                xstar,
                np.tile(weights, (x.size, 1)).T.flatten())

        mean, cov_params, noise_variance = unpack_kernel_params(params)

        if n == 0:
            # no data, return the prior
            prior_mean = mean * np.ones(xstar.size)
            prior_covariance = rbf_covariance(cov_params, xstar, xstar)
            return prior_mean, prior_covariance

        y_bar = np.dot(weights, y)
        weights_full = (np.logical_not(np.isnan(y)) *
                        weights[:, np.newaxis]).sum(axis=0)

        cov_f_f = rbf_covariance(cov_params, xstar, xstar)
        cov_y_f = weights_full[:, np.newaxis] * rbf_covariance(cov_params, x, xstar)

        cov_y_y = np.outer(weights_full, weights_full) * \
            rbf_covariance(cov_params, x, x) + \
            noise_variance * np.diag(weights_full)

        z = solve(cov_y_y, cov_y_f).T
        pred_mean = mean + np.dot(z, y_bar - mean).flatten()
        pred_cov = cov_f_f - np.dot(z, cov_y_f)
        if prediction_noise:
            pred_cov = pred_cov + noise_variance * np.eye(xstar.size)
        return pred_mean, pred_cov

    def log_marginal_likelihood_full(params, x, y, weights=None):
        if weights is None:
            weights = np.ones(len(y))
        mean, cov_params, noise_variance = unpack_kernel_params(params)
        cov_y_y = cov_func(cov_params, x, x) + \
            noise_variance * np.diag(1 / weights)
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)

    def log_marginal_likelihood_quad(params, x, y,
                                     weights=None, condense=True):
        """
        weights an NG array to compress observations at each timepoint
        y is NxT expression for line n gene g at time t
        x is T indicates time for each column of y
        weights is N weighs each sample for this GP
        """
        n, t = y.shape

        if weights is None:
            weights = np.ones(y.shape[0])

        if not condense:
            return log_marginal_likelihood_full(
                params,
                np.tile(x, y.shape[0]),
                y.flatten(),
                np.tile(weights, (x.size, 1)).T.flatten())

        mean, cov_params, noise_variance = unpack_kernel_params(params)

        def integrand(*args):
            """
            If we want avoid inverting a huge matrix we need to evaluate
            the log marginal through quadrature
            here observations are conditionally independent
            given function draw f
            """
            f = np.array(args)

            # UPDATE TO REFLECT DIMENSION
            likelihood = np.sum(mvn.logpdf(
                y, f, noise_variance * np.eye(t)))
            K = cov_func(cov_params, x, x)
            prior = mvn.logpdf(f, np.zeros(len(f)), K)
            return np.exp(likelihood + prior)

        return np.log(integrate.nquad(integrand, [[-100, 100] for _ in range(t)])[0])

    def log_marginal_likelihood(params, x, y, weights=None, condense=True):
        """
        weights an NG array to compress observations at each timepoint
        y is NxT expression for line n gene g at time t
        x is T indicates time for each column of y
        weights is N weighs each sample for this GP
        """
        n, t = y.shape

        if weights is None:
            weights = np.ones(y.shape[0])

        mask = ~np.isclose(weights, 0)
        y = y[mask]
        weights = weights[mask]

        if not condense:
            return log_marginal_likelihood_full(
                params,
                np.tile(x, y.shape[0]),
                y.flatten(),
                np.tile(weights, (x.size, 1)).T.flatten())

        mean, cov_params, noise_variance = unpack_kernel_params(params)

        c = y[0]
        C = np.eye(t) * (noise_variance / weights[0])
        ll = 0
        # import pdb; pdb.set_trace()
        for i in range(1, y.shape[0]):
            dat = y[i]
            Sigma = np.eye(t) * (noise_variance / weights[i])
            c, C, Z = gaussian_product(dat, Sigma, c, C,
                                       diagA=True, diagB=True)
            ll = ll + Z

        cov_f_f = cov_func(cov_params, x, x)
        c, C, Z = gaussian_product(c, C, np.zeros(t), cov_f_f)
        ll = ll + Z
        return ll

    return predict, log_marginal_likelihood


# Define an example covariance function.
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(
        x / lengthscales[:, np.newaxis], 2) - np.expand_dims(
        xp / lengthscales[:, np.newaxis], 1)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=0))


def build_toy_dataset(D=1, n_data=20, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs = np.concatenate([np.linspace(0, 3, num=n_data/2),
                            np.linspace(6, 8, num=n_data/2)])
    targets = (np.cos(inputs) + rs.randn(n_data) * noise_std) / 2.0
    inputs = (inputs - 4.0) / 2.0
    inputs = inputs.reshape((len(inputs), D))
    return inputs, targets


def diag_inv(X):
    """
    returns diagonal matrix with reciprocal
    of diagonal of X
    """
    return np.eye(X.shape[0]) * (1 / np.diagonal(X, offset=0, axis1=-1, axis2=-2))


def gaussian_product(a, A, b, B, diagA=False, diagB=False):
    if diagA:
        Ainv = diag_inv(A)
    else:
        Ainv = np.linalg.inv(A)

    if diagB:
        Binv = diag_inv(B)
    else:
        Binv = np.linalg.inv(B)

    Sigma = A + B
    if diagA and diagB:
        C = diag_inv(Ainv + Binv)
        Sigmainv = diag_inv(Sigma)
        detSigma = 1
        for i in range(Sigma.shape[0]):
            detSigma = detSigma * Sigma[i, i]

    else:
        C = np.linalg.inv(Ainv + Binv)
        Sigmainv = np.linalg.inv(Sigma)
        detSigma = np.linalg.det(Sigma)

    c = np.dot(C, np.dot(Ainv, a) + np.dot(Binv, b))
    D = a.size

    quad = np.dot(a - b, np.dot(Sigmainv, a - b))
    Zinv = -0.5 * (np.log(detSigma) + quad + D * np.log(2 * np.pi))
    return c, C, Zinv


def logpdf(x, mu, sigma2, weights=None):
    n, D = x.shape
    if weights is None:
        weights = np.ones(n)
    a = np.sum((weights * x - mu)**2)
    b = np.linalg.logdet(2 * np.pi * sigma2) * weights.sum() * D
    return -0.5 * (a + b)