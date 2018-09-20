# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

import gpflow
from gpflow import kullback_leiblers, features
from gpflow import settings
from gpflow import transforms
from gpflow.conditionals import conditional
from gpflow.features import Kuu
from gpflow.decors import params_as_tensors
from gpflow.models.model import GPModel
from gpflow.params import DataHolder
from gpflow.params import Minibatch
from gpflow.params import Parameter
from scipy.special import digamma


class MixtureSVGP(gpflow.models.GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is
    ::
      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }
    """

    def __init__(self, X, Y, assignments, weight_idx, kern, likelihood,
                 feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None, init_weights=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        num_weights = np.unique(weight_idx).size

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
            weight_idx = DataHolder(weight_idx)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
            weight_idx = Minibatch(weight_idx, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood,
                         mean_function, num_latent=num_latent, **kwargs)

        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)
        self.weight_idx = weight_idx

        # weights set externally
        self.assignments = assignments

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros(
            (num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones(
                    (num_inducing, self.num_latent),
                    dtype=settings.float_type),
                    transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(
                    num_inducing, dtype=settings.float_type)
                    for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(
                    q_sqrt, transform=transforms.LowerTriangular(
                        num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(
                    q_sqrt, transform=transforms.LowerTriangular(
                        num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern,
                    jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        Just compute likelihood for each model and scale each observation
        by current responsibility estimate
        """

        """
        fmean, fvar, input_idx = self._build_predict_condensed(
            self.X, full_cov=False, full_output_cov=False)
        Y_parts = tf.dynamic_partition(self.Y, input_idx, self.T)
        weight_parts = tf.dynamic_partition(self.weights, input_idx, self.T)
        return tf.reduce_sum(
        [tf.reduce_sum(self.likelihood.variational_expectations(
                fmean[t], fvar[t], Y_parts[t]
            ) * weight_parts[t]) for t in range(self.T)]) - KL
        fmean, fvar, idx = self._build_predict_condensed(
            self.X, full_cov=False, full_output_cov=False)

        likelihood = []
        for k in range(self.K):
            for l in range(self.L):
                weights = self.Phi[:, k][None, :] *
                    self.Lambda[:, l][:, None] * self.Gamma[l, 0]
                weights = tf.tile(weights[:, :, None], [1, 1, self.T])

                mean = tf.gather(fmean[k * l + l], idx)
                var = tf.gather(fmean[k * l + l], idx)
                weights = tf.boolean_mask(
                    tf.reshape(weights, [-1, 1]), self.mask)
                var_exp = self.likelihood.variational_expectations(
                    mean, var, self.Y)
                likelihood.append(tf.reduce_sum(weights * var_exp))
        return
        """
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(
            self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(
            fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(
            var_exp * tf.gather(self.assignments.compute_weights(),
                                self.weight_idx)) * scale - KL

    @gpflow.autoflow((settings.float_type, [None, None]),
                     (settings.float_type, [None, None]))
    def expected_density(self, Xnew, Ynew):
        """
        This gives a variational bound on the model likelihood.
        Just compute likelihood for each model and scale each observation
        by current responsibility estimate
        """
        fmean, fvar = self._build_predict(
            Xnew, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(
            fmean, fvar, Ynew)

        return var_exp

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        Xnew, input_idx = tf.unique(tf.reshape(Xnew, [-1]))
        mu, var = conditional(tf.reshape(Xnew, [-1, 1]),
                              self.feature, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        mu = mu + self.mean_function(tf.reshape(Xnew, [-1, 1]))
        mu = tf.gather(mu, input_idx, axis=0)
        var = tf.gather(var, input_idx, axis=0)
        return mu, var

    @params_as_tensors
    def _build_predict_condensed(
            self, Xnew, full_cov=False, full_output_cov=False):
        Xnew, input_idx = tf.unique(tf.reshape(Xnew, [-1]))
        mu, var = conditional(tf.reshape(Xnew, [-1, 1]),
                              self.feature, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        mu = mu + self.mean_function(Xnew)
        return mu, var, input_idx


def generate_updates(N, G, K, L, T, global_trajectories=True):
    def compute_weights(Phi, Lambda, Gamma=None):
        if Gamma is None:
            Gamma = np.ones((L, 2))
            Gamma[:, 1] = 0

        weights = np.einsum('nk,gl->klng', Phi, Lambda * Gamma[:, 0])
        if global_trajectories:
            global_weights = np.einsum(
                'na,gl->alng', np.ones(N).reshape(-1, 1), Lambda * Gamma[:, 1])
            weights = np.concatenate([weights, global_weights])
            weights = weights.reshape(K*L + L, -1).T
        else:
            weights = weights.reshape(K*L, -1).T
        return weights

    def update_assignments(m, X, Y, pi, psi, rho, Phi, Lambda, Gamma):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        if global_trajectories:
            Phi, Lambda, Gamma = _update_assignments_global(
                m, X, Y, pi, psi, rho, Phi, Lambda, Gamma)

        else:
            Phi, Lambda = _update_assignments(
                m, X, Y, pi, psi, Phi, Lambda)
            Gamma = np.ones((L, 2))
            Gamma[:, 1] = 0
        return Phi, Lambda, Gamma

    def _update_assignments(m, X, Y, pi, psi, Phi, Lambda):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        densities = m.predict_density(X.reshape(-1, 1), Y.reshape(-1, 1)).T
        densities = densities.reshape(K, L, N, G, T)

        for _ in range(10):
            Lambda_old = Lambda.copy()

            # sample assignment update
            logPhi = np.nansum(densities[:K]
                               * Lambda.T[None, :, None, :, None],
                               axis=(1, 3, 4)).T + np.log(pi)

            Phi = np.exp(logPhi - logsumexp(logPhi)[:, None])

            # gene assignment update
            logLambda = np.nansum(densities[:K]
                                  * Phi.T[:, None, :, None, None],
                                  axis=(0, 2, 4)).T
            logLambda = logLambda + np.log(psi)
            Lambda = np.exp(logLambda - logsumexp(logLambda)[:, None])

            # local/global gene cluster
            if np.allclose(Lambda, Lambda_old):
                break
        return Phi, Lambda

    def _update_assignments_dp(m, X, Y, pi, alpha, Phi, Lambda):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        densities = m.expected_density(X.reshape(-1, 1), Y.reshape(-1, 1)).T
        densities = densities.reshape(K, L, N, G, T)

        log_psi = _stick_breaking(alpha, Lambda)

        for _ in range(10):
            Lambda_old = Lambda.copy()

            # sample assignment update
            logPhi = np.nansum(densities[:K]
                               * Lambda.T[None, :, None, :, None],
                               axis=(1, 3, 4)).T + np.log(pi)

            Phi = np.exp(logPhi - logsumexp(logPhi)[:, None])

            # gene assignment update
            logLambda = np.nansum(densities[:K]
                                  * Phi.T[:, None, :, None, None],
                                  axis=(0, 2, 4)).T
            logLambda = logLambda + log_psi
            Lambda = np.exp(logLambda - logsumexp(logLambda)[:, None])

            # local/global gene cluster
            if np.allclose(Lambda, Lambda_old):
                break
            return Phi, Lambda

    def _stick_breaking(alpha, Z):
        """
        return expected values of stick lengths
        alpha concentration parameter
        Z [N, T] expected assignments on truncated DP
        """
        alpha = Z.sum(0)
        beta_alpha= np.flip(np.cumsum(np.flip(alpha, 0)), 0)

        digamma_alpha = digamma(alpha)
        digamma_beta = digamma(beta_alpha - alpha)
        digamma_alpha_beta = digamma_alpha(beta_alpha)

        lnv = digamma_alpha - digamma_alpha_beta
        ln1_v = digamma_beta - digamma_alpha_beta

        log_psi = lnv + np.cumsum(ln1_v) - ln1_v
        return log_psi

    def _update_assignments_global(m, X, Y, pi, psi, rho, Phi, Lambda, Gamma):
        """
        BE AWARE X and Y are the full data with NANS (DONT MASK)
        """
        densities = m.predict_density(X.reshape(-1, 1), Y.reshape(-1, 1)).T
        densities = densities.reshape(K+1, L, N, G, T)

        for _ in range(10):
            Lambda_old = Lambda.copy()

            # sample assignment update
            logPhi = np.nansum(densities[:K]
                               * Lambda.T[None, :, None, :, None]
                               * Gamma[:, 0][None, :, None, None, None],
                               axis=(1, 3, 4)).T + np.log(pi)

            Phi = np.exp(logPhi - logsumexp(logPhi)[:, None])

            # gene assignment update
            logLambda = np.nansum(densities[:K]
                                  * Phi.T[:, None, :, None, None]
                                  * Gamma[:, 0][None, :, None, None, None],
                                  axis=(0, 2, 4)).T
            logLambda = logLambda + np.nansum(
                densities[K] * Gamma[:, 1][:, None, None, None],
                axis=(1, 3)).T
            logLambda = logLambda + np.log(psi)
            Lambda = np.exp(logLambda - logsumexp(logLambda)[:, None])

            # local/global gene cluster
            logGamma = np.zeros((L, 2))
            logGamma[:, 0] = np.nansum(
                densities[:K] * Phi.T[:, None, :, None, None]
                * Lambda.T[None, :, None, :, None], axis=(0, 2, 3, 4))

            logGamma[:, 1] = np.nansum(
                densities[K] * Lambda.T[:, None, :, None], axis=(1, 2, 3))
            logGamma = logGamma + np.log(rho)
            Gamma = np.exp(logGamma - logsumexp(logGamma)[:, None])
            Gamma = (Gamma + 0.001)
            Gamma = Gamma / Gamma.sum(axis=1)[:, None]

            if np.allclose(Lambda, Lambda_old):
                break
            return Phi, Lambda, Gamma

    return compute_weights, update_assignments


def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x, axis=1)
    return max_x + np.log(np.sum(np.exp(x - max_x[:, np.newaxis]), axis=1))
