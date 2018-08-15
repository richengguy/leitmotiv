from abc import ABC, abstractmethod

import numpy as np

import scipy.linalg
import scipy.special

from leitmotiv.math import cholesky
from leitmotiv.math import linalg


__all__ = [
    'Distribution',
    'MultivariateNormalDistribution',
    'VariationalGaussianMixture'
]


class Distribution(ABC):
    '''Basic interface for a distribution used in data modelling.

    The :class:`Distribution` class provides a uniform interface for working
    with distributions that are used to model a data set.  The intended use if
    with mixture models where a model is actually comprised of multiple
    distributions of the same class.

    Attributes
    ----------
    weight : double
        the amount of influence this distribution has on the overall model;
        defaults to '1'
    samples : int
        number of samples in the distribution; read-only
    parameters : tuple of model parameters
        the contents and order of this tuple are model-specific, but it will
        contain all of the model parameters aside from the weight
    '''
    def __init__(self):
        self._weight = 1
        self._samples = 0

    @property
    def samples(self):
        return self._samples

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, w):
        if w >= 0 and w <= 1:
            self._weight = w
        else:
            raise ValueError('Weight can only be between 0 and 1.')

    @property
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def update(self, x, weights=None):
        '''Perform a batch update on the distribution.

        Note
        ----
        This is meant for relatively infrequent bulk updates where the entire
        data set changes.  If frequent, single sample updates are required then
        :meth:`add_sample` and :meth:`remove_sample` are better options.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containing the dataset used to update the
            distribution's parameters
        weights : numpy.ndarray
            a 1xN array containing the relative weighting of each component for
            this distribution; optional
        '''

    @abstractmethod
    def add_sample(self, x):
        '''Add a sample into the distribution.

        Parameters
        ----------
        x : numpy.ndarray
            a Dx1 vector containing the sample that is being added to the
            distribution
        '''

    @abstractmethod
    def remove_sample(self, x):
        '''Removed a sample from the distribution.

        Parameters
        ----------
        x : numpy.ndarray
            a Dx1 vector containing the sample being removed from the
            distribution
        '''

    @abstractmethod
    def loglikelihood(self, x):
        '''Compute the log-likelihood of the data under the model.

        For each sample :math:`\\vec{x}_i`, this finds
        :math:`\\ln P(\\vec{x}_i | \\vec{\\theta})`, where
        :math:`\\vec{\\theta}` are the parameters for this particular model.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array of samples that are being examined under the model

        Returns
        -------
        numpy.ndarray
            a 1xN array containing the log-likelihoods
        '''

    def likelihood(self, x):
        '''Compute the likelihood of some data under the model.

        Internally this calls :meth:`loglikelihood` and then takes the
        exponent.  It is more efficient to do the computations in the log
        domain and convert rather than working with the expressions directly.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array of samples that are being examined under the model

        Returns
        -------
        numpy.ndarray
            a 1xN array containing the likelihoods
        '''
        return np.exp(self.loglikelihood(x))

    def kl_divergence(self, other):
        '''Compute the KL-divergence between this distribution and another one.

        The Kullbeck-Leibler divergence is a measure of dissimilarity between
        two probability distributions.  The divergence is defined such that it
        is zero iff the two distributions are identical.  It is not the same as
        a distance metric and so it is not symmetric.  That is,
        :math:`D(A|B) \\ne D(B|A)`.

        Parameters
        ----------
        other : subclass of :class:`Distribution`
            another distribution to compare to

        Returns
        -------
        double
            the KL-divergence between the two distributions

        Raises
        ------
        TypeError
            if the other distribution is not of the same class as the one that
            the method is being called on
        '''
        raise NotImplementedError(
            'KL-divergence has not been implemented for this Distribution.')


class MultivariateNormalDistribution(Distribution):
    '''A multivariate Normal, or Gaussian, distribution.

    A :math:`D`-dimensional multivariate Gaussian is described by the
    expression

    .. math::
        f(\\vec{x}| \\vec{\\mu}, \\mathbf{\\Sigma}) =
            \\frac{1}{\\sqrt{(2\\pi)^D|\\Sigma|}}
            \\exp\\left\\{
                -\\frac{1}{2}(\\vec{x} - \\vec{\\mu})^T
                \\mathbf{\\Sigma}^{-1}
                (\\vec{x} - \\vec{\\mu})
            \\right\\},

    where :math:`\\vec{\\mu}` is the distribution's mean and
    :math:`\\mathbf{\\Sigma}` is its covariance.

    Attributes
    ----------
    L : numpy.ndarray
        a DxD lower-triangular matrix that is the Cholesky decomposition of the
        covariance matrix
    Sigma : numpy.ndarray
        a DxD, symmetric, positive-definite covariance matrix; this is computed
        from :attr:`L` and not stored directly
    mu : numpy.ndarray
        a Dx1 vector containing the distribution mean
    parameters : (:attr:`mu`, :attr:`Sigma`)
        a tuple containing the mean and covariance matrix
    '''
    def __init__(self, mu=None, Sigma=None):
        '''
        Parameters
        ----------
        mu : numpy.ndarray
            distribution mean
        Sigma : numpy.ndarray
            distribution covariance
        '''
        super().__init__()

        if mu is None:
            mu = np.zeros((1, 1))

        if Sigma is None:
            Sigma = np.eye(1)

        self.L = cholesky.from_matrix(Sigma)
        self.mu = mu.copy()

        detL = 2.0*np.sum(np.log(np.diag(self.L)))
        self._const = -(np.log(2*np.pi)*self.mu.shape[0] + detL) / 2.0

    @property
    def Sigma(self):
        return cholesky.to_matrix(self.L)

    @property
    def parameters(self):
        return self.mu, self.Sigma

    def update(self, x, weights=None):
        '''Batch-update the properties for the current distribution.

        This assumes that all samples are being provided at once.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containing the dataset used to update the
            distribution's parameters
        weights : numpy.ndarray
            a 1xN array containing the relative weighting of each component for
            this distribution; if not provided then assumed to all be '1'
        '''
        self._samples = x.shape[1]
        if weights is None:
            weights = np.ones((1, self._samples))

        # Compute new means.
        Nk = np.sum(weights)
        self.mu = np.sum(weights*x, axis=1)[:, np.newaxis] / Nk

        # Compute new covariances.
        Sigma = linalg.weighted_scatter_matrix(weights, x, self.mu) / Nk
        self.L = cholesky.from_matrix(Sigma)

        # Compute the new model weight.
        self.weight = Nk / self._samples

        # Update any variables that don't change during a log-likelihood
        # estimate.
        detL = 2.0*np.sum(np.log(np.diag(self.L)))
        self._const = -(np.log(2*np.pi)*x.shape[0] + detL) / 2.0

    def add_sample(self, x):
        '''Update the model given new observation.

        Note
        ----
        This is currently unimplemented.

        Raises
        ------
        NotImplemented
        '''
        raise NotImplemented('This operation is not supported.')

    def remove_sample(self, x):
        '''Update the model after removing an observation.

        Note
        ----
        This is currently unimplemented.

        Raises
        ------
        NotImplemented
        '''
        raise NotImplemented('This operation is not supported.')

    def loglikelihood(self, x):
        '''Compute the log-likelihood of data under the model.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array of sample vectors
        use_weight : bool
            if ``True`` then the model's weighting is incorporated into the
            likelihood

        Returns
        -------
        numpy.ndarray
            a Dx1 array of log-likelihoods
        '''
        gamma = scipy.linalg.solve_triangular(self.L, x - self.mu, lower=True)
        loglike = (self._const - 0.5*np.sum(gamma * gamma, axis=0) +
                   np.log(self.weight))
        return loglike

    def kl_divergence(self, other):
        '''Compute the KL-divergence between two Gaussians.

        For a multivariate Gaussian, this is defined to be

        .. math::
            D(N_A||N_B) =
                \\frac{1}{2}\\left\\{
                    \\mathrm{tr}\\left(
                        \\mathbf{\\Sigma}_B^{-1}\\mathbf{\\Sigma}_A
                    \\right) +
                    \\left( \\vec{\\mu}_B - \\vec{\\mu}_A \\right)^T
                    \\mathbf{\\Sigma}_B^{-1}
                    \\left( \\vec{\\mu}_B - \\vec{\\mu}_A \\right) -
                    k +
                    \\ln\\left(
                        \\frac{
                            | \\mathbf{\\Sigma}_B |
                        }{
                            | \\mathbf{\\Sigma}_A |
                        } \\right)
                \\right\\}.

        Parameters
        ----------
        other : :class:`MultivariateNormalDistribution`
            another multivariate normal distribution

        Returns
        -------
        double
            the KL-divergerence

        Raises
        ------
        TypeError
            if the other distribution is of the wrong type
        ValueError
            if the dimensionality is different between the two distributions
        '''
        if not isinstance(other, MultivariateNormalDistribution):
            raise TypeError(
                'Other distribution must also be a '
                'MultivariateNormalDistribution.')

        if self.mu.shape[0] != other.mu.shape[0]:
            raise ValueError('Dimensionality must be the same.')

        ndim = self.mu.shape[0]

        # 1. A = tr(SigmaB^-1*SigmaA)
        # This is computed by taking the inverses of the Cholesky
        # decompositions.
        LBinv, _ = scipy.linalg.lapack.dtrtri(other.L, True)
        A = np.trace(LBinv.T @ LBinv @ self.L @ self.L.T)

        # 2. B = (muB - muA)^T SigmaA^{-1} (muB - muA)
        B = scipy.linalg.solve_triangular(self.L, other.mu - self.mu,
                                          lower=True)
        B = np.sum(B*B)

        # 3. C = ln(det(SigmaB)/det(SigmaA))
        #      = ln(det(SigmaB)) - ln(det(SigmaA))
        lnDetSigmaA = 2.0*np.sum(np.log(np.diag(self.L)))
        lnDetSigmaB = 2.0*np.sum(np.log(np.diag(other.L)))
        C = lnDetSigmaB - lnDetSigmaA

        return 0.5*(A + B + C - ndim)


class VariationalGaussianMixture(Distribution):
    '''A distribution that discribes a variational Gaussian mixture model.

    A variational Gaussian mixture model is three separate distributions
    combined into one:

     * a Dirichlet prior over the model weights
     * a Gaussian prior over the mixture means
     * a Wishart prior over the mixture precisions (inverse covariances)

    The Dirichlet prior is controlled by a single concentration parameter
    :math:`\\alpha` that dictates the distribution of model weights.  In short,
    a value of :math:`\\alpha < 1` enforces sparsity so that the majority of
    the weights will be zero, or close to it.  A value of :math:`\\alpha > 1`
    will do the exact opposite.

    The means and precisions are actually combined into a single distribution
    known as the Normal-Wishart distribution.  It is the conjugate prior to the
    multivariate Gaussian and it describes the distribution of means and
    precisions (inverse-covariance) rather than the distribution of data.  The
    distribution is intended to be used when the number of mixtures in a model
    is known, only that it can be described by one or more Gaussians.

    A feature of this distribution is that it contains both a mean and
    covariance (inferred from the precision), just like a Gaussian
    distribution.  Sampling from a Normal-Wishart distribution provides an
    estimate of a particular Gaussian without requiring a value of the
    parameters directly.

    Attributes
    ----------
    alpha : float
        Dirichlet concentration parameter
    alpha_sum : float
        summed Dirichlet concentration parameter *across* models, defined as
        :math:`\\hat{\\alpha} = \\sum_k \\alpha`; this must be set for the
        likelihoods to computed correctly
    n : float
        degrees of freedom
    k : float
        a non-zero scaling parameter
    L : numpy.ndarray
        a DxD lower-triangular matrix that is the Cholesky decomposition of the
        Wishart scale matrix (unnormalized precision matrix)
    Sigma : numpy.ndarray
        a DxD, symmetric, positive-definite covariance matrix; this is computed
        from :attr:`L` and not stored directly
    mu : numpy.ndarray
        a Dx1 vector containing the distribution mean
    parameters : (:attr:`mu`, :attr:`Sigma`)
        a tuple containing the **maximum a posteriori** (MAP) of the mean and
        covariance
    '''
    def __init__(self, alpha, d, num_models, k=1, mu=None, Sigma=None):
        '''
        Parameters
        ----------
        alpha : float
            concentration parameter
        d : float
            dimensionality of the data
        num_models : int
            number of total models
        k : float
            initial scaling factor
        mu : numpy.ndarray
            initial prior on the distribution mean; if unknown then it is set
            to zero
        Sigma : numpy.ndarray
            initial prior on the covariance; if unknown then it is set to
            identity
        '''
        super().__init__()

        # Internal variables
        self._num_models = num_models
        self._alpha0 = alpha
        self._beta0 = k
        self._nu0 = d + 1

        if mu is None:
            self._mu0 = np.zeros((d, 1))
        else:
            if mu.shape[0] != d:
                raise ValueError(
                    'Mean prior should have a dimensionality of %d.' % d)
            self._mu0 = mu.copy()

        # Note: this stores the *covariance*, not the precision even though
        # that's how the Wishart distribution is defined.  The math in the
        # update equations is actually a little bit cleaner if the Cholesky
        # decomposition of the covariance matrix is used (basically it ends up
        # being nearly the same as the Gaussian version).
        if Sigma is None:
            self._Sigma0 = np.eye(d)
            self._L0 = np.eye(d)
        else:
            if Sigma.shape[0] != d and Sigma.shape[1] != d:
                raise ValueError(
                    'Covariance prior should be a %dx%d matrix.' % (d, d))
            self._Sigma0 = Sigma.copy()
            self._L0 = cholesky.from_matrix(self._Sigma0)

        # Public attributes
        self.alpha = self._alpha0
        self.alpha_sum = 0
        self.beta = self._beta0
        self.nu = self._nu0
        self.mu = self._mu0.copy()
        self.L = self._L0.copy()

    @property
    def Lambda(self):
        return cholesky.inverse_from_decomp(self.L) * self.nu

    @property
    def Sigma(self):
        return cholesky.to_matrix(self.L) / self.nu

    @property
    def parameters(self):
        return self.mu, self.Sigma

    def update(self, x, weights=None):
        '''Batch-update the properties for the current distribution.

        This assumes that all samples are being provided at once.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containing the dataset used to update the
            distribution's parameters
        weights : numpy.ndarray
            a 1xN array containing the relative weighting of each component for
            this distribution; if not provided then assumed to all be '1'
        '''
        self._samples = x.shape[1]
        # print('Samples: %d' % self._samples)
        if weights is None:
            weights = np.ones((1, self._samples)) / self._num_models

        # NOTE All equations are from the cited resource.
        # "Pattern Recognition and Machine Learning". Bishop. 2006

        # Gaussian Update
        Nk = np.sum(weights) + np.finfo(float).eps  # (10.51)
        x_mean = np.sum(weights*x, axis=1)[:, np.newaxis] / Nk  # (10.52)
        Sk = linalg.weighted_scatter_matrix(weights, x, x_mean) / Nk  # (10.53)

        # Dirichlet Update
        self.alpha = self._alpha0 + Nk  # (10.58)

        # Wishart Update
        self.nu = self._nu0 + Nk  # (10.63)
        self.beta = self._beta0 + Nk  # (10.60)
        self.mu = (self._beta0*self._mu0 + Nk*x_mean) / self.beta  # (10.61)

        delta = x_mean - self._mu0
        Winv = (self._Sigma0 + Nk*Sk +
                (self._beta0*Nk/self.beta) * (delta @ delta.T))
        self.L = cholesky.from_matrix(Winv)  # (10.62)

        # Update any variables that don't change during a log-likelihood
        # estimate.
        self._detW = -2.0*np.sum(np.log(np.diag(self.L)))

    def add_sample(self, x):
        '''Update the model given new observation.

        Note
        ----
        This is currently unimplemented.

        Raises
        ------
        NotImplemented
        '''
        raise NotImplemented('This operation is not supported.')

    def remove_sample(self, x):
        '''Update the model after removing an observation.

        Note
        ----
        This is currently unimplemented.

        Raises
        ------
        NotImplemented
        '''
        raise NotImplemented('This operation is not supported.')

    def loglikelihood(self, x):
        '''Compute the log-likelihood of data under the model.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array of sample vectors
        psi_alpha_hat : float
            the digamma value of the summed concentration parameters,
            i.e. :math:`\\psi \\left(\\sum_{i=1}^K \\alpha_i \\right)`;
            required as an additional parameter because it requires knowledge
            of other models

        Returns
        -------
        numpy.ndarray
            a Dx1 array of log-likelihoods
        '''
        gamma = scipy.linalg.solve_triangular(self.L, x - self.mu, lower=True)
        psi_alpha_hat = scipy.special.psi(self.alpha_sum)

        # NOTE All equations are from the cited resource.
        # "Pattern Recognition and Machine Learning". Bishop. 2006

        D = x.shape[0]
        i = np.arange(start=1, stop=D+1)
        psisum = np.sum(scipy.special.psi(0.5*(self.nu + 1 - i)))

        Egauss = D/self.beta + self.nu*np.sum(gamma * gamma, axis=0)  # (10.64)
        Eprec = psisum + D*np.log(2) + self._detW  # (10.65)
        Eweight = scipy.special.psi(self.alpha) - psi_alpha_hat  # (10.66)

        loglike = Eweight + 0.5*(Eprec - D*np.log(2*np.pi) - Egauss)  # (10.46)
        return loglike

    def __str__(self):
        repstr = ['Model:']
        repstr.append('   mu - ' + str(self.mu.T))
        repstr.append(' Sigma - \n' + str(self.Sigma))
        repstr.append(' alpha - ' + str(self.alpha))
        repstr.append('weight - ' + str(self.weight))
        return '\n'.join(repstr)
