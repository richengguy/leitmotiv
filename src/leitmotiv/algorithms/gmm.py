import abc
import functools
import logging

import numpy as np

import scipy.special
import sklearn.cluster

from leitmotiv.algorithms import MultivariateNormalDistribution
from leitmotiv.algorithms import VariationalGaussianMixture
from leitmotiv.exceptions import AlgorithmUntrainedError, LinearAlgebraError


__all__ = [
    'GMM',
    'AdaptiveGMM'
]


def _is_trained(fn):
    '''Raise exeception is an EM algorithm is untrained.

    This is an internal decorator that is meant to be used with
    :class:`EMAlgorithm` subclasses.  It should decorate any function that
    requires the algorithm to first be trained before it can be called.

    Parameters
    ----------
    fn : bound method
        method of an :class:`EMAlgorithm` subclass

    Returns
    -------
    decorated function
    '''
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._is_trained:
            raise AlgorithmUntrainedError('EM algorithm is untrained.')
        return fn(self, *args, **kwargs)
    return wrapper


class EMAlgorithm(abc.ABC):
    '''Base class for algorithms implementing Expectation-Maximization.

    The :class:`EMAlgorithm` class is used to provide the low-level plumbing
    for any algorithms that are using the expectation-maximization (EM)
    algorithm.  All EM algorithms have two stages:

     1. Compute the expected values for whatever parameters are being
        estimated.
     2. Maximize the log-likelihood of the data under the model.

    This is very similar to performing a gradient *ascent* on the model's
    likelihood function, except that the derivatives do not have to be
    computed.  In practice, this is a minimization of the *negative*
    log-likelihood, because that function reaches a minimum when the likelihood
    is at a local maximum and the connection between other optimization methods
    is a bit cleaner.

    Attributes
    ----------
    max_iterations : int
        the maximum number of iterations that the algorithm will
    tolerance : float
        the tolerance on the change in log-likelihood before the algorithm is
        considered to have converged
    model_type : :class:`~leitmotiv.algorithms.Distribution` class type
        the class type of the model that the EM algorithm is estimating; this
        parameter is read-only
    negloglikelihood : numpy.ndarray
        an array containing the per-iteration negative log-likelihood
    '''
    def __init__(self, model, max_iterations=1000, tolerance=1e-6):
        '''
        Parameters
        ----------
        model : :class:`~leitmotiv.algorithms.Distribution` class type
            model being estimated by the EM algorithm
        max_iterations : int
            maximum number of EM iterations
        tolerance : float
            convergence tolerance
        '''
        self._logger = logging.getLogger(__name__)
        self._model_type = model
        self._models = []
        self._is_trained = False
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._negloglikelihood = None

    @property
    def model(self):
        return self._model_type

    @property
    @_is_trained
    def weights(self):
        return np.array([model.weight for model in self._models])

    @property
    @_is_trained
    def negloglikelihood(self):
        return self._negloglikelihood

    @property
    def num_models(self):
        return len(self._models)

    def __len__(self):
        return len(self._models)

    @_is_trained
    def __getitem__(self, i):
        model = self._models[i]
        return (model.weight, *model.parameters)

    @_is_trained
    def __next__(self):
        for model in self._models:
            yield (model.weight, *model.parameters)

    @abc.abstractmethod
    def _initialize(self, x):
        '''Initialize the algorithm's internal state.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containing the data used in the model estimation; this
            may be useful in initializing the algorithm
        '''

    @abc.abstractmethod
    def _expectation(self, x):
        '''Perform the expectation step.

        Parameters
        ---------
        x : numpy.ndarray
            a DxN array containing the data used in the model estimation

        Returns
        -------
        float
            log-likelihood of the data under the model
        '''

    @abc.abstractmethod
    def _maximization(self, x):
        '''Perform the maximization step.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containing the data used in the model estimation
        '''

    def _cleanup(self):
        '''Perform any cleanup after the model training has finished.

        This is called immediately after the EM algorithm loop.  It is only
        necessary to be overidden if there is some post-processing to be done
        before returning the model results.
        '''

    def train(self, x):
        '''Train the algorithm on some data.

        The input data is assumed to exist in a D-dimensional feature space,
        where each column in the input array is a single data point in that
        feature space.  The training process will iterate over the data set at
        most :attr:`max_iterations` times but may terminate if the solution
        converges before that.  Convergence occurs when the relative change in
        negative log-likelihood is below some :attr:`tolerance`.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containing the data used in the model estimation

        Raises
        ------
        ValueError
            if the number of features is less than the number of dimensions
        '''
        if x.shape[0] >= x.shape[1]:
            raise ValueError(
                'Number of samples (%g) must be greater than the number of '
                'dimensions (%d)' % (x.shape[1], x.shape[0]))

        # Initialize the model estimator state.
        self._negloglikelihood = np.zeros((self.max_iterations, 1))
        self._initialize(x)

        # Run the EM algorithm loops.
        self._logger.debug('Starting EM iterations.')
        converged = False
        for n in range(self.max_iterations):
            self._negloglikelihood[n] = -self._expectation(x)
            self._maximization(x)

            if n % 100 == 0:
                self._logger.debug('Iteration %d - neg-log-likelihood %g', n,
                                   self._negloglikelihood[n])

            # If the algorith has already started, check for convergence by
            # looking at the log-likelihoods.  Once they stabilise, the
            # algorithm is assumed to have converged.
            if n > 1:
                current = self._negloglikelihood[n]
                previous = self._negloglikelihood[n-1]
                delta = np.abs(current - previous)
                if delta < self.tolerance:
                    self._logger.debug(
                        'EM algorithm converged after %d iterations.', n)
                    self._negloglikelihood = self._negloglikelihood[0:n]
                    converged = True
                    break

        # Perform any last-minute cleanup.
        self._cleanup()
        self._is_trained = True

        if not converged:
            self._logger.debug(
                'EM algorithm did not converge after %d iterations.',
                self.max_iterations)

        self._logger.debug('Final log-likelihood -> %g',
                           self._negloglikelihood[-1])

    @_is_trained
    def likelihoods(self, x):
        '''Compute the sample likelihoods under the model.

        Given a sample :math:`\\vec{x}`, its likelihood
        :math:`P(\\vec{x}|\\mathbf{\\Theta})` will be

        .. math::
            P(\\vec{x}|\\mathbf{\\Theta}) \\propto
                \\sum_{k=0}^{K-1} \\pi_k P(\\vec{x}|\\vec{\\theta}_k),

        where :math:`\\vec{\\theta}_k \\in \\mathbf{\\Theta}` is the set of
        trained model parameters and :math:`\\sum_{k} \\pi_k = 1` is the
        individual module weight.

        Parameters
        ----------
        x : numpy.ndarray
            DxN array containing the data to examine under the model

        Returns
        -------
        numpy.ndarray
            1xN array containing the set of likelihoods under the model

        Raises
        ------
        AlgorithmUntrainedError
            if the EM algorithm has not been trained yet
        '''
        loglike = np.array([model.loglikelihood(x) for model in self._models])
        loglike = scipy.special.logsumexp(loglike, axis=0)
        return np.exp(loglike)


class GMM(EMAlgorithm):
    '''The classic Gaussian Mixture Model.

    The :class:`GMM` class computes a classical GMM using the
    Expectation-Maximization (EM) algorithm.  The resulting K-mixture model is

    .. math::
        M\\left(\\vec{x}|\\pi, \\mu, \\Sigma \\right) =
            \\sum_{k=0} ^ {K-1} \\pi_k N\\left(\\vec{x}| \\vec{\\mu}_k,
                                               \\mathbf{\\Sigma}_k \\right),

    where :math:`N(\\vec{x}|\\vec{\\mu}, \\mathbf{\\Sigma})` is a Gaussian
    distribution with a mean :math:`\\vec{\\mu}` and covariance
    :math:`\\mathbf{\\Sigma}`.  :math:`\\pi_k` is the weight or
    contribution of Guassian to the model.  It is defined such that
    :math:`\\sum_{k} \\pi_k = 1`.

    Attributes
    ----------
    num_models : int
        the number of models that the algorithm was estimating
    means : numpy.ndarray
        a DxK array containing the set of model means
    covariances : numpy.ndarray
        a KxDxD array containing the set of model covariances
    weights : numpy.ndarray
        a D-length array containing the model weights; these will sum to '1'
    '''
    def __init__(self, num_models, max_iterations=1000, tolerance=1e-6):
        '''
        Parameters
        ----------
        num_models : int
            number of models/clusters in the GMM
        max_iterations : int
            maximum number iterations for the EM algorithm if convergence does
            not occur
        tolerance : float
            convergence tolerance
        '''
        super().__init__(MultivariateNormalDistribution, max_iterations,
                         tolerance)
        self._num_models = num_models
        self._loglike = None
        self._logsums = None
        self._responsibilities = None

    @property
    @_is_trained
    def means(self):
        return np.transpose(
            np.squeeze(np.array([model.mu for model in self._models])))

    @property
    @_is_trained
    def covariances(self):
        return np.array([model.Sigma for model in self._models])

    def _initialize(self, x):
        '''Initialize the GMM using K-means.

        This simply runs K-means with the number of components, or models, that
        are expected for the final GMM.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containg the dataset to model
        '''
        self._logger.debug(
            'Performing K-means initialization with %d clusters.',
            self._num_models)
        kmeans = sklearn.cluster.KMeans(n_clusters=self._num_models,
                                        n_jobs=2)
        kmeans.fit(x.T)

        for i in range(self._num_models):
            model = self._model_type()
            try:
                model.update(x[:, kmeans.labels_ == i])
                model.weight = 1 / self._num_models
                self._models.append(model)
            except LinearAlgebraError:
                self._logger.debug(
                    'Failed into initialize the %d-th model, decrementing '
                    'model count.', i)
                self._num_models -= 1

    def _expectation(self, x):
        '''Perform the expectation step.

        This will automatically update the log-likelihood as part of the
        expectation calculation.  The function doesn't return anything but
        instead updates the internal state variables.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containg the dataset

        Returns
        -------
        float
            log-likelihood under the model
        '''
        if self._loglike is None:
            self._logger.debug('Allocating memory for internal state.')

            self._loglike = np.zeros((self._num_models, x.shape[1]))
            self._logsum = np.zeros((1, x.shape[1]))
            self._responsibilities = np.zeros_like(self._loglike)

            self._logger.debug('Storage:')
            self._logger.debug('  Log-likelihood - %dx%d',
                               *self._loglike.shape)
            self._logger.debug('  Loglike-Sum - %dx%d',
                               *self._logsum.shape)
            self._logger.debug('  Model Responsibilities - %dx%d',
                               *self._loglike.shape)

        # Compute the per-sample likelihoods.
        for i, model in enumerate(self._models):
            self._loglike[i, :] = model.loglikelihood(x)

        self._logsum[:] = scipy.special.logsumexp(self._loglike, axis=0)
        self._responsibilities[:] = np.exp(self._loglike - self._logsum)
        return self._logsum.sum()

    def _maximization(self, x):
        '''Perform the maximization step.'''
        for i, model in enumerate(self._models):
            model.update(x, self._responsibilities[i, :])


class AdaptiveGMM(EMAlgorithm):
    '''A Gaussian Mixture Model with a Dirichlet prior for the mixture weights.

    Adding a Dirichlet prior onto the mixture weights allows it to
    automatically the number of models in the mixture.  This works nearly the
    same as the :class:`GMM` but has an extra expectation term for the weights,
    allowing them to be more closely tied to the optimization.  Once training
    is complete, all values with these expectations very close to zero will be
    automatically removed from the mixture model.

    Doing all of this doesn't actually require that much extra computation.
    Internally, the class has a "weight of weights" that keep track of how the
    weights are distributed.  This is then used when computing the actual model
    weights.  In other words, the main difference from the original
    :class:`GMM` is that there, the mixture weights are estimated after the
    model parameters.  Here, they are *part* of the model.

    Being able to handle this new condition requires the introduction of a new
    hyperparameter, :math:`\\alpha`.  This is known as a concentration
    parameter that controls how "big" the mixtures will be.  If
    :math:`\\alpha \\ll 1` then the algorithm will try to use as few mixtures
    as possible.  If :math:`\\alpha \\gg 1` then it does the exact opposite.

    Note
    ----
    The current implementation looks at the log-likelihood when checking for
    convergence.  This is different from other approaches, such as what
    scikit-learn does in its `BayesianGaussianMixture`_ class.  There, a lower
    bound is computed on the log-likelihood, a value that should increase over
    time.  The log-likelihood being estimated at any given iteration may
    actually decrease as the optimization moves between local minima.

    .. _BayesianGaussianMixture: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html

    Attributes
    ----------
    alpha : float
        the Dirichlet concentration parameter that controls how "clumpy" the
        mixture is
    max_models : int
        the maximum number of models to search for
    num_models : int
        the number of models that the algorithm estimated from the data
    means : numpy.ndarray
        a DxK array containing the set of model means
    covariances : numpy.ndarray
        a KxDxD array containing the set of model covariances
    weights : numpy.ndarray
        a D-length array containing the model weights; these will sum to '1'
    '''  # noqa: E501
    def __init__(self, alpha, max_models, max_iterations=1000,
                 tolerance=1e-6):
        '''
        Parameters
        ----------
        alpha : float
            concentration parameter
        max_models : int
            the *maximum* number of clusters that the mixture model will find
        max_iterations : int
            maximum number of EM algorithm iterations
        tolerance : float
            convergence tolerance
        '''
        super().__init__(VariationalGaussianMixture, max_iterations, tolerance)
        self.alpha = alpha
        self.max_models = max_models
        self._loglike = None
        self._logsums = None
        self._responsibilities = None

    @property
    @_is_trained
    def means(self):
        return np.transpose(
            np.squeeze(np.array([model.mu for model in self._models])))

    @property
    @_is_trained
    def covariances(self):
        return np.array([model.Sigma for model in self._models])

    def _initialize(self, x):
        '''Initialize the GMM using K-means.

        This simply runs K-means with the number of components, or models, that
        are expected for the final GMM.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containg the dataset to model
        '''
        self._logger.debug(
            'Performing K-means initialization with %d initial clusters.',
            self.max_models)
        kmeans = sklearn.cluster.KMeans(n_clusters=self.max_models,
                                        n_jobs=2)
        kmeans.fit(x.T)
        mu_init = np.mean(x, axis=1)[:, np.newaxis]
        weights_init = np.zeros((1, x.shape[1]))

        for i in range(self.max_models):
            weights_init[:] = 0
            weights_init[0, kmeans.labels_ == i] = 1 / self.max_models
            model = self._model_type(self.alpha, x.shape[0], self.max_models,
                                     mu=mu_init)
            try:
                model.update(x, weights_init)
                self._models.append(model)
            except LinearAlgebraError:
                self._logger.debug(
                    'Failed into initialize the %d-th model, decrementing '
                    'model count.', i)
                self.max_models -= 1

    def _expectation(self, x):
        '''Perform the expectation step.

        This will automatically update the log-likelihood as part of the
        expectation calculation.  The function doesn't return anything but
        instead updates the internal state variables.

        Parameters
        ----------
        x : numpy.ndarray
            a DxN array containg the dataset

        Returns
        -------
        float
            log-likelihood under the model
        '''
        if self._loglike is None:
            self._logger.debug('Allocating memory for internal state.')

            self._loglike = np.zeros((self.max_models, x.shape[1]))
            self._logsum = np.zeros((1, x.shape[1]))
            self._responsibilities = np.zeros_like(self._loglike)

            self._logger.debug('Storage:')
            self._logger.debug('  Log-likelihood - %dx%d',
                               *self._loglike.shape)
            self._logger.debug('  Loglike-Sum - %dx%d',
                               *self._logsum.shape)
            self._logger.debug('  Model Responsibilities - %dx%d',
                               *self._loglike.shape)

        # Generate summed-alpha value need for computing log-likelihoods.
        alpha_hat = np.sum(model.alpha for model in self._models)

        # Compute the per-sample likelihoods.
        for i, model in enumerate(self._models):
            model.alpha_sum = alpha_hat
            self._loglike[i, :] = model.loglikelihood(x)

        self._logsum[:] = scipy.special.logsumexp(self._loglike, axis=0)
        self._responsibilities[:] = np.exp(self._loglike - self._logsum)
        return self._logsum.sum()

    def _maximization(self, x):
        '''Perform the maximization step.'''
        for i, model in enumerate(self._models):
            model.update(x, self._responsibilities[i, :])

    def _cleanup(self):
        '''Update the model weights.

        Unlike a standard GMM, the model weights do not have to be calculated
        directly at any point in the EM algorithm.  The weights can be
        calculated after the training is complete.
        '''
        weights = np.sum(self._responsibilities, axis=1)
        weights = weights / np.sum(weights)
        for i, model in enumerate(self._models):
            model.weight = weights[i]

    @_is_trained
    def prune(self, th=1e-3):
        '''Remove any models with negligible weights.

        A model is considered to have negligible weighting if it is less than
        some fraction of the largest weight.  The :meth:`prune` method will
        remove these models since they most likely do not contribute anything
        substantial to the overall mixture.

        Parameters
        ----------
        th : float
            the threshold after which a model weight is considered numerically
            insignificant

        Raises
        ------
        :exc:`~leitmotiv.exceptions.AlgorithmUntrainedError`
            if the mixture model has not been trained yet
        ValueError
            if ``th`` is not less than '1'
        '''
        if not (th > 0 and th < 1):
            raise ValueError('Threshold must be a value between 0 and 1')

        # Compute the ratios in the log-domain to avoid working with really
        # small numbers.
        log_weights = np.log10(self.weights + np.finfo(float).eps)
        max_weight = np.max(log_weights)
        deltas = max_weight - log_weights

        # Remove all mixtures with small weights.
        keep = deltas < -np.log10(th)
        self._models = [model for i, model in enumerate(self._models) if keep[i]]  # noqa: E501

        # Renormalize the weights
        weights = self.weights
        weights = weights / np.sum(weights)
        for i, model in enumerate(self._models):
            model.weight = weights[i]
