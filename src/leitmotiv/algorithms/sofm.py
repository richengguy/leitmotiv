import logging

import numpy as np

from leitmotiv.exceptions import AlgorithmUntrainedError


__all__ = [
    'dist_euclsq',
    'init_random',
    'SOFM'
]


def dist_euclsq(x, y):
    '''Compute the square-Euclidean distance in a feature space.

    Parameters
    ----------
    x: numpy.ndarray
        first element
    y: numpy.ndarray
        second element

    Returns
    -------
    numpy.ndarray
        a vector containing the squared-Euclidean distance between elements in
        'x' and 'y'; standard numpy broadcasting rules are used
    '''
    return np.sum(np.square(x - y), axis=0)


def init_random(X):
    '''Randomly initialize the values inside the matrix 'X'.

    The initialization is performed in-place.

    Parameters
    ----------
    X: numpy.ndarray
        matrix being initialized
    '''
    values = np.random.random(size=X.shape)
    np.copyto(X, values)


class SOFM(object):
    '''Train a Self-Organizing Feature Map on a data set.

    The SOFM is a type of connected graph, usually 1D or 2D, that can map
    high-dimensional spaces to lower ones.  Because the graph is defined on the
    higher space, the topology of that space is preserved in the lower
    dimensional representation.  From a machine learning viewpoint, it is a
    form of unsupervised learning since no training data needs to be provided
    into the network.

    Attributes
    ----------
    iterations: int
        number of training iterations
    sigma: float
        algorithm learning rate
    init_sigma: float
        initial learning rate value
    term_thresh: float
        the relative error after which convergence is assumed
    '''
    def __init__(self, gridsz, distmeas=dist_euclsq, initfn=init_random,
                 iterations=500, sigma=75, init_sigma=5, term_thresh=1e-6):
        '''Initialize the SOFM.

        Parameters
        ----------
        gridsz: (width, height)
            tuple containing the width and height of the SOFM grid
        distmeas: functor
            an object, that when called, will compute distances within a
            feature space; default is the squared-Euclidean distance
        initfn: functor
            an object, that when called, can be used to initialize the nodes
            within the SOFM; default is random initialization
        iterations: int
            number of training iterations
        sigma: float
            algorithm learning rate
        init_sigma: float
            initial learning rate value
        '''
        if gridsz[0] < 1 or gridsz[1] < 1:
            raise ValueError('Grid dimensions must be positive and non-zero.')

        # Setup the internal properties
        self._nnodes = gridsz[0]*gridsz[1]
        self._gridsz = gridsz
        self._logger = logging.getLogger(__name__)
        self._distmeas = distmeas
        self._initfn = initfn
        self._nodes = None

        # Generate a look-up table for the node (x,y) coordinates
        self._gridmap = np.zeros((2, self._nnodes))
        for y in range(gridsz[1]):
            for x in range(gridsz[0]):
                i = x + y*gridsz[0]
                self._gridmap[0, i] = x
                self._gridmap[1, i] = y

        # Initialize the public properties
        self.iterations = iterations
        self.sigma = sigma
        self.init_sigma = init_sigma
        self.term_thresh = term_thresh

        self._logger.info('%dx%d SOFM ready for training.',
                          gridsz[0], gridsz[1])

    @property
    def gridsz(self):
        '''The size of the SOFM.'''
        return self._gridsz

    @property
    def dimensionality(self):
        '''The dimensionality of the SOFM nodes.'''
        if self._nodes is None:
            raise AlgorithmUntrainedError('SOFM has yet to be trained.')
        return self._nodes.shape[0]

    def node_coord(self, ind):
        '''Convert a node index or indices into a set of (x,y) coordinates.

        Parameters
        ----------
        ind: numpy.ndarray or int
            node index

        Returns
        -------
        numpy.ndarray
            a 2xN array, where 'N' is the same size as the input, containing
            the node coordinates
        '''
        return self._gridmap[:, ind]

    def node_value(self, *args):
        '''Get the value of a particular SOFM node.

        Parameters
        ----------
        ind: int
            the node index; must be passed on its own
        x, y: int
            the node's (x,y) coordinate; this must be passed on its own

        Returns
        -------
        numpy.ndarray
            a vector containing the nodes value in the input space of the data

        Raises
        ------
        leitmotiv.exceptions.AlgorithmUntrainedError
            if this is called before the SOFM has even been trained
        ValueError
            if the input arguments are incorrect; reason will be in the message
        '''
        if self._gridmap is None:
            raise AlgorithmUntrainedError('SOFM has yet to be trained.')

        nargs = len(args)

        if nargs == 1:
            ind = args[0]
        elif nargs == 2:
            ind = args[0] + self._gridsz[0]*args[1]
        else:
            raise ValueError(
                'Can only provide either a 2D coordinate or linear index.')

        return self._nodes[:, ind]

    def train(self, data):
        '''Train the SOFM on the provided data set.

        This performs a batch update on the data set by looking at all of the
        data in aggregate.  The individual SOFM nodes are updated in parallel
        but requires that the data be available ahead of time.

        The batch training assumes that the provided data is completely
        unrelated to any data that the SOFM may have already seen.  If a SOFM
        has already been trained then calling this function will reset the
        map and perform training from scratch.

        Parameters
        ----------
        data: numpy.ndarray
            a DxN numpy array containing the training data
        distmeas: function handle
            function handle that provides the distance measure that is most
            appropriate for the given data set
        '''
        nelem = data.shape[1]
        self._initialize(data)

        # Keep a copy of the previous set of the nodes from the prior
        # iteration.
        prev = self._nodes.copy()

        # Pre-allocate storage for the per-node weights.
        W = np.zeros((nelem, self._nnodes))

        # Batch SOFM Training Process
        self._logger.info('Perform SOFM batch training.')
        for n in range(self.iterations):
            sigma = self.init_sigma*np.exp(-n/self.sigma)

            # 1. Find the best-matching-unit (BMU) for all nodes.
            for i in range(nelem):
                # a. Find the best-matching-unit (BMU) in the input space.
                pt = data[:, i, np.newaxis]
                bmu = np.argmin(self._distmeas(pt, self._nodes))

                # b. Use the BMU to determine which way the SOFM needs to be
                # move during the update.
                pt = self._gridmap[:, bmu, np.newaxis]
                dn = np.sum(np.square(pt - self._gridmap), axis=0)
                W[i, :] = np.exp(-dn/(2.0*sigma))

            # 2. Update the nodes using the weights from the BMU step.
            for i in range(self._nnodes):
                S = np.sum(W[:, i])
                self._nodes[:, i] = np.sum(W[:, i].T*data, axis=1)/S

            # 3. Compare the current solution to the previous one.  If there
            # was little change, assume convergence and terminate the loop.  A
            # minimum number of iterations (100) must have already occurred
            # before this check is done.
            if n > 100:
                err = np.sum(np.sum(np.square(self._nodes - prev), axis=0))
                np.copyto(prev, self._nodes)
                if n % 10 == 0:
                    self._logger.debug('Iteration %d - Error %g', n, err)
                if err < self.term_thresh:
                    self._logger.info(
                        'SOFM converged after %d iterations with relative '
                        'error of %g.', n, err)
                    return

        # If this logging statement was hit then no convergence occurred.
        self._logger.info('SOFM batch training concluded without converging.')

    def _initialize(self, data):
        '''Initialize the SOFM using the provided initializer.

        This is called during training and is not intended to be used by an
        external caller.

        Parameters
        ----------
        data: numpy.ndarray
            the input data
        '''
        if self._nodes is not None:
            self._logger.debug(
                'Initialization has already happened...skipping.')
            return

        ndim = data.shape[0]
        nelem = data.shape[1]

        self._logger.info(
            'Initializing SOFM on %d-dimensional dataset with %d samples.',
            ndim, nelem)

        self._nodes = np.zeros((ndim, self._nnodes))
        self._initfn(self._nodes)
