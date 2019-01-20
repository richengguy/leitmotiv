import numpy as np


def distance_to_similarity(distances, alpha=None, beta=None):
    '''Convert a distance matrix into a similarity matrix.

    A distance matrix provides a mechanism for discribing the dissimilarity
    between a set of points in some space.  In general, this matrix is
    symmetric with zeros along the main diagonal as the distance of a point to
    itself should be zero.  The distances are converted into similarities by
    one of two methods, controlled by the ``param`` argument.

    If no value is provided (the default) then the similarity is

    .. math::

        S(i,j) = 1 - \\frac{D(i,j)}{D_{max}(i)},

    where :math:`D(i,j)` is the distance entry and :math:`D_{max}(i)` is the
    maximum distance value that is not ``inf`` for the i-th row..  If a value
    is provided then the similarity is

    .. math::

        S(i,j) = \\frac{
            1
        }{
            1 + \\exp\\left\\{ \\alpha (x - \\beta) \\right\\}
        }.

    Parameters
    ----------
    distances : numpy.ndarray
        an NxN distance matrix
    alpha, beta : double
        values for controlling the sigmoid adjustment curve

    Returns
    -------
    numpy.ndarray
        similarity matrix

    Raises
    ------
    ValueError
        if the matrix is not square
    '''
    if distances.ndim != 2:
        raise ValueError('Input must be an NxN matrix.')
    if distances.shape[0] != distances.shape[1]:
        raise ValueError('Input must be an NxN matrix.')

    has_param = alpha is not None and beta is not None

    if has_param:
        similarity = 1 / (1 + np.exp(alpha*(distances + beta)))
    else:
        Dmax = np.max(distances)
        similarity = 1 - distances / Dmax

    return similarity


def sparsify_graph(adjacency, th=1e-3, normalize=True):
    '''Sparsify a graph via thresholding.

    Sparsity is enforced by performing two operations: setting any values below
    some threshold to zero and scaling them to be between 0 and 1.  The choice
    of threshold and whether to normalize will depend on the application.  The
    normalization operation is performed first.

    Parameters
    ----------
    adjacency : numpy.ndarray
        a graph's adjacency matrix
    th : double
        threshold value
    normalize : bool
        rescale the values to be between 0 and 1

    Returns
    -------
    numpy.ndarray
        "sparse" matrix (storage type is unmodified)
    '''
    if normalize:
        minval = adjacency.min()
        maxval = adjacency.max()
        adjacency = (adjacency - minval)/(maxval - minval)

    sparse = adjacency.copy()
    sparse[adjacency < th] = 0

    return sparse
