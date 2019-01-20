import scipy.linalg

from leitmotiv.exceptions import LinearAlgebraError


def from_matrix(A):
    '''Converts a positive-definite matrix into its Cholesky decomposition.

    This is a convenience function around SciPy's ``cholesky`` method to ensure
    that the correct arguments are used.

    Parameters
    ----------
    A : numpy.ndarray
        a positive-definite matrix

    Returns
    -------
    numpy.ndarray
        a lower-triangular matrix :math:`\\mathbf{L}` such that
        :math:`\\mathbf{A} = \\mathbf{L}\\mathbf{L}^T`

    Raises
    ------
    :exc:`~leitmotiv.exceptions.LinearAlgebraError`
        if the matrix could not be decomposed (e.g. because it's not positive
        definite)
    '''
    try:
        return scipy.linalg.cholesky(A, lower=True)
    except Exception as e:
        raise LinearAlgebraError('Could not perform Cholesky decomposition.') from e  # noqa: E501


def to_matrix(L):
    '''Converts a Cholesky decompostion into the full matrix.

    This is a convenience function to make certain operations a bit clearer.

    Parameters
    ----------
    L : numpy.ndarray
        a DxD lower-triangular matrix such that
        :math:`\\mathbf{A} = \\mathbf{L}\\mathbf{L}^T`

    Returns
    -------
    numpy.ndarray
        :math:`\\mathbf{A}`
    '''
    return L @ L.T


def inverse(A):
    '''Compute the inverse of a positive-definite matrix.

    The uses a Cholesky decomposition to speed up the process.  See
    :func:`inverse_from_decomp` for more information.  This is approximately
    twice as fast as calling ``scipy.linalg.inv`` directly on a 5x5 matrix::

        In [10]: %timeit scipy.linalg.inv(S)
        51.6 µs ± 130 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

        In [11]: %timeit cholesky.inverse(S)
        20.9 µs ± 50.1 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    Parameters
    ----------
    A : numpy.ndarray
        a DxD positive-definite matrix

    Returns
    -------
    numpy.ndarray
        the inverse matrix

    Raises
    ------
    :exc:`~leitmotiv.exceptions.LinearAlgebraError`
        if the inverse failed
    '''  # noqa: E501
    try:
        L = scipy.linalg.cholesky(A, lower=True)
        return inverse_from_decomp(L)
    except Exception as e:
        raise LinearAlgebraError from e


def inverse_from_decomp(L):
    '''Compute the inverse of a matrix given its Cholesky decomposition.

    The inverse is performed by recognizing the following relationship.  First,
    the Cholesky decomposition of the positive-definite matrix
    :math:`\\mathbf{A}` is defined as

    .. math::
        \\mathbf{A} = \\mathbf{L}\\mathbf{L}^T,

    where :math:`\\mathbf{L}` is a lower-triangular matrix.  Therefore,

    .. math::
        \\mathbf{A}^{-1} &= \\left(\\mathbf{L}\\mathbf{L}^T\\right)^{-1} \\\\
                         &= \\mathbf{L}^{-T}\\mathbf{L}^{-1}.

    Because :math:`\\mathbf{L}` is lower-triangular, finding
    :math:`\\mathbf{L}^{-1}` is significantly faster than solving for
    :math:`\\mathbf{A}^{-1}` directly.  If the decomposition is cached at some
    previous stage then this is significantly faster than inverting directly::

        In [17]: %timeit scipy.linalg.inv(S)
        52.4 µs ± 1.88 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

        In [18]: %timeit cholesky.inverse(L)
        2.08 µs ± 11.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    Parameters
    ----------
    L : numpy.ndarray
        a DxD lower-triangular matrix such that
        :math:`\\mathbf{A} = \\mathbf{L}\\mathbf{L}^T`

    Returns
    -------
    numpy.ndarray
        the inverse of :math:`\\mathbf{A}`, i.e. :math:`\\mathbf{A}^{-1}`
    '''  # noqa: E501
    Linv, info = scipy.linalg.lapack.dtrtri(L, True)
    if info != 0:
        raise LinearAlgebraError('LAPACK had a non-zero exit value.')
    return Linv.T @ Linv


def update(L, x, beta=1.0):
    '''Perform a rank-1 update on a Cholesky decomposition.

    Todo
    ----
    Implement using a proper Cholesky rank-1 update algorithm, possibly in
    C/C++ if the speed is an issue.

    Parameters
    ----------
    L : numpy.ndarray
        a DxD lower-triangular matrix such that
        :math:`\\mathbf{A} = \\mathbf{L}\\mathbf{L}^T`
    x : numpy.ndarray
        a Dx1 vector that will be used to update :math:`\\mathbf{L}`
    beta : float
        scaling factor applied onto the update vector

    Returns
    -------
    numpy.ndarray
        an updated :math:`\\mathbf{L}` such that it is the Cholesky
        decomposition of :math:`\\mathbf{A} + \\beta\\vec{x}\\vec{x}^T`
    '''
    return scipy.linalg.cholesky(L @ L.T + beta*(x @ x.T), lower=True)


def downdate(L, x, beta=1.0):
    '''Perform a rank-1 downdate on a Cholesky decomposition.

    Todo
    ----
    Implement using a proper Cholesky rank-1 update algorithm, possibly in
    C/C++ if the speed is an issue.

    Parameters
    ----------
    L : numpy.ndarray
        a DxD lower-triangular matrix such that
        :math:`\\mathbf{A} = \\mathbf{L}\\mathbf{L}^T`
    x : numpy.ndarray
        a Dx1 vector that will be used to downdate :math:`\\mathbf{L}`
    beta : float
        scaling factor applied onto the update vector

    Returns
    -------
    numpy.ndarray
        an updated :math:`\\mathbf{L}` such that it is the Cholesky
        decomposition of :math:`\\mathbf{A} + \\beta\\vec{x}\\vec{x}^T`

    Raises
    ------
    :exc:`leitmotiv.exceptions.LinearAlgebraError`
        if the result matrix will no longer be positive-definite
    '''
    try:
        return scipy.linalg.cholesky(L @ L.T - beta*(x @ x.T), lower=True)
    except Exception as e:
        raise LinearAlgebraError('Resulting matrix is singular') from e
