def weighted_scatter_matrix(weights, x, alpha=None):
    '''Compute a weighted scatter matrix for some set of data.

    The weighted scatter matrix is defined as

    .. math::

        \\mathbf{S} = \\sum_{i=1}^N w_i \\vec{x}_i \\vec{x}_i^T,

    where :math:`w_i \\in \\mathcal{W}` is the i-th weight and

    .. math::

        \\mathbf{X} = \\begin{bmatrix}
            \\vec{x}_1 & \\vec{x}_2 & \\dots & \\vec{x}_N
        \\end{bmatrix}

    is the data the matrix is being computed from.  Note that if the weights
    are equal to one then :math:`\\mathbf{S} = \\mathbf{X}\\mathbf{X}^T`.  The
    purpose of this function is to use vectorized operators to quickly compute
    the sum above without requiring nested Python loops.

    Parameters
    ----------
    weights : numpy.ndarray
        a 1xD array of weights :math:`\\mathcal{W}`
    x : numpy.ndarray
        a DxN array containing :math:`\\mathbf{X}`
    alpha : numpy.ndarray
        a Dx1 array that may be subtracted from ``x``, i.e.
        :math:`\\vec{x}' = \\vec{x} - \\vec{\\alpha}`

    Returns
    -------
    numpy.ndarray
        the DxD scatter matrix :math:`\\mathbf{S}`
    '''
    if alpha is None:
        u = x
    else:
        u = x - alpha

    S = (weights * u) @ u.T
    return S
