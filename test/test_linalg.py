import numpy as np

import pytest

from leitmotiv.math import linalg


def test_scatter_no_alpha():
    '''Scatter matrix is generated correctly without a centering vector.'''
    np.random.seed(1111)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    w = np.random.normal(loc=0.5, scale=0.1, size=(1, 100))

    S = np.zeros((5, 5))
    for i in range(100):
        y = x[:, i][:, np.newaxis]
        S = S + (w[0, i]*y) @ y.T

    # Now compare against the other function.
    T = linalg.weighted_scatter_matrix(w, x)

    print(S)
    print(T)

    assert np.linalg.norm(S - T) == pytest.approx(0)


def test_scatter_with_alpha():
    '''Scatter matrix is generated correctly with a centering vector.'''
    np.random.seed(1111)

    mu = np.random.uniform(size=(3, 1))

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(3, 100)) + mu
    # w = np.random.normal(loc=0.5, scale=0.1, size=(1, 100))
    w = np.ones((1, 100))

    S = np.zeros((3, 3))
    for i in range(100):
        y = x[:, i][:, np.newaxis] - mu
        S = S + (w[0, i]*y) @ y.T

    # Now compare against the other function.
    T = linalg.weighted_scatter_matrix(w, x, mu)

    print(S)
    print(T)

    assert np.linalg.norm(S - T) == pytest.approx(0)
