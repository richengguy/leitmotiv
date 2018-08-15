import pytest

import numpy as np
from scipy import linalg

from leitmotiv.math import cholesky
from leitmotiv.exceptions import LinearAlgebraError


def test_decomp():
    '''Cholesky decomposition is called correctly.'''
    np.random.seed(1234)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    S = x @ x.T

    Lwrapper = cholesky.from_matrix(S)
    Lscipy = linalg.cholesky(S, lower=True)

    assert np.linalg.norm(Lwrapper - Lscipy) == pytest.approx(0)


def test_invalid_decomp():
    '''Invalid Cholesky decomposition raises an exception.'''
    S = np.zeros((3, 3))
    with pytest.raises(LinearAlgebraError):
        cholesky.from_matrix(S)


def test_reconst():
    '''Matrix is recovered from a Cholesky decomposition.'''
    np.random.seed(1234)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    S = x @ x.T

    L = cholesky.from_matrix(S)
    T = cholesky.to_matrix(L)

    print(S)
    print(L)
    print(T)

    assert np.linalg.norm(S - T) == pytest.approx(0)


def test_inverse():
    '''Cholesky inverse is calculated to machine epsilon.'''
    np.random.seed(1234)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    S = x @ x.T

    # Compare between a direct inverse and the Cholesky version.
    direct = linalg.inv(S)
    indirect = cholesky.inverse(S)

    assert np.linalg.norm(direct - indirect) == pytest.approx(0)


def test_inverse_from_decomp():
    '''Cholesky inverse is calculated to machine epsilon.'''
    np.random.seed(2345)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    S = x @ x.T
    L = linalg.cholesky(S, lower=True)

    # Compare between a direct inverse and from a Cholesky decomposition.
    direct = linalg.inv(S)
    indirect = cholesky.inverse_from_decomp(L)

    assert np.linalg.norm(direct - indirect) == pytest.approx(0)


def test_inverse_nonposdef():
    '''Expection raised if inverse is not for a positive-definite matrix.'''
    S = np.zeros((3, 3))
    with pytest.raises(LinearAlgebraError):
        cholesky.inverse(S)


def test_update():
    '''Updated decomposition is equivalent to a direct calculation.'''
    np.random.seed(1234)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    y = x[:, 1:]
    z = x[:, 1]

    A = y @ y.T
    Sfull = A + z @ z.T
    Lfull = linalg.cholesky(Sfull, lower=True)

    # Compute the reduced version and update it.
    Sred = y @ y.T
    Lred = linalg.cholesky(Sred, lower=True)
    Lupd = cholesky.update(Lred, z)

    print('Initial Matrices (Full and Reduced)')
    print(Sfull)
    print(Sred)

    print('Cholesky Matrices (Full and Reduced)')
    print(Lfull)
    print(Lred)

    print('Updated Cholesky Matrix')
    print(Lupd)

    # Compare the two.
    assert np.linalg.norm(Lupd - Lfull) == pytest.approx(0)


def test_downdate():
    '''Downdated decomposition is equivalent to a direct calculation.'''
    np.random.seed(1234)

    # Generate a positive-definite matrix.
    x = np.random.uniform(size=(5, 100))
    y = x[:, 1:]
    z = x[:, 1]

    A = y @ y.T
    Sfull = A - z @ z.T
    Lfull = linalg.cholesky(Sfull, lower=True)

    # Compute the reduced version and update it.
    Sred = y @ y.T
    Lred = linalg.cholesky(Sred, lower=True)
    Ldnd = cholesky.downdate(Lred, z)

    print('Initial Matrices (Full and Reduced)')
    print(Sfull)
    print(Sred)

    print('Cholesky Matrices (Full and Reduced)')
    print(Lfull)
    print(Lred)

    print('Downdated Cholesky Matrix')
    print(Ldnd)

    # Compare the two.
    assert np.linalg.norm(Ldnd - Lfull) == pytest.approx(0)
