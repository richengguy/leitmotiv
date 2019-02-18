import os.path

import pytest


@pytest.fixture
def samples_directory():
    '''Returns the directory where test samples are stored.'''
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, 'data'))
