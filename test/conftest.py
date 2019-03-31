import collections
import os.path
from unittest import mock

import pytest
import skimage

from leitmotiv.library import Library

IMAGES = collections.OrderedDict({
    'a': 'astronaut.png',
    'b': 'chelsea.png',
    'c': 'coffee.png',
    'd': 'ihc.png',
    'e': 'coins.png'  # this is a greyscale image, unlike the other images
})

KEYS = list(IMAGES.keys())


@pytest.fixture
def library():
    def get_path(query):
        return os.path.join(skimage.data_dir, IMAGES[query])

    library = mock.MagicMock(spec=Library)
    library.get_hashes.return_value = IMAGES.keys()
    library.get_path.side_effect = get_path
    return library


@pytest.fixture
def samples_directory():
    '''Returns the directory where test samples are stored.'''
    fdir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(fdir, 'data'))
