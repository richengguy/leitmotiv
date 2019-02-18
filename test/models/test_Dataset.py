import collections
import os.path
from unittest import mock

import pytest
import skimage
import torch

from leitmotiv.models import Dataset
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


class TestDataset(object):
    '''Test the leitmotiv.model.Dataset object.'''
    def test_init(self, library):
        dataset = Dataset(library, img_dim=256)
        library.get_hashes.assert_called()
        assert dataset._img_dim == 256
        assert len(dataset._hashes) == 5

    def test_len(self, library):
        dataset = Dataset(library)
        assert len(dataset) == 5

    @pytest.mark.parametrize('index', [0, 1, 2, 3, 4])
    @pytest.mark.parametrize('reps', [1, 2, 3])
    def test_getitem(self, library, index, reps):
        dataset = Dataset(library, img_dim=256)

        for i in range(reps):
            img = dataset[index]
            assert torch.is_tensor(img)
            assert img.shape == (3, 256, 256)

        # Check that the cache was used as expected.
        cache_info = dataset.__getitem__.cache_info()
        assert cache_info.hits == reps-1
        assert cache_info.misses == 1
        assert cache_info.maxsize == 256
        assert cache_info.currsize == 1

        # Need to clear cache to avoid it affecting other tests.
        dataset.__getitem__.cache_clear()
