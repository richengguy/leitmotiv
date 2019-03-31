import pytest
import torch

from leitmotiv.models import Dataset


class TestDataset(object):
    '''Test the leitmotiv.model.Dataset object.'''
    def test_init(self, library):
        dataset = Dataset(library, img_dim=256)
        library.get_hashes.assert_called()
        assert dataset.img_dim == 256
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

    def test_cuda_memoization(self, library):
        dataset = Dataset(library, img_dim=256, to_gpu=True)

        for i in range(3):
            img = dataset[0]
            assert torch.is_tensor(img)
            assert img.shape == (3, 256, 256)

        # Check that the cache was used as expected.
        cache_info = dataset.__getitem__.cache_info()
        assert cache_info.hits == 2
        assert cache_info.misses == 1
        assert cache_info.maxsize == 256
        assert cache_info.currsize == 1

        # Need to clear cache to avoid it affecting other tests.
        dataset.__getitem__.cache_clear()