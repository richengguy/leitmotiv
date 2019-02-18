import functools

import numpy as np
import PIL
import torch.utils.data

from leitmotiv.io import imread, to_ndarray


__all__ = [
    'Dataset'
]


class Dataset(torch.utils.data.Dataset):
    '''Allows a leitmotiv library to act as a dataset.

    The dataset uses an LRU cache to avoid hitting the database too many times,
    since that will also result in the application waiting for an image to be
    read from disk and then resized.
    '''
    def __init__(self, library, img_dim=512):
        '''
        Parameters
        ----------
        library : :class:`~leitmotiv.library.Library`
            initialized leitmotiv library
        img_dim : int
            width/height of the image as presented during training
        '''
        self._library = library
        self._img_dim = img_dim
        self._hashes = list(library.get_hashes(sort_by_date=False))

    @functools.lru_cache(maxsize=256)
    def __getitem__(self, index):
        '''Retrieve an image from the database and convert it into a tensor.

        Parameters
        ----------
        index : int
            zero-based index of a particular picture

        Returns
        -------
        image : :class:`torch.Tensor`
            an image tensor that is :math:`S \\times S` square; the value of
            :math:`S` is set when the dataset is first initialized
        aspect_ratio : float
            the aspect ratio of the original image, defined as
            :math:`A = \\frac{W}{H}`
        '''
        path = self._library.get_path(self._hashes[index])

        # Load the image and resize to the requested dimensions.
        img = imread(path).resize((self._img_dim, self._img_dim),
                                  resample=PIL.Image.BICUBIC)

        # Convert it first to a numpy array, followed by a conversion to a
        # PyTorch tensor.
        img = to_ndarray(img)
        if img.ndim == 2:
            img = np.dstack((img, img, img))

        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)

    def __len__(self):
        '''Return the number of images in the library.'''
        return len(self._hashes)
