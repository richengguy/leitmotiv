import base64
import os.path

import numpy as np
import pytest
import skimage

from leitmotiv.io import (deserialize_array, serialize_ndarray, imread,
                          to_ndarray)


def test_serialize():
    '''Numpy array properties are serialized correctly.'''
    array = np.eye(3)
    data = serialize_ndarray(array)
    assert len(data['shape']) == 2
    assert data['shape'] == (3, 3)
    assert data['dtype'] == array.dtype.name
    assert data['data'] == base64.b64encode(array.tobytes()).decode('utf-8')


def test_deserialize():
    '''Numpy array can be roundtripped.'''
    array = np.eye(3)
    data = serialize_ndarray(array)
    newarr = deserialize_array(data)
    assert len(newarr.shape) == 2
    assert newarr.shape == (3, 3)
    assert (newarr == np.eye(3)).all()


def test_load_png():
    '''Can load a PNG without any EXIF data.'''
    path = os.path.join(skimage.data_dir, 'astronaut.png')

    # Read the image.
    img, exif = imread(path, return_exif=True)
    assert img.size == (512, 512)
    assert exif is None

    # Convert to a numpy array.
    img = to_ndarray(img)
    assert img.shape == (512, 512, 3)


def test_load_jpeg(samples_directory):
    '''Can load a JPEG with some EXIF data.'''
    path = os.path.join(samples_directory, 'yonge-dundas.jpg')

    # Read the image.
    img, exif = imread(path, return_exif=True)
    assert img.size == (2000, 867)
    assert exif is not None

    # Convert to a numpy array.
    img = to_ndarray(img)
    assert img.shape == (867, 2000, 3)


def test_load_jpeg_with_stripped_exif(samples_directory):
    '''Can load a JPEG where the EXIF data has been removed.'''
    path = os.path.join(samples_directory, 'IMG_3247.jpg')

    # Read the image.
    img, exif = imread(path, return_exif=True)
    assert img.size == (128, 85)
    assert exif is None

    # Convert to a numpy array.
    img = to_ndarray(img)
    assert img.shape == (85, 128, 3)


def test_to_ndarray_without_PIL_image_raises_exception():
    '''Exception raised if to_ndarray() used incorrectly.'''
    with pytest.raises(ValueError):
        to_ndarray(np.zeros((128, 128, 3)))
