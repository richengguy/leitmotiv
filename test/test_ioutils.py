import base64

import numpy as np

from leitmotiv.io import deserialize_array
from leitmotiv.io import serialize_ndarray


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
