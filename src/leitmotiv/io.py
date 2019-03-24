import base64
import enum

import numpy as np
from PIL import Image


__all__ = [
    'imread',
    'imshow',
    'to_ndarray'
]


_EXIF_ORIENTATION = 274


class ExifOrientation(enum.IntEnum):
    '''Various orientations stored within EXIF data.

    The values used below come from the EXIF specification.  The enum names
    have been chosen to represent the equivalent clockwise rotation rather than
    the corner-corner mapping from the original specification.
    '''
    Original = 1
    Rotate90 = 8
    Rotate180 = 3
    Rotate270 = 6


def get_orientation(img):
    '''Check the EXIF data to see if the image has a preferred orientation.

    This is only possible if the image has associated EXIF data.  If there is
    no data then the image is treated as if it has a rotation flag of '0'.

    Parameters
    ----------
    img: PIL.Image
        a Pillow image object

    Returns
    -------
    ExifOrientation:
        enum representing the EXIF orientation flag
    '''
    try:
        exif = img._getexif()
    except AttributeError:
        return ExifOrientation.Original

    if _EXIF_ORIENTATION in exif:
        return ExifOrientation(exif[_EXIF_ORIENTATION])
    else:
        return ExifOrientation.Original


def imread(fname, return_exif=False, inspect_exif=True):
    '''Read in an image file.

    This provides a wrapper around the various libraries used to load image
    data.  It will always provide the final output using Pillow to keep the
    interface consistent.  Use :func:`to_ndarray` to convert the object into a
    numpy-compatible array.

    Parameters
    ----------
    fname : str
        path to the image file
    inspect_exif : bool
        look at the image EXIF data and apply any orientation changes post-load
    return_exif : bool
        return EXIF data alongside the original image

    Returns
    -------
    img : PIL.Image.Image
        the opened image
    exif : dict
        the image's EXIF data (only if ``return_exif`` is ``True``)
    '''
    img = Image.open(fname)
    try:
        exif = img._getexif()
    except AttributeError:
        exif = None
        inspect_exif = False

    if inspect_exif:
        orientation = get_orientation(img)
        if orientation == ExifOrientation.Rotate90:
            img = img.rotate(90, Image.BICUBIC, True)
        elif orientation == ExifOrientation.Rotate180:
            img = img.rotate(180, Image.BICUBIC)
        elif orientation == ExifOrientation.Rotate270:
            img = img.rotate(270, Image.BICUBIC, True)

    if return_exif:
        return img, exif
    else:
        return img


def imshow(img):
    '''Display an image using Matplotlib.

    Parameters
    ----------
    img: numpy.ndarray or PIL.Image.Image
        the image to display
    '''
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def to_ndarray(img, sz=None, ignore_exif=False):
    '''Convert an image into a numpy-compatible ndarray.

    The Image class stores a representation of the image, not the image itself,
    so some work is needed to convert it into a usable format.  Furthermore, in
    the case of camera images, these may have metadata, such as EXIF,
    indicating that the image requires a rotation.

    Note
    ----
    This may consume a lot of memory depending on the image size.  Pillow
    usually stores the compressed representation so this requires making an
    in-memory copy.

    Parameters
    ----------
    img: PIL.Image
        image object being converted
    sz: (width, height)
        if provided then this will specify the maximum width and height of the
        image
    ignore_exif: bool
        ignore any image manipulations based on the EXIF data

    Returns
    -------
    numpy.ndarray
        the numpy representation
    '''
    if not isinstance(img, Image.Image):
        raise ValueError('Expected a PIL.Image instance.')

    # Extract the EXIF data if it exists.
    try:
        exif = img._getexif()
    except AttributeError:
        ignore_exif = True

    # Create a copy of the image before resizing it.
    img = img.copy()
    if sz is not None:
        img.thumbnail(sz)

    # Convert into the numpy array.
    data = np.array(img)

    # Finally, check to see if a rotation is required.
    if not ignore_exif:
        if _EXIF_ORIENTATION in exif:
            rotation = ExifOrientation(exif[_EXIF_ORIENTATION])
        else:
            rotation = ExifOrientation.Original

        # Rotate the image depending on the EXIF tag value.
        if rotation == ExifOrientation.Rotate180:
            data = np.flip(data, 1)
        elif rotation == ExifOrientation.Rotate90:
            data = np.swapaxes(data, 0, 1)
            data = np.flip(data, 0)
        elif rotation == ExifOrientation.Rotate270:
            raise NotImplementedError('Need to implement EXIF Orientation 6')

    return data


def serialize_ndarray(array):
    '''Convert a Numpy ndarray into a JSON-compatible dictionary.

    The array is encoded into the following structure:

    .. code::

        {
            'data': << base64-encoded array >>
            'shape': << shape property >>
            'dtype': << dtype property >>
        }

    This has been adapted from the bokeh.serialization module:
    https://bokeh.pydata.org/en/latest/_modules/bokeh/util/serialization.html#encode_base64_dict

    Parameters
    ----------
    array : numpy.ndarray
        array being serialized

    Returns
    -------
    dict
        serializable dictionary
    '''
    data = array.tobytes()
    return {
        'data': base64.b64encode(data).decode('utf-8'),
        'shape': array.shape,
        'dtype': array.dtype.name
    }


def deserialize_array(data):
    '''Convert a formatted dictionary into a Numpy array.

    Parameters
    ----------
    data : dict
        dictionary representation of the particular array

    Returns
    -------
    numpy.ndarray
        array stored in the dictionary
    '''
    bytedata = base64.b64decode(data['data'])
    array = np.frombuffer(bytedata, dtype=data['dtype'])
    if len(data['shape']) > 1:
        array = np.reshape(array, data['shape'])
    return array
