import itertools
import struct

import numpy as np

from ._features import ImageDescriptor


__all__ = [
    'ColourHistogram'
]


def _rle_compress(buffer):
    '''Compress a byte array using Run Length Encoding (RLE).

    The colour histograms are quite sparse, so they can be easily compressed
    using RLE to remove most of the zeros.

    Parameters
    ----------
    buffer: bytes
        uncompressed byte array

    Returns
    -------
    bytes
        byte array in RLE format
    '''
    outbuffer = bytearray()
    for unique, values in itertools.groupby(buffer):
        nelem = len(list(values))
        outbuffer.extend(struct.pack('!IB', nelem, unique))
    return bytes(outbuffer)


def _rle_decompress(buffer):
    '''Uncompress an RLE byte array.

    This assumes that the input string is RLE-compressed.  If it is not then
    the output is undefined.

    Parameters
    ----------
    buffer: bytes
        input byte array in RLE format

    Returns
    -------
    bytes
        uncompressed byte array
    '''
    outbuffer = bytearray()
    for count, value in struct.iter_unpack('!IB', buffer):
        outbuffer.extend(count*(value.to_bytes(1, byteorder='big')))
    return outbuffer


class ColourHistogram(ImageDescriptor):
    '''Compute the joint histogram of an image.

    The ColourHistogram is a 3D structure that contains the number of times a
    particular (R,G,B) triplet, or equivalent, occurred.  In the simplest case,
    an image is described by a ``2^24``-element array (roughly 16.8 million
    entries).  While storing the complete histogram in memory is entirely
    feasible, the ColourHistogram supports quantizing the colours prior to the
    histogram construction minimize the in-database storage.
    '''
    GB_PLANE = 0
    RB_PLANE = 1
    RG_PLANE = 2

    __CH_FTYPE = 1
    __CH_DTYPE = np.int32

    # Header Definition:
    #    bits (1 byte) - number of bits for each image channel
    #    channels (1 byte) - number of image colour channels
    __CH_HEADER = 'BB'

    def __init__(self, img, nbits=8, bins=None):
        '''Initialize a ColourHistogram object.

        Parameters
        ----------
        img: numpy.ndarray
            input image with a depth of 8-bits per channel and '1' or '3'
            channels
        nbits: int
            the number of bits per channel in the histogram; this must be a
            value between 1 and 8
        bins: numpy.ndarray
            this histogram array; the image can be set to ``None`` if this is
            provided

        Raises
        ------
        ValueError:
            if the input image type is unsupported or number of bits is invalid
        '''
        if bins is None:
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]

            if nbits < 1 or nbits > 8:
                raise ValueError(
                    'Histogram bit depth must be between 1 and 8.')
            if img.dtype != np.uint8:
                raise ValueError('Only 8-bit, unsigned ints are supported.')
            if channels != 1 and channels != 3:
                raise ValueError('Image can only have 1 or 3 channels.')

            # Compute the number of histgram bins and the quantization factor.
            div = 2**(8-nbits)
            bins = 2**nbits

            # Allocate memory for the histogram.
            self._bins = np.zeros(tuple(bins for n in range(channels)),
                                  dtype=ColourHistogram.__CH_DTYPE)

            if channels == 1:
                for v in img.flatten():
                    self._bins[v // div] += 1
            else:
                for x in range(width):
                    for y in range(height):
                        i = img[y, x, 0] // div
                        j = img[y, x, 1] // div
                        k = img[y, x, 2] // div
                        self._bins[i, j, k] += 1
        else:
            if bins.dtype != np.int32:
                raise ValueError('Histogram must contain 32-bit values.')
            if len(bins.shape) != 1 and len(bins.shape) != 3:
                raise ValueError('Histogram must be 1D or 3D.')

            # Copy over the provided histogram.
            self._bins = bins.copy()

    def __getitem__(self, inds):
        '''Allow read-only access to the accumulator bins.'''
        return self._bins[inds]

    def points(self, expand=False):
        '''Return the coordinates of the non-zero histogram bins.

        Parameters
        ----------
        expand: bool
            if set then the returned points are expanded by the number of times
            that value occurred in the histogram

        Returns
        -------
        numpy.ndarray
            an 3xN array containing the point positions
        '''
        ind = self._bins.nonzero()
        if expand:
            cnt = self._bins[ind]
            pts = np.zeros((3, cnt.sum()))
            n = 0
            for i, val in enumerate(zip(*ind)):
                pts[:, n:(n+cnt[i])] = val
                n = n+cnt[i]
            return pts
        else:
            return np.array(ind)

    def serialize(self):
        '''Represent the histogram as a byte array.

        The histogram is represented using a simple format that is the
        histogram array with a small header.

        Returns
        -------
        bytes
            the ColourHistogram as a byte-array
        '''
        # Number of bits and channels can be derived from histogram shape.
        nbits = int(np.log2(self._bins.shape[0]))
        channels = len(self._bins.shape)

        # Pack that information into a header an append the actual histogram
        # data.
        data = bytearray(self._pack_header(nbits, channels))
        data.extend(_rle_compress(self._bins.tobytes()))

        # Finally, return the result to do something with it.
        return data

    def plane(self, channel):
        '''Return the marginal distribution for a particular plane.

        To make visualization a bit easier, this will calculate the marginal
        distributions (i.e. sums) for a particular column in the histogram
        volume.  For example, to calculate the red-green distribution then the
        summation is done along the blue channel (i.e. channel '2').

        Parameters
        ----------
        channel: int
            the direction of the summation

        Returns
        -------
        numpy.ndarray
            a 2D histgram containing the distribution along a particular plane

        Raises
        ------
        IndexError
            if the channel ID is not 0, 1 or 2
        '''
        if channel < 0 or channel > 2:
            raise IndexError('Channel must be 0, 1 or 2.')

        return np.sum(self._bins, axis=channel)

    def visualize(self):
        '''Visualize the histgram.

        This is shown in 3D on a scatter plot where the colours indicate how
        many times a particular colour value occurred.
        '''
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.pyplot as plt

        ind = self._bins.nonzero()
        val = self._bins[ind[0], ind[1], ind[2]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ind[0], ind[1], ind[2], c=val, s=val)
        plt.show()

    @staticmethod
    def from_bytes(buffer):
        '''Create a ColourHistgram object from a byte array.

        Parameters
        ----------
        buffer: bytes
            an array of bytes containing a serialized histogram

        Returns
        -------
        ColourHistogram
            the deserialized object
        '''
        ftype, data, offset = ColourHistogram._unpack_header(buffer)

        # Check the histogram type ID.
        if ftype != ColourHistogram.__CH_FTYPE:
            raise ValueError('Descriptor type is not for a ColourHistogram')

        # Verify that the number of bits and channels are correct.
        nbits, channels = data
        if nbits < 1 or nbits > 8:
            raise ValueError('Bin size must be between 1 and 8 bits.')
        if channels != 1 and channels != 3:
            raise ValueError('Histogram can only have 1 or 3 channels.')

        # Read in the actual data from the buffer.
        nbins = 2**nbits
        nelem = nbins**channels
        data = np.frombuffer(_rle_decompress(buffer[offset:]),
                             dtype=ColourHistogram.__CH_DTYPE)

        if data.size != nelem:
            raise ValueError('Expected %d elements, read in %d.' % (data.size,
                                                                    nelem))

        # Finally, reshape the buffer and return the actual ColourHistogram.
        shape = tuple(nbins for n in range(channels))
        return ColourHistogram(None, bins=np.reshape(data, shape))

    @staticmethod
    def ftype():
        return ColourHistogram.__CH_FTYPE

    @staticmethod
    def _hdrfmt():
        return ColourHistogram.__CH_HEADER
