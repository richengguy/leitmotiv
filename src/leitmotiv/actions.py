import datetime
import hashlib
import io
import logging

from leitmotiv.features import ImageGMM
from leitmotiv.library import Library, DistanceMeasure, DistanceType
from leitmotiv.io import imread


__all__ = [
    'InsertImages',
    'ExtractFeatures',
    'PairwiseDistance'
]


_EXIF_DATETIME_ORIGINAL = 36867


class InsertImages(object):
    '''Insert images into the application library.

    Attributes
    ----------
    config : dict-like object
        a dict-like object containing the runtime configuration
    '''
    def __init__(self, config):
        self.config = config['library']

    def __call__(self, path):
        '''Extract image metadata and prepare it for library insert.

        Parameters
        ----------
        path : path-like object
            image path

        Returns
        -------
        path : path-like object
            image path (passthrough)
        image_hash : str
            image's SHA256 hash
        dt : :class:`datetime.datetime`
            a datetime object containing the image's capture time
        thumb : bytes
            image thumbnail
        '''
        try:
            img, exif = imread(path, return_exif=True)
        except BaseException as e:
            raise RuntimeError('Failed to load %s' % path) from e

        # Generate the image hash.
        sha256 = hashlib.sha256()
        sha256.update(img.tobytes())
        image_hash = sha256.digest().hex()

        # Create the thumbnail.
        thumb = io.BytesIO()
        img.thumbnail((self.config['thumbnail']['width'],
                       self.config['thumbnail']['height']))
        img.save(thumb, 'jpeg', quality=90)

        # Obtain the date-time.
        dt = datetime.datetime.today()
        try:
            if exif is not None:
                dt = datetime.datetime.strptime(exif[_EXIF_DATETIME_ORIGINAL],
                                                '%Y:%m:%d %H:%M:%S')
        except KeyError:
            pass

        return path, image_hash, dt, thumb

    def reduce(self, args):
        '''Insert the image into the database.

        Parameters
        ----------
        args : tuple
            output from ``__call__``
        '''
        with Library() as lib:
            return lib.insert_image(*args)


class ExtractFeatures(object):
    '''Extract ImageGMM features given an image hash.

    Attributes
    ----------
    config : dict-like object
        a dict-like object containing the runtime configuration
    '''
    def __init__(self, config):
        '''
        Parameters
        ----------
        config : dict-like object
            a dict-like object containing the runtime configuration; this must
            have a 'GMM' key
        '''
        self.config = config['GMM']

    def __call__(self, imghash):
        '''Process the requested image.

        Parameters
        ----------
        imghash : str
            a SHA256 hash of the image that will be processed
        '''
        logger = logging.getLogger('leitmotiv')
        logger.info('Extracting features from %s' % imghash)

        with Library() as lib:
            img = imread(lib.get_path(imghash))
            gmm = ImageGMM(img, **self.config)
            lib.append_feature(imghash, gmm)


class PairwiseDistance(object):
    '''Compute the pairwise distance between two images.'''
    def __init__(self, config):
        '''
        Parameters
        ----------
        config : dict-like object
            a dict-like object containing the runtime configuration
        '''
        self.config = config['GMM']
        self._buffer = []

    def __call__(self, images):
        '''Process the pair of images.

        Parameters
        ----------
        images : (src, tgt)
            a tuple containing the hashes of the source and target images

        Returns
        -------
        double
            distance between the two images
        '''
        logger = logging.getLogger('leitmotiv')
        logger.info('Computing distances between %s and %s' % images)

        src, tgt = images

        with Library() as lib:
            src_desc = lib.get_features(src, ImageGMM.ftype())[0]
            tgt_desc = lib.get_features(tgt, ImageGMM.ftype())[0]

            src_desc.monte_carlo_samples = self.config['samples']
            tgt_desc.monte_carlo_samples = self.config['samples']

        return src, tgt, src_desc.compare(tgt_desc)

    def _flush_buffer(self):
        '''Flush the buffer contents into the library.'''
        with Library() as lib:
            lib.update_distances(DistanceMeasure(*args, ImageGMM.ftype(),
                                                 DistanceType.JENSEN_SHANNON)
                                 for args in self._buffer)
            self._buffer.clear()

    def reduce(self, args):
        '''Insert the distance pair into the library.

        This will store the distances in a temporary buffer until all of the
        processing is done.

        Parameters
        ----------
        args : tuple
            output from ``__call__``
        '''
        self._buffer.append(args)
        if len(self._buffer) > 1000:
            self._flush_buffer()

    def done(self):
        '''Insert any remaining distances into the library.'''
        if len(self._buffer) > 0:
            self._flush_buffer()
