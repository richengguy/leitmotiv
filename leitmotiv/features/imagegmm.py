import enum
import json

import numpy as np

import scipy.ndimage
import skimage.color

from leitmotiv.algorithms import AdaptiveGMM, MultivariateNormalDistribution
from leitmotiv.features._features import ImageDescriptor
from leitmotiv.io import to_ndarray, serialize_ndarray, deserialize_array


__all__ = [
    'ImageGMM'
]


class _ColourSpace(enum.IntEnum):
    rgb = 0
    yuv = 1
    lab = 2


def _normalization_transform(data):
    '''Compute the normalization transform for the sampled data.

    Parameters
    ----------
    data : numpy.ndarray
        sampled values

    Returns
    -------
    N : numpy.ndarray
        the normalization transform for image coordinates
    Nc : numpy.ndarray
        the normalization transform for RGB values
    '''
    mean = np.mean(data, axis=1)
    stddev = np.std(data, axis=1) + 10*np.finfo(float).eps

    N = np.eye(data.shape[0]+1)
    N[0:-1, -1] = -mean / stddev
    N[0:-1, 0:-1] = np.diag(1 / stddev)

    return N


def _generate_samples(width, height, samples):
    '''Generate the set of image positions.

    Parameters
    ----------
    width, height : int
        image dimensions
    samples : int
        number of samples to obtain from the image

    Returns
    -------
    numpy.ndarray
        2xN array containing the sample positions
    '''
    num_pixels = width*height
    indices = set(np.random.randint(num_pixels, size=samples))
    while len(indices) < samples:
        indices.add(np.random.randint(num_pixels, size=1)[0])

    y, x = np.unravel_index(list(indices), (height, width))

    return np.transpose(np.c_[x, y])


class ImageGMM(ImageDescriptor):
    '''Generate a spatiochromatic GMM for an image.

    A spatiochromatic GMM provides a mechanism for describing the content in an
    image using a statistical model.  The model assumes that the image is
    representable as a 5-dimensional space, :math:`(x,y,r,g,b)`, that is a
    combination of a pixel's spatial coordinate and its RGB, or equivalent,
    colour value.  Some normalization is assumed so that the relative scale of
    the spatial and chromatic features are the same.  In this case of a
    monochrome (greyscale) image, the mixture models the 3-dimensional space
    :math:`(x,y,I)`.

    To avoid large images from having an excess of data when compared to small
    images, the :class:`ImageGMM` randomly samples :math:`(x,y)` coordinates in
    the provided image.  The sampled :math:`\\vec{c} = (r,g,b)` value is
    computed as

    .. math::
        \\vec{c} = \\frac{1}{\\sum_{i,j}W_{i,j}}
                  \\sum_{i,j} W_{i,j}\\vec{I}(x + i, y + j),

    where

    .. math::
        W_{i,j} = \\exp\\left\\{
            -\\frac{1}{2}
            \\frac{i^2 + j^2}{\\sigma^2}
        \\right\\},

    :math:`\\sigma = \\frac{w_{S}}{6}` and :math:`w_{S}` is the width of the
    sampling region.  This is implemented by first convolving the image with a
    Gaussian having the desired :math:`\\sigma` and then sampling from the
    filtered image.

    Internally, the :class:`ImageGMM` uses an variational Bayesian GMM,
    computed using the :class:`~leitmotiv.algoriths.AdaptiveGMM`.  This allows
    the number of mixtures to be estimated automatically during the training
    process.  Once the training has completed, the resulting GMM can the be
    used to compare against another :class:`ImageGMM`.

    Note
    ----
    The :class:`ImageGMM` is automatically trained the moment that it is
    created.  Executing

    >>> desc = ImageGMM(img)

    will cause the caller to block until training is complete.

    Attributes
    ----------
    N : numpy.ndarray
        a read-only 3x3 transform used to normalize image coordinates
    models : list of :class:`leitmotiv.algorithms.MultivariateNormalDistribution`
        a list of distributions that contain the model parameters, including
        the mixture weights
    monte_carlo_samples : int
        number of samples used during the Monte Carlo simulation that computes
        the KL-divergence between two :class:`ImageGMM` objects
    '''  # noqa: E501
    __IG_FTYPE = 2
    __IG_DTYPE = np.float
    __IG_HEADER = ''

    def __init__(self, img, alpha=0.01, max_clusters=15, sigma=7, samples=1000,
                 seed=None, colour_space='lab', _contents=None):
        '''
        Parameters
        ----------
        img : numpy.ndarray
            greyscale or colour image
        alpha : float
            Dirichlet concentration parameter; the larger this number, the
            more likely the final mixture has a larger number of models in it
        max_clusters : int
            maximum number of clusters in the mixture
        sigma : float
            size of the Gaussian filter such values that are within
            :math:`\\pm 3\\sigma` of each other can be considered as being the
            same point
        samples : int
            number of samples to extract
        seed : int
            random seed used to control the state of the PRNG used in sampling;
            can be set to a known value if the PRNG sequence needs to be
            repeatible
        colour_space : {'yuv', 'lab', 'rgb'}
            convert the input data into one of the three colour spaces
        _contents : dict
            special data structure used during deserialization; this overrides
            all other parameters
        '''
        self.monte_carlo_samples = 10000

        if _contents is not None:
            self._cs = _contents['colourSpace']
            self._dim = _contents['dimensions']
            self.N = deserialize_array(_contents['N'])
            self._finish_init(deserialize_array(_contents['weights']),
                              deserialize_array(_contents['means']),
                              deserialize_array(_contents['covariances']))

            return

        if seed is not None:
            prng_state = np.random.get_state()
            np.random.seed(seed)

        img = to_ndarray(img)
        num_channels = img.shape[2] if len(img.shape) == 3 else 1

        # Prepare the image for processing, including applying the Gaussian
        # filter.
        sigma = [sigma]*num_channels
        sigma[-1] = 0
        filt = scipy.ndimage.gaussian_filter(img, sigma, mode='nearest',
                                             truncate=3)

        # Generate the set of samples.
        width = filt.shape[1]
        height = filt.shape[0]
        pts = _generate_samples(width, height, samples)
        rgb = np.squeeze(filt[pts[1, :], pts[0, :], :]).T

        self._dim = (width, height)
        self._cs = {
            'rgb': _ColourSpace.rgb,
            'yuv': _ColourSpace.yuv,
            'lab': _ColourSpace.lab
        }[colour_space]

        rgb = self._convert_colour(rgb, True)

        # Concatenate the results.
        data = np.vstack((pts, rgb, np.ones((1, samples))))
        del pts, rgb

        self.N = _normalization_transform(data[0:-1, :])
        data = (self.N @ data)[0:-1, :]

        # Train the GMM on the dataset.
        gmm = AdaptiveGMM(alpha, max_clusters)
        gmm.train(data)
        gmm.prune()

        # Store the pruned GMM's weights, means and covariances.  The weights
        # are renormalized to ensure that they sum to one, since the pruning
        # removes models.
        self._finish_init(gmm.weights, gmm.means, gmm.covariances)

        if seed is not None:
            np.random.set_state(prng_state)

    def _finish_init(self, weights, means, covariances):
        '''Sets the internal model storage of the object.

        Parameters
        ----------
        weights : numpy.ndarray
            1xN array of model weights
        means : numpy.ndarray
            DxN array of model means
        covariances : numpy.ndarray
            NxDxD array of model covariances
        '''
        # Used for ImageGMM to ImageGMM comparisons.
        self._mc_samples = None

        # All other ImageGMM properties.
        self._Ninv = np.linalg.inv(self.N)
        self._weights = weights / np.sum(weights)
        self.models = []
        for n in range(len(weights)):
            mu = means[:, n]
            sigma = np.squeeze(covariances[n, :, :])
            mvn = MultivariateNormalDistribution(mu[:, np.newaxis], sigma)
            mvn.weight = self._weights[n]
            self.models.append(mvn)

    def _convert_colour(self, rgb, fwd):
        '''Perform a colour conversion using a Scikit-Image function.

        This method is provided to create a "pseudo-image" that Scikit-Image
        can work with.  It won't work if the image is a 3xN array.

        Parameters
        ----------
        rgb : numpy.ndarray
            3xN array of sampled colour points
        fwd : bool
            if ``True`` then the transform maps RGB to some other colour space
            and if ``False`` then it's in from another colour space into RGB

        Returns
        -------
        numpy.ndarray
            sampled points in the new colour space
        '''
        if fwd:
            method = {
                _ColourSpace.yuv: skimage.color.rgb2yuv,
                _ColourSpace.lab: skimage.color.rgb2lab,
                _ColourSpace.rgb: lambda i: i.astype(np.float) / 255
            }[self._cs]
        else:
            method = {
                _ColourSpace.yuv: skimage.color.yuv2rgb,
                _ColourSpace.lab: skimage.color.lab2rgb,
                _ColourSpace.rgb: lambda i: i
            }[self._cs]

        rgb = rgb[:, np.newaxis, :]  # array is 3x1xN
        rgb = np.swapaxes(rgb, 0, 2)  # array is now Nx1x3

        # Perform the conversion.
        out = method(rgb)

        # Undo the warping.
        return np.squeeze(out).T

    @property
    def weights(self):
        return self._weights

    @property
    def means(self):
        return np.hstack([mvn.mu for mvn in self.models])

    @property
    def covariances(self):
        return np.array([mvn.Sigma for mvn in self.models])

    def visualize(self):
        '''Generate a visualization of the GMM.'''
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        means = self.means
        covariances = self.covariances
        weights = self.weights

        # Undo the scaling on the means.
        means = self._Ninv @ np.vstack((means, np.ones((1, len(self.models)))))
        means = means[0:-1, :]

        mu = means[0:2, :]
        rgb = self._convert_colour(means[2:, :], False)

        ax = plt.gca()
        ax.set_facecolor('black')
        for n in range(len(self.models)):
            sigma = self._Ninv[0:2, 0:2] @ covariances[n, 0:2, 0:2] @ self._Ninv[0:2, 0:2]  # noqa: E501

            eig_vals, eig_vecs = np.linalg.eigh(sigma)
            unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
            # Ellipse needs degrees
            angle = 180 * angle / np.pi
            # eigenvector normalization
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)

            ell = mpl.patches.Ellipse(mu[:, n],
                                      eig_vals[0], eig_vals[1],
                                      180 + angle)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(weights[n] / np.max(weights))
            ell.set_facecolor(rgb[:, n])
            ax.add_artist(ell)

        # plt.scatter(mu[0, :], mu[1, :])
        plt.axis([0, self._dim[0], 0, self._dim[1]])
        plt.gca().invert_yaxis()
        plt.title('Clusters %d' % len(self.models))

    def sample(self, nsamples=1, for_output=False):
        '''Sample from the GMM.

        The sampling process first selects a Gaussian using a multinomial
        distribution before sampling from that specific Gaussian.

        Parameters
        ----------
        nsamples : int
            number of samples to draw
        for_output : bool
            if ``True`` then the data is assumed to be for output, and so it
            will be cropped to be in the image bounds, colours converted to RGB
            and clamped to 0 to 255

        Returns
        -------
        numpy.ndarray
             a DxN array of samples drawn from the GMM
        '''
        counts = np.random.multinomial(nsamples, self._weights)
        pts = np.vstack([
            np.random.multivariate_normal(np.squeeze(mvn.mu), mvn.Sigma,
                                          sample)
            for mvn, sample in zip(self.models, counts)
        ]).T

        if for_output:
            ones = np.ones((1, pts.shape[1]))
            pts = self._Ninv @ np.vstack((pts, ones))
            pts = pts[0:-1, :]
            pts[2:, :] = self._convert_colour(pts[2:, :], False)

            # Points outside of the image.
            in_x = np.logical_and(pts[0, :] >= 0, pts[0, :] < self._dim[0])
            in_y = np.logical_and(pts[1, :] >= 0, pts[1, :] < self._dim[1])
            pts = pts[:, np.logical_and(in_x, in_y)]

            # Clamp RGB values to be between 0 and 255.
            pts[2:, :] = np.clip(255*pts[2:, :], 0, 255)

        return pts

    def compare(self, other, method='js'):
        '''Compare this instance to another GMM.

        This will compute an approximation to the Jensen-Shannon divergence
        between this instance and another :class:`ImageGMM` instance using the
        Monte Carlo method described in a StackOverflow question [so2018]_.

        A set of random draws from this distribution are generated when this
        method is first called.  They are cached for future use.  The number of
        samples is controlled by the :attr:`monte_carlo_samples` attribute.

        .. [so2018] https://stackoverflow.com/a/26079963

        Parameters
        ----------
        other : :class:`ImageGMM`
            the other GMM
        method : ``kl`` or ``js``
            select the type of comparison method to use; default is
            Jensen-Shannon but Kullbeck-Leibler is also supported

        Returns
        -------
        double
            the Jensen-Shannon or Kullbeck-Leibler divergence between the two
            GMMs

        Raises
        ------
        TypeError
            if the other object is not an :class:`ImageGMM` instance
        '''
        if not isinstance(other, ImageGMM):
            raise TypeError('Other object must be an ImageGMM.')

        if self.monte_carlo_samples != other.monte_carlo_samples:
            raise ValueError('Number of Monte Carlo samples must be the same.')

        # Make sure that both ImageGMMs have samples available.
        if self._mc_samples is None:
            self._mc_samples = self.sample(self.monte_carlo_samples)

        if other._mc_samples is None:
            other._mc_samples = other.sample(other.monte_carlo_samples)

        # Compute the divergence.
        def loglikelihoods(P, x):
            return scipy.special.logsumexp(np.array([
                mvn.loglikelihood(x) for mvn in P
            ]), axis=0)

        def kl_divergence(P, Q, x):
            Px = loglikelihoods(P, x)
            Qx = loglikelihoods(Q, x)
            return (Px - Qx).mean()

        def js_divergence(P, Q, x, y):
            Px = loglikelihoods(P, x)
            Qx = loglikelihoods(Q, x)
            Mx = scipy.special.logsumexp(np.vstack([Px, Qx]), axis=0)

            Py = loglikelihoods(P, y)
            Qy = loglikelihoods(Q, y)
            My = scipy.special.logsumexp(np.vstack([Py, Qy]), axis=0)

            return ((Px - Mx).mean() + (Qy - My).mean())/2 + np.log(2)

        if method == 'js':
            return js_divergence(self.models, other.models,
                                 self._mc_samples, other._mc_samples)
        elif method == 'kl':
            return kl_divergence(self.models, other.models, self._mc_samples)
        else:
            raise ValueError('Unknown comparison method "%s".' % method)

    # -- Serialization Methods ---------------------------------------------- #
    def serialize(self):
        '''Serialize the descriptor object.

        A subclass will need to implement this if the data it contains needs to
        be exported.

        Returns
        -------
        bytes
            a bytes object containing the serialized form of the descriptor
        '''
        # Pack the header data.
        data = bytearray(self._pack_header())

        # Now, construct a JSON object with the GMM parameters and convert it
        # into a byte-array to easy serialization.
        contents = {
            'colourSpace': self._cs,
            'dimensions': self._dim,
            'N': serialize_ndarray(self.N),
            'weights': serialize_ndarray(self.weights),
            'means': serialize_ndarray(self.means),
            'covariances': serialize_ndarray(self.covariances)
        }
        data.extend(json.dumps(contents).encode())

        # Finally, return the serialized result.
        return data

    @staticmethod
    def from_bytes(buffer):
        '''Generate a new ImageDescriptor from a byte array.

        This has no preferred implementation and it is the subclass'
        responsibility to provide an implementation.  The only requirement is
        that the buffer is converted into an :class:`ImageDescriptor` object.

        Parameters
        ----------
        buffer: bytes
            byte array containing the serialized descriptor

        Returns
        -------
        ImageDescriptor
            the initialized descriptor
        '''
        ftype, _, offset = ImageGMM._unpack_header(buffer)

        # Check the serialized type ID.
        if ftype != ImageGMM.__IG_FTYPE:
            raise ValueError('Descriptor type is not for an ImageGMM.')

        # Parse the JSON contents.
        contents = json.loads(buffer[offset:])
        return ImageGMM(None, _contents=contents)

    @staticmethod
    def ftype():
        '''Return the type's identifier.

        Returns
        -------
        int
            a unique ID value
        '''
        return ImageGMM.__IG_FTYPE

    @staticmethod
    def _hdrfmt():
        '''Return the header format used for storing descriptors.

        Returns
        -------
        str
            a ``struct``-compatible string
        '''
        return ImageGMM.__IG_HEADER
