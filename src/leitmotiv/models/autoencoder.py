import collections

import numpy as np
import torch
import torch.nn.functional as F

from leitmotiv.models import Model


class _SqueezeLayer(torch.nn.Module):
    '''Squeezes a layer to make it fully-connected.

    It also will apply batch normalization if the batch size is greater than 1.
    '''
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(num_features)

    def forward(self, x):
        x = torch.squeeze(torch.squeeze(x, 3), 2)
        if x.shape[0] > 1:
            x = self.batch_norm(x)
        return x


def _encoder_block(C_in, C_out, K, is_output=False):
    '''Creates an "encoder block".

    An encoder block is a sequence of layers that will process and then
    downsample an image.  Replication padding is used so that the the output is
    always half the size of the input.

    Parameters
    ----------
    C_in : int
        number of input channels
    C_out : int
        number of output channels
    K : int
        width of the filter kernel
    is_output : bool
        if this is the last output block, remove the normalization and
        activation functions (output has some nuances)

    Returns
    -------
    :class:`torch.nn.Sequential`
        the encoder "block"
    '''
    layers = []
    layers.append(('rep2', torch.nn.ReplicationPad2d(K // 2)))
    layers.append(('conv', torch.nn.Conv2d(C_in, C_out, K, stride=2,
                                           bias=False)))
    if is_output:
        layers.append(('squeeze', _SqueezeLayer(C_out)))
    else:
        layers.append(('norm', torch.nn.BatchNorm2d(C_out)))

    layers.append(('relu', torch.nn.ReLU()))
    return torch.nn.Sequential(collections.OrderedDict(layers))


def _decoder_block(C_in, C_out, K, is_output=False):
    '''Creates a "decoder block".

    A decoder block takes an input image and upsamples it using a
    fractionally-strided convolution.  This allows the decoder to reconstruct
    an input given some lower-dimensional representation.

    Parameters
    ----------
    C_in : int
        number of input channels
    C_out : int
        number of output channels
    K : int
        width of the filter kernel
    batchnorm : bool
        enable or disable the use of batch normalization
    is_output : bool
        indicate that this is an output layer, so it should have a tanh
        activation function

    Returns
    -------
    :class:`torch.nn.Sequential`
        the encoder "block"
    '''
    layers = []
    layers.append(('fconv', torch.nn.ConvTranspose2d(C_in, C_out, K,
                                                     stride=2, padding=1,
                                                     output_padding=1,
                                                     bias=is_output)))

    if is_output:
        layers.append(('sigmoid', torch.nn.Sigmoid()))
    else:
        layers.append(('norm', torch.nn.BatchNorm2d(C_out)))
        layers.append(('relu', torch.nn.ReLU()))

    return torch.nn.Sequential(collections.OrderedDict(layers))


class Encoder(torch.nn.Module):
    '''Encoder Module.

    The encoder module takes in an input image and then projects it down into
    an N-dimensional latent space.  Because the encoder performs variational
    inference, each dimension :math:`z_i` in the latent vector :math:`\\vec{z}`
    is assumed to distributed according to a univariate Gaussian
    :math:`z_i \\sim \\mathcal{N}(\\mu_i, \\sigma_i)`.

    The encoder has been set up so that the dimensionality of the latent space
    will be twice the original input image size.
    '''
    def __init__(self, width):
        '''
        Parameters
        ----------
        width : int
            width of the images being sent into the encoder; must be a power of
            two
        '''
        super().__init__()

        log2 = np.log2(width)
        nlayers = int(log2)
        if np.floor(log2) != np.ceil(log2):
            raise ValueError('Image dimension must be a power-of-two.')

        # The network halves the image size after each convolution.  It also
        # doubles the number of input channels, producing a structure that
        # looks something like:
        #
        #   width:      64 -> 32 -> 16 ->  8 ->  4 ->   2 ->   1
        #   channels:    3 ->  8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 128
        #                                                           |
        #                                                           V
        #                                                           64-mu
        #                                                           64-sigma
        #
        # If you generalizing the sequence for some value of 2**N, where N is
        # a positive integer, then the last tensor has 4x the number of
        # channels as the input width.  A fully-connected network then
        # compresses the space down to be 2x the width, so that the
        # dimensionality of the latent space is the same as the image width.

        layers = []
        layers.append(('input', _encoder_block(3, 8, 3)))
        for i in range(nlayers-2):
            C_in = 2**(i+3)
            C_out = 2**(i+4)
            layers.append(('layer%d' % (i+1), _encoder_block(C_in, C_out, 3)))
        layers.append(('output', _encoder_block(2**(nlayers+1), 4*width, 1,
                                                is_output=True)))
        self.convnet = torch.nn.Sequential(collections.OrderedDict(layers))
        self.compressor = torch.nn.Linear(4*width, 2*width, bias=False)

    def forward(self, x):
        '''Performs a forward pass of the encoder.'''
        conv = self.convnet(x)
        result = self.compressor(conv)

        # Extract the mean and stddev from the convolution output and apply the
        # different activation functions so the mean can be positive or
        # negative by the stddev can only be positive.
        mu, sigma = torch.chunk(result, 2, 1)
        log_sigma = F.softplus(sigma)

        return mu, torch.exp(log_sigma)


class Decoder(torch.nn.Module):
    '''Decoder Module.

    The decoder module takes an N-dimensional vector and then backprojects it
    back to an image.
    '''
    def __init__(self, width):
        '''
        Parameters
        ----------
        width : int
            width of the images being sent into the encoder; must be a power of
            two
        '''
        super().__init__()

        log2 = np.log2(width)
        nlayers = int(log2)
        if np.floor(log2) != np.ceil(log2):
            raise ValueError('Image dimensional must be a power-of-two.')

        # The decoder doublers the image size after each convolution.  It also
        # halves the number of input channels, doing the exact opposite as the
        # encoder.  The resulting structure is:
        #
        #   width:       1 ->  2 ->  4 ->  8 -> 16 -> 32 -> 64
        #   channels:   64 -> 64 -> 32 -> 16 ->  8 ->  4 ->  3
        #
        # The final output is a 3-channel RGB image.  The sequence is meant to
        # reverse what the encoder does so it takes some latent vector and puts
        # it back to the original image space.

        layers = []
        layers.append(('input', _decoder_block(width, 2**(nlayers+1), 3)))
        for i in range(nlayers-2):
            C_in = 2**(nlayers+1-i)
            C_out = 2**(nlayers-i)
            layers.append(('layer%d' % (i+1), _decoder_block(C_in, C_out, 3)))
        layers.append(('output', _decoder_block(8, 3, 3, is_output=True)))
        self.convnet = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        '''Performs a forward pass of the decoder.'''
        return self.convnet(x)


class VAEModel(torch.nn.Module):
    '''Model implementing a variational autoencoder.

    The :class:`VAEModel` is comprised of an encoder, which projects an image
    to a lower-dimensional latent space, and a decoder, which generates an
    image given a sample drawn from that latent space.

    Attributes
    ----------
    encoder : :class:`Encoder`
        VAE's encoding half
    decoder : :class:`Decoder`
        VAE's decoding half
    '''
    def __init__(self, width):
        '''
        Parameters
        ----------
        width : int
            width/height that images will be resized to, prior to being sent
            into the VAE
        '''
        super().__init__()
        self._width = width
        self.encoder = Encoder(width)
        self.decoder = Decoder(width)

    def forward(self, x):
        '''Apply the network onto some data set.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            a :math:`N \\times C \\times W \\times W` tensor with the input
            images

        Returns
        -------
        mu : :class:`torch.Tensor`
            a :math:`N \\times D` vector containing the distribution means
            within the latent space
        sigma : :class:`torch.Tensor`
            a :math:`N \\times D` vector contianing the diagonals of the
            covariance matrices for distributions within the latent space
        reconstruction : :class:`torch.Tensor`
            a :math:`N \\times C \\times W \\times W` tensor with the
            reconstructed images
        '''
        if x.shape[2] != self._width and x.shape[3] != self._width:
            raise ValueError('Input width and height must be %d (was %d x %d)'
                             % (self._width, x.shape[2], x.shape[3]))

        mu, sigma = self.encoder(x)
        z = mu + sigma*torch.randn_like(mu)
        z = torch.unsqueeze(torch.unsqueeze(z, -1), -1)
        reconstruction = self.decoder(z)
        return mu, sigma, reconstruction


class VariationalAutoencoder(Model):
    '''A variational autoencoder for finding the latent space of an image set.

    A variational autoencoder (VAE) takes an image and projects it down into a
    lower-dimensional latent space.  The encoder assumes that the latent space
    is modelled by the standard multivariate Normal distribution (zero mean,
    identity covariance).  Features within that space are themselves modelled
    as diagonal multivariate normal distributions.  The covariance of any
    particular image then represents the relative uncertainty of that image's
    representation within that space.

    Attributes
    ----------
    model : :class:`VAEModel`
        the model that implements the VAE
    image_width : int
        expected width/height of the images going into the VAE
    dimensionality : int
        dimensionality of the latent space
    '''
    def __init__(self, width, sigma=1.0, lr=2e-4, betas=(0.5, 0.999)):
        '''
        Parameters
        ----------
        width : int
            width/height that images will be resized to, prior to being sent
            into the VAE
        sigma : float
            hyperparameter used to control how close the reconstruction must be
            to the original image
        lr : float
            learning rate
        betas : tuple
            value of the two beta parameters used in the Adam optimizer
        '''
        self._ndim = width
        self._width = width
        self._in_training = False
        self._optim_type = torch.optim.Adam
        self._log_prob_scale = 1.0/(2.0*sigma**2)
        self._opt_args = {'lr': lr, 'betas': betas}

        self.model = VAEModel(width)
        self._init_optim()

    def _init_optim(self):
        '''Initialized the optimizer.'''
        self.optim = torch.optim.Adam(self.model.parameters(), **self._opt_args)  # noqa: E501

    @property
    def image_width(self):
        return self._width

    @property
    def dimensionality(self):
        return self._ndim

    def to_gpu(self):
        '''Move the model onto the GPU.'''
        if torch.cuda.is_available():
            self.model.cuda()
            self._init_optim()

    def to_cpu(self):
        '''Move the model onto the CPU.'''
        self.model.cpu()
        self._init_optim()

    def score(self, data):
        '''Compute the cost of the model on some data.

        The cost calculated with the Evidence Lower BOund (ELBO), which is the
        defined as the log-likelihood of the model being a good representation
        of the input data.

        Parameters
        ----------
        data : :class:`torch.Tensor`
            the images used for training, stored as a
            :math:`N \\times 3 \\times W \\times W` tensor
        beta : float
            regularization parameter used when computing the ELBO

        Returns
        -------
        elbo : float
            the ELBO value
        l2 : float
            value of the L2-norm between the data and the model's
            reconstruction
        kl : float
            value of the KL divergence between the latent representation and
            the standard Normal distribution
        '''
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, 0)

        batch_size = data.shape[0]
        mu, sigma, reconstruction = self.model(data)

        # Compute the log-likelihood of the reconstruction under a Gaussian
        # prior.
        l2 = torch.sum((reconstruction - data)**2)
        log_likelihood = -self._log_prob_scale*l2

        # Compute the Kullbeck-Leibler divergence.
        kl = 0.5*torch.sum((mu**2 + sigma**2 - torch.log(sigma**2) - 1.0))

        # Scale by the batch size.
        log_likelihood /= batch_size
        kl /= batch_size

        elbo = log_likelihood - kl
        return {'elbo': elbo, 'log_likelihood': log_likelihood, 'kl': kl}

    def train(self, data):
        '''Train the model on the provided images.

        This calls the optimizer to train the model on the provided image
        data.  In this case, the optimizer will attempt to minimize the
        reconstruction error between the input images and what the network
        thinks it should look like.  In the process, it will start creating a
        lower-dimensional representation of that image.

        Parameters
        ----------
        data : :class:`torch.Tensor`
            the images used for training, stored as a
            :math:`N \\times 3 \\times W \\times W` tensor
        beta : float
            regularization parameter used when computing the ELBO

        Returns
        -------
        dict
            a dictionary containing the internal costs
        '''
        if not self._in_training:
            self.model.train(True)
            self._in_training = True

        self.optim.zero_grad()
        scores = self.score(data)
        loss = -scores['elbo']
        loss.backward()
        self.optim.step()

        return scores

    def infer(self, data):
        '''Obtain the data's latent space representation.

        Parameters
        ----------
        data : :class:`torch.Tensor`
            a :math:`N \\times 3 \\times W \\times W` tensor containing one or
            more images to apply the model inference to

        Returns
        -------
        mu : :class:`torch.Tensor`
            a :math:`N \\times D` tensor containing the mean of the latent
            space representation(s)
        sigma : :class:`torch.Tensor`
            a :math:`N \\times D` tensor containing the standard deviations of
            the latent space representation(s)
        '''
        if self._in_training:
            self.model.train(False)
            self._in_training = False

        mu, sigma = self.model.encoder(data)

        mu = torch.reshape(mu, mu.shape[:2])
        sigma = torch.reshape(sigma, sigma.shape[:2])

        return mu, sigma

    def generate(self, latent):
        '''Generate an image sample, given some latent space representation.

        The latent space representation vector should have all of its
        components on [-6, 6].  This is not a hard requirement, as the space is
        continuous and without bound.  However, as the latent distribution is
        forced to be similar to a zero-mean Gaussian with unit variance, going
        outside this range may not produce reasonable results.

        Parameters
        ----------
        latent : :class:`torch.Tensor`
            a :math:`N \\times D` tensor containing the latent space
            representation(s)

        Returns
        -------
        :class:`torch.Tensor`
            a :math:`N \\times 3 \\times W \\times W` tensor containing an
            obtained from the latent space
        '''
        if self._in_training:
            self.model.train(False)
            self._in_training = False

        latent = torch.unsqueeze(torch.unsqueeze(latent, -1), -1)
        reconstructed = self.model.decoder(latent)
        return reconstructed

    def to_dict(self):
        '''Return the model's dictionary representation.'''
        return {
            'model': self.model.state_dict(),
            'width': self._width,
            'dimensions': self._ndim,
            'opt_args': self._opt_args,
            'log_prob_scale': self._log_prob_scale
        }

    @staticmethod
    def from_dict(model_dict):
        '''Load the model from its dictionary representation.'''
        vae = VariationalAutoencoder(model_dict['width'],
                                     **model_dict['opt_args'])
        vae._log_prob_scale = model_dict['log_prob_scale']
        vae.model.load_state_dict(model_dict['model'])
        return vae
