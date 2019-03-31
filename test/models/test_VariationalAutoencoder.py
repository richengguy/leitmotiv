import pytest

import torch

from leitmotiv.models import VariationalAutoencoder
from leitmotiv.models.autoencoder import (_decoder_block, _encoder_block,
                                          VAEModel, Encoder, Decoder)


def test_encoder_block_size_for_even_dims():
    '''Ensure encoder block shrinks image correctly for even dimensions.'''
    block = _encoder_block(3, 16, 3)
    input = torch.ones((5, 3, 100, 100))
    output = block(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 16
    assert output.shape[2] == 50
    assert output.shape[3] == 50


def test_decoder_block_size_for_even_dims():
    '''Ensure decoder block scales image correctly for even dimensions.'''
    block = _decoder_block(3, 16, 3)
    input = torch.ones((5, 3, 50, 50))
    output = block(input)
    assert output.shape[0] == 5
    assert output.shape[1] == 16
    assert output.shape[2] == 100
    assert output.shape[3] == 100


class TestEncoder(object):
    '''Test the Encoder class.'''
    def test_init(self):
        encoder = Encoder(64, 512)
        assert len(encoder.convnet) == 9

        for i in range(9):
            if i == 0:
                C_in = 3
                C_out = 8
            else:
                C_in = C_out
                C_out = C_in * 2

            if i == 8:
                C_out = 128

            print("%d, %d" % (encoder.convnet[i][1].in_channels,
                              encoder.convnet[i][1].out_channels))

            assert encoder.convnet[i][1].in_channels == C_in
            assert encoder.convnet[i][1].out_channels == C_out

    def test_forward(self):
        encoder = Encoder(64, 512)
        input = torch.ones((5, 3, 512, 512))
        mu, sigma = encoder(input)

        assert mu.dim() == 4
        assert mu.shape[0] == 5
        assert mu.shape[1] == 64
        assert mu.shape[2] == 1
        assert mu.shape[3] == 1

        assert sigma.dim() == 4
        assert sigma.shape[0] == 5
        assert sigma.shape[1] == 64
        assert sigma.shape[2] == 1
        assert sigma.shape[3] == 1

    def test_init_incorrect_width_raises_exception(self):
        with pytest.raises(ValueError):
            Encoder(64, 500)


class TestDecoder(object):
    '''Test the Decoder class.'''
    def test_init(self):
        decoder = Decoder(64, 512)
        assert len(decoder.convnet) == 9

        for i in range(9):
            if i == 0:
                C_in = 64
                C_out = 1024
            else:
                C_in = C_out
                C_out = C_in / 2

            if i == 8:
                C_out = 3

            print("%d: %d, %d" % (i, decoder.convnet[i][0].in_channels,
                                  decoder.convnet[i][0].out_channels))

            assert decoder.convnet[i][0].in_channels == C_in
            assert decoder.convnet[i][0].out_channels == C_out

    def test_forward(self):
        decoder = Decoder(64, 512)
        input = torch.ones((5, 64, 1, 1))
        output = decoder(input)

        assert output.shape[0] == 5
        assert output.shape[1] == 3
        assert output.shape[2] == 512
        assert output.shape[3] == 512

    def test_forward_squeezed_input_raises_exception(self):
        decoder = Decoder(64, 512)
        input = torch.ones((5, 64))
        with pytest.raises(RuntimeError):
            decoder(input)

    def test_init_incorrect_width_raises_exception(self):
        with pytest.raises(ValueError):
            Decoder(64, 500)


class TestVAEModel(object):
    '''Test the VAEModel class.'''
    def test_init(self):
        model = VAEModel(64, 512)
        assert len(model.encoder.convnet) == 9
        assert len(model.decoder.convnet) == 9

    def test_init_invalid_width_raises_exception(self):
        with pytest.raises(ValueError):
            VAEModel(64, 500)

    @pytest.mark.parametrize('ndim', [2, 4, 8, 16, 32, 64, 128])
    def test_forward(self, ndim):
        model = VAEModel(ndim, 512)
        input = torch.ones((5, 3, 512, 512))

        with torch.no_grad():
            mu, sigma, reconst = model(input)

        assert mu.shape == sigma.shape
        assert mu.dim() == 4
        assert mu.shape[0] == 5
        assert mu.shape[1] == ndim

        assert reconst.shape == input.shape

    def test_forward_invalid_input_raises_exception(self):
        input = torch.ones((8, 3, 64, 64))
        with pytest.raises(ValueError):
            model = VAEModel(8, 128)
            model(input)


class TestVariationalAutoencoder(object):
    '''Test the VariationalAutoencoder class.'''
    def test_init(self):
        vae = VariationalAutoencoder(64, 512)
        assert len(vae.model.encoder.convnet) == 9
        assert len(vae.model.decoder.convnet) == 9

    @pytest.mark.parametrize('nimg', [1, 2, 3, 4])
    def test_compute_cost(self, nimg):
        input = torch.ones((nimg, 3, 32, 32))

        with torch.no_grad():
            vae = VariationalAutoencoder(2, 32)
            scores = vae.score(input)

        assert scores['elbo'].dim() == 0
        assert scores['elbo'].allclose(scores['log_likelihood'] - scores['kl'])

    @pytest.mark.parametrize('nimg', [1, 2, 3, 4])
    def test_train(self, nimg):
        input = torch.ones((nimg, 3, 32, 32))
        vae = VariationalAutoencoder(2, 32)
        cost = vae.train(input)
        assert 'elbo' in cost
        assert cost['elbo'].allclose(cost['log_likelihood'] - cost['kl'])

    @pytest.mark.parametrize('nimg', [1, 2, 3, 4])
    def test_infer(self, nimg):
        input = torch.ones((nimg, 3, 32, 32))
        vae = VariationalAutoencoder(2, 32)
        mu, sigma = vae.infer(input)

        print(mu.shape)

        assert mu.dim() == 2
        assert mu.shape[0] == nimg and mu.shape[1] == 2
        assert mu.shape == sigma.shape

    @pytest.mark.parametrize('nimg', [1, 2, 3, 4])
    def test_generate(self, nimg):
        input = torch.ones((nimg, 2))
        vae = VariationalAutoencoder(2, 32)
        reconst = vae.generate(input)

        assert reconst.dim() == 4
        assert reconst.shape[0] == nimg
        assert reconst.shape[1] == 3
        assert reconst.shape[2] == 32
        assert reconst.shape[3] == 32

    def test_run_on_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available on this system.')

        def on_gpu(elem):
            if hasattr(elem, 'is_cuda'):
                assert elem.is_cuda

        def on_cpu(elem):
            if hasattr(elem, 'is_cuda'):
                assert not elem.is_cuda

        input = torch.ones((1, 3, 32, 32)).cuda()
        vae = VariationalAutoencoder(2, 32)
        vae.to_gpu()
        vae.model.apply(on_gpu)

        cost = vae.train(input)
        assert 'elbo' in cost
        assert cost['elbo'].allclose(cost['log_likelihood'] - cost['kl'])

        vae.to_cpu()
        vae.model.apply(on_cpu)

    def test_to_dict(self):
        vae = VariationalAutoencoder(2, 32)
        rep = vae.to_dict()

        assert rep['width'] == 32
        assert rep['dimensions'] == 2

    def test_from_dict(self):
        vae = VariationalAutoencoder(2, 32)
        alt = VariationalAutoencoder.from_dict(vae.to_dict())
        assert alt._ndim == 2
        assert alt._width == 32
