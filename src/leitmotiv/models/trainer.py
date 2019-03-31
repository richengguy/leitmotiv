import random
import statistics

import click

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from leitmotiv.models import Model


def _split_dataset(dataset, split):
    '''Split the dataset into a training and validation set.

    Parameters
    ----------
    dataset : :class:`~leitmotiv.models.Dataset`
        dataset being partitioned
    split : float
        a value between 0 and 1 indicating where the training split will be

    Returns
    -------
    training_set : :class:`torch.utils.data.Subset`
        the subset that will be used for training
    testing_set : :class:`torch.utils.data.Subset`
        the subset that will be used for validation
    '''
    nelem = len(dataset)
    ntesting = round(split*nelem)
    ntraining = nelem - ntesting

    indices = list(range(nelem))
    random.shuffle(indices)

    training_set = torch.utils.data.Subset(dataset, indices[0:ntraining])
    testing_set = torch.utils.data.Subset(dataset, indices[ntraining:nelem])

    return training_set, testing_set


def _train_sample(model, sample):
    '''Run a model's training method on a sample.'''
    loss = model.train(sample)
    if isinstance(loss, dict):
        return loss
    else:
        return {'loss': loss}


def _validate_sample(model, sample):
    '''Score a sample using a trained model.'''
    loss = model.score(sample)
    if isinstance(loss, dict):
        return loss
    else:
        return {'loss': loss}


def _training_mode(model, in_training):
    '''Put the model into training mode.'''
    try:
        model.model.train(in_training)
    except AttributeError:
        pass


def _loop_over_samples(fn, model, data, progbar):
    '''Run the training loop given a training set.'''
    _training_mode(model, True)
    losses = {}

    for sample in data:
        loss = fn(model, sample)
        for k, v in loss.items():
            if k in losses:
                losses[k].append(v.item())
            else:
                losses[k] = [v.item()]
        progbar.update(sample.shape[0])

    return {k: statistics.mean(v) for k, v in losses.items()}


class ModelTrainer(object):
    '''Train a machine learning model.

    The :class:`ModelTrainer` takes a :class:`~leitmotiv.models.Model` and
    trains it on some :class:`~leitmotiv.models.Dataset`.  It defines the
    training loop and provides the necessary hooks to monitor how the model is
    being trained.

    Attributes
    ----------
    batch_size : int
        batch size used during training
    epochs : int
        number of epochs
    split : float
        a value, between 0 and 1, indicating how much of the dataset is used
        for validation; if ``None`` then no validation set is created
    verbose : bool
        enable/disable verbose output
    use_gpu : bool
        enable/disable GPU training
    '''
    def __init__(self, batch_size, epochs, split=None, verbose=False,
                 use_gpu=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.split = split
        self.verbose = verbose
        self.use_gpu = use_gpu

    @property
    def split(self):
        return self._validation_split

    @split.setter
    def split(self, val):
        if val is not None and not (val > 0 and val < 1):
            raise ValueError('Validation split must be between 0 and 1.')
        self._validation_split = val

    def train(self, model, dataset):
        '''Train the model on some dataset.

        Parameters
        ----------
        model : :class:`~leitmotiv.models.Model`
            an instance of a :class:`~leitmotiv.models.Model` subclass
        dataset : :class:`~leitmotiv.models.Dataset`
            the dataset that the model is being trained on

        Returns
        -------
        dict
            a dictionary containing the various losses

        Raises
        ------
        TypeError
            if the model isn't a subclass of :class:`~leitmotiv.models.Model`
        ValueError
            if the split value is not between 0 and 1
        '''
        if not issubclass(model.__class__, Model):
            raise TypeError('Can only train a leitmotiv.models.Model.')

        # Set the correct training device.
        if torch.cuda.is_available() and self.use_gpu:
            model.to_gpu()
        else:
            model.to_cpu()

        # Set up the training and (optional) validation sets.
        if self.verbose:  # pragma: no cover
            click.echo('Input dataset contains %d samples' % len(dataset))

        loader_args = {'batch_size': self.batch_size, 'shuffle': False}
        if self.split is None:
            training_data = DataLoader(dataset, **loader_args)
            testing_data = []
        else:
            partitions = _split_dataset(dataset, self.split)
            training_data = DataLoader(partitions[0], **loader_args)
            testing_data = DataLoader(partitions[1], **loader_args)

            if self.verbose:  # pragma: no cover
                click.echo('Using ' + click.style('%d' % len(partitions[0]),
                                                  bold=True)
                           + ' for training; '
                           + click.style('%d' % len(partitions[1]), bold=True)
                           + ' for testing.')

        # Set up storage for the losses.
        losses = {k: [] for k in _train_sample(model, dataset[0])}
        validation = {k: [] for k in _validate_sample(model, dataset[0])}

        progbar = tqdm(total=self.epochs*len(dataset), desc='Sample',
                       disable=not self.verbose)

        for epoch in range(self.epochs):
            # Run over the training samples.
            epoch_loss = _loop_over_samples(_train_sample, model,
                                            training_data, progbar)

            for k, v in epoch_loss.items():
                losses[k].append(v)

            # Run over the validation samples.
            with torch.no_grad():
                epoch_loss = _loop_over_samples(_validate_sample, model,
                                                testing_data, progbar)

                if len(testing_data) > 0:
                    for k, v in epoch_loss.items():
                        validation[k].append(v)

        progbar.close()

        if self.split is None:
            return losses
        else:
            return losses, validation
