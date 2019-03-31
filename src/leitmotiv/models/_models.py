import abc
import pathlib

import torch


class Model(abc.ABC):
    '''High-level definition of a leitmotiv model.

    A model is something that is able to describe some data given a finite set
    of parameters.  For example, a linear regressor will describe a dataset via
    a hyperplane in some N-dimensional space.  The :class:`Model` describes the
    interface that all models must adhere to, such as being able to
    export/import their internal state and being trained on a sample from a
    data set.
    '''
    @abc.abstractmethod
    def to_gpu(self):
        '''Move the model to the GPU.

        It is the responsibility of the subclass to decide how this is done.
        This method may act as a no-op if no GPU is available or if the data is
        already on the GPU.
        '''

    @abc.abstractmethod
    def to_cpu(self):
        '''Move the model to the CPU.

        It is the responsibility of the subclass to decide how this is done.
        This method may act as a no-op if the data is already on the CPU.
        '''

    @abc.abstractmethod
    def score(self, data):
        '''Calculate the loss function on some given data.

        The score (or loss) gives an idea of how well the model describes, or
        fits the data that it has been trained on.  It can't indicate how good
        the model is on its own, as that usually requires data not in the
        training set.  The score is also used internally during training of the
        model.

        Parameters
        ----------
        data : :class:`torch.Tensor` or :class:`numpy.ndarray`
            a multi-dimensional array containing the training data

        Returns
        -------
        :class:`torch.Tensor` or dict
            the current training cost(s), all stored as PyTorch scalars
        '''

    @abc.abstractmethod
    def train(self, data):
        '''Train the model on some data set.

        The model will specify what exactly the data should be.  In general, it
        is expected to be a :class:`torch.Tensor` object since that's the
        math library being used internally.  A call to this method will cause
        the model to run one training iteration.  It is not idempotent, since
        training requires updating the model's internal state.

        Parameters
        ----------
        data : :class:`torch.Tensor` or :class:`numpy.ndarray`
            a multi-dimensional array containing the training data

        Returns
        -------
        :class:`torch.Tensor` or dict
            the current training cost(s), all stored as PyTorch scalars
        '''

    @abc.abstractmethod
    def infer(self, data):
        '''Apply the trained model onto some data.

        The model is expected to know how to apply its own interference on some
        sample data.

        Parameters
        ----------
        data : :class:`torch.Tensor` or :class:`numpy.ndarray`
            a multi-dimensional array containing the data to be processed by
            the model
        '''

    @abc.abstractmethod
    def to_dict(self):
        '''Convert the model into a dictionary representation.

        This must be implemented by the subclass to describe how the model is
        converted into a Python dictionary.

        Returns
        -------
        dict
            the model's internal state
        '''

    @abc.abstractstaticmethod
    def from_dict(model_dict):
        '''Generate a model from its internal state dictionary.

        Parameters
        ----------
        model_dict : dict
            dictionary representation of the model's internal state

        Returns
        -------
        :class:`Model` instance
            an instance of the particular model
        '''

    def save(self, path):
        '''Save the model to a file on disk.

        Parameters
        ----------
        path : pathlib.Path
            path to where the model will be saved
        '''
        path = pathlib.Path(path)
        model_dict = self.to_dict()
        torch.save(model_dict, path)

    @classmethod
    def load(cls, path):
        '''Load the model from the given file.

        Parameters
        ----------
        path : pathlib.Path
            path to where the model is saved

        Returns
        -------
        :class:`Model` instance
            an instance of the particular model
        '''
        path = pathlib.Path(path)
        model_dict = torch.load(path, map_location='cpu')
        return cls.from_dict(model_dict)
