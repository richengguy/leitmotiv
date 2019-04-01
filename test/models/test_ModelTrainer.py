import random

import pytest
import torch

from leitmotiv.models import Dataset, Model, ModelTrainer
from leitmotiv.models.trainer import _split_dataset


class TensorLoss(Model):
    def __init__(self, a):
        self.a = torch.Tensor([a])

    def to_gpu(self):
        pass

    def to_cpu(self):
        pass

    def score(self, data):
        return self.a

    def train(self, data):
        return self.score(data)

    def infer(self, data):
        return self.a

    def to_dict(self):
        print('Called to_dict()')
        return {'a': self.a}

    @staticmethod
    def from_dict(model_dict):
        print('Called from_dict()')
        return TensorLoss(**model_dict)


class DictLoss(Model):
    def __init__(self, a):
        self.a = torch.Tensor([a])

    def to_gpu(self):
        pass

    def to_cpu(self):
        pass

    def score(self, data):
        return {'a': self.a}

    def train(self, data):
        return self.score(data)

    def infer(self, data):
        return self.a

    def to_dict(self):
        print('Called to_dict()')
        return {'a': self.a}

    @staticmethod
    def from_dict(model_dict):
        print('Called from_dict()')
        return DictLoss(**model_dict)


class RandomObject(object):
    def train(self, data):
        return {'a': 1.0}

    def infer(self, data):
        return 1.0


def test_random_split(library):
    '''Ensure the random splitting is happening correctly.'''
    dataset = Dataset(library, img_dim=64)

    state = random.getstate()
    random.seed(1)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    random.seed(1)
    partitions = _split_dataset(dataset, 0.4)
    assert partitions[0].indices == indices[0:3]
    assert partitions[1].indices == indices[3:5]

    random.setstate(state)


class TestModelTrainer(object):
    '''Test the model trainer.'''
    def test_float_loss(self, library):
        dataset = Dataset(library, img_dim=64)
        model = TensorLoss(5.0)
        trainer = ModelTrainer(1, 2)
        losses, _ = trainer.train(model, dataset)
        assert losses['loss'][0] == pytest.approx(5.0)

    def test_dict_loss(self, library):
        dataset = Dataset(library, img_dim=64)
        model = DictLoss(5.0)
        trainer = ModelTrainer(1, 2)
        losses, _ = trainer.train(model, dataset)
        assert losses['a'][0] == pytest.approx(5.0)

    def test_exception_raised_with_incorrect_model_type(self, library):
        dataset = Dataset(library, img_dim=64)
        model = RandomObject()
        with pytest.raises(TypeError):
            trainer = ModelTrainer(1, 2)
            trainer.train(model, dataset)

    def test_validation_split_set_correctly(self, library):
        trainer = ModelTrainer(1, 2)
        assert trainer._validation_split is None

        with pytest.raises(ValueError):
            trainer.split = -1

        with pytest.raises(ValueError):
            trainer.split = 2

        trainer.split = 0.5
        assert trainer._validation_split == pytest.approx(0.5)

    def test_train_with_split(self, library):
        dataset = Dataset(library, img_dim=64)
        model = DictLoss(5.0)
        trainer = ModelTrainer(1, 2, split=0.4)
        losses, validation = trainer.train(model, dataset)
        assert losses['a'][0] == pytest.approx(5.0)
        assert validation['a'][0] == pytest.approx(5.0)
