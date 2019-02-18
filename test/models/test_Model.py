import pytest
import torch

from leitmotiv.models import Model


class SimpleModel(Model):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def train(self, data):
        return self.a + self.b

    def infer(self, data):
        return self.a*self.b

    def to_dict(self):
        print('Called to_dict()')
        return {
            'a': self.a,
            'b': self.b
        }

    @staticmethod
    def from_dict(model_dict):
        print('Called from_dict()')
        return SimpleModel(**model_dict)


class TestModel(object):
    '''Test the model object.'''
    def test_train(self):
        model = SimpleModel(1, 2)
        assert model.train(0) == pytest.approx(3)

    def test_infer(self):
        model = SimpleModel(1, 2)
        assert model.infer(0) == pytest.approx(2)

    def test_to_dict(self):
        model = SimpleModel(1, 2)
        rep = model.to_dict()
        assert rep['a'] == 1
        assert rep['b'] == 2

    def test_from_dict(self):
        rep = {
            'a': 1,
            'b': 2
        }
        model = SimpleModel.from_dict(rep)
        assert model.a == 1
        assert model.b == 2

    def test_save(self, tmp_path):
        model = SimpleModel(1, 2)
        model.save(tmp_path / 'saved.pth')

        data = torch.load(tmp_path / 'saved.pth')
        assert isinstance(data, dict)
        assert data['a'] == 1
        assert data['b'] == 2

    def test_load(self, tmp_path):
        torch.save({'a': 1, 'b': 2}, tmp_path / 'saved.pth')
        model = SimpleModel.load(tmp_path / 'saved.pth')
        assert model.a == 1
        assert model.b == 2
