import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class GaussianGenerator(nn.Module):

    def __init__(self, nz, nc):
        super(GaussianGenerator, self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            nn.Linear(nz, nc),
            nn.ReLU(),
            nn.Linear(nc, nc),
            nn.ReLU(),
            nn.Linear(nc, nc),
            nn.ReLU(),
            nn.Linear(nc, 2)
        )

    def forward(self, input):
        return self.main(input)

class GaussianDiscriminator(nn.Module):

    def __init__(self, nc):
        super(GaussianDiscriminator, self).__init__()
        self.nc = nc
        self.main = nn.Sequential(
            nn.Linear(2, nc),
            nn.ReLU(),
            nn.Linear(nc, nc),
            nn.ReLU(),
            nn.Linear(nc, nc),
            nn.ReLU(),
            nn.Linear(nc, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# copied from mh-gan released code https://github.com/obryniarski/metropolis-hastings-gans/blob/master/mhgan/classification.py
class Calibrator(object):
    def fit(self, y_pred, y_true):
        raise NotImplementedError

    def predict(self, y_pred):
        raise NotImplementedError

    @staticmethod
    def validate(y_pred, y_true=None):
        y_pred = np.asarray(y_pred)
        assert y_pred.ndim == 1
        assert y_pred.dtype.kind == 'f'
        assert np.all(0 <= y_pred) and np.all(y_pred <= 1)

        if y_true is not None:
            y_true = np.asarray(y_true)
            assert y_true.shape == y_pred.shape
            assert y_true.dtype.kind == 'b'

        return y_pred, y_true


class Identity(Calibrator):
    def fit(self, y_pred, y_true):
        assert y_true is not None
        Calibrator.validate(y_pred, y_true)

    def predict(self, y_pred):
        Calibrator.validate(y_pred)
        # Could make copy to be consistent with other methods, but prob does
        # not matter.
        return y_pred


class Linear(Calibrator):
    def __init__(self):
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred[:, None], y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict_proba(y_pred[:, None])[:, 1]
        return y_calib


class Isotonic(Calibrator):
    def __init__(self):
        self.clf = IsotonicRegression(y_min=0.0, y_max=1.0,
                                      out_of_bounds='clip')

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred, y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict(y_pred)
        return y_calib
