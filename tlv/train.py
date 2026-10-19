"""Training and teacher-forced prediction."""
import torch

from qutils.ml.regression import create_datasets, trainModel  # noqa: F401  (re-exported)
from qutils.plot import plotStatePredictions

STATES = ['x', 'y', 'z', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
UNITS = ["km", "km", "km", "km/s", "km/s", "km/s"]


def fit(model, datasets, epochs, lr=1e-3, criterion=None, verbose=False):
    """Train with Adam + Huber (ASC hyper-parameters). Return seconds spent training.

    SmoothL1Loss() is Huber with delta = beta = 1; HuberLoss itself needs torch >= 1.9.
    """
    criterion = torch.nn.SmoothL1Loss() if criterion is None else criterion
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    return trainModel(model, epochs, list(datasets), criterion, optimizer,
                      printOutAcc=verbose, printOutToc=True)


def predict_teacher_forced(model, t, seq, train_in, test_in, train_size, test_size,
                           DU=None, TU=None, **plot_kw):
    """Teacher-forced (T, 6) prediction: each step sees the true previous state.

    Dimensional (km, km/s) when DU/TU are given.
    """
    # qutils returns a tuple if `states` is passed together with DU/TU.
    kw = dict(timeLabel="Periods")
    if DU is None and TU is None:
        kw.update(states=STATES, units=UNITS)
    kw.update(plot_kw)
    return plotStatePredictions(model, t, seq, train_in, test_in, train_size, test_size,
                                DU=DU, TU=TU, **kw)

