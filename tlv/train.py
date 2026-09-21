"""Training, teacher-forced prediction, and the closed-loop rollout (HANDOFF section 3.5, F3)."""
import numpy as np
import torch

from qutils.ml.regression import create_datasets, trainModel  # noqa: F401  (re-exported)
from qutils.plot import plotStatePredictions

STATES = ['x', 'y', 'z', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
UNITS = ["km", "km", "km", "km/s", "km/s", "km/s"]


def fit(model, datasets, epochs, lr=1e-3, criterion=None, verbose=False):
    """Train with Adam + Huber (ASC hyper-parameters). Return seconds spent training."""
    criterion = torch.nn.HuberLoss() if criterion is None else criterion
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    return trainModel(model, epochs, list(datasets), criterion, optimizer,
                      printOutAcc=verbose, printOutToc=True)


def predict_teacher_forced(model, t, seq, train_in, test_in, train_size, test_size,
                           DU=None, TU=None, **plot_kw):
    """(T, 6) prediction from qutils.plot.plotStatePredictions.

    Teacher-forced: every step is fed the *true* previous state (F3). Reported only
    alongside rollout(). With DU/TU the returned array is dimensional (km, km/s).
    """
    # qutils quirk: passing `states` replaces nonDim2Dim with a no-op that returns a tuple
    # when DU/TU are also given. With states=None it uses ['x'..'zdot'] + km/km/s anyway.
    kw = dict(timeLabel="Periods")
    if DU is None and TU is None:
        kw.update(states=STATES, units=UNITS)
    kw.update(plot_kw)
    return plotStatePredictions(model, t, seq, train_in, test_in, train_size, test_size,
                                DU=DU, TU=TU, **kw)


def rollout(model, seq, train_size, device, seq_length=1):
    """Closed-loop prediction from the forecast origin: the model is fed its own output.

    The forecast origin is index `train_size`, the last *observed* state: with lookback 1,
    create_datasets trains on the transitions x_0->x_1 ... x_{train_size-1}->x_{train_size},
    so state train_size is a training target and the first unseen state is train_size + 1
    (HANDOFF REV-02/REV-04). Seeding one step earlier re-predicts a training target and
    throws the last observation away.

    Return (T, 6) with NaN for j <= train_size. This is the honest propagation metric (F3).
    """
    seq = np.asarray(seq, dtype=float)
    out = np.full_like(seq, np.nan)
    model.eval()
    origin = train_size
    x = torch.tensor(seq[origin + 1 - seq_length:origin + 1][None, :, :]).double().to(device)
    with torch.no_grad():
        for j in range(origin + 1, len(seq)):
            y = model(x)[:, -1, :]
            out[j] = y.cpu().numpy()[0]
            x = torch.cat([x[:, 1:, :], y[:, None, :]], dim=1)
    return out
