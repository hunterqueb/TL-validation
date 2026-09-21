"""Guards HANDOFF F1: the ASC script optimized a model whose gradients were None, so the
target model never changed. These tests fail if that ever comes back."""
import numpy as np
import torch

from tlv.models import return_model, transfer
from tlv.train import fit
from qutils.ml.regression import create_datasets

DEVICE = "cpu"


def _tiny_datasets(n=40):
    rng = np.random.default_rng(0)
    seq = np.cumsum(rng.normal(0, 0.01, (n, 6)), axis=0) + 1.0
    return create_datasets(seq, 1, n // 2, DEVICE)


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_full_transfer_actually_trains_the_target_model():
    torch.manual_seed(0)
    src = return_model("mamba", DEVICE)
    tgt = transfer(src, return_model("mamba", DEVICE), "full")

    before, src_before = _snapshot(tgt), _snapshot(src)
    fit(tgt, _tiny_datasets(), epochs=1)

    assert any(not torch.allclose(before[k], v) for k, v in tgt.state_dict().items()), \
        "target model did not change -- F1 has come back"
    assert all(torch.allclose(src_before[k], v) for k, v in src.state_dict().items()), \
        "source model changed during target training"


def test_feature_transfer_keeps_in_proj_and_A_log_frozen():
    torch.manual_seed(0)
    src = return_model("mamba", DEVICE)
    tgt = transfer(src, return_model("mamba", DEVICE), "feature")
    before = _snapshot(tgt)

    fit(tgt, _tiny_datasets(), epochs=1)

    assert torch.allclose(before["layers.0.mixer.in_proj.weight"],
                          tgt.state_dict()["layers.0.mixer.in_proj.weight"])
    assert torch.allclose(before["layers.0.mixer.A_log"], tgt.state_dict()["layers.0.mixer.A_log"])
    assert not torch.allclose(before["layers.0.mixer.dt_proj.weight"],
                              tgt.state_dict()["layers.0.mixer.dt_proj.weight"])


def test_lstm_full_transfer_copies_every_layer():
    """transferLSTM leaves `fc` random; transfer() must use transferModelAll (HANDOFF 7)."""
    torch.manual_seed(0)
    src = return_model("lstm", DEVICE)
    tgt = transfer(src, return_model("lstm", DEVICE), "full")
    for k, v in src.state_dict().items():
        assert torch.allclose(v, tgt.state_dict()[k]), k


def test_rollout_starts_at_the_last_observed_state():
    """A model that returns its input must reproduce the forecast origin forever.

    The origin is state `train_size` -- with lookback 1, create_datasets trains on the
    transitions up to x_{train_size-1} -> x_{train_size}, so that state is observed and
    the first unseen one is train_size + 1 (HANDOFF REV-02/REV-04). Seeding one step
    earlier, as the code did before, re-predicts a training target and drops the last
    observation.
    """
    from tlv.train import rollout

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    seq = np.arange(60, dtype=float).reshape(10, 6)
    out = rollout(Identity(), seq, train_size=4, device=DEVICE)
    assert np.all(np.isnan(out[:5])), "nothing is forecast at or before the origin"
    assert np.allclose(out[5:], seq[4]), "the rollout is seeded from the last observed state"


def test_horizons_are_measured_from_the_forecast_origin():
    """1T must be one full period after the last observation, not after the window start."""
    from tlv.metrics import position_error_at

    period, dt, train_size = 86165.2, 60.0, 1436
    t_sec = np.arange(8620) * dt
    truth = np.zeros((len(t_sec), 6))
    pred = np.full_like(truth, np.nan)
    pred[train_size + 1:] = 1.0                      # unit error everywhere it forecasts

    err = position_error_at(pred, truth, t_sec, period, periods=(1, 5), origin=train_size)
    assert np.isfinite(err[1]) and np.isfinite(err[5]), "both horizons fit in the window"

    # the sample scored at 1T sits one period after the origin, within one dt
    j = int(np.searchsorted(t_sec, t_sec[train_size] + period))
    assert abs((t_sec[j] - t_sec[train_size]) - period) <= dt
    assert t_sec[j] - t_sec[train_size] > 0.99 * period, "not the 120 s rollout of REV-02"

    # a window that stops short reports NaN rather than a closer, flattering sample
    short = position_error_at(pred[:train_size + 100], truth[:train_size + 100],
                              t_sec[:train_size + 100], period, periods=(1,), origin=train_size)
    assert np.isnan(short[1])
