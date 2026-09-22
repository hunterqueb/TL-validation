"""Guards the ASC 2025 bug: the script optimized a model whose gradients were None, so the
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
        "target model did not change -- the ASC 2025 bug has come back"
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
    """transferLSTM leaves `fc` random; transfer() must use transferModelAll."""
    torch.manual_seed(0)
    src = return_model("lstm", DEVICE)
    tgt = transfer(src, return_model("lstm", DEVICE), "full")
    for k, v in src.state_dict().items():
        assert torch.allclose(v, tgt.state_dict()[k]), k

