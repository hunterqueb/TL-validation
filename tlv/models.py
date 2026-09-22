"""Model construction and transfer, as in the ASC 2025 script."""
import torch

from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import LSTM, transferMamba, transferModelAll


def return_model(kind, device, d_model=6, n_layers=1, d_conv=4, lstm_hidden=30):
    """Build Mamba or LSTM. d_conv=16 for the DDDAS-style low-data runs."""
    kind = kind.lower()
    if kind == "mamba":
        return Mamba(MambaConfig(d_model=d_model, n_layers=n_layers, d_conv=d_conv)).to(device).double()
    if kind == "lstm":
        return LSTM(d_model, lstm_hidden, d_model, n_layers, 0).double().to(device)
    raise ValueError(f"unknown model {kind!r}")


def transfer(source, target, mode):
    """'full': every parameter trainable. 'feature': in_proj and A_log frozen.

    LSTM always uses transferModelAll: transferLSTM leaves `fc` random.
    """
    if mode.startswith("full"):
        return transferModelAll(source, target)
    if mode == "feature":
        if not hasattr(target, "layers"):
            return transferModelAll(source, target)
        return transferMamba(source, target, [True, True, True])
    raise ValueError(f"unknown tl mode {mode!r}")


def param_count(model):
    return sum(p.numel() for p in model.parameters())
