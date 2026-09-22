"""Differentiable two-body integration with bounded, history-dependent corrections."""
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn

MU = 398600.4418
R = 42164.0
V = math.sqrt(MU / R)
ACCEL_LIMIT = 1e-3 * MU / R**2


class Correction(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "mamba":
            from qutils.ml.mamba import Mamba, MambaConfig
            self.backbone = Mamba(MambaConfig(d_model=6, n_layers=1, d_conv=4))
            hidden = 6
        elif kind == "lstm":
            self.backbone = nn.LSTM(6, 30, batch_first=True)
            hidden = 30
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("scale", torch.tensor([R] * 3 + [V] * 3, dtype=torch.float64))
        self.double()

    def forward(self, context):
        features = self.backbone(context / self.scale)
        if self.kind == "lstm":
            features = features[0]
        return ACCEL_LIMIT * torch.tanh(self.head(features[:, -1]))


def rhs(state, correction):
    r, v = state[..., :3], state[..., 3:]
    gravity = -MU * r / torch.linalg.vector_norm(r, dim=-1, keepdim=True)**3
    return torch.cat((v, gravity + correction), dim=-1)


def step(state, correction=0.0, dt=60.0, substeps=1):
    """RK4; correction stays constant across this observation interval."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = rhs(state, correction)
        k2 = rhs(state + h / 2 * k1, correction)
        k3 = rhs(state + h / 2 * k2, correction)
        k4 = rhs(state + h * k3, correction)
        state = state + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return state


def unroll(model, context, count, dt=60.0, substeps=1):
    """No teacher forcing or graph detaches. Shape: (batch, count, six)."""
    predictions = []
    for _ in range(count):
        acceleration = 0.0 if model is None else model(context)
        state = step(context[:, -1], acceleration, dt, substeps)
        predictions.append(state)
        context = torch.cat((context[:, 1:], state[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


def forecast(model, observed, count, cfg):
    """Only the observed prefix is accepted; future reference states cannot leak in."""
    context = torch.as_tensor(observed[-cfg["lookback"]:]).double()[None]
    out = np.full((count, 6), np.nan)
    saturation = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for j in range(count):
            acceleration = 0.0 if model is None else model(context)
            state = step(context[:, -1], acceleration, cfg["dt"], cfg["substeps"])
            if not torch.isfinite(state).all():
                break
            out[j] = state[0].numpy()
            if model is not None:
                saturation.append(float((acceleration.abs() > 0.95*ACCEL_LIMIT).double().mean()))
            context = torch.cat((context[:, 1:], state[:, None]), dim=1)
    return out, float(np.mean(saturation)) if saturation else 0.0


def reference(initial, elapsed):
    def f(t, state):
        r = state[:3]
        return np.r_[state[3:], -MU*r / np.linalg.norm(r)**3]
    solution = solve_ivp(f, (0, float(elapsed[-1])), initial, t_eval=elapsed,
                         method="DOP853", rtol=1e-11, atol=1e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def eligible_starts(tle_index, fit_end, lookback, horizon):
    starts = np.arange(lookback - 1, fit_end - horizon + 1)
    # Prefix counts catch every hand-off, including an index changing back later.
    changes = np.r_[0, np.cumsum(np.diff(tle_index) != 0)]
    return starts[changes[starts + horizon] == changes[starts - lookback + 1]]


def training_batch(observed, starts, lookback, horizon):
    inputs = starts[:, None] + np.arange(1-lookback, 1)
    targets = starts[:, None] + np.arange(1, horizon+1)
    return torch.as_tensor(observed[inputs]).double(), torch.as_tensor(observed[targets]).double()


def trajectory_loss(pred, target, cfg):
    r = ((pred[..., :3] - target[..., :3]) / cfg["position_scale_km"]).square().mean()
    v = ((pred[..., 3:] - target[..., 3:]) / cfg["velocity_scale_km_s"]).square().mean()
    return r + v


def diagnostics(states, initial):
    r, v = states[:, :3], states[:, 3:]
    energy = np.sum(v*v, axis=1)/2 - MU/np.linalg.norm(r, axis=1)
    initial_energy = np.dot(initial[3:], initial[3:])/2 - MU/np.linalg.norm(initial[:3])
    momentum = np.cross(r, v)
    initial_momentum = np.cross(initial[:3], initial[3:])
    return ((energy-initial_energy)/abs(initial_energy)*100,
            np.linalg.norm(momentum-initial_momentum, axis=1)/np.linalg.norm(initial_momentum)*100)


def metrics(pred, truth, elapsed, period):
    finite = np.isfinite(pred).all(axis=1)
    failed = not finite.all()
    error = np.linalg.norm(pred[:, :3]-truth[:, :3], axis=1)
    result = dict(status="diverged" if failed else "ok",
                  last_valid_step=int(np.flatnonzero(finite)[-1]+1) if finite.any() else 0,
                  position_rmse_km=None if failed else float(np.sqrt(np.mean(error**2))),
                  velocity_rmse_km_s=None if failed else float(np.sqrt(np.mean(np.sum(
                      (pred[:, 3:]-truth[:, 3:])**2, axis=1)))),
                  max_position_error_km=None if failed else float(error.max()))
    horizons = {"15min":900, "1h":3600, "6h":21600,
                "0.25T":period/4, "0.5T":period/2, "1T":period, "2T":2*period, "5T":5*period}
    result["horizons"] = {}
    for name, seconds in horizons.items():
        j = int(np.searchsorted(elapsed, seconds))
        available = j < len(pred)
        result["horizons"][name] = dict(
            elapsed_sec=float(elapsed[j]) if available else None,
            position_error_km=float(error[j]) if available and finite[j] else None)
    return result
