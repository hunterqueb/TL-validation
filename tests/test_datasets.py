"""HANDOFF section 9.4: non-dim round-trip, synthetic shapes, period split."""
import numpy as np
import pytest

from tlv import datasets as ds

SYNTH = dict(minutes=450, n_sys=4000)   # 450min-10000 / 900min-10000 are empty dirs


def test_nondim_roundtrip():
    """Requires the qutils constants fix (F4, R1-01): the mu typo made this fail by 0.23 %."""
    x = np.random.default_rng(0).uniform(-8000, 8000, (50, 6))
    assert np.allclose(ds.redim(ds.nondim(x)), x, atol=1e-12)


def test_nondim_copies():
    """dim2NonDim6 mutates its input; nondim() must not (guardrail 9.3.1)."""
    x = np.ones((3, 6))
    ds.nondim(x)
    assert np.all(x == 1.0)


def test_split_periods():
    period = 6040.0
    for p in (1, 2, 5):
        train_size = ds.split_periods(None, period, p)
        assert abs(train_size * 60.0 / period - p) < 0.01


def test_nearest_synthetic_finds_itself():
    bank = np.array([[7000.0, 0.001, 1.7], [7100.0, 0.02, 0.9], [6900.0, 0.0005, 1.71]])
    idx, d = ds.nearest_synthetic(bank[2], bank, k=2)
    assert idx[0] == 2 and d[0] == 0.0


@pytest.mark.skipif(not ds.pathlib.Path("data/synthetic").exists(),
                    reason="data/synthetic symlink not set up (P0-05)")
def test_load_synthetic_shape_and_time_base():
    states, t_sec, _ = ds.load_synthetic(**SYNTH)
    assert states.ndim == 3 and states.shape[2] == 6
    assert states.shape[1] == SYNTH["minutes"]
    assert t_sec[0] == 60.0          # GMAT sample j is after the (j+1)-th step (F5)
