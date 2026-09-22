"""Local checks for the scientific and isolation boundaries."""
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
import demo
from dynamics import (Correction, R, V, eligible_starts, forecast, metrics,
                      reference, step, training_batch, trajectory_loss, unroll)

torch.set_num_threads(1)
CFG = json.loads((demo.ROOT/"config.json").read_text())


def orbit(count=400):
    initial = np.array([R, 0, 0, 0, V, 0], dtype=float)
    return np.vstack([initial, reference(initial, np.arange(1,count)*60.)])


@pytest.mark.parametrize("kind", ["mamba", "lstm"])
def test_zero_correction_and_temporal_gradients(kind):
    torch.manual_seed(3)
    model = Correction(kind)
    x = orbit(30)
    context = torch.tensor(x[:8])[None]
    physical = unroll(None,context,4)
    learned = unroll(model,context,4)
    assert torch.equal(physical,learned)
    target = physical.detach().clone()
    target[:,:,0] += 0.1
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    loss = trajectory_loss(learned,target,CFG)
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().max() > 0
    optimizer.step()
    optimizer.zero_grad()
    trajectory_loss(unroll(model,context,4),target,CFG).backward()
    temporal = (model.backbone.layers[0].mixer.A_log if kind=="mamba"
                else model.backbone.weight_hh_l0)
    assert torch.isfinite(temporal.grad).all() and temporal.grad.abs().max() > 0
    changed = context.clone()
    changed[:,:-1,0] += 1000
    assert (model(changed)-model(context)).abs().max() > 1e-16


def test_rk4_full_five_orbits_agrees_with_dop853():
    x = orbit(8)
    period = 2*np.pi*R/V
    count = int(np.ceil(5*period/60))
    pred,_ = forecast(None,x,count,CFG)
    exact = reference(x[-1],np.arange(1,count+1)*60.)
    assert np.linalg.norm(pred[:,:3]-exact[:,:3],axis=1).max() < .01


def test_origin_horizons_and_no_future_leakage():
    x = orbit(200)
    origin = 40
    pred,_ = forecast(None,x[:origin+1],80,CFG)
    assert np.allclose(pred[0],step(torch.tensor(x[origin])).numpy())
    altered = x.copy()
    altered[origin+1:] *= 100
    second,_ = forecast(None,altered[:origin+1],80,CFG)
    assert np.array_equal(pred,second)
    starts = np.array([7,20])
    a = training_batch(x[:origin+1],starts,8,4)
    b = training_batch(altered[:origin+1],starts,8,4)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
    score = metrics(pred,x[origin+1:origin+81],np.arange(1,81)*60,3600)
    assert score["horizons"]["1T"]["elapsed_sec"] == 3600
    assert score["horizons"]["2T"]["position_error_km"] is None


def test_starts_respect_context_targets_and_handoffs():
    tle = np.r_[np.zeros(20),np.ones(20),np.zeros(20)]
    starts = eligible_starts(tle,50,8,8)
    assert len(starts)>0
    for s in starts:
        assert s-7>=0 and s+8<=50
        assert np.unique(tle[s-7:s+9]).size==1
    assert eligible_starts(tle,10,8,8).size==0


def test_snapshot_hash_and_external_output_boundaries():
    demo.load_data()
    manifest=demo.read(demo.local("data/manifest.json"))
    for name in demo.FILES:
        assert demo.sha(demo.local("data/"+name))==manifest["hashes"][name]
    with pytest.raises(ValueError):
        demo.local("../outside.json")
    with tempfile.TemporaryDirectory(dir=demo.local(".cache")) as tmp:
        escape=Path(tmp)/"escape"
        escape.symlink_to(demo.ROOT.parent,target_is_directory=True)
        with pytest.raises(ValueError):
            demo.local(escape/"outside.json")


def test_nonfinite_forecast_is_retained_as_failure():
    x=orbit(20)
    pred=x[1:].copy()
    pred[5:]=np.nan
    result=metrics(pred,x[1:],np.arange(1,20)*60,3600)
    assert result["status"]=="diverged"
    assert result["position_rmse_km"] is None
    assert result["last_valid_step"]==5


def test_freeze_uses_three_seed_medians_and_rejects_changed_protocol(monkeypatch):
    cfg = dict(CFG, steps_per_stage=50, horizons=[8,32])
    ident = {"config":cfg, "code":"test-only"}
    with tempfile.TemporaryDirectory(dir=demo.local(".cache")) as tmp:
        monkeypatch.setattr(demo,"ROOT",Path(tmp))
        with pytest.raises(ValueError,match="Missing complete"):
            demo.freeze(cfg,ident)
        for variant in demo.VARIANTS:
            for seed in cfg["seeds"]:
                run_id=f"{variant}-{seed}"
                history=[dict(step=step,position_rmse_km=value,status="ok")
                         for step,value in zip((0,50,100),(3+seed,1+seed,1+seed))]
                demo.write(f"runs/{run_id}/run.json",dict(run_id=run_id,phase="develop",
                    variant=variant,window=0,seed=seed,status="ok",history=history,
                    identity_hash=demo.digest(ident)))
        demo.freeze(cfg,ident)
        protocol=demo.frozen(ident)
        assert set(protocol["selected_steps"].values())=={50}
        assert len(protocol["development_runs"])==12
        with pytest.raises(ValueError,match="differ from frozen"):
            demo.frozen(dict(ident,code="changed"))
