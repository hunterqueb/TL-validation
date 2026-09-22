#!/usr/bin/env python
"""Standalone, local-only rollout experiment. See README.md for the walkthrough."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def local(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output escapes standalone directory: {path}")
    return path


os.environ["MPLCONFIGDIR"] = str(local(".cache/matplotlib"))
os.environ["TMPDIR"] = str(local(".cache/tmp"))
local(".cache/tmp").mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from dynamics import (Correction, diagnostics, eligible_starts, forecast, metrics,
                      reference, training_batch, trajectory_loss, unroll)

VARIANTS = ("mamba_one_step", "mamba_rollout", "lstm_one_step", "lstm_rollout")
FILES = ("statesArrayReal.npz", "meta.json")
NOTE = "reference = SGP4 from space-track TLEs"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def outside_state():
    repo = ROOT.parents[1]
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    hashes = {n: sha(repo/n) for n in names if n and (repo/n).is_file()
              and not (repo/n).resolve().is_relative_to(ROOT)}
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                     cwd=repo).decode().splitlines()
    status = [s for s in status if "experiments/branch_b_rollout/" not in s]
    return dict(tracked_hashes=hashes, outside_git_status=status)


def validate_data(data, meta):
    x, t = data["statesArrayReal"], data["t_sec"]
    if x.shape != (8, 8620, 6) or t.shape != x.shape[:2]:
        raise ValueError("Expected eight 8620-sample six-state windows")
    for key in ("tle_index", "handoff_jump_km"):
        if data[key].shape != t.shape:
            raise ValueError(f"Bad shape: {key}")
    if data["epoch_jd"].shape != (8,):
        raise ValueError("Bad epoch_jd shape")
    if not all(np.isfinite(data[k]).all() for k in data):
        raise ValueError("Nonfinite input data")
    if meta["dt"] != 60 or not np.allclose(np.diff(t), 60, rtol=0, atol=1e-7):
        raise ValueError("Expected uniform 60 s samples")
    if not ("GCRS" in meta["frame_chain"] and meta["norad"] == 28089):
        raise ValueError("Unexpected target/frame")
    radius, speed = np.linalg.norm(x[..., :3], axis=-1), np.linalg.norm(x[..., 3:], axis=-1)
    if not ((radius > 40000).all() and (radius < 44000).all()
            and (speed > 2).all() and (speed < 4).all()):
        raise ValueError("GEO range/unit sanity check failed (expected km and km/s)")
    absolute = data["epoch_jd"][:, None]*86400 + t
    if not np.all(absolute[1:, 0] > absolute[:-1, -1]):
        raise ValueError("Windows overlap or epochs are not ordered")
    origin = round(meta["period_sec"]/60)
    if not np.all(t[:, -1]-t[:, origin] >= 5*meta["period_sec"]):
        raise ValueError("Missing full five-period forecast horizon")


def load_data():
    manifest = read(local("data/manifest.json"))
    for name in FILES:
        if sha(local("data/"+name)) != manifest["hashes"][name]:
            raise ValueError("Snapshot hash mismatch: " + name)
    with np.load(local("data/statesArrayReal.npz"), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = read(local("data/meta.json"))
    validate_data(data, meta)
    return data, meta


def prepare(source):
    source = (ROOT / source).resolve()
    if local("data/manifest.json").exists():
        load_data()
        print("Existing snapshot verified; not overwritten.", flush=True)
        return
    if not local("isolation.json").exists():
        write("isolation.json", outside_state())
    hashes = {name: sha(source/name) for name in FILES}
    local("data").mkdir(exist_ok=True)
    for name in FILES:
        shutil.copyfile(source/name, local("data/"+name))
    write("data/manifest.json", dict(source=str(source), hashes=hashes))
    data, meta = load_data()
    print(f"Snapshot verified: {data['statesArrayReal'].shape}; {meta['frame_chain']}", flush=True)


def identity(cfg):
    import qutils.ml.mamba as mamba
    return dict(config=cfg, code={name: sha(ROOT/name) for name in ("demo.py", "dynamics.py")},
                qutils_mamba_sha256=sha(inspect.getfile(mamba)),
                packages=environment()["packages"],
                data=read(local("data/manifest.json"))["hashes"])


def environment():
    return dict(python=sys.version, device="cpu", dtype="float64", threads=torch.get_num_threads(),
                packages={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "torch", "matplotlib")})


def runs():
    return [read(p) for p in sorted(local("runs").glob("*/run.json"))]


def new_run(phase, variant, window, seed, cfg, ident):
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = local("runs/"+run_id)
    path.mkdir(parents=True)
    row = dict(run_id=run_id, phase=phase, variant=variant, window=window, seed=seed,
               identity=ident, identity_hash=digest(ident), environment=environment(),
               status="running", config=cfg)
    write(path/"run.json", row)
    return path, row


def fit_model(variant, seed, observed, tle_index, cfg, steps, validation=None, log_path=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Correction(variant.split("_")[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    horizons = [1] if variant.endswith("one_step") else cfg["horizons"]
    starts = {k: eligible_starts(tle_index, len(observed)-1, cfg["lookback"], k) for k in horizons}
    if any(len(v) == 0 for v in starts.values()):
        raise ValueError(f"No hand-off-free training starts: { {k:len(v) for k,v in starts.items()} }")
    history, training_states, network_calls = [], 0, 0
    start_time = time.perf_counter()

    def validate(step):
        if validation is None:
            return
        pred, sat = forecast(model, observed, len(validation), cfg)
        score = metrics(pred, validation, np.arange(1, len(pred)+1)*cfg["dt"], 1e20)
        entry = dict(step=step, position_rmse_km=score["position_rmse_km"],
                     velocity_rmse_km_s=score["velocity_rmse_km_s"], status=score["status"],
                     saturation_fraction=sat)
        history.append(entry)
        if log_path:
            write(log_path/"history.json", history)
        print(f"{variant} seed={seed} step={step}/{steps} validation_rmse={entry['position_rmse_km']}", flush=True)

    validate(0)
    for i in range(steps):
        k = horizons[min(i//cfg["steps_per_stage"], len(horizons)-1)]
        chosen = rng.choice(starts[k], cfg["batch_size"], replace=True)
        context, target = training_batch(observed, chosen, cfg["lookback"], k)
        model.train()
        optimizer.zero_grad()
        pred = unroll(model, context, k, cfg["dt"], cfg["substeps"])
        loss = trajectory_loss(pred, target, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at optimizer step {i+1}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"], error_if_nonfinite=True)
        optimizer.step()
        training_states += k*cfg["batch_size"]
        network_calls += k
        if (i+1) % cfg["validate_every"] == 0:
            validate(i+1)
            if validation is None:
                print(f"{variant} seed={seed} step={i+1}/{steps} loss={loss.item():.5g}", flush=True)
            if log_path:
                write(log_path/"progress.json", dict(step=i+1, loss=float(loss), grad_norm=float(grad)))
    return model, history, dict(optimizer_steps=steps, training_seconds=time.perf_counter()-start_time,
                               observed_states=len(observed), sampled_unrolled_states=training_states,
                               training_network_calls=network_calls,
                               training_force_evaluations=4*cfg["substeps"]*training_states,
                               validation_network_calls=len(history)*(len(validation) if validation is not None else 0),
                               eligible_starts={str(k):len(v) for k,v in starts.items()})


def existing(phase, variant, window, seed, ident, protocol_hash=None):
    matches = [r for r in runs() if r["phase"] == phase and r["variant"] == variant
               and r["window"] == window and r["seed"] == seed and r["status"] == "ok"
               and r["identity_hash"] == digest(ident) and r.get("protocol_hash") == protocol_hash]
    return matches[-1] if matches else None


def evaluate_cell(phase, variant, window, seed, cfg, ident, data, meta, protocol=None):
    protocol_hash = None if protocol is None else protocol["protocol_hash"]
    old = existing(phase, variant, window, seed, ident, protocol_hash)
    if old:
        print("Verified completed cell:", old["run_id"], phase, variant, window, seed, flush=True)
        return old
    path, row = new_run(phase, variant, window, seed, cfg, ident)
    row["protocol_hash"] = protocol_hash
    origin = round(meta["period_sec"]/cfg["dt"])
    x, t, tle = data["statesArrayReal"][window], data["t_sec"][window], data["tle_index"][window]
    try:
        if phase == "develop":
            end = int(0.75*origin)
            model, history, cost = fit_model(variant, seed, x[:end+1], tle[:end+1], cfg,
                cfg["steps_per_stage"]*len(cfg["horizons"]), x[end+1:origin+1], path)
            torch.save(model.state_dict(), local(path/"checkpoint.pt"))
            row.update(status="ok", history=history, cost=cost, origin=end,
                       validation_end=origin, tle_sha256=meta["tle_sha256"])
        else:
            observed, elapsed = x[:origin+1], t[origin+1:]-t[origin]
            model, cost = None, None
            if variant in VARIANTS:
                steps = protocol["selected_steps"][variant]
                model, _, cost = fit_model(variant, seed, observed, tle[:origin+1], cfg, steps,
                                           log_path=path)
                torch.save(model.state_dict(), local(path/"checkpoint.pt"))
            if variant == "persistence":
                pred, saturation = np.broadcast_to(x[origin], (len(elapsed),6)).copy(), 0.0
            else:
                pred, saturation = forecast(model, observed, len(elapsed), cfg)
            score = metrics(pred, x[origin+1:], elapsed, meta["period_sec"])
            pred_energy, pred_h = diagnostics(pred, x[origin])
            ref_energy, ref_h = diagnostics(x[origin+1:], x[origin])
            handoff = data["handoff_jump_km"][window]
            jumps = handoff[handoff > 0]
            row.update(score, cost=cost, saturation_fraction=saturation, origin=origin,
                       origin_sec=float(t[origin]), tle_sha256=meta["tle_sha256"],
                       handoff_count=int(len(jumps)), handoff_median_km=float(np.median(jumps)),
                       handoff_p95_km=float(np.percentile(jumps,95)),
                       zero_correction=bool(cost is not None and cost["optimizer_steps"] == 0))
            row["forecast_network_calls"] = len(elapsed) if model is not None else 0
            row["forecast_force_evaluations"] = 0 if variant=="persistence" else 4*cfg["substeps"]*len(elapsed)
            if variant == "two_body":
                exact = reference(x[origin], elapsed)
                discrepancy = float(np.linalg.norm(pred[:,:3]-exact[:,:3],axis=1).max())
                row["rk4_dop853_max_km"] = discrepancy
                if discrepancy >= .01:
                    raise ValueError(f"RK4 error {discrepancy} km: set substeps=2 and rerun development")
            np.savez_compressed(local(path/"forecast.npz"), predicted=pred, truth=x[origin+1:],
                elapsed=elapsed, initial=x[origin], handoff=handoff[origin+1:],
                prediction_energy_drift_pct=pred_energy, reference_energy_drift_pct=ref_energy,
                prediction_momentum_drift_pct=pred_h, reference_momentum_drift_pct=ref_h)
            if score["status"] == "ok":
                row["max_energy_drift_pct"] = float(np.max(np.abs(pred_energy)))
                row["max_momentum_drift_pct"] = float(np.max(pred_h))
            print(f"{phase} {variant} window={window} seed={seed}: RMSE={score['position_rmse_km']}", flush=True)
    except Exception as exc:
        row.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        print(row["traceback"], flush=True)
    write(path/"run.json", row)
    return row


def freeze(cfg, ident):
    selected, ids = {}, []
    for variant in VARIANTS:
        cells = [existing("develop", variant, cfg["development_window"], s, ident) for s in cfg["seeds"]]
        if any(c is None for c in cells):
            raise ValueError("Missing complete development cells: " + variant)
        candidates = []
        steps = range(0, cfg["steps_per_stage"]*len(cfg["horizons"])+1, cfg["validate_every"])
        for step_number in steps:
            values = []
            for cell in cells:
                entry = next(e for e in cell["history"] if e["step"] == step_number)
                values.append(entry["position_rmse_km"] if entry["status"] == "ok" else float("inf"))
            candidates.append((float(np.median(values)), step_number))
        selected[variant] = min(candidates)[1]
        ids.extend(c["run_id"] for c in cells)
    obj = dict(identity=ident, selected_steps=selected, development_runs=ids,
               selection="minimum median validation trajectory RMSE across three seeds; earliest tie")
    obj["protocol_hash"] = digest(obj)
    if local("frozen.json").exists() and read(local("frozen.json")) != obj:
        raise ValueError("An immutable frozen protocol already exists; do not overwrite")
    write("frozen.json", obj)
    print(json.dumps(obj["selected_steps"], indent=2), flush=True)


def frozen(ident):
    obj = read(local("frozen.json"))
    supplied = obj["protocol_hash"]
    if digest({k:v for k,v in obj.items() if k != "protocol_hash"}) != supplied:
        raise ValueError("Frozen protocol hash mismatch")
    if obj["identity"] != ident:
        raise ValueError("Code/config/dependency/data differ from frozen protocol")
    return obj


def plot_window(rows, window, period, phase):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)
    colors = dict(zip(("persistence", "two_body")+VARIANTS, plt.get_cmap("tab10").colors))
    reference_drawn = False
    for row in rows:
        if row["window"] != window or row["phase"] != phase or row["status"] != "ok":
            continue
        with np.load(local("runs/"+row["run_id"]+"/forecast.npz")) as f:
            truth, pred, elapsed = f["truth"], f["predicted"], f["elapsed"]/period
            delta = pred[:,:3]-truth[:,:3]
            radial = truth[:,:3]/np.linalg.norm(truth[:,:3],axis=1)[:,None]
            normal = np.cross(truth[:,:3],truth[:,3:])
            normal /= np.linalg.norm(normal,axis=1)[:,None]
            along = np.cross(normal,radial)
            name, color = row["variant"], colors[row["variant"]]
            label = name + ("" if row["seed"] is None else f" s{row['seed']}")
            axes[0,0].semilogy(elapsed, np.maximum(np.linalg.norm(delta,axis=1),1e-8), color=color, alpha=.7, label=label)
            axes[0,1].plot(elapsed,np.linalg.norm(pred[:,:3],axis=1),color=color,alpha=.7)
            axes[1,0].plot(elapsed,np.sum(delta*radial,axis=1),color=color,alpha=.7)
            axes[1,1].plot(elapsed,np.sum(delta*along,axis=1),color=color,alpha=.7)
            axes[2,0].plot(elapsed,np.sum(delta*normal,axis=1),color=color,alpha=.7)
            axes[2,1].plot(elapsed,f["prediction_energy_drift_pct"],color=color,alpha=.7)
            axes[3,0].plot(elapsed,f["prediction_momentum_drift_pct"],color=color,alpha=.7)
            axes[3,1].plot(elapsed,np.linalg.norm(pred[:,3:]-truth[:,3:],axis=1),color=color,alpha=.7)
            if not reference_drawn:
                axes[0,1].plot(elapsed,np.linalg.norm(truth[:,:3],axis=1),"k--",label="reference")
                axes[2,1].plot(elapsed,f["reference_energy_drift_pct"],"k--",label="reference")
                axes[3,0].plot(elapsed,f["reference_momentum_drift_pct"],"k--",label="reference")
                for a in axes.flat:
                    for at in elapsed[f["handoff"] > 0]:
                        a.axvline(at,color="gray",alpha=.12,lw=.7)
                reference_drawn = True
    for a, label in zip(axes.flat, ("Position error (km)","Radius (km)","Radial error (km)",
                                    "Along-track error (km)","Cross-track error (km)","Energy drift (%)",
                                    "Angular momentum drift (%)", "Velocity error (km/s)")):
        a.set_ylabel(label)
        a.set_xlabel("Periods since forecast origin")
        a.grid(alpha=.15)
    axes[0,0].legend(fontsize=6,ncol=2)
    fig.suptitle(f"{phase} window {window} — {NOTE}\nGray lines: TLE hand-offs")
    fig.tight_layout()
    out = local("figures/"+rows[0]["protocol_hash"][:12])
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(local(out/f"{phase}-window-{window}.png"),dpi=130)
    plt.close(fig)
    return out


def plot_development(protocol):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for variant, ax in zip(VARIANTS, axes.flat):
        for run_id in protocol["development_runs"]:
            row = read(local("runs/"+run_id+"/run.json"))
            if row["variant"] == variant:
                ax.plot([e["step"] for e in row["history"]],
                        [e["position_rmse_km"] for e in row["history"]], label=f"seed {row['seed']}")
        ax.axvline(protocol["selected_steps"][variant],color="k",ls="--",label="selected step")
        ax.set(title=variant,xlabel="Optimizer steps",ylabel="Validation position RMSE (km)")
        ax.legend(fontsize=7)
    fig.suptitle("Reserved quarter-orbit validation — "+NOTE,fontsize=10)
    fig.tight_layout()
    out = local("figures/"+protocol["protocol_hash"][:12])
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(local(out/"validation.png"),dpi=130)
    plt.close(fig)
    data, meta = load_data()
    x,t = data["statesArrayReal"][0],data["t_sec"][0]
    origin=round(meta["period_sec"]/60)
    fig,ax=plt.subplots(figsize=(10,3))
    ax.plot((t-t[origin])/meta["period_sec"],x[:,0],lw=.8)
    ax.axvspan(-1,0,color="green",alpha=.15,label="Observed: 1,437 states")
    ax.axvline(0,color="k",ls="--",label="Forecast origin")
    ax.set(xlabel="Periods relative to forecast origin",ylabel="Reference x (km)",
           title="One orbit observed, five forecast — "+NOTE)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(local(out/"observation-budget.png"),dpi=130)
    plt.close(fig)


def report(cfg, ident):
    protocol = frozen(ident)
    plot_development(protocol)
    rows = [r for r in runs() if r.get("protocol_hash") == protocol["protocol_hash"]
            and r["phase"] in ("walkthrough", "evaluate")]
    # Successful retries supersede failed attempts for aggregation, but all attempts stay listed.
    cells = {}
    for row in rows:
        key = (row["phase"],row["window"],row["variant"],row["seed"])
        if key not in cells or row["status"] == "ok":
            cells[key] = row
    chosen = list(cells.values())
    _, meta = load_data()
    lines = ["# Standalone rollout results", "", NOTE + ". Hand-offs measure consistency, not accuracy.", "",
             "Protocol: `"+protocol["protocol_hash"]+"`. Three development seeds on the first orbit of window 0; "
             "later-window models each observe only their own first orbit. CPU float64. "
             "The forecast spans five orbits with no observations.", "",
             "Selected optimizer steps: `"+json.dumps(protocol["selected_steps"],sort_keys=True)+"`.", "",
             "Zero selected steps means the learned correction was not selected over physics. "
             "Budgets are not compute-matched; per-run costs are saved.", "",
             f"[Observation budget](figures/{protocol['protocol_hash'][:12]}/observation-budget.png) · "
             f"[Validation curves](figures/{protocol['protocol_hash'][:12]}/validation.png)", "",
             "| Phase | Window | Method | Successful / expected | Position RMSE median ± IQR (km) | RMSE / physics |",
             "|---|---:|---|---:|---:|---:|"]
    aggregates, figure_links = [], []
    for phase, windows in (("walkthrough",[0]),("evaluate",cfg["evaluation_windows"])):
        for window in windows:
            base = next((r["position_rmse_km"] for r in chosen if r["phase"]==phase and
                         r["window"]==window and r["variant"]=="two_body" and r["status"]=="ok"),None)
            for variant in ("persistence","two_body")+VARIANTS:
                group = [r for r in chosen if (r["phase"],r["window"],r["variant"])==(phase,window,variant)]
                values = [r["position_rmse_km"] for r in group if r["status"]=="ok"]
                expected = 3 if variant in VARIANTS else 1
                complete = len(values)==expected
                med = float(np.median(values)) if complete else None
                iqr = float(np.diff(np.percentile(values,[25,75]))[0]) if complete else None
                score = f"{med:.4f} ± {iqr:.4f}" if complete else "incomplete; failures/missing retained"
                ratio = f"{med/base:.4f}" if complete and base else "—"
                lines.append(f"| {phase} | {window} | {variant} | {len(values)}/{expected} | {score} | {ratio} |")
                aggregates.append(dict(phase=phase,window=window,variant=variant,median=med,iqr=iqr,
                                       expected=expected,successes=len(values),ratio_physics=med/base if complete and base else None))
            if any(r["phase"]==phase and r["window"]==window for r in chosen):
                out = plot_window(chosen,window,meta["period_sec"],phase)
                figure_links.append(f"- [Trajectory diagnostics, {phase} window {window}]({out.relative_to(ROOT)}/{phase}-window-{window}.png)")
    lines.extend(["", "## Trajectory figures", ""] + figure_links)
    lines.extend(["", "## Paired rollout / one-step RMSE ratios", "",
                  "Ratios below one favor rollout training. Each pair uses the same window and seed.", ""])
    paired = []
    for row in chosen:
        if row["variant"].endswith("_rollout") and row["status"]=="ok":
            control = cells.get((row["phase"],row["window"],row["variant"].replace("_rollout","_one_step"),row["seed"]))
            if control and control["status"]=="ok":
                ratio = row["position_rmse_km"]/control["position_rmse_km"]
                paired.append(dict(phase=row["phase"],window=row["window"],seed=row["seed"],variant=row["variant"],ratio=ratio))
                lines.append(f"- {row['phase']} window {row['window']}, {row['variant']}, seed {row['seed']}: {ratio:.4f}")
    lines.extend(["", "## Across later windows", "",
                  "These summaries use the four window-level medians, not pooled seed rows.", ""])
    for variant in VARIANTS:
        a = [r for r in aggregates if r["phase"]=="evaluate" and r["variant"]==variant]
        if len(a)==4 and all(r["median"] is not None for r in a):
            values = [r["median"] for r in a]
            wins = sum(r["ratio_physics"] < 1 for r in a if r["ratio_physics"] is not None)
            lines.append(f"- {variant}: median window RMSE {np.median(values):.4f} km; "
                         f"range {min(values):.4f}–{max(values):.4f}; lower than physics on {wins}/4 windows.")
        else:
            lines.append(f"- {variant}: incomplete later-window evaluation; no aggregate claim.")
    lines.extend(["", "## Provenance and limitations", "",
                  "One object, SGP4-derived reference, additional window-0 development data. "
                  "Correction is constant within each RK4 observation step and bounded per component. "
                  "Energy/angular momentum drift are diagnostics, not targets for perfect conservation. "
                  "All metric and hand-off details, actual endpoint times, predictions, costs, and failures "
                  "are in the run directories below. No original-demo result rows were mixed in.", "",
                  "Development runs: " + ", ".join(protocol["development_runs"]), "", "All evaluation attempts:", ""])
    lines += [f"- `{r['run_id']}`: {r['phase']} {r['variant']} w{r['window']} s{r['seed']} — {r['status']}" for r in rows]
    local("REPORT.md").write_text("\n".join(lines)+"\n")
    write("runs/summary-"+protocol["protocol_hash"][:12]+".json", dict(aggregates=aggregates,paired=paired,run_ids=[r["run_id"] for r in rows]))
    print("Wrote local REPORT.md and figures",flush=True)


def audit():
    before = read(local("isolation.json"))
    after = outside_state()
    if before != after:
        raise ValueError("Outside tracked files or git status changed; inspect without reverting user work")
    load_data()
    manifest = read(local("data/manifest.json"))
    for name in FILES:
        if sha(Path(manifest["source"])/name) != manifest["hashes"][name]:
            raise ValueError("Original source changed: "+name)
    for path in ROOT.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
            raise ValueError("Symlink escapes experiment: "+str(path))
    print("PASS: outside tracked files/status unchanged; original and copied data hashes match; no escaping symlinks.",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare").add_argument("--source",required=True)
    for name in ("baselines","develop","walkthrough","evaluate"):
        p = sub.add_parser(name)
        if name == "evaluate":
            p.add_argument("--windows",type=int,nargs="+",default=[4,5,6,7])
        else:
            p.add_argument("--window",type=int,default=0)
        p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
        p.add_argument("--variants",choices=VARIANTS,nargs="+",default=list(VARIANTS))
    for name in ("freeze","report","audit"):
        sub.add_parser(name)
    args = parser.parse_args()
    cfg = read(local("config.json"))
    torch.set_num_threads(cfg["threads"])
    if args.command == "prepare":
        prepare(args.source)
        return
    if args.command == "audit":
        audit()
        return
    data, meta = load_data()
    ident = identity(cfg)
    if args.command == "freeze":
        freeze(cfg,ident)
        return
    if args.command == "report":
        report(cfg,ident)
        return
    protocol = frozen(ident) if args.command in ("walkthrough","evaluate") else None
    windows = args.windows if args.command == "evaluate" else [args.window]
    if args.command == "evaluate" and not set(windows) <= set(cfg["evaluation_windows"]):
        raise ValueError("Evaluation windows must follow the frozen protocol")
    if args.command in ("develop","walkthrough") and windows != [cfg["development_window"]]:
        raise ValueError("Development/walkthrough must use window 0")
    if not set(args.seeds) <= set(cfg["seeds"]):
        raise ValueError("Seeds must follow the configured protocol")
    if args.command == "develop" and not existing("baselines","two_body",0,None,ident):
        raise ValueError("Run baselines with the current code/config before development")
    failed = 0
    for window in windows:
        if args.command != "develop":
            for variant in ("persistence","two_body"):
                row = evaluate_cell(args.command,variant,window,None,cfg,ident,data,meta,protocol)
                failed += row["status"] != "ok"
        if args.command != "baselines":
            for variant in args.variants:
                for seed in args.seeds:
                    row = evaluate_cell(args.command,variant,window,seed,cfg,ident,data,meta,protocol)
                    failed += row["status"] != "ok"
    if failed:
        raise SystemExit(f"{failed} failed cells; artifacts retained")


if __name__ == "__main__":
    main()
