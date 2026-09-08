"""CPU-only predictive-coding experiments; no model downloads or runtime imports.

Run from the repository root:
  .venv/bin/python research/predictive_coding/demo.py
Local learning uses explicit tensor updates under no_grad. Autograd is used
only by the named backpropagation control and the independent correctness test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@torch.no_grad()
def settle_senses(matrix, observed, mask, prior, prior_precision, steps=40):
    """Minimize E = .5 ||mask*(Az-y)||² + .5 precision*||z-prior||².

    Each sensory error drives its incident latent units through A.T. Fixed
    symmetric feedback and an analytic stable step are simplifying assumptions.
    Missing sensor values have zero influence on inference.
    """
    hessian = matrix.T @ (mask[:, None] * matrix)
    hessian += prior_precision * torch.eye(matrix.shape[1], dtype=matrix.dtype)
    step = 0.9 / torch.linalg.eigvalsh(hessian).max()
    z = prior.clone()
    trace = []
    for i in range(steps + 1):
        prediction = matrix @ z
        error = mask * (observed - prediction)
        energy = 0.5 * (error.square().sum() + prior_precision * (z-prior).square().sum())
        trace.append({"step": i, "energy": float(energy),
                      "prediction": prediction.tolist(), "latent": z.tolist(),
                      "error": error.tolist()})
        z += step * (matrix.T @ error - prior_precision * (z-prior))
    return trace


def sensory_demo(seed):
    rng = np.random.default_rng(seed)
    matrix = np.linalg.qr(rng.normal(size=(12, 3)))[0]
    latent = np.array([0.8, -0.6, 0.5])
    clean = matrix @ latent
    observed = clean + rng.normal(0, 0.12, 12)
    mask = np.ones(12)
    mask[[2, 5, 8, 11]] = 0
    traces = {}
    for name, prior, precision in [("neutral", np.zeros(3), 0.04),
                                    ("strong_wrong", -latent, 2.0)]:
        traces[name] = settle_senses(torch.tensor(matrix), torch.tensor(observed),
                                    torch.tensor(mask), torch.tensor(prior), precision)
    prediction = np.array(traces["neutral"][-1]["prediction"])
    return {"seed": seed, "kind": "inference only; supplied linear world model",
            "clean": clean.tolist(), "observed": [float(y) if m else None for y, m in zip(observed, mask)],
            "mask": mask.tolist(), "traces": traces,
            "metrics": {"observed_noisy_mse": float(np.mean((observed[mask == 1]-clean[mask == 1])**2)),
                        "observed_inferred_mse": float(np.mean((prediction[mask == 1]-clean[mask == 1])**2)),
                        "missing_inferred_mse": float(np.mean((prediction[mask == 0]-clean[mask == 0])**2))}}


def world_step(state, force=1.0):
    """Dimensionless nonlinear oscillator, semi-implicit discrete dynamics."""
    x, v = state[..., 0], state[..., 1]
    v_next = 0.99 * v - 0.15 * force * np.sin(2 * x)
    return np.stack((x + 0.15 * v_next, v_next), axis=-1)


def trajectories(seed, n, force=1.0, length=50):
    rng = np.random.default_rng(seed)
    state = rng.uniform([-0.9, -0.5], [0.9, 0.5], (n, 2))
    rows = [state]
    for _ in range(length):
        state = world_step(state, force)
        rows.append(state)
    return np.stack(rows, axis=1).astype(np.float64)


def pairs(rows):
    x = torch.from_numpy(rows[:, :-1].reshape(-1, 2).copy())
    y = torch.from_numpy((rows[:, 1:]-rows[:, :-1]).reshape(-1, 2).copy())
    return x, y


def initial_weights(seed, width=32):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, width, generator=g, dtype=torch.float64)*0.3,
            torch.zeros(width, dtype=torch.float64),
            torch.randn(width, 2, generator=g, dtype=torch.float64)*0.05,
            torch.zeros(2, dtype=torch.float64)]


def forward(weights, x):
    w1, b1, w2, b2 = weights
    return torch.tanh(x @ w1+b1) @ w2+b2


@torch.no_grad()
def local_gradients(weights, x, y, beta=0.1, settling_steps=20):
    """Finite-nudging predictive coding for a one-hidden-layer predictor.

    E(h) = .5||h-f(x)||² + .5 beta ||y-(h W2+b2)||².
    Settle hidden activity with weights fixed, then use only neighboring
    activity/error products. Divide weight gradients by beta to compare with
    the BP objective .5 * mean_batch(sum_output(error²)). No graph/backward.
    As beta -> 0 and settling converges, these approach BP gradients.
    """
    w1, b1, w2, b2 = weights
    prediction = torch.tanh(x @ w1+b1)
    h = prediction.clone()
    # Frobenius norm is a cheap upper bound on spectral curvature.
    step = 0.9 / (1 + beta * w2.square().sum())
    for _ in range(settling_steps):
        output_error = y - (h @ w2+b2)
        h += step * (prediction-h + beta * output_error @ w2.T)
    output_error = h @ w2+b2-y
    input_error = ((prediction-h)/beta) * (1-prediction.square())
    n = x.shape[0]
    return [x.T @ input_error/n, input_error.mean(0),
            h.T @ output_error/n, output_error.mean(0)]


def bp_gradients(weights, x, y):
    loss = 0.5 * (forward(weights, x)-y).square().sum(1).mean()
    return torch.autograd.grad(loss, weights)


def fit(initial, x, y, batches, mode, lr, beta, settling_steps, seconds=None):
    weights = [w.clone().requires_grad_(mode == "backprop") for w in initial]
    # Warm each learning path without modifying weights, outside its timer.
    if mode == "local_pc":
        local_gradients(weights, x[:64], y[:64], beta, settling_steps)
    else:
        bp_gradients(weights, x[:64], y[:64])
    history = []
    started = time.perf_counter()
    updates = 0
    while (updates < len(batches) if seconds is None else time.perf_counter()-started < seconds):
        ids = batches[updates % len(batches)]
        xb, yb = x[ids], y[ids]
        if mode == "local_pc":
            gradients = local_gradients(weights, xb, yb, beta, settling_steps)
        else:
            gradients = bp_gradients(weights, xb, yb)
        with torch.no_grad():
            for w, grad in zip(weights, gradients):
                w -= lr * grad
        updates += 1
        if updates == 1 or updates % 100 == 0:
            with torch.no_grad():
                history.append({"updates": updates, "batch_mse": float((forward(weights, xb)-yb).square().mean())})
    elapsed = time.perf_counter()-started
    return [w.detach() for w in weights], {"updates": updates, "train_seconds": elapsed, "history": history}


@torch.no_grad()
def score(weights, rows):
    x, y = pairs(rows)
    predicted = forward(weights, x)
    state = torch.from_numpy(rows[:, 0].copy())
    roll = [state.numpy().copy()]
    for _ in range(rows.shape[1]-1):
        state = state + forward(weights, state)
        roll.append(state.numpy().copy())
    roll = np.stack(roll, axis=1)
    mse = float(np.mean((roll[:, 1:]-rows[:, 1:])**2))
    started = time.perf_counter()
    for _ in range(300):
        forward(weights, x[:1])
    return {"one_step_mse": float((predicted-y).square().mean()),
            "free_rollout_mse_50_steps": mse,
            "forward_mean_us": (time.perf_counter()-started)*1e6/300,
            "example_rollout": roll[0].tolist()}


def learning_demo(seed, args):
    rows = {"train": trajectories(seed+1000, 64), "validation": trajectories(seed+2000, 16),
            "test": trajectories(seed+3000, 16), "shifted_test": trajectories(seed+4000, 16, force=1.5)}
    x, y = pairs(rows["train"])
    initial = initial_weights(seed)
    g = torch.Generator().manual_seed(seed+5000)
    batches = torch.randint(len(x), (args.steps, 64), generator=g)
    arms = {}
    for mode in ("local_pc", "backprop", "backprop_matched_time"):
        budget = arms["local_pc"]["train_seconds"] if mode.endswith("matched_time") else None
        w, training = fit(initial, x, y, batches, "local_pc" if mode == "local_pc" else "backprop",
                          args.lr, args.beta, args.settling_steps, budget)
        arms[mode] = {**training, **{s: score(w, rows[s]) for s in ("validation", "test", "shifted_test")}}
    persistence = float(np.mean((rows["test"][:, 1:]-rows["test"][:, :-1])**2))
    return {"seed": seed, "parameters": sum(w.numel() for w in initial),
            "split_sha256": {s: digest(r) for s, r in rows.items()},
            "initial_sha256": digest(*(w.numpy() for w in initial)),
            "batch_sha256": digest(batches.numpy()), "arms": arms,
            "untrained": score(initial, rows["test"]), "persistence_one_step_mse": persistence,
            "test_example": rows["test"][0].tolist(), "shifted_example": rows["shifted_test"][0].tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"research/candidate_evals/predictive-coding-demo-20260907.json")
    p.add_argument("--html-output", type=Path, default=ROOT/"artifacts/predictive-coding/demo.html")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 41])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--settling-steps", type=int, default=20)
    args = p.parse_args()
    if min(args.steps, args.lr, args.beta, args.settling_steps) <= 0:
        p.error("steps, learning rate, beta and settling steps must be positive")
    torch.set_num_threads(1)
    report = {"schema": 1, "purpose": "educational experiments, not production qualification",
              "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
              "python": sys.version, "platform": platform.platform(),
              "command": [sys.executable, *sys.argv],
              "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                for name in ("demo.py", "render.py", "view.html", "test_demo.py", "check_view.mjs")},
              "config": {k: v for k, v in vars(args).items() if k not in ("output", "html_output")},
              "sensory": sensory_demo(args.seeds[0]), "learning": [],
              "limitations": ["Synthetic worlds, no speech or biological measurements.",
                               "Fixed symmetric feedback, continuous activations, supervised future states.",
                               "Local PC is finite-nudging, not the fixed-prediction algorithm of Millidge et al.",
                               "Equal-update and approximate equal-time controls are reported separately.",
                               "Matched-time update counts and timing vary by host load; no checkpoint selection.",
                               "World prediction at deployment is one feedforward pass; settling is training-only."]}
    for seed in args.seeds:
        run = learning_demo(seed, args)
        report["learning"].append(run)
        print(json.dumps({"seed": seed, "test_mse": {a: r["test"]["one_step_mse"] for a, r in run["arms"].items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    from render import render
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    render(report, args.html_output)
    print(f"Evidence: {args.output}\nInteractive demo: {args.html_output}")


if __name__ == "__main__":
    main()
