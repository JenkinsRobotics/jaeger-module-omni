# Predictive coding lab

Two small, runnable CPU demos for Jaeger's biologically inspired research lane.
They use the repository's existing **PyTorch and NumPy**, plus Python's standard
library. The offline visual uses ordinary browser JavaScript. No special
predictive-coding library, model download, GPU, or microphone is required.

```sh
.venv/bin/python research/predictive_coding/demo.py
open artifacts/predictive-coding/demo.html
.venv/bin/python -m unittest discover -s research/predictive_coding -p 'test_*.py' -v
```

The first command runs both experiments for three seeds and writes numerical
evidence to `research/candidate_evals/predictive-coding-demo-20260907.json`.
The generated HTML and its inline fragment live under ignored `artifacts/`.
No weights are saved. Use `--output PATH --html-output PATH` for another run.
The local-learning algorithm is in `local_gradients()` and uses no autograd.

Optional developer UI verification uses Playwright and an installed Chrome in
a separate headless process. Neither is a Python/demo training dependency:

```sh
npm install --prefix artifacts/predictive-coding/browser-check --no-save --package-lock=false playwright
node research/predictive_coding/check_view.mjs "$PWD/artifacts/predictive-coding/browser-check/node_modules/playwright/index.mjs"
```

## First measured run · 2026-09-07

All three seeds (7, 19, 41) learn the oscillator with local updates. Held-out
next-state MSE ranges from **4.18e-5 to 7.80e-5**, essentially identical to
matched-update backprop. Backprop achieves slightly lower next-step error with
the same approximate time budget. Neither method solves the changed dynamics:
local-PC free-rollout MSE grows from **0.00415–0.00732** to **0.188–0.246**
when the restoring force changes. These are small synthetic experiments, not
evidence that one learning rule is generally better.

The fixed sensory model's observed-channel reconstruction MSE changes from
**0.00980** for the noisy observations to **0.00174** after neutral-prior
inference. Missing-channel MSE is **0.00966**; a strong incorrect prior makes
it **0.2807** despite decreasing internal energy. This illustrates the dependence
on an appropriate model and prior, not a general denoising benchmark.

Evidence: `predictive-coding-demo-20260907.json`,
`predictive-coding-checks-20260907.json`, and
`predictive-coding-ui-20260907.json` under `research/candidate_evals/`.

## 1. Explaining noisy senses

A supplied three-dimensional latent state predicts twelve sensory channels
through a **fixed linear generative model**. Eight observations are available,
four are missing. Starting from a prior belief, latent activity repeatedly
changes to reduce local sensory errors. Slide the settling step to inspect
predictions, missing-channel reconstruction, and error energy.

Switch to a strong incorrect prior to see why a model's internal error can fall
without its interpretation becoming accurate. Ground truth is displayed for
evaluation only; neither missing-channel truth nor the hidden true state enters
inference. This is an inference demonstration, **not a trained sensory NN**.

For matrix A, observations y, binary availability mask M and latent state z:

```
E(z) = 0.5 ||M(Az - y)||² + 0.5 precision ||z - prior||²
z <- z + step [Aᵀ M(y - Az) - precision (z - prior)]
```

The test compares the iteration with an independently solved linear system,
checks decreasing energy, and changes unavailable observations to extreme values
to prove they cannot influence inference.

## 2. Learning a moving world

A 162-parameter, 2→32→2 tanh network learns the next position/velocity increment
of a nonlinear damped oscillator. Training examples are consecutive synthetic
states. Independent initial conditions generate 64 training, 16 validation and
16 test trajectories of 50 steps. A separate shifted test strengthens the
restoring force by 50%. Each split and initialization is hashed in the report.

The local predictive-coding arm and backprop arm start with identical weights
and receive identical batches, learning rate, and 1,200 updates. Both use SGD
without momentum; changing optimizer would confound this small learning-rule
comparison. A third backprop arm gets approximately the local arm's measured
training time and cycles the same predetermined batch stream. That arm's update
count varies by machine load, so it is not a bitwise reproducibility claim.
Each timed path is warmed without changing weights. Fixed settings are used;
there is no validation selection or test-driven tuning.

The finite-nudging PC rule is:

```
f = tanh(x W1 + b1)
E(h) = 0.5 ||h - f||² + 0.5 beta ||y - (h W2 + b2)||²
h <- h + step [f - h + beta (y - h W2 - b2) W2ᵀ]
```

Weights stay fixed during hidden-activity settling. Afterwards, weights receive
local activity/error-product updates, scaled by beta. At small beta and
converged activity, these approach the ordinary backprop gradients. Tests
independently check that limit and the energy derivative at analytically solved
hidden equilibrium. The PC path runs under `torch.no_grad()`; autograd appears
only in the backprop control and independent mathematical checks.

Evaluation reports both next-step error using observed current state and a
50-step rollout that receives **only the initial state**. The visual shows the
first held-out trajectory for each seed; numerical scores cover all sixteen.
Switch the world to examine distribution shift. Training needs hidden settling;
deployment here uses one feedforward pass, with no target or settling loop.

## What this says about biology

The intended research direction is local learning, internal generative models,
prediction errors, and recurrent correction of beliefs. These demos retain
strong simplifications: continuous activations, exact transposed feedback
weights, synchronized updates, a chosen energy function, and (in demo 2)
observed future-state supervision. They do not implement spiking neurons,
neuromodulation, biological time constants, or establish human-brain equivalence.
Predictive coding is a theory and family of algorithms, not a settled account
of all human neural computation.

Primary references:

- [Whittington & Bogacz (2017), local predictive coding and backprop approximation](https://doi.org/10.1162/NECO_a_00949).
- [Millidge, Tschantz & Buckley (2020), predictive coding on computation graphs](https://arxiv.org/abs/2006.04182).

This implementation is a small finite-nudging educational formulation, not a
reproduction of Millidge's fixed-prediction algorithm or its reported benchmarks.
Predicting future data is the **task**; local prediction-error updates are the
**learning mechanism**. Both demo-2 arms perform the same predictive task.

## Next experiment for Jaeger

Treat biological inspiration and practical benefit as separately testable goals.
Next, replace the oscillator data with causal audio features from the already
partitioned overlap corpus, preserving source/speaker disjointness. Compare
local-PC learning with backprop under equal capacity, updates and compute.
Include a nonpredictive baseline and an identical predictive objective trained
with backprop, so the objective is not mistaken for a learning-rule advantage.

Predictive error by itself is not an overlap label: a door slam, a new speaker,
or a room change can all be surprising. Only held-out overlap F1, false events
per minute and onset delay can qualify a perception candidate. A future candidate
stays an independent audio branch; Gemma and the semantic adapter remain frozen.
No node, production profile, duplex policy, GUI file, or existing gate changes
as a result of these educational demos.
