"""Independent mathematical checks for the educational local-learning rules."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import (bp_gradients, initial_weights, local_gradients, pairs,
                  settle_senses, trajectories)


class PredictiveCodingTests(unittest.TestCase):
    def test_sensory_settling_matches_closed_form(self):
        g = torch.Generator().manual_seed(8)
        a = torch.randn(8, 3, generator=g, dtype=torch.float64)
        y = torch.randn(8, generator=g, dtype=torch.float64)
        mask = torch.tensor([1., 1., 0., 1., 0., 1., 1., 1.], dtype=torch.float64)
        prior = torch.tensor([.3, -.1, .2], dtype=torch.float64)
        precision = .4
        trace = settle_senses(a, y, mask, prior, precision, 200)
        exact = torch.linalg.solve(a.T @ (mask[:, None]*a)+precision*torch.eye(3),
                                   a.T @ (mask*y)+precision*prior)
        np.testing.assert_allclose(trace[-1]["latent"], exact.numpy(), atol=1e-7)
        self.assertTrue(all(b["energy"] <= a["energy"]+1e-12 for a, b in zip(trace, trace[1:])))
        y[mask == 0] = 1e6
        other = settle_senses(a, y, mask, prior, precision, 200)
        self.assertEqual(trace, other)  # Missing truth cannot leak through.

    def test_local_gradients_approach_backprop_without_autograd(self):
        w = initial_weights(5)
        x, y = pairs(trajectories(77, 2))
        x, y = x[:16], y[:16]
        expected = bp_gradients([v.clone().requires_grad_() for v in w], x, y)
        errors = []
        snapshots = [v.clone() for v in w]
        with patch("torch.autograd.grad", side_effect=AssertionError("local path used autograd")):
            for beta in (.1, .001, .00001):
                actual = local_gradients(w, x, y, beta, 40)
                errors.append(float(torch.cat([(a-b).flatten() for a, b in zip(actual, expected)]).norm()))
                self.assertTrue(all(not a.requires_grad for a in actual))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 1e-6)
        for before, after in zip(snapshots, w):
            self.assertTrue(torch.equal(before, after))  # Inference freezes weights.

    def test_local_update_is_energy_derivative_at_fixed_activity(self):
        w = initial_weights(13)
        x, y = pairs(trajectories(80, 1))
        beta = .1
        # Solve hidden equilibrium analytically; differentiate E at fixed h.
        f = torch.tanh(x @ w[0]+w[1])
        h = torch.linalg.solve(torch.eye(32)+beta*w[2]@w[2].T,
                               (f+beta*(y-w[3])@w[2].T).T).T.detach()
        v = [a.clone().requires_grad_() for a in w]
        energy = ((h-torch.tanh(x@v[0]+v[1])).square().sum()
                  + beta*(y-h@v[2]-v[3]).square().sum())/(2*beta*len(x))
        expected = torch.autograd.grad(energy, v)
        actual = local_gradients(w, x, y, beta, 100)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
