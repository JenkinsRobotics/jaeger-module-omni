"""Causality and event scoring checks for the independent raw-audio NN trial."""
import unittest

import numpy as np
import torch

from training.parallel_overlap import OverlapNet, intervals, logmel, metrics, split_for


class ParallelOverlapTests(unittest.TestCase):
    def test_features_and_network_never_read_future_audio(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        wave = np.random.default_rng(0).normal(0, .1, 32000).astype(np.float32)
        features, prefix = logmel(wave), logmel(wave[:16000])
        np.testing.assert_allclose(prefix, features[:len(prefix)], atol=1e-5)
        model = OverlapNet(16)
        x = torch.from_numpy(features[None])
        output = model(x)
        torch.testing.assert_close(model(x[:, :25]), output[:, :25], atol=1e-5, rtol=1e-5)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_false_bursts_count_as_events_and_misses_stay_visible(self):
        self.assertEqual(intervals([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        truth = np.array([0, 1, 1, 0, 0, 1, 1, 0.])
        predicted = np.array([0, 0, 1, 0, 1, 0, 0, 0.])
        score = metrics([(predicted, truth)], .5)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 1, 3))
        self.assertEqual(score["false_alarm_events"], 1)
        self.assertEqual(score["detected_events"], 1)
        self.assertEqual(score["event_recall"], .5)
        self.assertEqual(score["detection_delay_p95_ms"], 40)
        self.assertIsNone(metrics([(np.zeros(8), truth)], .5)["detection_delay_p95_ms"])

    def test_cross_fold_speakers_are_discarded(self):
        from training.parallel_overlap import fold

        representatives = {}
        for i in range(100):
            representatives.setdefault(fold(i), i)
        self.assertEqual(split_for([representatives[0]]), "test")
        self.assertEqual(split_for([representatives[1]]), "validation")
        self.assertEqual(split_for([representatives[2], representatives[3]]), "train")
        self.assertIsNone(split_for([representatives[0], representatives[2]]))
