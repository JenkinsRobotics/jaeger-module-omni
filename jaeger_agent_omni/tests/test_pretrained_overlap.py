import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from training.pretrained_overlap import Head, extract, buffered_detection_delays


class PretrainedOverlapTests(unittest.TestCase):
    def test_head_cannot_read_future_frames(self):
        torch.manual_seed(7)
        model=Head().eval()
        x=torch.randn(1,24,768)
        with torch.no_grad():
            expected=model(x)[:,:12]
            x[:,12:]=torch.randn_like(x[:,12:])*100
            actual=model(x)[:,:12]
        torch.testing.assert_close(expected,actual)

    def test_feature_interpolation_keeps_constant_signal(self):
        class FixedEncoder:
            def __call__(self,x,output_hidden_states):
                class Output: pass
                result=Output()
                result.hidden_states=[torch.ones(1,49,768)*i for i in range(13)]
                return result
        result=extract(FixedEncoder(),np.zeros(16000,np.float32),25,'cpu')
        self.assertEqual(result.shape,(25,768))
        np.testing.assert_array_equal(result,np.full((25,768),8,dtype=np.float16))

    def test_detection_is_charged_at_feature_availability(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'window.npz'
            np.savez(path,available_at=np.array([.16,.16,.16,.16]))
            probability=np.array([.9,.9,0,0])
            labels=np.array([1,1,0,0])
            rows=[({'archive':str(path)},np.zeros((4,768)),labels)]
            result=buffered_detection_delays([(probability,labels)],rows,.5)
            self.assertAlmostEqual(result['p95_ms'],160)
            self.assertFalse(result['includes_compute'])
