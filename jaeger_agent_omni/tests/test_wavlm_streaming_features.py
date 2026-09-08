import unittest
import numpy as np
from training.wavlm_streaming_features import windows, target_centers, WINDOW


class StreamingFeatureTest(unittest.TestCase):
    def test_appended_future_cannot_change_published_windows(self):
        audio = np.arange(640*12,dtype=np.float32)
        original = list(windows(audio[:640*8],8))
        extended = list(windows(audio,12))
        for before,after in zip(original,extended):
            self.assertEqual(before[:3],after[:3])
            np.testing.assert_array_equal(before[3],after[3])
            end = round(before[2]*16000)
            np.testing.assert_array_equal(before[3][-end:],audio[:end])

    def test_partial_block_preserves_last_frame_center(self):
        audio = np.arange(640*5,dtype=np.float32)
        first,count,available,chunk = list(windows(audio,5))[-1]
        self.assertEqual((first,count),(4,1))
        self.assertAlmostEqual(available,.2)
        self.assertAlmostEqual(target_centers(first,available)[0],1.26)
        self.assertEqual(len(chunk),WINDOW)

    def test_all_features_in_block_wait_until_end(self):
        blocks = list(windows(np.ones(640*8,np.float32),8))
        self.assertEqual([b[2] for b in blocks],[.16,.32])
        self.assertEqual([b[1] for b in blocks],[4,4])
