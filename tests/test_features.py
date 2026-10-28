"""特徵處理：舊格式相容、正規化、增強、手部濾波。"""
import types
import unittest

import numpy as np

from src import features as F


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_all_stored_widths_become_126(self):
        for w in F.VALID_DIMS:
            a = self.rng.normal(size=(30, w)).astype(np.float32)
            self.assertEqual(F.to_model_input(a).shape, (30, F.FEAT_DIM))
            self.assertEqual(F.clean_hand_tracks(a).shape, (30, F.FEAT_DIM))
            self.assertEqual(F.augment_sequence(a, self.rng).shape, (30, F.FEAT_DIM))

    def test_too_narrow_is_rejected(self):
        with self.assertRaises(ValueError):
            F.hands_only(np.zeros((30, 100)))

    def test_legacy_face_columns_are_ignored(self):
        base = self.rng.normal(size=(30, 126)).astype(np.float32)
        wide = np.concatenate([base, self.rng.normal(size=(30, 44)).astype(np.float32) * 50], axis=1)
        np.testing.assert_allclose(F.to_model_input(base), F.to_model_input(wide))
        self.assertEqual(F.hand_motion(base), F.hand_motion(wide))

    def test_augment_keeps_absent_hand_zero_and_input_untouched(self):
        seq = np.zeros((30, 126), np.float32)
        seq[5:25, :63] = self.rng.normal(size=(20, 63)) * 0.1     # 只有左手
        before = seq.copy()
        for i in range(50):
            out = F.augment_sequence(seq, np.random.default_rng(i))
            self.assertTrue(np.isfinite(out).all())
            self.assertEqual(np.abs(out[:, 63:]).max(), 0.0)
        np.testing.assert_array_equal(seq, before)

    def test_clean_tracks_fills_gaps_and_removes_ghosts(self):
        s = np.zeros((30, 126), np.float32)
        s[:, :63] = 1.0
        s[10:12, :63] = 0          # 掉 2 幀 → 補回
        s[20, 63:] = 1.0           # 右手只閃 1 幀 → 幽靈
        c = F.clean_hand_tracks(s)
        self.assertGreater(np.abs(c[10:12, :63]).sum(), 0)
        self.assertEqual(np.abs(c[:, 63:]).sum(), 0)
        self.assertEqual(F.longest_hand_gap(s), 2)

    def test_extract_two_hands(self):
        def hand(off):
            return types.SimpleNamespace(landmark=[types.SimpleNamespace(x=off + i * .01, y=.5, z=0)
                                                   for i in range(21)])
        def cls(label):
            return types.SimpleNamespace(classification=[types.SimpleNamespace(label=label)])
        res = types.SimpleNamespace(multi_hand_landmarks=[hand(.7)], multi_handedness=[cls("Right")])
        v = F.extract_two_hands(res)
        self.assertEqual(len(v), 126)
        self.assertEqual(sum(abs(x) for x in v[:63]), 0)          # 左手槽空著
        none = types.SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
        self.assertEqual(F.extract_two_hands(none), [0.0] * 126)


if __name__ == "__main__":
    unittest.main()
