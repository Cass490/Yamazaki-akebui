"""Bounded component regressions: python -m unittest -v test_even_word."""
import unittest
from unittest.mock import patch

import numpy as np

import gold_local as G


class RunnerWiringRegression(unittest.TestCase):
    """Integration checks that do not execute numerical synthesis."""

    def test_paired_words_request_even_t_without_fallback(self):
        target = np.eye(2, dtype=complex)
        with patch.object(G, "cyclosynth_synth", return_value="") as native:
            with patch.object(G, "pygridsynth_synth", side_effect=AssertionError("fallback used")):
                synth = G.make_synth("cyclosynth")
                self.assertEqual(G.even_word(target, synth, 1e-8), "")
        native.assert_called_once()
        self.assertIs(native.call_args.args[0], target)
        self.assertEqual(native.call_args.args[1], 1e-8)
        self.assertEqual(native.call_args.kwargs, {"even_t": True})


class SynthesisRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            G.require_cyclosynth()
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if not hasattr(G.require_cyclosynth().Synthesizer, "even_t"):
            raise RuntimeError("Rebuild cyclosynth with even_t support before running these regressions")

    def assert_approximates(self, word, target, eps):
        matrix = G.mat(word)
        phase = np.angle(np.trace(target.conj().T @ matrix))
        self.assertLess(np.linalg.norm(np.exp(-1j * phase) * matrix - target, 2), eps)

    def test_joint_u3_angle_convention(self):
        target = G.su2_exp(np.array([0.21, -0.37, 0.49]))
        eps = 1e-3
        self.assert_approximates(G.cyclosynth_synth(target, eps), target, eps)

    def test_python_even_t_api(self):
        cls = G.require_cyclosynth().Synthesizer
        self.assertFalse(cls(1e-3).even_t)
        self.assertTrue(cls(1e-3, even_t=True).even_t)
        with self.assertRaisesRegex(ValueError, "even_t requires Clifford"):
            cls(1e-3, even_t=True, sqrt_t=True)

    def test_generic_even_t_does_not_use_pygridsynth(self):
        target = G.su2_exp(np.array([0.21, -0.37, 0.49]))
        synth = G.make_synth("cyclosynth")
        with patch.object(G, "pygridsynth_synth", side_effect=AssertionError("fallback used")):
            word = G.even_word(target, synth, 1e-3)
        self.assertEqual(G.tcount(word) % 2, 0)
        self.assert_approximates(word, target, 1e-3)

    def test_odd_exact_rotations_and_pair(self):
        synth = G.make_synth("cyclosynth")
        eps = 4e-9
        for exact_word in ("T", "HTH", "SHTHs"):
            with self.subTest(axis_word=exact_word):
                target = G.mat(exact_word)
                target /= np.sqrt(np.linalg.det(target))
                self.assertEqual(G.tcount(synth(target, eps)) % 2, 1)
                with patch.object(G, "pygridsynth_synth", side_effect=AssertionError("fallback used")):
                    word = G.even_word(target, synth, eps)
                self.assertEqual(G.tcount(word) % 2, 0)
                self.assert_approximates(word, target, eps)
                if exact_word == "T":
                    pair = G.compile_pair_best(word)
                    a = G.mat(pair["A"])
                    expected = np.zeros((4, 4), dtype=complex)
                    expected[:2, :2] = a
                    expected[2:, 2:] = a @ G.mat(word)
                    self.assertLess(np.max(np.abs(G.pair_matrix(pair) - expected)), 1e-11)


if __name__ == "__main__":
    unittest.main()
