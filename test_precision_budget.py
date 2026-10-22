"""Precision regressions using exact words and tiny matrices; no lattice search."""
import unittest
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np

import gold_local as G


class PrecisionBudgetRegression(unittest.TestCase):
    @staticmethod
    def grader_helpers():
        namespace = {"__name__": "grader"}
        source = Path(G.__file__).with_name("challenge.py").read_text()
        exec(source.split("def submission_circ")[0], namespace)
        return namespace

    def test_budget_limits_even_coherent_product_errors(self):
        budget = 8e-9
        for n in (2, 3, 4, 5):
            count = (2**n - 1) * 2**(n - 1)
            eps = G.word_epsilon(n, 4e-9, budget)
            # All residual rotations point the same way: no random cancellation.
            residual = G.Ry(2 * eps)
            product = np.linalg.matrix_power(residual, count)
            self.assertLessEqual(np.linalg.norm(product - np.eye(2), 2), budget / 2 + 1e-14)
        self.assertAlmostEqual(G.word_epsilon(5, 4e-9, budget), 8e-9 / 992)

    def test_per_word_cap_and_legacy_settings(self):
        self.assertEqual(G.word_epsilon(5, 4e-9), 4e-9)
        self.assertEqual(G.word_epsilon(5, 1e-12, 8e-9), 1e-12)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                G.word_epsilon(5, invalid)
            with self.assertRaises(ValueError):
                G.word_epsilon(5, 4e-9, invalid)

    def test_word_check_detects_error_lost_by_trace_overlap(self):
        target = G.Ry(4e-8)
        old_check = abs(abs(np.trace(target.conj().T)) / 2 - 1)
        self.assertLess(old_check, 1e-12)
        self.assertGreater(G.word_error("", target), 1e-8)
        with patch.object(G, "pygridsynth_synth", return_value=""):
            synth = G.make_synth("pygridsynth")
            with self.assertRaisesRegex(RuntimeError, "phase-aligned word error"):
                synth(target, 1e-9)

    def test_word_check_preserves_global_phase_freedom(self):
        target = np.exp(0.713j) * G.mat("H")
        with patch.object(G, "pygridsynth_synth", return_value="H"):
            self.assertEqual(G.make_synth("pygridsynth")(target, 1e-10), "H")

    def test_mux_bound_covers_measured_block_error(self):
        blocks = np.array([np.eye(2), G.Ry(4e-10)], dtype=complex)
        result = G.synth_mux(blocks, lambda target, eps: "", 1e-9)
        gates, phase = G.mux_gates(result, 1, [0])
        actual = np.exp(1j * phase) * G.simulate(gates, 2)
        target = np.zeros((4, 4), complex)
        for x in range(2):
            target[2*x:2*x+2, 2*x:2*x+2] = (
                np.exp(-1j * result["ctrl_phase"][x]) * blocks[x]
            )
        self.assertLessEqual(np.linalg.norm(actual - target, 2), result["error_bound"] + 1e-13)
        self.assertGreater(result["error_bound"], 1e-10)

    def test_bad_factorization_is_rejected_before_synthesis(self):
        blocks = np.array([np.eye(2), G.Ry(0.2)], dtype=complex)
        with patch.object(G, "factorize", return_value=(np.eye(2), [np.eye(2)])):
            with self.assertRaisesRegex(RuntimeError, "factorization matrix error"):
                G.synth_mux(blocks, lambda target, eps: self.fail("synthesis started"), 1e-9)

    def test_budget_reaches_all_words_in_a_full_small_circuit(self):
        scope = self.grader_helpers()
        calls = []

        def exact_identity(target, eps):
            self.assertLess(np.linalg.norm(target - np.eye(2)), 1e-12)
            calls.append(eps)
            return ""

        target = np.eye(4, dtype=complex)
        gates, phase = G.synthesize(target, scope["cs_decomposition"], exact_identity,
                                    4e-9, optimize=False, error_budget=8e-9)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(eps == G.word_epsilon(2, 4e-9, 8e-9) for eps in calls))
        tape = scope["compile_circuit"](lambda u: G.queue_ops(gates, phase, scope["qp"]), target)
        self.assertLess(np.max(abs(scope["tape_matrix"](tape, 2) - target)), 1e-12)

    def test_full_circuit_budget_rejects_bad_custom_backend(self):
        scope = self.grader_helpers()
        with self.assertRaisesRegex(RuntimeError, "exceeds requested circuit budget"):
            G.synthesize(np.eye(4), scope["cs_decomposition"], lambda target, eps: "H",
                         4e-9, optimize=False, error_budget=8e-9)

    def test_dry_run_does_not_construct_backend(self):
        output = io.StringIO()
        with patch("sys.argv", ["gold_local.py", "--sizes", "5", "--error-budget", "8e-9", "--dry-run"]):
            with patch("sys.stdout", output), patch.object(G, "make_synth") as backend:
                G.main()
        backend.assert_not_called()
        self.assertIn("496 words", output.getvalue())
        self.assertIn("8.064516e-12", output.getvalue())

    def test_cache_only_miss_does_not_call_synthesis(self):
        output = io.StringIO()
        with patch("sys.argv", ["gold_local.py", "--sizes", "2", "--seed", "1234",
                                "--samples", "1", "--synth", "pygridsynth", "--cache-only"]):
            with patch("sys.stdout", output), patch.object(G, "make_synth") as backend:
                with patch("circuit_cache.CircuitCache") as cache:
                    cache.return_value.load.return_value = None
                    with self.assertRaisesRegex(RuntimeError, "prevented a new synthesis run"):
                        G.main()
        backend.return_value.assert_not_called()


if __name__ == "__main__":
    unittest.main()
