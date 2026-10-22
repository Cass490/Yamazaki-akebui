"""11-mask, folding, and carry checks with no lattice synthesis."""
from pathlib import Path
import io
import unittest
from unittest.mock import patch

import numpy as np

import gold_local as G
import phase_factorization as F


def su2_word(word):
    matrix = G.mat(word)
    return matrix / np.sqrt(np.linalg.det(matrix))


class Phase11Regression(unittest.TestCase):
    def test_production_factorizer_reconstructs_haar_and_structured_blocks(self):
        rng = np.random.default_rng(802)
        haar = []
        for _ in range(16):
            z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            q, r = np.linalg.qr(z)
            q *= np.diag(r) / abs(np.diag(r))
            haar.append(q / np.sqrt(np.linalg.det(q)))
        structured = [G.Ry(rng.uniform(0, np.pi)) @ G.su2_exp(np.array([0., 0., rng.uniform(-2, 2)]))
                      for _ in range(16)]
        for blocks in (np.array(haar), np.array(structured)):
            common, betas, phases, error = F.factorize_phase(blocks)
            self.assertEqual(len(betas), 11)
            self.assertLess(error, 1e-12)
            self.assertLess(np.max(abs(F.reconstruct(common, betas, phases) - blocks)), 1e-12)

    def test_numerical_failure_uses_baseline_before_any_word_synthesis(self):
        blocks = np.tile(np.eye(2, dtype=complex), (16, 1, 1))
        with patch.object(F, "factorize_phase", side_effect=F.PhaseFactorizationError("forced stall")):
            common, betas, masks, phases, error, method = G.factor_mux_blocks(blocks, "phase11")
        self.assertEqual(method, "baseline")
        self.assertEqual(masks, tuple(range(1, 16)))
        self.assertTrue(np.all(phases == 0))
        self.assertLess(error, 1e-12)

    def test_final_mux_does_not_call_phase_solver(self):
        blocks = np.tile(np.eye(2, dtype=complex), (16, 1, 1))
        with patch.object(F, "factorize_phase", side_effect=AssertionError("phase solver used")):
            result = G.factor_mux_blocks(blocks, "phase11", allow_output_z=False)
        self.assertEqual(len(result[2]), 15)

    def test_full_word_pairs_keep_independent_global_phases_and_lengths(self):
        for word0, word1 in (("WHTHT", "WWWSTHTH"), ("HTHT", "HTHTHTHT"),
                             ("WWWH", "WWS"), ("", "HTHT"), ("HT", "WST")):
            result = G.compile_word_pair_best(word0, word1)
            expected = np.zeros((4, 4), complex)
            expected[:2, :2], expected[2:, 2:] = G.mat(word0), G.mat(word1)
            self.assertLess(np.max(abs(G.pair_matrix(result) - expected)), 1e-12)
        with self.assertRaisesRegex(ValueError, "matching T parity"):
            G.compile_word_pair("T", "")

    def test_phase11_compilation_carries_z_and_folds_both_branch_phases(self):
        common_word = "SH"
        beta_words = ["S", "", "", "", "", "", "", "", "H", "HS", "X"]
        common = su2_word(common_word)
        betas = [su2_word(word) for word in beta_words]
        phases = np.linspace(-0.7, 0.9, 16)
        blocks = F.reconstruct(common, betas, phases)

        for folded in (False, True):
            with self.subTest(folded=folded):
                gauge_word, calls = "", []
                remaining = list(range(10, 0 if folded else -1, -1))
                original_compile = G.compile_pair_best

                def exact_backend(target, eps):
                    nonlocal remaining
                    if remaining:
                        j = remaining.pop(0)
                        word = gauge_word + beta_words[j] + G.inverse(gauge_word)
                    elif not folded or len(calls) == 10:
                        word = common_word + G.inverse(gauge_word)
                    else:
                        word = common_word + beta_words[0] + G.inverse(gauge_word)
                    # Deliberately distinct global phases in the two folded words.
                    word = "W" * (len(calls) % 4) + G.ma_normal_form(word)
                    self.assertLess(G.word_error(word, target), 1e-11)
                    if folded or len(calls) < 11:
                        self.assertEqual(G.tcount(word) % 2, 0)
                    calls.append(word)
                    return word

                exact_backend.even = exact_backend

                def tracking_compile(word):
                    nonlocal gauge_word
                    result = original_compile(word)
                    gauge_word = G.ma_normal_form(result["A"] + gauge_word)
                    return result

                with patch.object(F, "factorize_phase", return_value=(common, betas, phases, 0.0)):
                    with patch.object(G, "compile_pair_best", side_effect=tracking_compile):
                        result = G.synth_mux(blocks, exact_backend, 1e-9,
                                             mux_method="phase11", fold_common=folded)
                gates, phase = G.mux_gates(result, 4, [0, 1, 2, 3])
                actual = np.exp(1j * phase) * G.simulate(gates, 5)
                carry = np.array([result["ctrl_phase"][x] + (2*b-1)*phases[x]
                                  for x in range(16) for b in (0, 1)])
                actual = np.exp(1j * carry[:, None]) * actual
                expected = np.zeros((32, 32), complex)
                for x in range(16):
                    expected[2*x:2*x+2, 2*x:2*x+2] = blocks[x]
                self.assertLess(np.max(abs(actual - expected)), 1e-11)
                self.assertEqual([m for m, _ in result["pairs"]], list(reversed(F.MASKS)))
                self.assertEqual(len(calls), 12)
                if folded:
                    self.assertEqual(result["C"], "")

    def test_full_csd_propagates_relative_phases_across_different_targets(self):
        scope = {}
        exec(Path(G.__file__).with_name("challenge.py").read_text().split("def submission_circ")[0], scope)
        rng = np.random.default_rng(991)
        n, dim = 3, 8
        target = np.asarray(scope["haar_unitary"](n, 321))
        seen = []

        def exact_mux(blocks, synth, eps, **kwargs):
            z = rng.uniform(-0.8, 0.8, len(blocks)) if kwargs["allow_output_z"] else np.zeros(len(blocks))
            ctrl = rng.uniform(-0.5, 0.5, len(blocks))
            f = blocks.copy()
            f[:, 0, :] *= np.exp(1j*z[:, None])
            f[:, 1, :] *= np.exp(-1j*z[:, None])
            seen.append(kwargs["allow_output_z"])
            return {"blocks": f, "output_z": z, "ctrl_phase": ctrl, "error_bound": 0.0}

        def exact_emit(result, target, controls):
            matrix = np.zeros((dim, dim), complex)
            for x, block in enumerate(result["blocks"]):
                indices = [G.full_index(n, target, controls, x, b) for b in (0, 1)]
                matrix[np.ix_(indices, indices)] = block * np.exp(-1j*result["ctrl_phase"][x])
            return [("MATRIX", tuple(range(n)), matrix)], 0.0

        with patch.object(G, "synth_mux", side_effect=exact_mux), patch.object(G, "mux_gates", side_effect=exact_emit):
            gates, phase = G.synthesize(target, scope["cs_decomposition"], None, 1e-9,
                                        mux_method="phase11", fold_common=True, optimize=False)
        actual = np.eye(dim, dtype=complex)
        for gate in gates:
            if gate[0] == "MATRIX":
                matrix = gate[2]
            elif gate[0] == "RZ":
                matrix = np.array([[1]], complex)
                local = np.diag(np.exp(np.array([-0.5j, 0.5j])*gate[2]))
                for wire in range(n):
                    matrix = np.kron(matrix, local if wire == gate[1][0] else np.eye(2))
            else:
                matrix = G.simulate([gate], n)
            actual = matrix @ actual
        self.assertLess(np.max(abs(np.exp(1j*phase)*actual-target)), 1e-12)
        self.assertEqual(seen, [True]*6 + [False])
        self.assertEqual(sum(g[0] == "RZ" for g in gates), 3)

    def test_planned_count_keeps_fallback_budget_safe(self):
        self.assertEqual(G.synthesis_word_counts(5, "phase11"), (376, 496))
        self.assertEqual(G.synthesis_word_counts(3, "phase11"), (28, 28))

    def test_cli_cache_keys_separate_both_optimization_flags(self):
        from circuit_cache import CircuitCache
        signatures = []
        for method, fold in (("baseline", False), ("phase11", False),
                             ("baseline", True), ("phase11", True)):
            argv = ["gold_local.py", "--sizes", "5", "--seed", "2026", "--samples", "1",
                    "--synth", "pygridsynth", "--cache-only", "--mux-method", method]
            if fold:
                argv.append("--fold-common")
            with patch("sys.argv", argv), patch("sys.stdout", io.StringIO()):
                with patch.object(G, "make_synth") as backend, patch("circuit_cache.CircuitCache") as cache:
                    cache.return_value.load.return_value = None
                    with self.assertRaisesRegex(RuntimeError, "prevented a new synthesis run"):
                        G.main()
                    signatures.append(cache.call_args.args[1])
                    backend.return_value.assert_not_called()
        keys = [CircuitCache("unused", signature).key(np.eye(32)) for signature in signatures]
        self.assertEqual(len(set(keys)), 4)
        self.assertEqual(signatures[0]["synthesis_revision"], "paired_even_t_gate_count_v2")


if __name__ == "__main__":
    unittest.main()
