"""Exact gate-algebra and checkpoint tests; no single-qubit synthesis searches."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from circuit_cache import CircuitCache
from gate_cleanup import cleanup_gates
import gold_local as G


def dense_matrix(gates, n):
    result = np.eye(2**n, dtype=complex)
    for gate in gates:
        if gate[0] == "RZ":
            angle = gate[2]
            local = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
            matrix = np.array([[1]], dtype=complex)
            for wire in range(n):
                matrix = np.kron(matrix, local if wire == gate[1][0] else np.eye(2))
        else:
            matrix = G.simulate([gate], n)
        result = matrix @ result
    return result


class GateCleanupRegression(unittest.TestCase):
    def verify(self, gates, n):
        reduced, phase = cleanup_gates(gates)
        error = np.max(np.abs(dense_matrix(gates, n) - np.exp(1j * phase) * dense_matrix(reduced, n)))
        self.assertLess(error, 1e-11)
        self.assertLessEqual(len(reduced), len(gates))
        self.assertLessEqual(sum(g[0] == "T" for g in reduced), sum(g[0] == "T" for g in gates))
        self.assertEqual(sum(g[0] == "RZ" for g in reduced), sum(g[0] == "RZ" for g in gates))
        return reduced, phase

    def test_removed_scalar_phase_is_preserved(self):
        gates = [(name, (0,)) for name in "SHSHSH"]
        reduced, phase = self.verify(gates, 1)
        self.assertEqual(reduced, [])
        self.assertAlmostEqual(phase, np.pi / 4)

    def test_control_phases_and_target_x_allow_cnot_cancellation(self):
        gates = [("CNOT", (0, 1)), ("T", (0,)), ("X", (1,)), ("CNOT", (0, 1))]
        reduced, _ = self.verify(gates, 2)
        self.assertEqual(len(reduced), 2)

    def test_random_two_qubit_clifford_blocks(self):
        rng = np.random.default_rng(571)
        for _ in range(50):
            gates = []
            for _ in range(20):
                name = str(rng.choice(["H", "S", "X", "Y", "Z", "CNOT"]))
                wires = (0, 1) if name == "CNOT" else (int(rng.integers(2)),)
                if name == "CNOT" and rng.integers(2):
                    wires = wires[::-1]
                gates.append((name, wires))
            self.verify(gates, 2)

    def test_random_mixed_gates_with_rotations_and_three_wires(self):
        rng = np.random.default_rng(811)
        for _ in range(30):
            gates = []
            for _ in range(60):
                name = str(rng.choice(["H", "S", "T", "X", "Y", "Z", "CNOT", "RZ"]))
                if name == "CNOT":
                    wires = tuple(int(x) for x in rng.choice(3, size=2, replace=False))
                    gates.append((name, wires))
                elif name == "RZ":
                    gates.append((name, (int(rng.integers(3)),), float(rng.normal())))
                else:
                    gates.append((name, (int(rng.integers(3)),)))
            self.verify(gates, 3)

    def test_diagonal_lifts_and_deferred_control_phase(self):
        omega = np.exp(1j * np.pi / 4)
        cnot = G.simulate([("CNOT", (0, 1))], 2)
        for x in range(8):
            for y in range(8):
                if (x-y) % 2:
                    continue
                a, d = G.diagonal_params(x, y)
                circuit = np.kron(np.eye(2), np.diag([1, omega**a])) @ cnot
                circuit = circuit @ np.kron(np.eye(2), np.diag([1, omega**(-d)])) @ cnot
                corrected = np.kron(np.diag([1, omega**d]), np.eye(2)) @ circuit
                expected = np.diag([1, omega**x, 1, omega**y])
                self.assertLess(np.max(np.abs(corrected-expected)), 1e-12)
                self.assertEqual((a % 2) + (d % 2), G.dcost(x, y))

    def test_paired_word_compiler_and_cleanup(self):
        rng = np.random.default_rng(92)
        for _ in range(2):
            word = "".join("HT" + ("S" if rng.integers(2) else "") for _ in range(80))
            pair = G.compile_pair_best(word)
            a = G.mat(pair["A"])
            expected = np.zeros((4, 4), dtype=complex)
            expected[:2, :2], expected[2:, 2:] = a, a @ G.mat(word)
            self.assertLess(np.max(np.abs(G.pair_matrix(pair)-expected)), 1e-11)
            gates = [(name, tuple(0 if wire == "c" else 1 for wire in wires))
                     for name, wires in pair["gates"]]
            reduced, _ = self.verify(gates, 2)
            self.assertLess(len(reduced), len(gates))

    def test_cleanup_phase_matches_grader_global_phase_convention(self):
        namespace = {"__name__": "grader"}
        source = Path(G.__file__).with_name("challenge.py").read_text()
        exec(source.split("def submission_circ")[0], namespace)
        gates = [(name, (0,)) for name in "SHSHSH"] + [("T", (1,)), ("CNOT", (0, 1))]
        reduced, gph = G.optimize_circuit(gates, 0.17)
        tape = namespace["compile_circuit"](
            lambda _: G.queue_ops(reduced, gph, namespace["qp"]), np.eye(4),
        )
        expected = np.exp(0.17j) * dense_matrix(gates, 2)
        self.assertLess(np.max(np.abs(namespace["tape_matrix"](tape, 2)-expected)), 1e-12)


class CircuitCacheRegression(unittest.TestCase):
    def test_round_trip_and_configuration_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = np.eye(4, dtype=complex)
            gates = [("H", (0,)), ("CNOT", (0, 1)), ("RZ", (1,), -0.2)]
            cache = CircuitCache(directory, {"eps": 4e-9, "backend": "test"})
            self.assertIsNone(cache.load(target))
            cache.save(target, gates, 0.3)
            self.assertEqual(cache.load(target), (gates, 0.3))
            self.assertIsNone(CircuitCache(directory, {"eps": 2e-9, "backend": "test"}).load(target))
            self.assertIsNone(cache.load(target * 1j))
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_corrupt_cache_stops_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            target = np.eye(4, dtype=complex)
            cache = CircuitCache(directory, {"eps": 4e-9})
            cache.save(target, [("T", (0,))], 0.0)
            path = Path(directory) / f"{cache.key(target)}.json"
            path.write_text('{"schema":')
            with self.assertRaisesRegex(ValueError, "Invalid circuit checkpoint"):
                cache.load(target)


if __name__ == "__main__":
    unittest.main()
