import json
import pennylane as qp
import pennylane.numpy as np

# The unitary sizes your circuit is graded on, and the wire budget.
TEST_SIZES = (2, 3, 4, 5)
MAX_TOTAL_WIRES = 8
# The grader has to finish inside the challenge platform's time limit, so a decomposed
# circuit may hold at most this many gates, which is more than a good solution should need.
MAX_GATES = 100_000

ATOL = 1e-8
GRADES = {"Bronze": 5.0, "Silver": 20.0, "Gold": 60.0}  # minimum saving at TEST_SIZES[-1]

# ---------------------------------------------------------------------------------------
# The cost model. You do not need to change any of this, but read it: it is the scoring.
#
# 1. Your circuit is decomposed with qp.transforms.decompose into the gate set below.
#
# 2. Every surviving gate must be one of these exact PennyLane classes. Anything else, a
#    subclass, a look-alike with a borrowed name, or an operator with no decomposition,
#    is rejected rather than charged.
#
# 3. Each gate is charged in T-gate equivalents: a rotation 44, a T gate 1, the rest 0.
#
# 4. Qubits are charged too. Every wire beyond the n data wires multiplies the whole cost
#    by AUX_WIRE_FACTOR. A circuit on more than MAX_TOTAL_WIRES wires is rejected without
#    being simulated.
# ---------------------------------------------------------------------------------------
GATE_COSTS = {
    qp.RX: 44, qp.RY: 44, qp.RZ: 44,  
    qp.T: 1,
    qp.S: 0, qp.Hadamard: 0, qp.PauliX: 0, qp.PauliY: 0, qp.PauliZ: 0,
    qp.CNOT: 0, qp.GlobalPhase: 0,
}
TARGET_GATE_SET = {"RX", "RY", "RZ", "T", "S", "H", "X", "Y", "Z", "CNOT", "GlobalPhase"}
ROTATION_COST = 44
AUX_WIRE_FACTOR = 2


class IllegalGateError(ValueError):
    """Raised when a decomposed circuit contains a gate outside the cost model."""


def compile_circuit(circuit, u_mat):
    """Decomposes `circuit(u_mat)` into the target gate set and checks every gate.

    Returns:
        qp.tape.QuantumScript: the decomposed circuit, which is what is charged and
            what is simulated.
    """
    tape = qp.tape.make_qscript(circuit)(u_mat)
    (decomposed,), _ = qp.transforms.decompose(tape, gate_set=TARGET_GATE_SET)
    for op in decomposed.operations:
        if type(op) not in GATE_COSTS:
            raise IllegalGateError(
                f"{type(op).__name__} on wires {list(op.wires)} is not in the gate set. Only "
                "the exact PennyLane classes RX, RY, RZ, T, S, H, X, Y, Z, CNOT and "
                "GlobalPhase are accepted after decomposition; subclasses and custom "
                "operators are rejected, whatever they are called."
            )
    return decomposed


def wire_usage(tape, num_qubits):
    """(total wires, auxiliary wires) used by a tape whose data wires are 0 to num_qubits-1."""
    data = set(range(num_qubits))
    used = set(tape.wires.tolist())
    return len(used | data), len(used - data)


def tape_cost(tape, num_qubits):
    """The T-gate-equivalent cost of an already compiled tape, (44 N_rot + N_T) * 2**N_aux.
    A tape on more than MAX_TOTAL_WIRES wires is not charged: the grader rejects it."""
    gates = sum(GATE_COSTS[type(op)] for op in tape.operations)
    _, aux_wires = wire_usage(tape, num_qubits)
    return int(gates * AUX_WIRE_FACTOR ** aux_wires)


def cost_of(circuit, u_mat):
    """The T-gate-equivalent cost of `circuit(u_mat)`. Call this on your own circuit."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_cost(compile_circuit(circuit, u_mat), num_qubits)


def _plain(array):
    """A plain NumPy array, without pennylane.numpy's autodiff bookkeeping (faster)."""
    array = np.array(array, dtype=complex, requires_grad=False)
    return array.unwrap() if hasattr(array, "unwrap") else array


_FIXED_GATES = {
    qp.T: _plain([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    qp.S: _plain([[1, 0], [0, 1j]]),
    qp.Hadamard: _plain([[2 ** -0.5, 2 ** -0.5], [2 ** -0.5, -(2 ** -0.5)]]),
    qp.PauliX: _plain([[0, 1], [1, 0]]),
    qp.PauliY: _plain([[0, -1j], [1j, 0]]),
    qp.PauliZ: _plain([[1, 0], [0, -1]]),
    qp.CNOT: _plain([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}


def _gate_matrix(op):
    """The grader's own matrix for each gate. Gates are never asked for their matrices."""
    kind = type(op)
    if kind in (qp.RX, qp.RY, qp.RZ):
        half = float(op.data[0]) / 2
        c, s = float(np.cos(half)), float(np.sin(half))
        if kind is qp.RX:
            return _plain([[c, -1j * s], [-1j * s, c]])
        if kind is qp.RY:
            return _plain([[c, -s], [s, c]])
        return _plain([[complex(c, -s), 0], [0, complex(c, s)]])
    return _FIXED_GATES[kind]


def tape_matrix(tape, num_qubits):
    """The matrix a compiled tape implements on wires 0 to num_qubits-1, every other
    wire starting in |0>. Simulated with the grader's own gate matrices.

    Runs of single-qubit gates on a wire are multiplied together before they are applied,
    and a CNOT is applied as a swap of two slices, which keeps grading fast."""
    order = list(range(num_qubits)) + sorted(
        (w for w in tape.wires.tolist() if w not in range(num_qubits)), key=str
    )
    position = {wire: k for k, wire in enumerate(order)}
    dim = 2 ** num_qubits
    states = _plain(np.zeros((dim,) + (2,) * len(order)))
    for column in range(dim):
        bits = [(column >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)]
        states[(column, *bits) + (0,) * (len(order) - num_qubits)] = 1.0
    every = [slice(None)] * states.ndim

    def axis_slice(fixed):
        index = list(every)
        for axis, value in fixed.items():
            index[axis] = value
        return tuple(index)

    def apply_single(gate, axis):
        zero, one = states[axis_slice({axis: 0})], states[axis_slice({axis: 1})]
        new_zero = gate[0, 0] * zero + gate[0, 1] * one
        one *= gate[1, 1]
        one += gate[1, 0] * zero
        zero[...] = new_zero

    pending = {}  # wire -> product of the single-qubit gates not applied yet

    def flush(wire):
        gate = pending.pop(wire, None)
        if gate is not None:
            apply_single(gate, 1 + position[wire])

    phase = 1.0 + 0j
    for op in tape.operations:
        kind = type(op)
        if kind is qp.GlobalPhase:
            phase *= complex(np.exp(-1j * float(op.data[0])))
            continue
        wires = op.wires.tolist()
        if len(wires) == 1:
            gate = _gate_matrix(op)
            pending[wires[0]] = gate @ pending[wires[0]] if wires[0] in pending else gate
            continue
        for wire in wires:
            flush(wire)
        axes = [1 + position[w] for w in wires]
        if kind is qp.CNOT:
            low = axis_slice({axes[0]: 1, axes[1]: 0})
            high = axis_slice({axes[0]: 1, axes[1]: 1})
            swapped = states[low].copy()
            states[low] = states[high]
            states[high] = swapped
        else:
            gate = _gate_matrix(op).reshape((2,) * (2 * len(axes)))
            states = np.tensordot(gate, states, axes=(list(range(len(axes), 2 * len(axes))), axes))
            states = np.moveaxis(states, list(range(len(axes))), axes)
    for wire in list(pending):
        flush(wire)
    kept = states.reshape(dim, dim, -1)[:, :, 0]
    return np.array(phase * kept.T, requires_grad=False)


def circuit_matrix(circuit, u_mat):
    """The matrix of `circuit(u_mat)` on wires 0 to n-1, every other wire starting in |0>."""
    num_qubits = int(round(np.log2(len(u_mat))))
    return tape_matrix(compile_circuit(circuit, u_mat), num_qubits)


def baseline_cost(num_qubits):
    """The cost of the 2006 Shannon-decomposition baseline, 44 * (21/16 * 4**n - 3 * 2**(n-1))."""
    return ROTATION_COST * ((21 * 4 ** num_qubits) // 16 - 3 * 2 ** (num_qubits - 1))


def haar_unitary(num_qubits, seed):
    """A Haar-random unitary of dimension 2**num_qubits. You do not need to change this."""
    rng = np.random.default_rng(seed)
    dim = 2 ** num_qubits
    ginibre = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q_mat, r_mat = np.linalg.qr(ginibre)
    diagonal = np.diag(r_mat)
    return np.array(q_mat * (diagonal / np.abs(diagonal)), requires_grad=False)


# Two pieces of linear algebra you are given. You do not need to change these.
_TOL = 1e-8
_MIXERS = (0.6180339887498949, 1.3027756377319946, 0.2360679774997896, 2.7320508075688772)


def diagonalize_unitary(d_mat):
    """Eigendecomposes a unitary in an orthonormal basis. Returns (v_mat, phases) with

        d_mat == v_mat @ diag(exp(1j * phases)) @ v_mat.conj().T
    """
    herm = d_mat + d_mat.conj().T
    skew = -1j * (d_mat - d_mat.conj().T)
    # herm and skew are Hermitian and commute with d_mat, so an eigenbasis of a generic
    # real combination of the two is an orthonormal eigenbasis of d_mat itself.
    best = None
    for mixer in _MIXERS:
        _, v_mat = np.linalg.eigh(herm + mixer * skew)
        rotated = v_mat.conj().T @ d_mat @ v_mat
        residual = np.abs(rotated - np.diag(np.diag(rotated))).max()
        if best is None or residual < best[0]:
            best = (residual, v_mat, np.angle(np.diag(rotated)))
        if residual < 1e-13:
            break
    return best[1], best[2]


def cs_decomposition(u_mat):
    """Cosine-sine decomposition of a unitary of even dimension 2m. Returns
    (l_0, l_1, theta, r_0, r_1) such that, in m x m blocks,

        u_mat == [[l_0, 0], [0, l_1]] @ [[C, -S], [S, C]] @ [[r_0, 0], [0, r_1]]

    with C = diag(cos(theta)) and S = diag(sin(theta)).
    """
    half = len(u_mat) // 2
    u_00, u_01 = u_mat[:half, :half], u_mat[:half, half:]
    u_10, u_11 = u_mat[half:, :half], u_mat[half:, half:]
    l_0, cosines, r_0 = np.linalg.svd(u_00)
    lower = u_10 @ r_0.conj().T
    sines = np.linalg.norm(lower, axis=0)
    theta = np.arctan2(sines, cosines)
    big_s, big_c = sines > _TOL, cosines > _TOL
    l_1 = _complete_basis(lower / np.where(big_s, sines, 1.0), big_s)
    r_1 = np.where(
        big_s[:, None],
        -(l_0.conj().T @ u_01) / np.where(big_s, sines, 1.0)[:, None],
        (l_1.conj().T @ u_11) / np.where(big_c, cosines, 1.0)[:, None],
    )
    # l_1 and r_1 come from dividing by small sines, so they drift from unitarity; snap
    # each to the nearest unitary so the drift is not amplified by a deep recursion.
    return l_0, _nearest_unitary(l_1), theta, r_0, _nearest_unitary(r_1)


def _nearest_unitary(mat):
    """The unitary closest to `mat` in Frobenius norm (its polar factor)."""
    left, _, right = np.linalg.svd(mat)
    return left @ right


def _complete_basis(partial, mask):
    """Fills the columns of `partial` where `mask` is False with an orthonormal completion."""
    missing = int((~mask).sum())
    if missing == 0:
        return partial
    good = partial[:, mask]
    extra, _, _ = np.linalg.svd(np.eye(len(partial)) - good @ good.conj().T)
    out = np.array(partial, dtype=complex)
    out[:, ~mask] = extra[:, :missing]
    return out


def submission_circ(u_mat):
    """Queues a circuit equal to u_mat to within ATOL, global phase included, at low cost.

    Args:
        u_mat (np.ndarray): A 2**n x 2**n unitary matrix, with n in TEST_SIZES. It is
            Haar-random, so you may assume no special structure whatsoever.

    Returns:
        None. The gates you queue on wires 0 to n-1 are the answer. Auxiliary wires are
        allowed, up to MAX_TOTAL_WIRES in total, but each one doubles your cost.
    """


  
    n = int(round(np.log2(len(u_mat))))
    wires = list(range(n))

    def split_blocks(blocks, target):
        if len(blocks[0]) == 2:
            return [
                ("B", np.array(
                    blocks, dtype=complex, requires_grad=False
                ))
            ]

        left, right, angles = [], [], []

        for block in blocks:
            l0, l1, theta, r0, r1 = cs_decomposition(block)
            left.extend([l0, l1])
            right.extend([r0, r1])
            angles.extend(2.0 * theta)

        return (
            split_blocks(left, target + 1)
            + [("Y", target, np.array(
                angles, requires_grad=False
            ))]
            + split_blocks(right, target + 1)
        )

    def euler_angles(blocks):
        determinant = (
            blocks[:, 0, 0] * blocks[:, 1, 1]
            - blocks[:, 0, 1] * blocks[:, 1, 0]
        )

        delta = np.angle(determinant) / 2.0
        special = blocks * np.exp(-1j * delta)[:, None, None]

        p = np.angle(special[:, 0, 0])
        q = np.angle(special[:, 1, 0])

        beta = 2.0 * np.arctan2(
            np.abs(special[:, 1, 0]),
            np.abs(special[:, 0, 0])
        )

        return q - p, beta, -q - p, delta

    def uniformly_controlled(axis, angles, controls, target):
        count = len(angles)
        gray = [j ^ (j >> 1) for j in range(count)]
        rotation = qp.RY if axis == "Y" else qp.RZ

        for j in range(count):
            angle = sum(
                (-1) ** ((i & gray[j]).bit_count()) * angles[i]
                for i in range(count)
            ) / count

            rotation(float(angle), wires=target)

            if controls:
                change = gray[j] ^ gray[(j + 1) % count]
                bit = change.bit_length() - 1

                qp.CNOT(wires=[
                    controls[len(controls) - 1 - bit],
                    target
                ])

    def diagonal_phases(phases, selected_wires):
        if not selected_wires:
            qp.GlobalPhase(-float(phases[0]))
            return

        half = len(phases) // 2
        first = phases[:half]
        second = phases[half:]

        uniformly_controlled(
            "Z",
            second - first,
            selected_wires[1:],
            selected_wires[0]
        )

        diagonal_phases(
            (first + second) / 2.0,
            selected_wires[1:]
        )

    factors = split_blocks([u_mat], 0)
    network = []
    last = n - 1

    for j in range(0, len(factors) - 1, 2):
        alpha, beta, gamma, delta = euler_angles(factors[j][1])
        target = factors[j + 1][1]

        network.extend([
            ("Z", last, alpha),
            ("Y", last, beta)
        ])

        phases = np.stack([
            delta - gamma / 2.0,
            delta + gamma / 2.0
        ], axis=1).reshape(-1)

        paired = np.moveaxis(
            phases.reshape([2] * n), target, 0
        ).reshape(2, -1)

        network.extend([
            ("Z", target, paired[1] - paired[0]),
            ("Y", target, factors[j + 1][2])
        ])

        # Move commuting diagonal phases into the next block.
        mean = (paired[0] + paired[1]) / 2.0
        repeated = np.stack([mean, mean]).reshape([2] * n)

        propagated = np.moveaxis(
            repeated, 0, target
        ).reshape(-1)

        next_blocks = factors[j + 2][1]
        next_blocks[:] = (
            next_blocks
            * np.exp(1j * propagated).reshape(-1, 2, 1)
        )

    alpha, beta, gamma, delta = euler_angles(factors[-1][1])

    network.extend([
        ("Z", last, alpha),
        ("Y", last, beta),
        ("Z", last, gamma)
    ])

    diagonal_phases(delta, wires[:-1])

    for axis, target, angles in reversed(network):
        controls = [wire for wire in wires if wire != target]
        uniformly_controlled(axis, angles, controls, target)


# These functions are responsible for testing the solution.
def _measure(seed, num_samples):
    """Mean cost and auxiliary wires per size, and the largest error. Raises ValueError on
    a circuit the grader cannot accept (illegal gate, too many gates, too many wires)."""
    costs, aux_by_size, max_error = [], [], 0.0
    for num_qubits in TEST_SIZES:
        total, size_aux = 0, 0
        for sample in range(num_samples):
            sample_seed = seed + 1000 * num_qubits + sample
            try:
                tape = compile_circuit(submission_circ, haar_unitary(num_qubits, sample_seed))
            except IllegalGateError as exc:
                raise ValueError(f"Illegal gate. {exc}") from None
            if len(tape.operations) > MAX_GATES:
                raise ValueError(
                    f"Too many gates: {len(tape.operations):,} at {num_qubits} qubits, and "
                    f"the limit is {MAX_GATES:,}."
                )
            total_wires, aux_wires = wire_usage(tape, num_qubits)
            if total_wires > MAX_TOTAL_WIRES:  # rejected, and never simulated
                raise ValueError(
                    f"Too many wires: {total_wires} at {num_qubits} qubits, and the limit is "
                    f"{MAX_TOTAL_WIRES}."
                )
            size_aux = max(size_aux, aux_wires)
            total += tape_cost(tape, num_qubits)
            u_mat = haar_unitary(num_qubits, sample_seed)
            matrix = tape_matrix(tape, num_qubits)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"Not accurate: the matrix of your circuit at {num_qubits} qubits has NaN "
                    "or infinite entries. Check your rotation angles."
                )
            max_error = max(max_error, float(np.abs(matrix - u_mat).max()))
        costs.append(total / num_samples)
        aux_by_size.append(size_aux)
    return costs, aux_by_size, max_error


def _percent(value):
    """A percentage rounded down to one decimal, so the display never overstates a saving."""
    return f"{np.floor(10 * value) / 10:.1f}%"


def _shown(saving, cost, base):
    """A saving for display; a circuit at or past twice the baseline shows its multiple."""
    return _percent(saving) if saving > -100 else f"none, {cost / base:.1f}x the baseline"


def run(test_case_input: str) -> str:
    seed, num_samples = json.loads(test_case_input)
    try:
        costs, aux_by_size, max_error = _measure(seed, num_samples)
    except ValueError as exc:
        return json.dumps({"grade": "None", "reason": str(exc)})

    baseline = [baseline_cost(n) for n in TEST_SIZES]
    savings = [100 * (1 - c / b) for c, b in zip(costs, baseline)]
    saving = savings[-1]
    over = [(n, c, b, a) for n, c, b, a in zip(TEST_SIZES, costs, baseline, aux_by_size)
            if c > b]
    shrinks = [i for i in range(1, len(TEST_SIZES)) if savings[i] < savings[i - 1] - 1e-9]
    if max_error > ATOL:
        reason = (f"Not accurate: the largest entrywise error is {max_error:.1e}, above "
                  f"{ATOL:.0e}. Check the global phase, and that every auxiliary wire comes "
                  "back to |0>.")
    elif over:
        n, c, b, a = over[0]
        reason = f"Over the baseline at {n} qubits: cost {c:,.0f} against {b:,}."
        if a:
            reason += f" Your {a} auxiliary wire(s) multiply the cost by {AUX_WIRE_FACTOR ** a}."
    elif shrinks:
        i = shrinks[0]
        small, large = TEST_SIZES[i - 1], TEST_SIZES[i]
        reason = (f"Does not scale: you save {savings[i - 1]:.2f}% at {small} qubits but only "
                  f"{savings[i]:.2f}% at {large}. Your cost grows by a factor of "
                  f"{costs[i] / costs[i - 1]:.3f} from {small} to {large} qubits, and the "
                  f"baseline's by {baseline[i] / baseline[i - 1]:.3f}.")
    else:
        reason = None

    valid = reason is None
    reached = [g for g, need in GRADES.items() if valid and saving >= need]
    grade = max(reached, key=GRADES.get) if reached else "None"
    if valid and not reached:
        reason = (f"Valid, but below Bronze: it saves {_percent(saving)} against the baseline "
                  f"at {TEST_SIZES[-1]} qubits.")
    report = {"grade": grade}
    if reason:
        report["reason"] = reason
    accurate = max_error <= ATOL
    report["saving_at_%d_qubits" % TEST_SIZES[-1]] = (
        _shown(saving, costs[-1], baseline[-1]) if accurate
        else "not graded, the circuit is not accurate"
    )
    better = [(g, need) for g, need in GRADES.items() if need > saving]
    if better and valid:
        nxt, need = min(better, key=lambda item: item[1])
        report["next_grade"] = f"{nxt} needs {need:g}%"
    if accurate:
        report["saving_by_size"] = [_shown(s, c, b) for s, c, b in zip(savings, costs, baseline)]
    report["cost"] = [round(c, 1) if c % 1 else int(c) for c in costs]
    report["baseline"] = baseline
    report["max_error"] = f"{max_error:.1e}"
    report["aux_wires"] = max(aux_by_size)
    return json.dumps(report)


def check(solution_output: str, expected_output: str) -> None:
    report = json.loads(solution_output)
    required = json.loads(expected_output)["required_grade"]
    ranks = {"None": -1, **{g: i for i, g in enumerate(sorted(GRADES, key=GRADES.get))}}
    grade = report["grade"]
    if grade == "None":
        raise AssertionError(f"{required} needed, and your circuit has no grade. "
                             + report["reason"])
    key = "saving_at_%d_qubits" % TEST_SIZES[-1]
    assert ranks[grade] >= ranks[required], (
        f"{required} needed, you reached {grade}: you save {report[key]} against the "
        f"baseline at {TEST_SIZES[-1]} qubits, and {required} needs {GRADES[required]:g}%."
    )


# These are the public test cases
test_cases = [
    ('[2026, 2]', '{"required_grade": "Gold"}'),
    ('[7, 2]', '{"required_grade": "Gold"}')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
