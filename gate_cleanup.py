"""Exact local Clifford+T cleanup; no approximate synthesis or angle rounding."""
from collections import deque
from functools import lru_cache


_CLIFFORDS = frozenset("HSXYZ")
_SINGLE = _CLIFFORDS | {"T"}
_BASIC = {"Z": "SS", "Y": "WWXSS"}


def _basic(word):
    return "".join(_BASIC.get(gate, gate) for gate in word)


def _key(unitary):
    """Canonical exact cyclotomic coefficients, including the global phase."""
    def component(value):
        value = value.reduce_denomexp()
        return (value.u.a, value.u.b, value.u.c, value.u.d, value.k)
    return (component(unitary.z), component(unitary.w), unitary.n)


@lru_cache(maxsize=1)
def _clifford_table():
    """Shortest Clifford words: physical gates cost one, scalar W costs zero."""
    from pygridsynth.domega_unitary import DOmegaUnitary

    identity = DOmegaUnitary.identity()
    table = {_key(identity): (0, "")}
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        cost, word = table[_key(current)]
        for gate in "WHSXYZ":
            candidate_word = gate + word
            candidate = DOmegaUnitary.from_gates(_basic(candidate_word))
            key = _key(candidate)
            candidate_cost = cost + (gate != "W")
            if key in table and table[key][0] <= candidate_cost:
                continue
            table[key] = (candidate_cost, candidate_word)
            if gate == "W":
                queue.appendleft(candidate)
            else:
                queue.append(candidate)
    if len(table) != 192:
        raise RuntimeError(f"unexpected Clifford group size: {len(table)}")
    return table


@lru_cache(maxsize=1024)
def _short_clifford(word):
    from pygridsynth.domega_unitary import DOmegaUnitary
    _, result = _clifford_table()[_key(DOmegaUnitary.from_gates(_basic(word)))]
    return result.replace("W", ""), result.count("W") % 8


def _clifford_chunks(word):
    """Simplify Clifford runs in a matrix-order word; return scalar W exponent."""
    out, run, phase = [], [], word.count("W")
    for gate in word.replace("W", "") + "!":
        if gate in _CLIFFORDS:
            run.append(gate)
            continue
        if run:
            reduced, scalar = _short_clifford("".join(run))
            out.append(reduced)
            phase += scalar
            run.clear()
        if gate != "!":
            out.append(gate)
    return "".join(out), phase % 8


@lru_cache(maxsize=8192)
def _reduce_run(time_word):
    """Choose an exact shorter representative, with no increase in T-count."""
    from pygridsynth.domega_unitary import DOmegaUnitary
    from pygridsynth.synthesis_of_cliffordT import decompose_domega_unitary

    matrix_word = time_word[::-1]
    candidates = [(matrix_word, 0), _clifford_chunks(matrix_word)]
    normal = decompose_domega_unitary(
        DOmegaUnitary.from_gates(_basic(matrix_word)), wires=[0], up_to_phase=False,
    ).to_simple_str()
    candidates.append(_clifford_chunks(normal))
    original_t = time_word.count("T")
    viable = [(word, phase) for word, phase in candidates
              if len(word) <= len(time_word) and word.count("T") <= original_t]
    word, phase = min(viable, key=lambda item: (len(item[0]), item[0].count("T"), item[0]))
    return word[::-1], phase


def _pass(gates):
    pending, nodes, last = {}, [], {}
    phase = 0

    def append(gate):
        name, wires = gate[:2]
        prior = last.get(wires[0])
        if name in {"H", "X", "Y", "Z", "CNOT"} and prior is not None:
            previous, links = nodes[prior]
            if previous == gate and all(last.get(wire) == prior for wire in wires):
                nodes[prior] = (None, links)
                for wire, link in zip(wires, links):
                    if link is None:
                        last.pop(wire, None)
                    else:
                        last[wire] = link
                return
        links = tuple(last.get(wire) for wire in wires)
        index = len(nodes)
        nodes.append((gate, links))
        for wire in wires:
            last[wire] = index

    def flush(wire, commuting=frozenset()):
        nonlocal phase
        run = pending.pop(wire, "")
        if not run:
            return
        run, scalar = _reduce_run(run)
        phase = (phase + scalar) % 8
        split = len(run)
        while split and run[split - 1] in commuting:
            split -= 1
        for gate in run[:split]:
            append((gate, (wire,)))
        if split < len(run):
            pending[wire] = run[split:]

    for gate in gates:
        name, wires = gate[:2]
        if name in _SINGLE and len(wires) == 1:
            wire = wires[0]
            pending[wire] = pending.get(wire, "") + name
        elif name == "CNOT":
            control, target = wires
            # Z-axis phase gates on controls and X on targets commute with CX.
            flush(control, frozenset("TSZ"))
            flush(target, frozenset("X"))
            append(gate)
        else:
            for wire in wires:
                flush(wire)
            append(gate)
    for wire in list(pending):
        flush(wire)
    return [gate for gate, _ in nodes if gate is not None], phase


def _pauli_pass(gates):
    """Propagate an exact X^x Z^z frame, removing internal Pauli gates.

    The invariant is processed U = omega^phase * P_frame * emitted U.
    No Hermitian-Pauli convention is assumed: XZ = -iY is tracked explicitly.
    """
    frame, out, phase = {}, [], 0
    for gate in gates:
        name, wires = gate[:2]
        if name == "CNOT":
            control, target = wires
            xc, zc = frame.get(control, (0, 0))
            xt, zt = frame.get(target, (0, 0))
            frame[control] = (xc, zc ^ zt)
            frame[target] = (xt ^ xc, zt)
            out.append(gate)
            continue
        if len(wires) != 1:
            raise ValueError(f"unsupported cleanup gate: {gate}")
        wire = wires[0]
        x, z = frame.get(wire, (0, 0))
        if name == "X":
            x ^= 1
        elif name == "Z":
            phase += 4 * x
            z ^= 1
        elif name == "Y":
            phase += 2 + 4 * x
            x, z = x ^ 1, z ^ 1
        elif name == "H":
            phase += 4 * x * z
            x, z = z, x
            out.append(gate)
        elif name == "S":
            phase += 2 * x
            z ^= x
            out.append(gate)
        elif name == "T":
            # T X = omega X T†; T† = Z S T. Absorb Z into the frame.
            phase += x
            z ^= x
            out.append(gate)
            if x:
                out.append(("S", (wire,)))
        elif name == "RZ":
            out.append(("RZ", wires, -gate[2] if x else gate[2]))
        else:
            raise ValueError(f"unsupported cleanup gate: {gate}")
        frame[wire] = (x, z)
    for wire, (x, z) in frame.items():
        if x and z:
            out.append(("Y", (wire,)))
            phase += 6  # XZ = -iY.
        elif x:
            out.append(("X", (wire,)))
        elif z:
            out.append(("Z", (wire,)))
    return out, phase % 8


_TWO_GENERATORS = tuple(
    (name, (wire,)) for wire in (0, 1) for name in "HSXYZ"
) + (("CNOT", (0, 1)), ("CNOT", (1, 0)))
_IDENTITY_ACTION = ((1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0))


def _conjugate_action(action, gate):
    """Exact images of X0,Z0,X1,Z1, represented by i^p X^x Z^z."""
    name, wires = gate
    result = []
    for x, z, phase in action:
        if name == "CNOT":
            c, t = wires
            if x & (1 << c):
                x ^= 1 << t
            if z & (1 << t):
                z ^= 1 << c
        else:
            mask = 1 << wires[0]
            a, b = bool(x & mask), bool(z & mask)
            if name == "H":
                phase += 2 * a * b
                if a != b:
                    x ^= mask
                    z ^= mask
            elif name == "S":
                phase += a
                if a:
                    z ^= mask
            elif name == "X":
                phase += 2 * b
            elif name == "Z":
                phase += 2 * a
            elif name == "Y":
                phase += 2 * (a != b)
        result.append((x, z, phase % 4))
    return tuple(result)


@lru_cache(maxsize=1)
def _two_clifford_table():
    """Shortest words in the 11,520 projective two-qubit Clifford elements."""
    table = {_IDENTITY_ACTION: ()}
    queue = deque([_IDENTITY_ACTION])
    while queue:
        state = queue.popleft()
        word = table[state]
        for gate in _TWO_GENERATORS:
            next_state = _conjugate_action(state, gate)
            if next_state not in table:
                table[next_state] = word + (gate,)
                queue.append(next_state)
                if len(table) > 11520:
                    raise RuntimeError("two-qubit Clifford action escaped the finite group")
    if len(table) != 11520:
        raise RuntimeError(f"unexpected two-qubit Clifford group size: {len(table)}")
    return table


def _exact_two_matrix(gates):
    """Evaluate a small Clifford word using exact dyadic cyclotomic entries."""
    from pygridsynth.ring import DOmega

    rows = [[DOmega.from_int(int(i == j)) for j in range(4)] for i in range(4)]
    def divide_sqrt2(value):
        return DOmega(value.u, value.k + 1)
    for name, wires in gates:
        if name == "CNOT":
            c, t = wires
            rows = [rows[i ^ (1 << t) if i & (1 << c) else i] for i in range(4)]
            continue
        mask = 1 << wires[0]
        if name in {"H", "X", "Y"}:
            for i in range(4):
                if i & mask:
                    continue
                a, b = rows[i], rows[i | mask]
                if name == "H":
                    rows[i] = [divide_sqrt2(x + y) for x, y in zip(a, b)]
                    rows[i | mask] = [divide_sqrt2(x - y) for x, y in zip(a, b)]
                elif name == "X":
                    rows[i], rows[i | mask] = b, a
                else:
                    rows[i] = [x.mul_by_omega_power(6) for x in b]
                    rows[i | mask] = [x.mul_by_omega_power(2) for x in a]
        else:
            power = 2 if name == "S" else 4
            for i in range(4):
                if i & mask:
                    rows[i] = [x.mul_by_omega_power(power) for x in rows[i]]
    return [entry.reduce_denomexp() for row in rows for entry in row]


@lru_cache(maxsize=8192)
def _short_two_clifford(gates):
    action = _IDENTITY_ACTION
    for gate in gates:
        action = _conjugate_action(action, gate)
    reduced = _two_clifford_table()[action]
    if len(reduced) >= len(gates):
        return gates, 0
    old, new = _exact_two_matrix(gates), _exact_two_matrix(reduced)
    for phase in range(8):
        if all(a == b.mul_by_omega_power(phase) for a, b in zip(old, new)):
            return reduced, phase
    raise RuntimeError("two-qubit Clifford rewrite failed exact phase verification")


def _two_qubit_pass(gates):
    """Resynthesize zero-T blocks using at most two wires; preserve all T/RZ barriers."""
    out, block, wires, phase = [], [], [], 0

    def flush():
        nonlocal phase
        if not block:
            return
        if len(wires) == 1:
            word, scalar = _short_clifford("".join(gate[0] for gate in reversed(block)))
            out.extend((name, (wires[0],)) for name in reversed(word))
            phase = (phase + scalar) % 8
            block.clear()
            wires.clear()
            return
        mapping = {wire: i for i, wire in enumerate(wires)}
        symbolic = tuple((name, tuple(mapping[w] for w in ws)) for name, ws in block)
        reduced, scalar = _short_two_clifford(symbolic)
        out.extend((name, tuple(wires[i] for i in ws)) for name, ws in reduced)
        phase = (phase + scalar) % 8
        block.clear()
        wires.clear()

    for gate in gates:
        name, ws = gate[:2]
        is_clifford = name in _CLIFFORDS or name == "CNOT"
        new_wires = [wire for wire in ws if wire not in wires]
        if not is_clifford or len(wires) + len(new_wires) > 2:
            flush()
            new_wires = list(ws)
        if is_clifford:
            wires.extend(wire for wire in new_wires if wire not in wires)
            block.append(gate)
        else:
            out.append(gate)
    flush()
    return out, phase


def cleanup_gates(gates):
    """Return (reduced time-order gates, scalar phase in radians).

    Original U = exp(i*phase) * reduced U. All Clifford rewrites use exact
    cyclotomic arithmetic. RZ angles are kept unchanged. Iterate only while
    gate count decreases, so the cleanup terminates without a precision knob.
    """
    from math import pi

    current, phase = list(gates), 0
    while True:
        local, local_phase = _pass(current)
        framed, frame_phase = _pauli_pass(current)
        framed, final_phase = _pass(framed)
        two_local, two_local_phase = _two_qubit_pass(local)
        two_framed, two_frame_phase = _two_qubit_pass(framed)
        candidates = [(current, 0), (local, local_phase),
                      (framed, (frame_phase + final_phase) % 8),
                      (two_local, (local_phase + two_local_phase) % 8),
                      (two_framed, (frame_phase + final_phase + two_frame_phase) % 8)]
        original_t = sum(gate[0] == "T" for gate in current)
        candidates = [(ops, scalar) for ops, scalar in candidates
                      if len(ops) <= len(current)
                      and sum(gate[0] == "T" for gate in ops) <= original_t]
        reduced, scalar = min(candidates, key=lambda item: (
            len(item[0]), sum(gate[0] == "T" for gate in item[0]),
        ))
        phase = (phase + scalar) % 8
        if len(reduced) >= len(current):
            return reduced, phase * pi / 4
        current = reduced
