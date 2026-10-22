#!/usr/bin/env python3
"""NumPy-only exploratory phase-gauged multiplexor test; no gate synthesis.

Model: V[x] = exp(-i*z[x]*Z) R product_j beta[j]**parity(x, mask[j]).
Run: python phase_factor_probe.py --targets 20
This is research evidence, not a universal factorization guarantee.
"""
import argparse
import json
import time

import numpy as np


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)


def exp(vector):
    angle = np.linalg.norm(vector)
    return (np.cos(angle) * np.eye(2)
            - 1j * np.sinc(angle / np.pi) * np.einsum("a,aij->ij", vector, PAULI))


def log(matrix):
    scalar = np.trace(matrix).real / 2
    vector = (0.5j * np.einsum("aij,ji->a", PAULI, matrix)).real
    sine = np.linalg.norm(vector)
    if sine < 1e-300:
        if scalar < 0:
            raise ValueError("SU(2) logarithm is ambiguous at -I")
        return np.zeros(3)
    return vector * (np.arctan2(sine, scalar) / sine)


def parities(masks):
    return np.array([[(x & int(mask)).bit_count() % 2 for mask in masks]
                     for x in range(16)], bool)


def evaluate(state, masks):
    common, betas, phases = state
    out = np.tile(common, (16, 1, 1))
    for selected, beta in zip(parities(masks).T, betas):
        out = np.where(selected[:, None, None], out @ beta, out)
    out[:, 0, :] *= np.exp(-1j * phases[:, None])
    out[:, 1, :] *= np.exp(1j * phases[:, None])
    return out


def adjoint(matrices):
    """Columns are coefficients of M† sigma[b] M in the Pauli basis."""
    result = np.empty((len(matrices), 3, 3))
    for b in range(3):
        image = matrices.conj().transpose(0, 2, 1) @ PAULI[b] @ matrices
        result[:, :, b] = (0.5 * np.einsum("aij,xji->xa", PAULI, image)).real
    return result


def jacobian(state, masks):
    """Exact differential in right SU(2) coordinates and left output-Z angles.

    dF[x] = F[x] * (-i sum_a J[x,a,j] sigma[a] dt[j]).
    The output has 16*3 real coordinates, including all output-Z freedoms.
    """
    common, betas, _ = state
    selected = parities(masks)
    count = len(masks)
    result = np.zeros((16, 3, 3 + 3 * count + 16))
    suffix = np.tile(np.eye(2, dtype=complex), (16, 1, 1))
    for j in range(count - 1, -1, -1):
        result[:, :, 3+3*j:6+3*j] = adjoint(suffix) * selected[:, j, None, None]
        suffix = np.where(selected[:, j, None, None], betas[j] @ suffix, suffix)
    result[:, :, :3] = adjoint(suffix)
    z_axes = adjoint(common @ suffix)[:, :, 2]
    for x in range(16):
        result[x, :, 3 + 3 * count + x] = z_axes[x]
    return result.reshape(48, -1)


def update(state, step, scale=1):
    common, betas, phases = state
    count = len(betas)
    return (common @ exp(scale * step[:3]),
            [beta @ exp(scale * step[3+3*j:6+3*j]) for j, beta in enumerate(betas)],
            phases + scale * step[3 + 3 * count:])


def random_state(rng, masks):
    return (exp(rng.normal(size=3)), [exp(rng.normal(size=3)) for _ in masks],
            rng.normal(size=16))


def haar_su2(rng):
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion[0] * np.eye(2) - 1j * np.einsum("a,aij->ij", quaternion[1:], PAULI)


def rank(state, masks):
    singular = np.linalg.svd(jacobian(state, masks), compute_uv=False)
    return int(np.count_nonzero(singular > 1e-9 * singular[0])), float(singular[-1])


def finite_difference_error(state, masks):
    base = evaluate(state, masks)
    columns = []
    h = 1e-5
    for column in range(3 + 3 * len(masks) + 16):
        delta = np.zeros(3 + 3 * len(masks) + 16)
        delta[column] = h
        derivative = (evaluate(update(state, delta), masks)
                      - evaluate(update(state, -delta), masks)) / (2 * h)
        tangent = base.conj().transpose(0, 2, 1) @ derivative
        coefficients = (0.5j * np.einsum("aij,xji->xa", PAULI, tangent)).real
        columns.append(coefficients.ravel())
    return float(np.max(abs(np.column_stack(columns) - jacobian(state, masks))))


def gauge_errors(state, masks):
    """Two independent exact freedoms defeat the 49-parameter/48-rank count."""
    common, betas, phases = state
    original = evaluate(state, masks)
    angle = 0.37
    output_z = exp(np.array([0.0, 0.0, angle]))
    common_gauge = (output_z @ common, betas, phases - angle)
    axis = adjoint(common[None, :, :])[0, :, 2]
    changed_betas = [exp(angle * axis) @ betas[0], *betas[1:]]
    first_gauge = (common, changed_betas, phases - angle * parities(masks)[:, 0])
    return {"common_phase_gauge_error": float(np.max(abs(evaluate(common_gauge, masks) - original))),
            "first_factor_gauge_error": float(np.max(abs(evaluate(first_gauge, masks) - original)))}


def folding_error(rng):
    """Verify C P(A,A W) = P(R G†, R beta G†) before word approximation."""
    common, gauge, beta, a = [exp(rng.normal(size=3)) for _ in range(4)]
    w = gauge @ beta @ gauge.conj().T
    c = common @ gauge.conj().T @ a.conj().T
    before = np.array([c @ a, c @ a @ w])
    after = np.array([common @ gauge.conj().T, common @ beta @ gauge.conj().T])
    return float(np.max(abs(before - after)))


def residual(state, masks, target):
    error = target.conj().transpose(0, 2, 1) @ evaluate(state, masks)
    return np.concatenate([log(matrix) for matrix in error])


def continue_to_target(rng, masks, target, deadline):
    # Start from a known generic model, not the rank-deficient all-identity
    # point. A straight identity-start Newton method has missing tangent modes.
    state = random_state(rng, masks)
    start = evaluate(state, masks)
    directions = [log(a.conj().T @ b) for a, b in zip(start, target)]
    position, ds, accepted = 0.0, 0.05, 0
    while position < 1 - 1e-15:
        if time.monotonic() > deadline:
            return None, "time limit", accepted
        next_position = min(1.0, position + ds)
        stage_target = np.array([a @ exp(next_position * v) for a, v in zip(start, directions)])
        trial = state
        converged = False
        for _ in range(30):
            if time.monotonic() > deadline:
                return None, "time limit", accepted
            r = residual(trial, masks, stage_target)
            if np.max(abs(r)) < 3e-14:
                converged = True
                break
            step = np.linalg.lstsq(jacobian(trial, masks), -r, rcond=1e-11)[0]
            length = max(np.linalg.norm(step[3*j:3*j+3]) for j in range(1 + len(masks)))
            scale = min(1.0, 0.5 / max(length, 1e-300))
            for _ in range(10):
                candidate = update(trial, step, scale)
                if np.linalg.norm(residual(candidate, masks, stage_target)) < np.linalg.norm(r):
                    trial = candidate
                    break
                scale *= 0.5
            else:
                break
        if converged:
            state, position = trial, next_position
            accepted += 1
            ds = min(0.15, ds * 1.3)
        else:
            ds *= 0.5
            if ds < 1e-5:
                return None, "continuation stalled", accepted
    error = float(np.max(abs(evaluate(state, masks) - target)))
    return error, "converged", accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--rank-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--seconds", type=float, default=45,
                        help="wall-clock cap for the continuation experiment")
    parser.add_argument("--candidate", choices=("mixed", "weight"), default="mixed",
                        help="11-mask continuation candidate; weight adds mask 7 to ordered ten")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    ten = [m for m in range(1, 16) if m.bit_count() <= 2]
    eleven = ([9, 5, 4, 6, 12, 13, 10, 1, 15, 7, 2] if args.candidate == "mixed"
              else sorted(ten, key=lambda m: (m.bit_count(), m)) + [7])
    sets = [("ten_numeric", ten),
            ("ten_weight_order", sorted(ten, key=lambda m: (m.bit_count(), m))),
            (f"eleven_{args.candidate}", eleven), ("baseline_fifteen", list(range(1, 16)))]
    for name, masks in sets:
        ranks, least = [], []
        for _ in range(args.rank_samples):
            r, small = rank(random_state(rng, masks), masks)
            ranks.append(r)
            least.append(small)
        identity = (np.eye(2, dtype=complex), [np.eye(2, dtype=complex) for _ in masks], np.zeros(16))
        print(json.dumps({"model": name, "masks": masks, "ranks": ranks,
                          "smallest_singular_min": min(least), "identity_rank": rank(identity, masks)[0]}), flush=True)
    print(json.dumps({"analytic_vs_finite_difference_max":
                      finite_difference_error(random_state(rng, eleven), eleven)}), flush=True)
    deadline = time.monotonic() + args.seconds
    outcomes = []
    for index in range(args.targets):
        target = np.array([haar_su2(rng) for _ in range(16)])
        error, status, steps = continue_to_target(rng, eleven, target, deadline)
        outcomes.append({"target": index, "status": status, "max_error": error, "steps": steps})
        print(json.dumps(outcomes[-1]), flush=True)
        if status == "time limit":
            break
    print(json.dumps({"completed": len(outcomes),
                      "passed_1e-13": sum(r["max_error"] is not None and r["max_error"] < 1e-13 for r in outcomes),
                      "requested": args.targets,
                      "note": "Local rank and finite random tests do not prove global solvability."}), flush=True)
    print(json.dumps(gauge_errors(random_state(rng, ten), ten)), flush=True)
    print(json.dumps({"common_word_fold_matrix_error": folding_error(rng)}), flush=True)


if __name__ == "__main__":
    main()
