"""Numerical 11-mask SU(2) multiplexor factors with output Z-phase carry.

No gate synthesis is performed here. Failure is explicit so callers can use
the existing 15-mask construction. Local rank does not imply global coverage.
"""
import hashlib
import time

import numpy as np


MASKS = (1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7)
_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)


class PhaseFactorizationError(RuntimeError):
    pass


def _exp(vector):
    angle = np.linalg.norm(vector)
    return (np.cos(angle) * np.eye(2)
            - 1j * np.sinc(angle / np.pi) * np.einsum("a,aij->ij", vector, _PAULI))


def _log(matrix):
    scalar = np.trace(matrix).real / 2
    vector = (0.5j * np.einsum("aij,ji->a", _PAULI, matrix)).real
    sine = np.linalg.norm(vector)
    if sine < 1e-300:
        if scalar < 0:
            raise PhaseFactorizationError("ambiguous SU(2) logarithm at -I")
        return np.zeros(3)
    return vector * (np.arctan2(sine, scalar) / sine)


_SELECTED = np.array([[(x & m).bit_count() % 2 for m in MASKS] for x in range(16)], bool)


def reconstruct(common, betas, phases):
    """D(phases[x]) common prod beta_j**parity(x,mask_j), D(z)=exp(-izZ)."""
    out = np.tile(common, (16, 1, 1))
    for selected, beta in zip(_SELECTED.T, betas):
        out = np.where(selected[:, None, None], out @ beta, out)
    out[:, 0, :] *= np.exp(-1j * phases[:, None])
    out[:, 1, :] *= np.exp(1j * phases[:, None])
    return out


def _adjoint(matrices):
    result = np.empty((16, 3, 3))
    for b in range(3):
        image = matrices.conj().transpose(0, 2, 1) @ _PAULI[b] @ matrices
        result[:, :, b] = (0.5 * np.einsum("aij,xji->xa", _PAULI, image)).real
    return result


def _jacobian(state):
    common, betas, _ = state
    result = np.zeros((16, 3, 52))
    suffix = np.tile(np.eye(2, dtype=complex), (16, 1, 1))
    for j in range(10, -1, -1):
        result[:, :, 3+3*j:6+3*j] = _adjoint(suffix) * _SELECTED[:, j, None, None]
        suffix = np.where(_SELECTED[:, j, None, None], betas[j] @ suffix, suffix)
    result[:, :, :3] = _adjoint(suffix)
    z_axes = _adjoint(common @ suffix)[:, :, 2]
    for x in range(16):
        result[x, :, 36+x] = z_axes[x]
    return result.reshape(48, 52)


def _update(state, step, scale):
    common, betas, phases = state
    return (common @ _exp(scale * step[:3]),
            [beta @ _exp(scale * step[3+3*j:6+3*j]) for j, beta in enumerate(betas)],
            phases + scale * step[36:])


def _residual(state, target):
    relative = target.conj().transpose(0, 2, 1) @ reconstruct(*state)
    return np.concatenate([_log(matrix) for matrix in relative])


def _attempt(target, rng, deadline):
    state = (_exp(rng.normal(size=3)), [_exp(rng.normal(size=3)) for _ in MASKS],
             rng.normal(size=16))
    start = reconstruct(*state)
    directions = [_log(a.conj().T @ b) for a, b in zip(start, target)]
    position, ds, accepted = 0.0, 0.05, 0
    while position < 1 - 1e-15:
        if time.monotonic() > deadline or accepted >= 128:
            raise PhaseFactorizationError("11-mask continuation work limit")
        next_position = min(1.0, position + ds)
        stage = np.array([a @ _exp(next_position * v) for a, v in zip(start, directions)])
        trial, converged = state, False
        for _ in range(30):
            if time.monotonic() > deadline:
                raise PhaseFactorizationError("11-mask continuation time limit")
            r = _residual(trial, stage)
            if np.max(abs(r)) < 3e-14:
                converged = True
                break
            step = np.linalg.lstsq(_jacobian(trial), -r, rcond=1e-11)[0]
            length = max(np.linalg.norm(step[3*j:3*j+3]) for j in range(12))
            scale = min(1.0, 0.5 / max(length, 1e-300))
            for _ in range(10):
                candidate = _update(trial, step, scale)
                if np.linalg.norm(_residual(candidate, stage)) < np.linalg.norm(r):
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
                raise PhaseFactorizationError("11-mask continuation stalled")
    return state


def factorize_phase(target, *, attempts=2, max_seconds=5.0):
    """Return R, betas, output-Z angles, measured block operator error.

    A target-derived seed makes repeat runs reproducible. Two generic starts
    share a bounded time allowance; callers handle failure with 15 masks.
    """
    target = np.ascontiguousarray(target, dtype=np.complex128)
    if target.shape != (16, 2, 2) or not np.all(np.isfinite(target)):
        raise ValueError("11-mask factorization needs 16 finite 2x2 blocks")
    seed = int.from_bytes(hashlib.sha256(target.tobytes()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + max_seconds
    reason = "no attempts"
    for _ in range(attempts):
        try:
            state = _attempt(target, rng, deadline)
            error = max(float(np.linalg.norm(a-b, 2))
                        for a, b in zip(reconstruct(*state), target))
            if not np.isfinite(error) or error > 1e-12:
                raise PhaseFactorizationError(f"11-mask reconstruction error {error:.2e}")
            return (*state, error)
        except (PhaseFactorizationError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
        if time.monotonic() > deadline:
            break
    raise PhaseFactorizationError(reason)
