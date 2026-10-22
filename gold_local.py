#!/usr/bin/env python3
"""
Unitary-synthesis challenge (QHack 2026 Q8) -- Gold attempt, LOCAL RUN.

Method
  1. Flag / recursive cosine-sine decomposition -> 2^n - 1 uniformly controlled single-qubit gates (minimal count).
  2. Each multiplexor: parity factorisation  V_x = R . prod_j beta_j^{p_j(x)}  (solved numerically),
     then Yamazaki & Akibue (arXiv 2603.14202, Thm 4 / Lemma 19): every beta_j becomes a *paired word*
     costing ~ T(W)/2 + ~3, plus one common word per multiplexor.
  3. Single-qubit words: optimal-T Clifford+T synthesis of general SU(2) (Morisaki-Sano-Akibue,
     arXiv 2510.05816) via cyclosynth (Rust).  All leftover diagonal phases are absorbed by the next
     multiplexor for free; the final diagonal uses 2^(n-1)-1 rotations.

Setup (once):
  pip install numpy pennylane pygridsynth mpmath maturin
  git clone https://github.com/mtweiden/cyclosynth && cd cyclosynth
  #   Debian/Ubuntu: sudo apt-get install libgmp-dev libmpfr-dev m4      macOS: brew install gmp mpfr
  maturin develop --release && cd ..

Run:
  python gold_local.py --template challenge.py              # the challenge file with the grader (any version)
  python smoke_test.py --mux-method phase11 --fold-common --eps 1e-9
  python gold_local.py --mux-method phase11 --fold-common --eps 1e-9 --sizes 5 --seed 2026 --samples 1 --verbose
  python gold_local.py --template challenge.py --error-budget 8e-9  # conservative accuracy allocation; costs more
  python gold_local.py --template challenge.py --synth pygridsynth --sizes 2 3   # quick correctness check, no Rust

Notes
  * This uses external packages, so it is for measuring locally -- it is not a platform submission.
  * --eps is a PER-WORD tolerance, not a circuit tolerance. A five-qubit sample at 4e-9 failed accuracy.
  * --error-budget allocates a conservative sum-of-errors budget across words; final grading is still required.
    This can increase T cost and does not establish Gold's 60% saving.
  * Runtime is dominated by cyclosynth (~1-3 s per word, ~500 words per 5-qubit sample).
  * --mux-method phase11 plans 376 source words at n=5, with a safe 496-word fallback maximum.
    --fold-common pairs two full words in place of the final split and common word.
    These options require measurement; no Gold score is guaranteed.
"""
import argparse, functools, itertools as _it, sys, time
from pathlib import Path
import numpy as np

def resolve_template(value):
    """Resolve a template from the current directory, then beside this script."""
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = Path(__file__).resolve().parent / path
    return path.resolve()

def require_cyclosynth():
    """Reject a source-directory namespace import without the compiled API."""
    import cyclosynth
    if not callable(getattr(cyclosynth, "Synthesizer", None)):
        raise ImportError(
            "cyclosynth.Synthesizer is unavailable; Python found a module or source "
            "folder without the compiled API. Install the cyclosynth Python extension "
            "into the interpreter running this script."
        )
    return cyclosynth

# ======================================================================
# words.py
# ======================================================================
"""Single-qubit Clifford+T word utilities.
Convention: a word is a string over {H,S,T,X,Y,Z,s,t,W}; s=S^dag, t=T^dag, W=global phase e^{i pi/4}.
The string 'ABC' means the matrix A @ B @ C (leftmost = applied last)."""
import numpy as np

OM = np.exp(1j * np.pi / 4)
GATE = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]), "s": np.diag([1, -1j]),
    "T": np.diag([1, OM]), "t": np.diag([1, np.conj(OM)]),
    "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    "W": np.eye(2) * OM, "I": np.eye(2),
}

def mat(word):
    M = np.eye(2, dtype=complex)
    for g in word:
        M = M @ GATE[g]
    return M

INV = {"H": "H", "S": "s", "s": "S", "T": "t", "t": "T", "X": "X", "Y": "Y", "Z": "Z", "I": "I"}
def inverse(word):
    """Inverse word (exact up to global phase W^-k, tracked separately by callers via skeleton phase)."""
    out = []
    for g in reversed(word):
        if g == "W":
            out.append("WWWWWWW")
        else:
            out.append(INV[g])
    return "".join(out)

def tcount(word):
    return sum(g in "Tt" for g in word)

def skeleton(word):
    """Reduce a word to  phase * T^e0 H T^e1 H ... H T^er  (e_i mod 8).
    Returns (exps, phase_k) with global phase OM**phase_k.  Exact identities used:
    S=T^2, Z=T^4, s=T^6, t=T^7, X = H T^4 H, Y = i X Z = OM^2 * H T^4 H T^4, W = OM."""
    toks = []      # list of ints (T powers) and 'H'
    ph = 0
    for g in word:
        if g == "H": toks.append("H")
        elif g == "T": toks.append(1)
        elif g == "t": toks.append(7)
        elif g == "S": toks.append(2)
        elif g == "s": toks.append(6)
        elif g == "Z": toks.append(4)
        elif g == "X": toks += ["H", 4, "H"]
        elif g == "Y": toks += ["H", 4, "H", 4]; ph += 2
        elif g == "W": ph += 1
        elif g == "I": pass
        else: raise ValueError(g)
    # normalise into alternating exps: e0 H e1 H ... (merge powers, cancel H H)
    exps = [0]
    for tk in toks:
        if tk == "H":
            if len(exps) >= 2 and exps[-1] % 8 == 0:
                exps.pop()               # ... H T^0 H  ->  ... (H H = I)
            else:
                exps.append(0)
        else:
            exps[-1] = (exps[-1] + tk) % 8
    return [e % 8 for e in exps], ph % 8

def skeleton_word(exps, ph=0):
    """Word realising OM^ph * T^e0 H T^e1 H ... H T^er using only H,S,T,Z (T^k as T/S/Z combos)."""
    def tp(k):
        k %= 8
        return ("T" if k & 1 else "") + ("S" if k & 2 else "") + ("Z" if k & 4 else "")
    w = "W" * (ph % 8)
    for i, e in enumerate(exps):
        if i: w += "H"
        w += tp(e)
    return w

def to_basic(word):
    """Rewrite into pygridsynth's alphabet {H,T,S,X,W} exactly."""
    rep = {"s": "SSS", "t": "TTTTTTT", "Z": "SS", "Y": "WWXSS", "I": ""}
    return "".join(rep.get(g, g) for g in word)

def ma_normal_form(word):
    """Exact Matsumoto-Amano normal form (as a word over H,T,S,X,W) via pygridsynth exact synthesis."""
    from pygridsynth.domega_unitary import DOmegaUnitary
    from pygridsynth.synthesis_of_cliffordT import decompose_domega_unitary
    u = DOmegaUnitary.from_gates(to_basic(word))
    return decompose_domega_unitary(u, wires=[0], up_to_phase=False).to_simple_str()

# ======================================================================
# parity.py
# ======================================================================
"""Parity factorisation of a uniformly controlled SU(2) gate (Yamazaki-Akibue eq. 127 form):
    V_x = R . prod_{j=1..q} beta_j^{p_j(x)}      (ordered product, j=1 leftmost),
where x ranges over 2^c control patterns, masks m_j = 1..2^c-1, p_j(x) = popcount(m_j & x) mod 2.
R = V_0. Solved by continuation from the identity with Newton steps (analytic structure, 3 params per beta)."""
import numpy as np

PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]

def su2_exp(v):
    th = np.linalg.norm(v)
    if th < 1e-300:
        return np.eye(2, dtype=complex)
    n = v / th
    return np.cos(th) * np.eye(2) - 1j * np.sin(th) * (n[0] * PAULI[0] + n[1] * PAULI[1] + n[2] * PAULI[2])

def su2_log(U):
    """v with su2_exp(v) = U (U in SU(2)), principal branch, full precision near identity."""
    a = np.real(U[0, 0] + U[1, 1]) / 2
    v = np.array([-np.imag(U[0, 1] + U[1, 0]) / 2,
                  np.real(U[1, 0] - U[0, 1]) / 2,
                  -np.imag(U[0, 0] - U[1, 1]) / 2])
    s = np.linalg.norm(v)
    if s < 1e-300:
        return np.zeros(3)
    th = np.arctan2(s, a)
    return th * v / s

def parities(c):
    q = 2 ** c - 1
    P = np.zeros((2 ** c, q), dtype=int)
    for x in range(2 ** c):
        for j, m in enumerate(range(1, 2 ** c)):
            P[x, j] = bin(x & m).count("1") & 1
    return P

def model(R, betas, P):
    out = []
    for x in range(P.shape[0]):
        M = R.copy()
        for j in range(P.shape[1]):
            if P[x, j]:
                M = M @ betas[j]
        out.append(M)
    return out

def factorize(V, max_steps=400, tol=1e-13):
    """V: array (2^c, 2, 2) of SU(2) blocks. Returns (R, betas[q]) with model == V to ~1e-13."""
    n = V.shape[0]; c = int(round(np.log2(n))); q = n - 1
    P = parities(c)
    R = V[0].copy()
    Rinv = R.conj().T
    L = [su2_log(Rinv @ V[x]) for x in range(n)]          # targets relative to R
    th = np.zeros((q, 3))                                 # beta_j = exp(th_j)
    def resid(th, s):
        betas = [su2_exp(t) for t in th]
        r = []
        for x in range(1, n):
            M = np.eye(2, dtype=complex)
            for j in range(q):
                if P[x, j]:
                    M = M @ betas[j]
            Tgt = su2_exp(s * L[x])
            r.append(su2_log(Tgt.conj().T @ M))
        return np.concatenate(r)
    def jac(th, s, h=1e-7):
        f0 = resid(th, s); J = np.zeros((f0.size, th.size))
        flat = th.ravel()
        for k in range(flat.size):
            t2 = flat.copy(); t2[k] += h
            J[:, k] = (resid(t2.reshape(q, 3), s) - f0) / h
        return f0, J
    s = 0.0; ds = 0.05
    while s < 1.0 - 1e-15:
        s_new = min(1.0, s + ds)
        th_try = th.copy(); ok = False
        for it in range(30):
            f, J = jac(th_try, s_new)
            if np.abs(f).max() < (tol if s_new == 1.0 else 1e-10):
                ok = True; break
            try:
                step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break
            th_try = th_try + step.reshape(q, 3)
            if np.abs(step).max() > 1.0:
                break
        if ok:
            th, s = th_try, s_new
            ds = min(0.25, ds * 1.5)
        else:
            ds /= 2
            if ds < 1e-6:
                raise RuntimeError("parity factorisation continuation failed")
    betas = [su2_exp(t) for t in th]
    # polish with exact-ish Newton at s=1
    return R, betas

# ======================================================================
# pair.py
# ======================================================================
"""Parity-controlled pair P(A,B) = |0><0|_c (x) A + |1><1|_c (x) B with T-cost ~ max(T(A),T(B)) (+ small overhead).
Layer rule (Yamazaki-Akibue Thm 2 idea): P(T^x, T^y) with x = y mod 2 costs 1 T if odd, 0 or 2 if even,
up to a phase on the control which is returned (to be absorbed downstream for free)."""
import numpy as np

V_CH = "SHT"           # V X V^dag = H exactly; controlled-H = (I(x)V) CNOT (I(x)V^dag), T-cost 2

def dcost(x, y):
    k1 = ((x + y) // 2) % 8
    d = ((y - x) // 2) % 8
    return (k1 & 1) + (d & 1)

def diagonal_params(x, y):
    """Equivalent half-angle lifts; a control-only phase is deferred by callers."""
    a = ((x + y) // 2) % 8
    d = ((y - x) // 2) % 8
    def gate_count(pair):
        k, delta = pair
        return k.bit_count() + (2 + ((-delta) % 8).bit_count() if delta else 0)
    return min(((a, d), ((a + 4) % 8, (d + 4) % 8)), key=gate_count)

def diagonal_gate_count(x, y):
    a, d = diagonal_params(x, y)
    return a.bit_count() + (2 + ((-d) % 8).bit_count() if d else 0)

def align(a, b, band=2):
    """DP alignment of skeleton segments a (word A) and b (word B), matrix order.
    Returns (cost, ops) with ops in matrix order: ('D',x,y) | ('H',) | ('HA',) (H only in A) | ('HB',)."""
    N, M = len(a) - 1, len(b) - 1
    INF = 10 ** 9
    sys.setrecursionlimit(10000)
    @functools.lru_cache(maxsize=None)
    def best(i, j, ra, rb):
        if abs(i - j) > band:
            return INF, INF, ()
        res = (INF, INF, ())
        if i == N and j == M:
            if (ra - rb) % 2 == 0:
                return dcost(ra, rb), diagonal_gate_count(ra, rb), (("D", ra, rb),)
            return res
        if i < N and j < M and (ra - rb) % 2 == 0:
            c, ng, ops = best(i + 1, j + 1, a[i + 1], b[j + 1])
            c += dcost(ra, rb)
            ng += diagonal_gate_count(ra, rb) + 1
            if (c, ng) < res[:2]:
                res = (c, ng, (("D", ra, rb), ("H",)) + ops)
        if i < N:
            for x in range(ra % 2, 8, 2):
                c, ng, ops = best(i + 1, j, a[i + 1], (rb - x) % 8)
                c += dcost(ra, x) + 2
                ng += diagonal_gate_count(ra, x) + 12
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", ra, x), ("HA",)) + ops)
        if j < M:
            for x in range(rb % 2, 8, 2):
                c, ng, ops = best(i, j + 1, (ra - x) % 8, b[j + 1])
                c += dcost(x, rb) + 2
                ng += diagonal_gate_count(x, rb) + 10
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", x, rb), ("HB",)) + ops)
        return res
    out = best(0, 0, a[0], b[0])
    best.cache_clear()
    return out[0], out[2]

def split_options(word, max_pos=3):
    """Ways to write word = X Y at a T-exponent position, giving (A,B) = (X^-1, Y) skeletons.
    Only near-balanced split points (equal T in both halves) and one exponent split per parity class."""
    f, ph = skeleton(word)
    nodd = sum(e & 1 for e in f)
    cands = []
    for j in range(len(f)):
        left_odd = sum(e & 1 for e in f[:j])
        for alpha in range(8):
            beta = (f[j] - alpha) % 8
            ta = left_odd + (alpha & 1); tb = nodd - left_odd - (f[j] & 1) + (beta & 1)
            if ta != tb:
                continue
            cands.append((abs(j - len(f) / 2), j, alpha))
    cands.sort()
    opts, seen = [], set()
    for _, j, alpha in cands:
        key = (j, alpha & 1, ((f[j] - alpha) % 8) & 1)
        if key in seen: continue
        seen.add(key)
        a = [(-alpha) % 8] + [(-e) % 8 for e in reversed(f[:j])]
        b = [(f[j] - alpha) % 8] + f[j + 1:]
        opts.append((a, b, ph))
        if len(opts) >= max_pos * 2: break
    return opts

def _emit_pair_ops(ops, ctrl_phase=0):
    """Emit a skeleton alignment, deferring its exact control phase."""
    gates = []            # time order list of (name, wires)
    def tpow(k, w):
        k %= 8
        out = []
        if k & 1: out.append(("T", (w,)))
        if k & 2: out.append(("S", (w,)))
        if k & 4: out.append(("Z", (w,)))
        return out
    def word_gates(wd, w):   # time order gates for word wd (matrix order string)
        out = []
        for g in reversed(wd):
            if g == "t": out += tpow(7, w)
            elif g == "s": out += tpow(6, w)
            else: out.append((g, (w,)))
        return out
    for op in reversed(ops):              # time order = reverse matrix order
        if op[0] == "D":
            x, y = op[1], op[2]
            k1, d = diagonal_params(x, y)
            gates += tpow(k1, "q")
            if d % 8:
                gates += [("CNOT", ("c", "q"))] + tpow(-d, "q") + [("CNOT", ("c", "q"))]
            ctrl_phase += d
        elif op[0] == "H":
            gates.append(("H", ("q",)))
        else:
            ch = word_gates(inverse(V_CH), "q") + [("CNOT", ("c", "q"))] + word_gates(V_CH, "q")
            if op[0] == "HA":                 # H only when parity = 0
                gates += [("X", ("c",))] + ch + [("X", ("c",))]
            else:
                gates += ch
    tc = sum(g[0] == "T" for g in gates)
    return {"gates": gates, "t": tc, "ctrl_phase": ctrl_phase % 8}

def compile_pair(word, band=2):
    """Best split pair with A^-1 B == word; retain exact deferred phases."""
    best = None
    for a, b, phB in split_options(word):
        cost, ops = align(a, b, band)
        ng = sum(diagonal_gate_count(op[1], op[2]) if op[0] == "D"
                 else {"H": 1, "HA": 12, "HB": 10}[op[0]] for op in ops)
        if best is None or (cost, ng) < best[:2]:
            best = (cost, ng, ops, a, phB)
    if best is None or best[0] >= 10 ** 9:
        raise RuntimeError("no parity-compatible split-word alignment")
    _, _, ops, a, phB = best
    result = _emit_pair_ops(ops, phB)
    result["A"] = skeleton_word(a)
    return result

def compile_word_pair(word0, word1, band=2):
    """Compile two full words; P(w0,w1)=exp(i global) D(ctrl) raw gates."""
    a, ph0 = skeleton(word0)
    b, ph1 = skeleton(word1)
    if sum(e & 1 for e in a) % 2 != sum(e & 1 for e in b) % 2:
        raise ValueError("full paired words must have matching T parity")
    band = max(band, abs(len(a) - len(b)) + 1)
    cost, ops = align(a, b, band)
    if cost >= 10 ** 9:
        cost, ops = align(a, b, max(len(a), len(b)))
    if cost >= 10 ** 9:
        raise RuntimeError("no full-word pair alignment")
    result = _emit_pair_ops(ops, ph1 - ph0)
    result["global"] = ph0 * np.pi / 4
    return result

def pair_matrix(res):
    """4x4 matrix (c,q) of the compiled gates including the deferred control phase."""
    I = np.eye(2); M = np.eye(4, dtype=complex)
    CN = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex)
    for name, w in res["gates"]:
        if name == "CNOT": G = CN
        elif w == ("c",): G = np.kron(GATE[name], I)
        else: G = np.kron(I, GATE[name])
        M = G @ M
    ph = np.exp(1j * np.pi / 4 * res["ctrl_phase"])
    return np.exp(1j * res.get("global", 0.0)) * np.kron(np.diag([1, ph]), I) @ M

# ---- Clifford conjugation variants (free): P(A'K^-1, A'K^-1 w) = P(A', A' K^-1 w K) (I (x) K^-1) ----
_CLIFF = []
def cliffords():
    if not _CLIFF:
        mats = []
        for L in range(0, 6):
            for w in _it.product("HS", repeat=L):
                w = "".join(w); M = mat(w)
                if not any(abs(abs(np.trace(M.conj().T @ m)) - 2) < 1e-9 for m in mats):
                    _CLIFF.append(w); mats.append(M)
    return _CLIFF

def compile_pair_best(word):
    """Best over the 24 Clifford conjugations. Returns res with 'gates' (time order, incl. the free
    uncontrolled K^-1 applied first), 't', 'ctrl_phase', and 'A' (= A' K^-1) such that the gates realise
    P(A, A word) up to the returned control phase."""
    best = None
    for K in cliffords():
        Kinv = inverse(K)
        w2 = ma_normal_form(Kinv + word + K)
        res = compile_pair(w2)
        pre_count = sum(2 if gate == "s" else 1 for gate in Kinv)
        if best is None or (res["t"], len(res["gates"]) + pre_count) < best[2]:
            best = (res, Kinv, (res["t"], len(res["gates"]) + pre_count))
    res, Kinv, _ = best
    pre = []
    for g in reversed(Kinv):
        if g == "s": pre += [("S", ("q",)), ("Z", ("q",))]
        else: pre.append((g, ("q",)))
    return {"gates": pre + res["gates"], "t": res["t"], "ctrl_phase": res["ctrl_phase"], "A": res["A"] + Kinv}

def compile_word_pair_best(word0, word1):
    """Search simultaneous Clifford conjugations of two full words."""
    best = None
    for K in cliffords():
        Kinv = inverse(K)
        result = compile_word_pair(ma_normal_form(Kinv + word0 + K),
                                   ma_normal_form(Kinv + word1 + K))
        pre = [(name, ("q",)) for name, _ in word_gates(Kinv, 0)]
        post = [(name, ("q",)) for name, _ in word_gates(K, 0)]
        result["gates"] = pre + result["gates"] + post
        score = result["t"], len(result["gates"])
        if best is None or score < best[0]:
            best = score, result
    return best[1]

# ======================================================================
# synths.py
# ======================================================================
"""Single-qubit synthesizer backends: word = synth(U, eps) with mat(word) ~ U up to global phase."""
import numpy as np

def euler_zyz(U):
    """U ~ e^{i phi} Rz(a) Ry(b) Rz(g), Rz(t)=diag(e^{-it/2},e^{it/2}), Ry(t)=[[c,-s],[s,c]]."""
    U = U / np.sqrt(np.linalg.det(U))
    b = 2 * np.arctan2(abs(U[1, 0]), abs(U[0, 0]))
    s = np.angle(U[1, 1]) ; d = np.angle(U[1, 0])          # U11 = e^{i(a+g)/2} cos, U10 = e^{i(a-g)/2} sin
    if abs(U[0, 0]) < 1e-14: s = -np.angle(U[0, 1]) + np.pi if False else d
    if abs(U[1, 0]) < 1e-14: d = s
    a = s + d; g = s - d
    return a, b, g

def pygridsynth_synth(U, eps):
    """Stand-in (exact Clifford+T, but ~7 log2(1/eps) T instead of the optimal ~3)."""
    import mpmath
    from pygridsynth.unitary_approximation import approximate_one_qubit_unitary
    U = U / np.sqrt(np.linalg.det(U))
    circ, _ = approximate_one_qubit_unitary(
        mpmath.matrix(U.tolist()), mpmath.mpf(eps), up_to_phase=False
    )
    return circ.to_simple_str()

_CYC = {}
def cyclosynth_synth(U, eps, *, even_t=False):
    """Joint SU(2) synthesis via cyclosynth (build locally: maturin develop --release)."""
    cyclosynth = require_cyclosynth()
    key = (eps, even_t)
    if key not in _CYC:
        _CYC[key] = cyclosynth.Synthesizer(eps, even_t=even_t)
    a, b, g = euler_zyz(U)
    # U3(theta, phi, lambda) represents Rz(phi) Ry(theta) Rz(lambda) up to phase.
    # synthesize_zyz instead approximates three rotations separately in this API.
    r = _CYC[key].synthesize_u3(float(b), float(a), float(g))
    if r is None or r.gates is None:
        raise RuntimeError("cyclosynth found no circuit")
    return r.gates

# ======================================================================
# mux.py
# ======================================================================
"""Uniformly controlled SU(2) gate via parity factorisation + Yamazaki-Akibue paired words (Thm 4).
    V_x = R . prod_j beta_j^{p_j(x)},   implemented as   U~ = C . P_1 ... P_q   (P_q first in time),
    P_j = P_{p_j}(A_j, A_j W^_j),   W_j = G_{j+1} beta_j G_{j+1}^-1,   G_j = A_j G_{j+1},   C ~ R G_1^-1."""
import numpy as np

def even_word(U, synth, eps):
    """Use the backend's even-T search for paired words when available."""
    even_synth = getattr(synth, "even", None)
    if even_synth is not None:
        w = even_synth(U, eps)
        if tcount(w) % 2:
            raise RuntimeError("even-T backend returned an odd T-count word")
        return w
    # Custom/stand-in backends can still use phase-preserving synthesis.
    w = synth(U, eps)
    if tcount(w) % 2 == 0:
        return w
    w = pygridsynth_synth(U, eps)
    if tcount(w) % 2:
        raise RuntimeError("phase-preserving SU(2) synthesis returned an odd T-count word")
    return w

def phase_of(word, U):
    """gamma with mat(word) ~ exp(i gamma) U."""
    return float(np.angle(np.trace(U.conj().T @ mat(word))))

def word_error(word, U):
    """Phase-aligned operator-norm error, stable for errors near 1e-8.

    The trace overlap alone loses small errors quadratically and can round to
    exactly one even when the matrix error exceeds the grader's tolerance.
    """
    M = mat(word)
    gamma = np.angle(np.trace(U.conj().T @ M))
    return float(np.linalg.norm(np.exp(-1j * gamma) * M - U, 2))

def synthesis_word_counts(n, mux_method="baseline"):
    """Planned source words and safe worst case if every reduced solve falls back."""
    if mux_method not in ("baseline", "phase11"):
        raise ValueError(f"unknown multiplexor method: {mux_method}")
    worst = (2 ** n - 1) * 2 ** (n - 1)
    planned = 30 * 12 + 16 if mux_method == "phase11" and n == 5 else worst
    return planned, worst

def word_epsilon(n, eps, error_budget=None):
    """Allocate optional circuit budget over (2**n-1)*2**(n-1) words.

    Unitary products satisfy ||prod U - prod V|| <= sum ||U_j-V_j||.
    Reserve half the budget for numerical decomposition/roundoff and allow
    the words to consume the other half. This is conservative, not a cost
    optimum or a rigorous floating-point certificate.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if error_budget is None:
        return float(eps)
    if not np.isfinite(error_budget) or error_budget <= 0:
        raise ValueError("error budget must be finite and positive")
    words = (2 ** n - 1) * 2 ** (n - 1)
    return min(float(eps), float(error_budget) / (2 * words))

def factor_mux_blocks(V, mux_method="baseline", allow_output_z=True, verbose=False):
    """Return factors, actual masks, and output-Z carry; fall back before synthesis."""
    if mux_method not in ("baseline", "phase11"):
        raise ValueError(f"unknown multiplexor method: {mux_method}")
    count = len(V)
    c = int(round(np.log2(count)))
    if mux_method == "phase11" and c == 4 and allow_output_z:
        from phase_factorization import MASKS, PhaseFactorizationError, factorize_phase
        try:
            R, betas, z, error = factorize_phase(V)
            if verbose:
                print(f"    phase11: 11 masks, factor error={error:.2e}", flush=True)
            return R, betas, MASKS, z, error, "phase11"
        except PhaseFactorizationError as exc:
            if verbose:
                print(f"    phase11 fallback to 15 masks: {exc}", flush=True)
    R, betas = factorize(V)
    error = max(float(np.linalg.norm(M - target, 2))
                for M, target in zip(model(R, betas, parities(c)), V))
    if error > 1e-11:
        raise RuntimeError(f"parity factorization matrix error {error:.2e} exceeds 1e-11")
    return R, betas, tuple(range(1, count)), np.zeros(count), error, "baseline"

def synth_mux(V, synth, eps, verbose=False, mux_method="baseline", fold_common=False,
              allow_output_z=True):
    """V: (2^c,2,2) SU(2) blocks. Returns dict:
       pairs: list (time order) of (mask, gates) with gates on symbolic wires 'c','q'
       C: word for the common gate (applied last), global: global phase (radians, multiply by e^{i*global})
       ctrl_phase: array over x (2^c) of control-pattern phases (radians) to push into the carry."""
    n = V.shape[0]; c = int(round(np.log2(n)))
    R, betas, masks, output_z, factor_error, method = factor_mux_blocks(
        V, mux_method, allow_output_z, verbose)
    q = len(masks)
    P = np.array([[(x & mask).bit_count() % 2 for mask in masks] for x in range(n)])
    error_bound = factor_error
    G = np.eye(2, dtype=complex)
    pairs_rev = []
    ctrl = np.zeros(n)
    tsum = 0
    folded = fold_common and q > 0
    for j in range(q - 1, 0 if folded else -1, -1):
        W = G @ betas[j] @ G.conj().T
        w = even_word(W, synth, eps)
        gam = phase_of(w, W)
        error_bound += word_error(w, W)
        res = compile_pair_best(w)
        tsum += res["t"]
        # exactness: pair realises P(A, A w) * phase(ctrl_phase*pi/4 on parity 1); we want P(A, A W):
        corr = res["ctrl_phase"] * np.pi / 4 - gam
        ctrl += corr * P[:, j]
        pairs_rev.append((masks[j], res["gates"]))
        G = mat(res["A"]) @ G
    if folded:
        target0 = R @ G.conj().T
        target1 = R @ betas[0] @ G.conj().T
        word0, word1 = even_word(target0, synth, eps), even_word(target1, synth, eps)
        gamma0, gamma1 = phase_of(word0, target0), phase_of(word1, target1)
        last_pair = compile_word_pair_best(word0, word1)
        ctrl += (last_pair["ctrl_phase"] * np.pi / 4 - gamma1 + gamma0) * P[:, 0]
        pairs_rev.append((masks[0], last_pair["gates"]))
        global_phase = last_pair["global"] - gamma0
        error_bound += max(word_error(word0, target0), word_error(word1, target1))
        tsum += last_pair["t"]
        wc = ""
    else:
        Cmat = R @ G.conj().T
        wc = synth(Cmat, eps)
        global_phase = -phase_of(wc, Cmat)
        error_bound += word_error(wc, Cmat)
    if verbose:
        print(f"    mux c={c}: {method}, {q} masks, pairs T={tsum}, "
              f"common={'folded' if folded else str(tcount(wc)) + ' T'}", flush=True)
    return {"pairs": pairs_rev, "C": wc, "global": global_phase, "ctrl_phase": ctrl, "c": c,
            "output_z": output_z, "factorization": method, "error_bound": error_bound}

# ======================================================================
# emit.py
# ======================================================================
"""Turn synthesized multiplexors into concrete gate lists and simulate small cases."""
import numpy as np

def word_gates(word, w):
    """Time-ordered gates for a matrix-order word on wire w (t,s expanded into T/S/Z)."""
    out = []
    for g in reversed(word):
        if g == "t": out += [("T", (w,)), ("S", (w,)), ("Z", (w,))]
        elif g == "s": out += [("S", (w,)), ("Z", (w,))]
        elif g == "W": pass                       # global phase handled by caller
        else: out.append((g, (w,)))
    return out

def word_phase(word):
    return np.pi / 4 * word.count("W")

def mux_gates(res, target, controls):
    """controls: list of wires, big-endian w.r.t. mask bits. Returns (gates time order, global phase rad)."""
    c = len(controls); gates = []
    gph = res["global"] + word_phase(res["C"])
    for mask, pg in res["pairs"]:
        mem = [controls[i] for i in range(c) if (mask >> (c - 1 - i)) & 1]
        pw = mem[-1]
        ladder = [("CNOT", (o, pw)) for o in mem[:-1]]
        gates += ladder
        for name, ws in pg:
            gates.append((name, tuple(pw if x == "c" else target for x in ws)))
        gates += ladder[::-1]
    gates += word_gates(res["C"], target)
    return gates, gph

def simulate(gates, n):
    """Full 2^n matrix of a gate list (big-endian wires 0..n-1)."""
    N = 2 ** n; M = np.eye(N, dtype=complex)
    for name, ws in gates:
        if name == "CNOT":
            a, b = ws; G = np.zeros((N, N))
            for i in range(N):
                j = i ^ (1 << (n - 1 - b)) if (i >> (n - 1 - a)) & 1 else i
                G[j, i] = 1
        else:
            (w,) = ws; ops = [np.eye(2)] * n; ops = list(ops); ops[w] = GATE[name]
            G = ops[0]
            for o in ops[1:]: G = np.kron(G, o)
        M = G @ M
    return M

# ======================================================================
# flagsynth.py
# ======================================================================
"""Gold attempt: flag (recursive CSD) circuit, every multiplexor built from Yamazaki-Akibue paired words."""
import numpy as np

def csd_items(u_mat, cs_decomposition):
    n = int(round(np.log2(len(u_mat))))
    def rec(blocks, k):
        if k == n - 1:
            return [("G", blocks)]
        lefts, rights, cs = [], [], []
        for block in blocks:
            l0, l1, th, r0, r1 = cs_decomposition(block)
            lefts += [l0, l1]; rights += [r0, r1]; cs.append(2 * np.asarray(th))
        return rec(rights, k + 1) + [("CS", k, np.concatenate(cs))] + rec(lefts, k + 1)
    return rec([np.array(u_mat, dtype=complex)], 0), n

def Ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)

def full_index(n, t, controls, x, tb):
    idx = 0
    for i, w in enumerate(controls):
        if (x >> (len(controls) - 1 - i)) & 1:
            idx |= 1 << (n - 1 - w)
    if tb:
        idx |= 1 << (n - 1 - t)
    return idx

def uc_rz(angles, controls, target, out):
    """Gray-code uniformly controlled RZ; skips (near-)zero rotations."""
    angles = np.asarray(angles, float); k = len(controls)
    if k == 0:
        if abs(angles[0]) > 1e-13: out.append(("RZ", (target,), float(angles[0])))
        return
    size = 2 ** k
    gray = [i ^ (i >> 1) for i in range(size)]
    signs = np.array([[(-1) ** bin(j & g).count("1") for j in range(size)] for g in gray])
    th = signs @ angles / size
    for i in range(size):
        if abs(th[i]) > 1e-13: out.append(("RZ", (target,), float(th[i])))
        bit = (gray[i] ^ gray[(i + 1) % size]).bit_length() - 1
        out.append(("CNOT", (controls[k - 1 - bit], target)))

def diagonal_gates(phases, wires, out):
    """diag(exp(i*phases)) on wires (big-endian); returns leftover global phase (radians)."""
    phases = np.asarray(phases, float); m = len(wires)
    while m > 0:
        even, odd = phases[0::2], phases[1::2]
        uc_rz(odd - even, wires[:m - 1], wires[m - 1], out)
        phases = (even + odd) / 2; m -= 1
    return float(phases[0])

def optimize_circuit(gates, gph, verbose=False):
    """Reduce physical gates exactly, retaining all removed scalar phases."""
    from gate_cleanup import cleanup_gates
    started = time.monotonic()
    if verbose:
        print(f"  cleaning {len(gates) + 1:,} emitted gates (including global phase)", flush=True)
    reduced, phase = cleanup_gates(gates)
    if verbose:
        print(f"  gate cleanup: {len(gates) + 1:,} -> {len(reduced) + 1:,}; "
              f"T={sum(g[0] == 'T' for g in reduced)} "
              f"({time.monotonic() - started:.1f}s)", flush=True)
    return reduced, gph + phase

def synthesize(u_mat, cs_decomposition, synth, eps, verbose=False, optimize=True,
               error_budget=None, mux_method="baseline", fold_common=False):
    """Returns (gates, global_phase): gates time-ordered tuples ('NAME', wires[, angle])."""
    items, n = csd_items(u_mat, cs_decomposition)
    eps = word_epsilon(n, eps, error_budget)
    if verbose:
        expected, worst = synthesis_word_counts(n, mux_method)
        print(f"  n={n}: {expected} planned words ({worst} with fallbacks); "
              f"per-word epsilon={eps:.3e}; fold_common={fold_common}", flush=True)
    N = 2 ** n
    carry = np.zeros(N)
    gates, gph = [], 0.0
    error_bound = 0.0
    t0 = time.time()
    for item_index, it in enumerate(items):
        if it[0] == "G":
            t = n - 1; blocks = list(it[1])
        else:
            t = it[1]; blocks = [Ry(a) for a in it[2]]
        controls = [w for w in range(n) if w != t]
        c = len(controls)
        V = np.zeros((2 ** c, 2, 2), complex); phi = np.zeros(2 ** c)
        for x in range(2 ** c):
            i0 = full_index(n, t, controls, x, 0); i1 = full_index(n, t, controls, x, 1)
            B = blocks[x] @ np.diag([np.exp(1j * carry[i0]), np.exp(1j * carry[i1])])
            d = np.linalg.det(B); phi[x] = np.angle(d) / 2
            V[x] = B / np.exp(1j * phi[x])
        # Keep the final carry independent of the last target bit, avoiding
        # 16 additional arbitrary RZ rotations at five qubits.
        res = synth_mux(V, synth, eps, verbose=verbose, mux_method=mux_method,
                        fold_common=fold_common, allow_output_z=item_index < len(items) - 1)
        error_bound += res["error_bound"]
        g, ph = mux_gates(res, t, controls)
        gates += g; gph += ph
        carry = np.zeros(N)
        for x in range(2 ** c):
            for tb in (0, 1):
                carry[full_index(n, t, controls, x, tb)] = (
                    phi[x] + res["ctrl_phase"][x] + (2 * tb - 1) * res["output_z"][x])
    gph += diagonal_gates(carry, list(range(n)), gates)
    if verbose:
        print(f"  sum of multiplexor error bounds: {error_bound:.3e} "
              "(excludes CSD/roundoff)", flush=True)
    if error_budget is not None and error_bound > error_budget:
        raise RuntimeError(f"sum of multiplexor error bounds {error_bound:.2e} "
                           f"exceeds requested circuit budget {error_budget:.2e}")
    if optimize:
        gates, gph = optimize_circuit(gates, gph, verbose=verbose)
    if verbose:
        print(f"  synthesized n={n} in {time.time()-t0:.0f}s")
    return gates, gph

def queue_ops(gates, gph, qp):
    for gt in gates:
        name, ws = gt[0], gt[1]
        if name == "CNOT": qp.CNOT(wires=list(ws))
        elif name == "RZ": qp.RZ(gt[2], wires=ws[0])
        elif name == "H": qp.Hadamard(wires=ws[0])
        elif name == "S": qp.S(wires=ws[0])
        elif name == "T": qp.T(wires=ws[0])
        elif name == "X": qp.PauliX(wires=ws[0])
        elif name == "Y": qp.PauliY(wires=ws[0])
        elif name == "Z": qp.PauliZ(wires=ws[0])
        else: raise ValueError(name)
    qp.GlobalPhase(-gph)

# ======================================================================
# driver
# ======================================================================
_CHECKED = {}
def make_synth(kind, eps_scale=1.0):
    if kind == "cyclosynth" and not hasattr(require_cyclosynth().Synthesizer, "even_t"):
        raise RuntimeError(
            "The installed cyclosynth extension predates even_t support. "
            "Rebuild the local cyclosynth checkout into qenv before running synthesis."
        )
    base = {"cyclosynth": cyclosynth_synth, "pygridsynth": pygridsynth_synth}[kind]
    def checked(U, eps, even_t=False):
        w = cyclosynth_synth(U, eps, even_t=True) if even_t else base(U, eps)
        err = word_error(w, U)
        # Dense products of long words have a small floating-point floor.
        roundoff = 32 * np.finfo(float).eps * max(1, len(w))
        if not np.isfinite(err) or err > eps + roundoff:
            raise RuntimeError(f"{kind}: phase-aligned word error {err:.2e} "
                               f"exceeds epsilon {eps:.2e} (roundoff {roundoff:.1e})")
        return w
    def synth(U, eps):
        return checked(U, eps)
    if kind == "cyclosynth":
        def even(U, eps):
            return checked(U, eps, even_t=True)
        synth.even = even
    return synth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=resolve_template,
                    default=Path(__file__).resolve().with_name("challenge.py"),
                    help="grader file (default: challenge.py beside this script)")
    ap.add_argument("--synth", default="cyclosynth", choices=["cyclosynth", "pygridsynth"])
    ap.add_argument("--mux-method", choices=["baseline", "phase11"], default="baseline",
                    help="phase11 reduces 4-control factors, with a 15-mask fallback")
    ap.add_argument("--fold-common", action="store_true",
                    help="jointly compile the final pair and common word in each multiplexor")
    ap.add_argument("--eps", type=float, default=4e-9,
                    help="per-word tolerance cap; NOT a whole-circuit error tolerance")
    ap.add_argument("--error-budget", type=float, default=None,
                    help="optional conservative whole-circuit error allocation; increases T cost")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="override TEST_SIZES, e.g. --sizes 2 3")
    ap.add_argument("--seed", type=int, default=None, help="run one seed instead of the public test cases")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(__file__).resolve().parent / ".run_cache",
                    help="local synthesized-circuit checkpoints (default: .run_cache beside this script)")
    ap.add_argument("--no-cache", action="store_true", help="disable persistent circuit checkpoints")
    ap.add_argument("--cache-only", action="store_true",
                    help="reuse checkpoints; stop on a cache miss instead of synthesizing")
    ap.add_argument("--dry-run", action="store_true",
                    help="show word counts and precision settings without synthesis or grading")
    args = ap.parse_args()
    if not args.template.is_file():
        ap.error(f"template file not found: {args.template}")
    if args.cache_only and args.no_cache:
        ap.error("--cache-only cannot be combined with --no-cache")
    try:
        word_epsilon(2, args.eps, args.error_budget)
    except ValueError as exc:
        ap.error(str(exc))

    src = args.template.read_text(encoding="utf-8")
    g = {"__name__": "grader"}
    exec(src.split("def submission_circ")[0], g)            # grader helpers & constants
    exec("def submission_circ(u_mat):\n    pass\n" + src.split("# These functions are responsible for testing the solution.")[1].split("# These are the public test cases")[0], g)
    qp = g["qp"]
    if args.sizes:
        g["TEST_SIZES"] = tuple(args.sizes)
    if args.dry_run:
        for n in g["TEST_SIZES"]:
            count, worst = synthesis_word_counts(n, args.mux_method)
            eps = word_epsilon(n, args.eps, args.error_budget)
            print(f"n={n}: {count} words; per-word epsilon={eps:.6e}; "
                  f"sum of word tolerance caps={count * eps:.6e}; "
                  f"mux_method={args.mux_method}; fold_common={args.fold_common}; "
                  f"fallback maximum={worst} words")
        if args.error_budget is None:
            print("No whole-circuit error budget selected; local epsilon does not ensure final accuracy.")
        else:
            print("Budget allocation is conservative; T cost can rise. Gold and final accuracy need grading.")
        return
    synth = make_synth(args.synth)
    cache = {}
    disk_caches = {}
    if not args.no_cache:
        import hashlib
        from circuit_cache import CircuitCache
        if args.synth == "cyclosynth":
            from importlib.machinery import EXTENSION_SUFFIXES
            module_dir = Path(require_cyclosynth().__file__).parent
            native = sorted(path for path in module_dir.iterdir()
                            if path.name.endswith(tuple(EXTENSION_SUFFIXES)))
            if not native:
                raise RuntimeError("cannot fingerprint the installed cyclosynth extension")
            backend_digest = hashlib.sha256(b"".join(path.read_bytes() for path in native)).hexdigest()
        else:
            from importlib.metadata import version
            backend_digest = version("pygridsynth")
        for n in g["TEST_SIZES"]:
            signature = {
                "synthesis_revision": "paired_even_t_gate_count_v2",
                "backend": args.synth, "backend_digest": backend_digest,
                "eps": word_epsilon(n, args.eps, args.error_budget),
                "grader_helpers": hashlib.sha256(src.split("def submission_circ")[0].encode()).hexdigest(),
            }
            if args.mux_method != "baseline" or args.fold_common:
                signature.update(synthesis_revision="paired_phase11_fold_v1",
                                 mux_method=args.mux_method, fold_common=args.fold_common)
            disk_caches[n] = CircuitCache(args.cache_dir, signature)
    def submission_circ(u_mat):
        key = np.asarray(u_mat).tobytes()
        if key not in cache:
            n = int(round(np.log2(len(u_mat))))
            t0 = time.time()
            disk_cache = disk_caches.get(n)
            saved = disk_cache.load(u_mat) if disk_cache is not None else None
            if saved is None:
                if args.cache_only:
                    raise RuntimeError(f"No checkpoint for n={n} at these settings; "
                                       "--cache-only prevented a new synthesis run")
                gates, gph = synthesize(np.array(u_mat, dtype=complex), g["cs_decomposition"],
                                        synth, args.eps, verbose=args.verbose, optimize=False,
                                        error_budget=args.error_budget, mux_method=args.mux_method,
                                        fold_common=args.fold_common)
                if disk_cache is not None:
                    disk_cache.save(u_mat, gates, gph)
                print(f"  [n={n}] synthesized in {time.time()-t0:.0f}s", flush=True)
            else:
                gates, gph = saved
                print(f"  [n={n}] reused synthesized circuit checkpoint", flush=True)
            gates, gph = optimize_circuit(gates, gph, verbose=args.verbose)
            cache[key] = (gates, gph)
        gates, gph = cache[key]
        queue_ops(gates, gph, qp)
    g["submission_circ"] = submission_circ

    import json
    cases = [(f"[{args.seed}, {args.samples}]", '{"required_grade": "Gold"}')] if args.seed is not None else None
    if cases is None:
        tc = src.split("test_cases = [")[1].split("]\n")[0]
        cases = [(a, b) for a, b in __import__("re").findall(r"\('(\[.*?\])',\s*'(\{.*?\})'\)", tc)]
    failed = False
    for inp, exp in cases:
        print(f"Running test case {inp} ...", flush=True)
        t0 = time.time()
        out = g["run"](inp)
        print(out, flush=True)
        try:
            g["check"](out, exp); print(f"  -> PASS ({time.time()-t0:.0f}s)")
        except AssertionError as e:
            failed = True
            print(f"  -> FAIL: {e}  ({time.time()-t0:.0f}s)")
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
