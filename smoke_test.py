#!/usr/bin/env python3
"""Smoke tests for gold_local.py. Usage: python smoke_test.py
The grader defaults to challenge.py beside this script; --template overrides it.
Each stage prints PASS/FAIL. Stages 1-5 take ~1-3 minutes; stage 6 (real small-size score) ~5-10 minutes."""
import argparse, sys, time
from pathlib import Path
import numpy as np
import gold_local as G

failures = []
def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        failures.append(msg)
    return cond

ap = argparse.ArgumentParser()
ap.add_argument("--template", type=G.resolve_template,
                default=Path(__file__).resolve().with_name("challenge.py"),
                help="grader file (default: challenge.py beside this script)")
ap.add_argument("--skip-score", action="store_true", help="skip stage 6")
ap.add_argument("--mux-method", choices=["baseline", "phase11"], default="baseline")
ap.add_argument("--fold-common", action="store_true")
ap.add_argument("--eps", type=float, default=4e-9, help="per-word tolerance for grader stages")
args = ap.parse_args()
if not args.template.is_file():
    ap.error(f"template file not found: {args.template}")
rng = np.random.default_rng(123)
def haar(d):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))
def haar_su2():
    U = haar(2); return U / np.sqrt(np.linalg.det(U))

print("[1] imports")
mods = {}
for m in ("numpy", "pennylane", "pygridsynth", "mpmath", "cyclosynth"):
    try:
        module = G.require_cyclosynth() if m == "cyclosynth" else __import__(m)
        mods[m] = module; ok(True, f"import {m}")
    except Exception as e:
        ok(False, f"import {m}: {e}")
have_cyc = "cyclosynth" in mods
ok(True, "import gold_local")
kind = "cyclosynth" if have_cyc else "pygridsynth"
if not have_cyc:
    print("  NOTE  cyclosynth missing -> later stages use the pygridsynth stand-in (correct, but long words)")

print(f"\n[2] single-qubit synthesis ({kind})")
synth = G.make_synth(kind)
allgood = True; ts = []; t0 = time.time()
for _ in range(5):
    U = haar_su2()
    w = synth(U, 1e-8)              # make_synth raises if the angle convention is wrong
    err = G.word_error(w, U)
    allgood &= err <= 1e-8 + 1e-12; ts.append(G.tcount(w))
ok(allgood, f"5 random targets at eps=1e-8 match (convention OK); T-counts {ts}, {(time.time()-t0)/5:.1f}s each")
if have_cyc:
    ok(70 <= np.mean(ts) <= 82, f"optimal-ish T-count (expect ~76 at 1e-8, got mean {np.mean(ts):.1f})")
w = G.even_word(haar_su2(), synth, 1e-8)
ok(G.tcount(w) % 2 == 0, f"even-T word helper (T={G.tcount(w)})")

print("\n[3] exact word algebra")
rw = "".join(rng.choice(list("HSTXYZst"), size=60))
e, ph = G.skeleton(rw)
ok(np.allclose(G.mat(G.skeleton_word(e, ph)), G.mat(rw), atol=1e-12), "skeleton rewrite exact")
ok(np.allclose(G.mat(G.ma_normal_form(rw)), G.mat(rw), atol=1e-10), "Matsumoto-Amano normal form exact (pygridsynth)")

print("\n[4] parity factorisation (16-block multiplexor)")
V = np.array([haar_su2() for _ in range(16)])
t0 = time.time()
R, betas, masks, z, _, method = G.factor_mux_blocks(V, args.mux_method, verbose=True)
P = np.array([[(x & m).bit_count() % 2 for m in masks] for x in range(16)])
Mx = np.array(G.model(R, betas, P))
Mx[:, 0, :] *= np.exp(-1j * z[:, None])
Mx[:, 1, :] *= np.exp(1j * z[:, None])
err = max(np.abs(Mx[x] - V[x]).max() for x in range(16))
ok(err < 1e-12, f"{method}, {len(masks)} masks: reconstructs to {err:.1e} ({time.time()-t0:.1f}s)")

print("\n[5] paired-word multiplexor, simulated exactly (2 controls)")
V = np.array([haar_su2() for _ in range(4)])
t0 = time.time()
res = G.synth_mux(V, synth, 1e-8, mux_method=args.mux_method, fold_common=args.fold_common)
gates, gph = G.mux_gates(res, target=2, controls=[0, 1])
Msim = G.simulate(gates, 3) * np.exp(1j * gph)
D = np.zeros((8, 8), complex)
for x in range(4):
    D[2*x:2*x+2, 2*x:2*x+2] = (
        np.diag(np.exp(1j * np.array([res["output_z"][x], -res["output_z"][x]])))
        @ V[x] * np.exp(-1j * res["ctrl_phase"][x]))
err = np.abs(Msim - D).max()
nT = sum(g[0] == "T" for g in gates)
ok(err < 1e-7, f"multiplexor matches target to {err:.1e}; T={nT} ({time.time()-t0:.0f}s)")
ok(all(g[0] in ("H","S","T","X","Y","Z","CNOT") for g in gates), "only Clifford+T gates emitted")

if args.skip_score:
    sys.exit(bool(failures))
print("\n[6] real grader on sizes 2 and 3 (one sample each)")
src = args.template.read_text(encoding="utf-8")
g = {"__name__": "grader"}; exec(src.split("def submission_circ")[0], g)
for n in (2, 3):
    U = g["haar_unitary"](n, 777 + n)
    t0 = time.time()
    gs, gp = G.synthesize(U, g["cs_decomposition"], synth, args.eps,
                          mux_method=args.mux_method, fold_common=args.fold_common)
    tape = g["compile_circuit"](lambda u: G.queue_ops(gs, gp, g["qp"]), U)
    errn = float(np.abs(g["tape_matrix"](tape, n) - U).max()); cost = g["tape_cost"](tape, n); base = g["baseline_cost"](n)
    ok(errn <= 1e-8, f"n={n}: grader max error {errn:.1e} (<= 1e-8)")
    sav = 100 * (1 - cost / base)
    if have_cyc:
        ok(sav > 0, f"n={n}: cost {cost} vs baseline {base} -> saving {sav:.1f}%  [{time.time()-t0:.0f}s]")
    else:
        print(f"  INFO  n={n}: cost {cost} vs baseline {base} (stand-in words are ~2.7x longer, so over baseline is expected)")
print("\nDone. " + ("FAILures remain; do not run the full job." if failures
                    else "All smoke stages PASS. Next check one five-qubit sample."))
sys.exit(bool(failures))
