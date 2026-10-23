# QHack 2026 Q8: Unitary Synthesis

A local compiler for the unitary-synthesis challenge, using parity-controlled
Clifford+T paired words, output-phase carry, and common-word folding.
The entry point is [`gold_local.py`](gold_local.py).

## Recorded result

The included challenge grader returned **Gold** for public case `[7, 2]`
(two targets at each size):

| Qubits | Mean cost | Baseline | Saving |
| ---: | ---: | ---: | ---: |
| 2 | 315 | 660 | 52.2% |
| 3 | 1,425 | 3,168 | 55.0% |
| 4 | 5,869.5 | 13,728 | 57.2% |
| 5 | 18,048 | 57,024 | **68.3%** |

The maximum entrywise error across that case was reported as **5.4e-9**, with
zero auxiliary wires. The final five-qubit sample contained **66,326 gates**
after cleanup, including GlobalPhase. The case took **1,543 seconds
(25 minutes 43 seconds)**; this is not the duration of both public cases.

The transcribed grader output, command, and measurement scope are in
[`results/public_seed7.json`](results/public_seed7.json). The complete first
case report and raw terminal logs are not included.

The score is `N_T + 44*N_rot`, as defined by the challenge. Five-qubit circuits
retain **15 arbitrary RZ rotations**; this result is a weighted-cost saving,
not a pure Clifford+T T-count comparison. Final accuracy includes global phase.

## Method

1. Recursive cosine-sine decomposition produces a network of uniformly
   controlled one-qubit gates.
2. Joint SU(2) synthesis supplies Clifford+T words using
   [cyclosynth](https://github.com/mtweiden/cyclosynth), patched to constrain
   paired words to even T count.
3. Paired-word compilation shares aligned gate layers. Common-word folding
   incorporates the final common gate into a directly compiled branch pair.
4. At five qubits, eligible multiplexors use an ordered 11-mask numerical
   factorization with output Z phases carried into subsequent blocks.
   Numerical failures use the original 15-mask factorization. The final
   multiplexor also retains 15 masks to avoid additional arbitrary rotations.
5. Exact Clifford rewrites and phase-aware cancellation reduce physical gate
   count before the grader simulates the full circuit.

The compiler synthesizes the supplied matrix rather than selecting a stored
public-target circuit. Optional checkpoints reuse completed synthesis work.
The reduced factorization is not proved to cover every target, and the
recorded score is not a guarantee for unseen seeds.

## Setup

The recorded environment used **Python 3.10.19**. Use Python 3.10 with the
pinned dependencies. Native compilation requires Git, Rust/rustup, a C build
toolchain, `make`, `m4`, and GMP/MPFR development packages.

```bash
git clone https://github.com/Cass490/Yamazaki-akebui.git
cd Yamazaki-akebui
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/setup_cyclosynth.sh
```

The setup script checks out the revision pinned in
[`native/cyclosynth-base.txt`](native/cyclosynth-base.txt), applies the even-T
patch, builds into the active virtual environment, and checks the installed
API. It does not run the benchmark. An unpatched upstream installation does
not provide the required `even_t=True` option. If needed, load the Rust shell
environment with `source "$HOME/.cargo/env"` before setup.

## Run

Start with the small-size smoke test:

```bash
python smoke_test.py --mux-method phase11 --fold-common --eps 1e-9
```

Run a single five-qubit target:

```bash
python -u gold_local.py --mux-method phase11 --fold-common --eps 1e-9 --sizes 5 --seed 2026 --samples 1 --verbose
```

Run both public cases at sizes 2 through 5 with the recorded settings:

```bash
python -u gold_local.py --mux-method phase11 --fold-common --eps 1e-9 --verbose
```

`--eps` is a per-word tolerance, not a whole-circuit error bound. The optional
`--error-budget` policy is more conservative and was not used for this result.
The grader's final matrix comparison determines accuracy.

Completed raw circuits are saved in `.run_cache/`. Add `--no-cache` to force
fresh synthesis, or `--cache-only` to reject missing checkpoints. Add
`--dry-run` to inspect planned word counts without synthesis or grading.
No checkpoint is saved inside an unfinished circuit.

This implementation requires a compiled dependency and runs outside the
platform's Python submission block. It has not demonstrated compliance with
the platform's 300-second limit.

## Validation and source layout

Focused checks that exclude native lattice synthesis:

```bash
python -B -m unittest -v test_phase11 test_precision_budget test_gate_cleanup test_even_word.RunnerWiringRegression
```

Separate native/backend checks:

```bash
python -B -m unittest -v test_even_word
cd cyclosynth
cargo test --release --lib
cargo test --release --lib -- --ignored deep_eps_canary
```

- `gold_local.py`: compiler and local grader runner.
- `phase_factorization.py`: bounded 11-mask solver.
- `gate_cleanup.py`: exact gate reduction with scalar-phase tracking.
- `circuit_cache.py`: validated JSON checkpoints.
- `challenge.py`: challenge template and grader. Its decomposition call uses
  `qp.transforms.decompose` for PennyLane 0.42.3 compatibility; scoring and
  accuracy thresholds are unchanged. The runner supplies the optimized circuit.
- `smoke_test.py` and `test_*.py`: integration and component checks.
- `native/` and `scripts/`: pinned native patch, setup, and API verification.

## References and attribution

- Yamazaki, Akibue, and Sano:
  [Multi-qubit controlled gate synthesis without T-count overhead in the small-error limit](https://arxiv.org/abs/2603.14202v3).
- Morisaki, Sano, and Akibue:
  [Optimal ancilla-free Clifford+T synthesis for general single-qubit unitaries](https://arxiv.org/abs/2510.05816).
- Shende, Bullock, and Markov:
  [Synthesis of Quantum Logic Circuits](https://arxiv.org/abs/quant-ph/0406176v5).

The native dependency is maintained at
[mtweiden/cyclosynth](https://github.com/mtweiden/cyclosynth). Its license is
retained in [`third_party/cyclosynth-LICENSE`](third_party/cyclosynth-LICENSE).
