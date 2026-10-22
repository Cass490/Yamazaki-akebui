# Yamazaki-akebui

Local Clifford+T unitary synthesis using parity-controlled paired words,
an optional 11-mask factorization with output-phase carry, and common-word
folding. The implementation retains the 15-mask construction as a fallback.

## Completed public run

The public test run has finished. Its final public case (`[7, 2]`, two
samples at each size) received **Gold** with the following averaged costs:

| Qubits | Cost | Baseline | Saving |
| ---: | ---: | ---: | ---: |
| 2 | 315 | 660 | 52.2% |
| 3 | 1,425 | 3,168 | 55.0% |
| 4 | 5,869.5 | 13,728 | 57.2% |
| 5 | 18,048 | 57,024 | **68.3%** |

The grader reported maximum entrywise error **5.4e-9** and zero auxiliary
wires. Its final five-qubit sample cleaned from 95,124 to **66,326** gates,
including GlobalPhase. That public case took **1,543 seconds**; this is not
the total runtime of both public cases. The exact report and run settings
are retained in [`results/public_seed7.json`](results/public_seed7.json).
The first public case's complete final JSON was not captured here, so this
table specifically documents the final case rather than combining reports.

The score is the grader's **weighted cost**, `N_T + 44*N_rot`, not a measured
pure T-count improvement against the best published compiler. In particular,
the five-qubit circuits retain 15 arbitrary RZ rotations.

### Initial single-sample result

One five-qubit sample, run locally with seed `2026`, one sample, and per-word
epsilon `1e-9`, received **Gold** from the included grader:

| Cost | Baseline | Saving | Maximum entrywise error | Emitted gates | Auxiliary wires |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18,057 | 57,024 | 68.3% | 5.8e-9 | 66,121 | 0 |

Synthesis took 547 seconds; the reported case finished in 556 seconds.
Separate smoke samples at two and three qubits saved 50.3% and 54.7%.
The single-sample and smoke measurements use different target selections
from the averaged public case above; they should not be combined as one dataset.

## Setup

The measured environment used Python **3.10.19**. Use Python 3.10 for the
pinned dependencies. Building the native dependency also requires Git, Rust
with rustup (the upstream toolchain is pinned), a C build toolchain, `make`,
`m4`, and GMP/MPFR development dependencies.

```bash
git clone https://github.com/Cass490/Yamazaki-akebui.git
cd Yamazaki-akebui
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/setup_cyclosynth.sh
```

The setup script clones [cyclosynth](https://github.com/mtweiden/cyclosynth)
at the revision in `native/cyclosynth-base.txt`, applies
`native/cyclosynth-even-t.patch`, and builds it into the active venv. The
patch exposes the native `Synthesizer(even_t=True)` constraint used by paired
words. Installing an unpatched upstream build is insufficient for this code.
If Rust is installed but unavailable in the shell, source `~/.cargo/env` first.

The source checkout, compiled extension, virtual environment, and generated
caches are deliberately excluded from this repository. The setup script
does not run the benchmark or native tests.

## Run

Inspect the optimized configuration without synthesis or grading:

```bash
python gold_local.py --mux-method phase11 --fold-common --eps 1e-9 --sizes 5 --dry-run
```

Run the smoke test first, including real grader checks at sizes 2 and 3:

```bash
python smoke_test.py --mux-method phase11 --fold-common --eps 1e-9
```

Then run one five-qubit sample:

```bash
python -u gold_local.py --mux-method phase11 --fold-common --eps 1e-9 --sizes 5 --seed 2026 --samples 1 --verbose
```

To check all public cases, across sizes 2 through 5:

```bash
python -u gold_local.py --mux-method phase11 --fold-common --eps 1e-9 --verbose
```

The full run can take around an hour, depending on hardware and available
checkpoints. The default `.run_cache/` preserves complete raw synthesized
circuits before cleanup and grading. Identical targets and settings reuse
those checkpoints; changing precision or the optimization flags selects a
different cache entry. `--cache-only` stops on a miss rather than synthesizing.
There is no checkpoint inside an unfinished circuit.

`--eps` is a **per-word** tolerance, not a guarantee on final circuit error.
The optional `--error-budget` allocation is conservative and can substantially
increase cost; it was not used for the reported Gold sample. Final acceptance
always comes from the grader's full matrix comparison and cost checks.

At five qubits, `phase11` plans 376 source words, with up to 496 if all reduced
factorizations fall back. The final multiplexor retains the original form to
avoid extra arbitrary rotations in the final diagonal. Common-word folding
also works at the smaller sizes. No Gold percentage is guaranteed for new
targets.

## Files and focused checks

- `gold_local.py`: canonical local synthesis runner.
- `phase_factorization.py`: bounded 11-mask continuation solver and explicit failures.
- `gate_cleanup.py`: exact gate cleanup with scalar-phase tracking.
- `circuit_cache.py`: validated JSON checkpoints.
- `challenge.py`: included local grader/template; its PennyLane decomposition
  call uses the installed `qp.transforms.decompose` namespace. The optimized
  circuit is supplied by the runner; it is not inserted into `challenge.py`.
- `smoke_test.py`: backend conventions, algebra, factorization, and small-size grading.
- `phase_factor_probe.py`: standalone NumPy rank/continuation research checks.
- `patch_parity.py`: read-only source and installed native-API checks.
- `gold_local_fix.py`: compatibility alias for the canonical runner.

The following focused checks do not perform lattice synthesis:

```bash
python -B -m unittest -v test_phase11 test_precision_budget test_gate_cleanup test_even_word.RunnerWiringRegression
```

They passed 29 tests locally. Native/backend regressions and Rust checks are
separate, potentially slower commands:

```bash
python -B -m unittest -v test_even_word
cd cyclosynth
cargo test --release --lib
cargo test --release --lib -- --ignored deep_eps_canary
```

The native checks above are provided for reproduction; they were not rerun
as part of packaging this repository.

## Attribution

The controlled-gate construction builds on Yamazaki, Akibue, and Sano,
[Multi-qubit controlled gate synthesis without T-count overhead in the small-error limit](https://arxiv.org/abs/2603.14202v3).
The original two-author version has the title
[Multi-qubit controlled gate with optimal T-count](https://arxiv.org/abs/2603.14202v1).
The reduced phase-gauged numerical factorization here is an experimental
implementation choice, not a claim of universal 11-mask solvability.

The native dependency is maintained upstream at
[mtweiden/cyclosynth](https://github.com/mtweiden/cyclosynth). Its source license
is retained in `third_party/cyclosynth-LICENSE`; the tracked patch contains
the local even-T modifications against the pinned upstream revision.
