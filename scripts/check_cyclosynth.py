#!/usr/bin/env python3
"""Check the source and installed native even-T API without synthesis.

Usage: python scripts/check_cyclosynth.py [path/to/cyclosynth]
"""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "cyclosynth",
        help="cyclosynth checkout (default: cyclosynth in the project root)",
    )
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    paths = (
        repo / "src/synthesis/synthesizer.rs",
        repo / "src/synthesis/clifford_t/mod.rs",
    )
    if not all(path.is_file() for path in paths):
        parser.error(f"not a cyclosynth checkout: {repo}")
    unified, backend = (path.read_text(encoding="utf-8") for path in paths)
    if "WANT_T_PARITY" in backend:
        print("FAIL: legacy process-global parity patch detected; use the instance-owned even_t API.")
        return 1
    if "fn with_even_t(" not in unified or "even_t: bool" not in backend:
        print("FAIL: this checkout does not declare the native even_t implementation.")
        print("No patch applied. Use the checkout containing the even_t Rust/Python API.")
        return 1
    print(f"PASS source API declarations: {repo}")
    try:
        import cyclosynth
        cls = getattr(cyclosynth, "Synthesizer", None)
        if cls is None or not hasattr(cls, "even_t"):
            raise RuntimeError("installed module does not expose even_t")
        configured = cls(1e-3, even_t=True)
        if not configured.even_t:
            raise RuntimeError("installed module did not retain even_t=True")
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        print(f"FAIL installed API: {exc}")
        print("Rebuild this checkout into the Python environment running this command.")
        return 1
    print(f"PASS installed API: {cyclosynth.__file__}")
    print("PASS even_t=True configuration; no synthesis or file changes performed.")
    print("Next: python -m unittest -v test_even_word && python smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
