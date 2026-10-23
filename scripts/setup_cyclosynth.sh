#!/usr/bin/env bash
# Clone the pinned upstream, apply this project's even-T patch, and build.
# Does not run synthesis, Rust tests, or the challenge benchmark.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${PYTHON:-python}"
task_python="$("$task_python" -c 'import sys; print(sys.executable)')"
task_venv="$("$task_python" -c 'import sys; print(sys.prefix if sys.prefix != sys.base_prefix else "")')"
if [[ -z "$task_venv" ]]; then
    echo 'Activate a Python virtual environment and install requirements.txt first.' >&2
    exit 1
fi

task_source="${CYCLOSYNTH_SOURCE_DIR:-$task_root/cyclosynth}"
task_pin="$(tr -d '\r\n' < "$task_root/native/cyclosynth-base.txt")"
task_patch="$task_root/native/cyclosynth-even-t.patch"
if [[ ! -e "$task_source" ]]; then
    git clone --no-checkout https://github.com/mtweiden/cyclosynth.git "$task_source"
    git -C "$task_source" checkout --detach "$task_pin"
elif [[ ! -d "$task_source/.git" ]]; then
    echo "Existing path is not a standalone Git checkout: $task_source" >&2
    exit 1
fi

task_head="$(git -C "$task_source" rev-parse HEAD)"
if [[ "$task_head" != "$task_pin" ]]; then
    if [[ -n "$(git -C "$task_source" status --porcelain)" ]]; then
        echo "Checkout has changes at another revision; preserving it: $task_source" >&2
        exit 1
    fi
    git -C "$task_source" checkout --detach "$task_pin"
fi

if git -C "$task_source" apply --reverse --check "$task_patch" 2>/dev/null; then
    echo 'Even-T patch is already applied.'
elif git -C "$task_source" apply --check "$task_patch"; then
    git -C "$task_source" apply "$task_patch"
else
    echo 'Patch does not apply cleanly; existing files were preserved.' >&2
    exit 1
fi

echo "Building cyclosynth into $task_venv (this can take several minutes)."
(
    cd -- "$task_source"
    # A venv created from Conda can inherit both variables; maturin needs one.
    env -u CONDA_PREFIX VIRTUAL_ENV="$task_venv" "$task_python" -m maturin develop --release
)
"$task_python" "$task_root/scripts/check_cyclosynth.py" "$task_source"
