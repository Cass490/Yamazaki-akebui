"""Local JSON checkpoints for expensive synthesized circuits; no pickle or execution."""
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np


class CircuitCache:
    def __init__(self, directory, signature):
        self.directory = Path(directory).expanduser().resolve()
        self.signature = json.dumps(signature, sort_keys=True, allow_nan=False).encode()

    def key(self, target):
        matrix = np.ascontiguousarray(target, dtype=np.complex128)
        digest = hashlib.sha256(self.signature)
        digest.update(str(matrix.shape).encode())
        digest.update(matrix.tobytes())
        return digest.hexdigest()

    def load(self, target):
        key = self.key(target)
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["schema"] != 1 or record["key"] != key:
                raise ValueError("checkpoint identity mismatch")
            phase = float(record["global_phase"])
            if not math.isfinite(phase):
                raise ValueError("non-finite phase")
            n = int(round(math.log2(len(target))))
            gates = []
            for gate in record["gates"]:
                name, wires = gate[:2]
                if name not in {"H", "S", "T", "X", "Y", "Z", "CNOT", "RZ"}:
                    raise ValueError("unsupported gate")
                count = 2 if name == "CNOT" else 1
                if len(wires) != count or len(set(wires)) != count:
                    raise ValueError("invalid wire count")
                if any(type(wire) is not int or not 0 <= wire < n for wire in wires):
                    raise ValueError("invalid wire")
                if len(gate) != (3 if name == "RZ" else 2):
                    raise ValueError("invalid gate parameters")
                if name == "RZ":
                    angle = float(gate[2])
                    if not math.isfinite(angle):
                        raise ValueError("non-finite rotation")
                    gates.append((name, tuple(wires), angle))
                else:
                    gates.append((name, tuple(wires)))
            return gates, phase
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid circuit checkpoint {path}: {exc}; remove it or use --no-cache") from exc

    def save(self, target, gates, global_phase):
        self.directory.mkdir(parents=True, exist_ok=True)
        key = self.key(target)
        record = {"schema": 1, "key": key, "gates": gates, "global_phase": float(global_phase)}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=f"{key}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, allow_nan=False, separators=(",", ":"))
                stream.write("\n")
            temporary.replace(self.directory / f"{key}.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
