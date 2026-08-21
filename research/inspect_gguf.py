"""Read a GGUF header — tensor names, shapes, and metadata. Header only.

Used to answer "how does MiniCPM actually do it" from the artifact rather than
from memory. Never loads weights, so it is instant on a 9 GB file.

    python research/inspect_gguf.py ~/models/MiniCPM-o-4_5-gguf/audio/*.gguf
    python research/inspect_gguf.py <file> --filter proj

ponytail: 80 lines of struct.unpack instead of a gguf/llama-cpp dependency.
We only need the header, and the format is stable.
"""

import struct
import sys
from pathlib import Path

# GGUF value type enum.
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIXED = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i",
          F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 12: "Q3_K", 13: "Q4_K", 14: "Q5_K",
              15: "Q6_K", 16: "Q8_K", 30: "BF16"}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.f.read(n))[0]

    def string(self):
        return self.f.read(self.raw("Q")).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _FIXED:
            return self.raw(_FIXED[vtype])
        if vtype == STR:
            return self.string()
        if vtype == ARR:
            etype, n = self.raw("I"), self.raw("Q")
            vals = [self.value(etype) for _ in range(n)]
            return vals if len(vals) <= 8 else vals[:8] + ["..."]
        raise ValueError(f"unknown gguf value type {vtype}")


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {magic!r}")
        version, n_tensors, n_kv = r.raw("I"), r.raw("Q"), r.raw("Q")

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.raw("I"))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.raw("Q") for _ in range(r.raw("I"))]
            ttype, _offset = r.raw("I"), r.raw("Q")
            tensors.append((name, dims, GGML_TYPES.get(ttype, str(ttype))))

    return version, meta, tensors


def main():
    argv = sys.argv[1:]
    filt, args, all_meta = None, [], False
    i = 0
    while i < len(argv):
        if argv[i] == "--all-meta":
            all_meta = True
            i += 1
        elif argv[i] == "--filter":
            filt = argv[i + 1]
            i += 2          # consume the value too, or it reads as a path
        else:
            args.append(argv[i])
            i += 1

    for path in args:
        version, meta, tensors = read_header(path)
        print(f"\n=== {Path(path).name}  (gguf v{version}, {len(tensors)} tensors) ===")

        for k, v in meta.items():
            if k.startswith("tokenizer.ggml.token"):
                continue                      # 150k-entry vocab arrays
            if all_meta or any(t in k for t in (
                    "arch", "name", "audio", "vision", "projector", "embedding",
                    "block_count", "head", "context", "type", "pool", "token")):
                print(f"  {k:<44} {v}")

        shown = [t for t in tensors if not filt or filt in t[0]]
        print(f"\n  {len(shown)} tensors" + (f" matching '{filt}'" if filt else ""))
        params = 0
        for name, dims, ttype in shown:
            n = 1
            for d in dims:
                n *= d
            params += n
            print(f"    {name:<52} {str(dims):<22} {ttype}")
        print(f"\n  params shown: {params/1e6:.1f}M")


if __name__ == "__main__":
    main()


# --- tensor extraction -------------------------------------------------------
# Reading actual weights, not just the header. Enough to lift a trained
# projector out of a released checkpoint and run it in our own code — which is
# how we verify the architecture port is correct rather than assuming it.

_GGML_BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2)}


def read_tensor(path, name: str):
    """Load one F32/F16/BF16 tensor by name. Quantised types are refused —
    dequantising them correctly is a whole subsystem and silently guessing
    would produce weights that look plausible and are wrong."""
    import numpy as np

    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _version, n_tensors, n_kv = r.raw("I"), r.raw("Q"), r.raw("Q")
        for _ in range(n_kv):
            r.string()
            r.value(r.raw("I"))

        infos, found = [], None
        for _ in range(n_tensors):
            tname = r.string()
            dims = [r.raw("Q") for _ in range(r.raw("I"))]
            ttype, offset = r.raw("I"), r.raw("Q")
            infos.append((tname, dims, ttype, offset))
            if tname == name:
                found = (dims, ttype, offset)

        if found is None:
            raise KeyError(f"{name!r} not in {Path(path).name}; "
                           f"have e.g. {[i[0] for i in infos[:3]]}")
        dims, ttype, offset = found
        kind = GGML_TYPES.get(ttype, str(ttype))
        if kind not in _GGML_BLOCK:
            raise NotImplementedError(
                f"{name} is {kind}; only F32/F16/BF16 are read directly. "
                "Dequantising is out of scope — use an F16 export.")

        # Tensor data starts after the header, aligned to general.alignment
        # (default 32). Offsets in the header are relative to that base.
        base = f.tell()
        align = 32
        base = (base + align - 1) // align * align

        count = 1
        for d in dims:
            count *= d
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[kind]
        f.seek(base + offset)
        raw = np.frombuffer(f.read(count * _GGML_BLOCK[kind][1]), dtype=dtype)
        if kind == "BF16":
            raw = (raw.astype("<u4") << 16).view("<f4")
        # ggml stores ne[0] fastest-varying; numpy wants the reverse.
        return raw.astype("float32").reshape(tuple(reversed(dims)))


def read_tensors(path, names=None):
    """Read many tensors in one pass — header parsed once, dict returned.

    `names=None` reads every F32/F16/BF16 tensor in the file.
    """
    import numpy as np

    want = set(names) if names is not None else None
    out = {}
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _v, n_tensors, n_kv = r.raw("I"), r.raw("Q"), r.raw("Q")
        for _ in range(n_kv):
            r.string()
            r.value(r.raw("I"))
        infos = []
        for _ in range(n_tensors):
            tname = r.string()
            dims = [r.raw("Q") for _ in range(r.raw("I"))]
            ttype, offset = r.raw("I"), r.raw("Q")
            infos.append((tname, dims, ttype, offset))
        base = (f.tell() + 31) // 32 * 32

        for tname, dims, ttype, offset in infos:
            if want is not None and tname not in want:
                continue
            kind = GGML_TYPES.get(ttype, str(ttype))
            if kind not in _GGML_BLOCK:
                continue
            count = 1
            for d in dims:
                count *= d
            dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[kind]
            f.seek(base + offset)
            raw = np.frombuffer(f.read(count * _GGML_BLOCK[kind][1]), dtype=dtype)
            if kind == "BF16":
                raw = (raw.astype("<u4") << 16).view("<f4")
            out[tname] = raw.astype("float32").reshape(tuple(reversed(dims)))
    if want is not None and (missing := want - set(out)):
        raise KeyError(f"missing tensors: {sorted(missing)[:5]}")
    return out
