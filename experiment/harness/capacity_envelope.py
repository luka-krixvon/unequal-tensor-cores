"""Capacity-feasibility envelope — implements preregistration §3.5.

The nominal decode grid (concurrency x ISL) contains cells whose resident KV
cache cannot physically fit. Rule (locked, pre-data):

  B_max(S) = floor( M_KV / (c_KV * S * q) )

where M_KV is the ENGINE-REPORTED usable KV memory in bytes measured after
weight loading on the actual machine (never the spec VRAM), c_KV is the
model's KV bytes/token at q=1, S the context length, q the KV bytes/value
(1 = FP8 KV, 2 = BF16 KV).

Cells with B > B_max(S) are INFEASIBLE-BY-CAPACITY: removed from BOTH arms,
logged separately, never counted as censored or failed, never entered into
estimation. Paired analysis runs on the intersection of both arms' feasible
regions. Held-out points falling infeasible are recorded, excluded from the
scoring denominator, and counted (never moved or replaced).

c_KV for pinned models:
  Qwen3-32B : 64 layers * 2 (K,V) * 8 KV heads * 128 head_dim = 131072 B/token
  Llama-3.1-405B: 126 layers * 2 * 8 * 128 = 258048 B/token

Pure standard-library Python. Locked by SHA-256 via amendment A-5.
"""
from __future__ import annotations

C_KV = {
    "qwen3-32b": 64 * 2 * 8 * 128,        # 131072
    "llama-405b": 126 * 2 * 8 * 128,      # 258048
}


def b_max(m_kv_bytes: int, c_kv: int, isl: int, q: int) -> int:
    """Max feasible concurrency for one (model, KV dtype, context) cell."""
    if m_kv_bytes <= 0:
        return 0
    return int(m_kv_bytes // (c_kv * isl * q))


def load_axis(spec):
    """Parse a generated run spec's load field into (axis, integer value).

    P0: crop_grid previously required integer keys 'concurrency'/'isl', but
    gen_sweep emits load='concurrency=1' and isl='128' as strings, so the
    envelope could not consume the pipeline it was written for. Normalise here
    instead of expecting callers to hand-massage every spec.
    """
    load = spec.get("load")
    if isinstance(load, str) and "=" in load:
        axis, _, val = load.partition("=")
        return axis, int(val)
    return "concurrency", int(load)


def resident_context(spec) -> int:
    """Context length whose KV must be resident. Decode holds ISL+OSL, not ISL.

    P0: the envelope previously sized decode cells on ISL alone, understating
    resident KV by the whole generated sequence.
    """
    isl = int(spec["isl"]) if str(spec.get("isl", "")).isdigit() else 0
    osl = int(spec["osl"]) if str(spec.get("osl", "")).isdigit() else 0
    return isl + (osl if "decode" in str(spec.get("phase", "")) else 0)


def crop_grid(cells, m_kv_bytes: int, c_kv: int, q: int, block: int = 16,
              tp: int = 1):
    """Split cells into (feasible, infeasible) per the locked rule.

    Accepts either integer-keyed cells ({'concurrency','isl'}) or generated run
    specs (load='concurrency=N', isl='128', osl=..., phase=...). KV is rounded
    up to whole paged-attention blocks and divided by the tensor-parallel
    degree, since each rank holds only its shard of the KV heads.
    """
    feasible, infeasible = [], []
    for c in cells:
        if "concurrency" in c and isinstance(c.get("isl"), int):
            axis, val, S = "concurrency", c["concurrency"], c["isl"]
        else:
            axis, val = load_axis(c)
            S = resident_context(c)
        if axis not in ("concurrency", "batch") or S <= 0:
            feasible.append({**c, "b_max": None,
                             "capacity_note": "not-capacity-bound"})
            continue
        S_blocks = -(-S // block) * block          # round up to block multiple
        limit = b_max(m_kv_bytes * max(tp, 1), c_kv, S_blocks, q)
        rec = {**c, "b_max": limit, "resident_context": S_blocks}
        if val <= limit:
            feasible.append(rec)
        else:
            infeasible.append({**rec, "reason": "INFEASIBLE-BY-CAPACITY"})
    return feasible, infeasible


def paired_feasible(cells_a, cells_b, *args, **kw):
    """Intersection of two arms' feasible regions, per preregistration 3.5.

    P0: the envelope only ever cropped a single arm, but the protocol requires
    paired analysis on the INTERSECTION (a BF16-weights arm has a strictly
    smaller feasible region than a W8 arm). Returns
    (paired_feasible_a, paired_feasible_b, ledger) where ledger records every
    dropped cell with the arm that made it infeasible.
    """
    fa, ia = crop_grid(cells_a, *args, **kw)
    fb, ib = crop_grid(cells_b, *args, **kw)
    def key(c):
        return (c.get("experiment_id"), c.get("isl"), c.get("osl"),
                c.get("load"), c.get("concurrency"))
    ok = {key(c) for c in fa} & {key(c) for c in fb}
    ledger = ([{**c, "dropped_by": "arm_a"} for c in ia] +
              [{**c, "dropped_by": "arm_b"} for c in ib] +
              [{**c, "dropped_by": "not-in-intersection",
                "reason": "INFEASIBLE-BY-CAPACITY"}
               for c in fa + fb if key(c) not in ok])
    return ([c for c in fa if key(c) in ok],
            [c for c in fb if key(c) in ok], ledger)


def spec_preview(vram_gb: float, weights_gb: float, c_kv: int, q: int,
                 isls=(2048, 8192, 32768), workspace_factor: float = 1.10):
    """Pre-data spec-based preview (documentation only; NEVER a substitute
    for the measured M_KV). Model matches the §3.5 registered expectation:
    M_KV = VRAM - workspace_factor * weights."""
    m_kv = (vram_gb - workspace_factor * weights_gb) * 1e9
    return {isl: b_max(int(m_kv), c_kv, isl, q) for isl in isls}


if __name__ == "__main__":
    # Self-test: reproduce the §3.5 registered pre-data expectations.
    ck = C_KV["qwen3-32b"]
    b300 = spec_preview(270, 33, ck, q=1)
    h200 = spec_preview(141, 33, ck, q=1)
    print("B300 270GB W8 q=1:", b300)   # §3.5 registered: ≈867/216/54
    print("H200 141GB W8 q=1:", h200)   # §3.5 registered: ≈394/98/24
    assert b300[32768] < 64 and h200[32768] < 32, "envelope sanity"
    f, inf = crop_grid(
        [{"concurrency": b, "isl": s} for b in (1, 64, 256)
         for s in (2048, 32768)],
        int(200e9), ck, 1)
    assert any(c["concurrency"] == 256 and c["isl"] == 32768 for c in inf)
    print("self-test OK:", len(f), "feasible /", len(inf), "infeasible")
