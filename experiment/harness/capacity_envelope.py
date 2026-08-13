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


def crop_grid(cells, m_kv_bytes: int, c_kv: int, q: int):
    """Split grid cells into (feasible, infeasible) per the locked rule.

    cells: iterable of dicts with integer keys 'concurrency' and 'isl'.
    Returns (feasible, infeasible) lists; caller logs the infeasible list
    verbatim into the run manifest (INFEASIBLE-BY-CAPACITY, not censored).
    """
    feasible, infeasible = [], []
    for c in cells:
        limit = b_max(m_kv_bytes, c_kv, c["isl"], q)
        (feasible if c["concurrency"] <= limit else infeasible).append(
            {**c, "b_max": limit})
    return feasible, infeasible


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
