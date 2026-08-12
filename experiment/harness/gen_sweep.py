"""Expand config_matrix.csv into executable sweep specs (JSONL).

Usage:  python3 gen_sweep.py            # writes sweeps/<experiment_id>.jsonl
        python3 gen_sweep.py --list     # print run counts per experiment

Each output line is one run spec: every axis fixed, plus derived GEMM shapes
for MICRO experiments (from the pinned Qwen3-32B config in this directory and
Llama-3.1-405B dims), repeat index, and a stable run_id for manifests.

This is plumbing only: the executor that consumes these specs (vLLM bench /
GEMM microbench) lands in Phase 2. Keeping generation separate and
deterministic means the sweep itself is part of the pre-registered design.
"""
import csv, json, hashlib, os, sys, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "config_matrix.csv")
OUT = os.path.join(HERE, "sweeps")

# ---- model dims -----------------------------------------------------------
QWEN = json.load(open(os.path.join(HERE, "qwen3_32b_config.json")))
LLAMA405B = {  # from HF config, meta-llama/Llama-3.1-405B-Instruct
    "hidden_size": 16384, "intermediate_size": 53248,
    "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 128,
}

def gemm_shapes(cfg):
    """Per-layer projection GEMMs as (name, N, K); M = active-token axis."""
    h = cfg["hidden_size"]; inter = cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or h // nq
    return [
        ("qkv_proj",  (nq + 2 * nkv) * hd, h),
        ("attn_out",  h,                  nq * hd),
        ("gate_up",   2 * inter,          h),
        ("down_proj", h,                  inter),
    ]

ACTIVE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

def split(v):
    return [s.strip() for s in v.strip('"').split("|")] if "|" in v else [v.strip('"')]

def expand(row):
    """Yield fully-specified run dicts for one matrix row."""
    axes = {
        "format": split(row["format"]),
        "isl": split(row["isl_tokens"]),
        "osl": split(row["osl_tokens"]),
        "load": split(row["load_or_concurrency"]),
    }
    # MICRO rows additionally sweep GEMM shapes x active tokens
    shapes = [None]
    if row["tier"] == "mechanism" and "gemm" in row["phase"]:
        model_cfgs = [("qwen3_32b", QWEN), ("llama405b_layer", LLAMA405B)]
        shapes = [
            {"model": mname, "gemm": g[0], "N": g[1], "K": g[2], "M": m}
            for mname, mcfg in model_cfgs
            for g in gemm_shapes(mcfg)
            for m in ACTIVE_TOKENS
        ]
        axes["isl"] = axes["osl"] = axes["load"] = ["NA"]
    reps = int(row["repeats"])
    for fmt, isl, osl, load, shape in itertools.product(
            axes["format"], axes["isl"], axes["osl"], axes["load"], shapes):
        for rep in range(reps):
            spec = {
                "experiment_id": row["experiment_id"],
                "tier": row["tier"], "hardware": row["hardware"],
                "model": row["model"], "format": fmt, "phase": row["phase"],
                "tp": row["tp"], "isl": isl, "osl": osl, "load": load,
                "rep": rep, "priority": row["priority"],
            }
            if shape:
                spec["shape"] = shape
            spec["run_id"] = hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            yield spec

def main():
    rows = list(csv.DictReader(open(MATRIX)))
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for row in rows:
        specs = list(expand(row))
        total += len(specs)
        if "--list" in sys.argv:
            print(f"{row['experiment_id']:24s} {row['priority']:3s} {len(specs):6d} runs")
            continue
        with open(os.path.join(OUT, row["experiment_id"] + ".jsonl"), "w") as f:
            for s in specs:
                f.write(json.dumps(s, sort_keys=True) + "\n")
    print(f"TOTAL runs: {total}")

if __name__ == "__main__":
    main()
