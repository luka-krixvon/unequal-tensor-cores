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
def _resolve_matrix():
    """Locate config_matrix.csv without ever silently reading a stale copy.

    Search order: same directory as this script, then the parent directory
    (the repo layout experiment/harness/ -> experiment/). The resolved path
    and its SHA-256 are printed on every run so a mismatch against the
    pre-registration cannot go unnoticed -- a flat deployment (e.g. a VM
    working directory) previously resolved ".." to an unrelated directory
    and read an out-of-date matrix while every SHA check passed.
    """
    import hashlib
    for cand in (os.path.join(HERE, "config_matrix.csv"),
                 os.path.join(HERE, "..", "config_matrix.csv")):
        if os.path.isfile(cand):
            p = os.path.realpath(cand)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            print(f"[gen_sweep] matrix: {p}\n[gen_sweep] sha256[:16]: {h}")
            return p
    raise SystemExit("config_matrix.csv not found beside or above " + HERE)


MATRIX = _resolve_matrix()

# P0: A-7 made the resolved matrix visible but never fail-closed. Printing a
# digest that nobody compares is not a control. EXPECTED_MATRIX_SHA is the
# value pinned in experiment/preregistration.md section 0; set
# GEN_SWEEP_ALLOW_UNPINNED=1 only for a deliberate pre-data matrix revision,
# which must be recorded as an amendment.
EXPECTED_MATRIX_SHA = "17d20f1db167d994"


def _verify_matrix_pin(path):
    import hashlib
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    if h != EXPECTED_MATRIX_SHA:
        if os.environ.get("GEN_SWEEP_ALLOW_UNPINNED") == "1":
            print(f"[gen_sweep] WARNING: matrix {h} != pinned "
                  f"{EXPECTED_MATRIX_SHA} (override in effect)")
        else:
            raise SystemExit(
                f"[gen_sweep] REFUSING TO RUN: matrix sha {h} does not match "
                f"the pinned {EXPECTED_MATRIX_SHA}. Either restore the pinned "
                f"matrix or record an amendment and update "
                f"EXPECTED_MATRIX_SHA.")
    return h


_verify_matrix_pin(MATRIX)

# Data roles, transcribed from preregistration section 3.1. Kept here so the
# generated specs and the protocol cannot drift apart silently.
DATA_ROLE = {
    "MICRO-H200-QWEN": "TRAINING-ELIGIBLE",
    "MICRO-B300-QWEN": "TRAINING-ELIGIBLE",
    "QWEN-PREFILL-H200": "TRAINING-ELIGIBLE",
    "QWEN-PREFILL-B300": "TRAINING-ELIGIBLE",
    "QWEN-DECODE-H200": "TRAINING-ELIGIBLE",
    "QWEN-DECODE-B300": "TRAINING-ELIGIBLE",
    "BACKEND-PENALTY-H200": "CALIBRATION-ONLY",
    "BACKEND-PENALTY-DECODE-H200": "CALIBRATION-ONLY",
    "MICRO-H200-405B": "HELD-OUT",
    "MICRO-B300-405B": "HELD-OUT",
    "LLAMA405-H200": "HELD-OUT",
    "LLAMA405-B300": "HELD-OUT",
    "SESSION-RETEST-H200": "HELD-OUT",
    "SESSION-RETEST-B300": "HELD-OUT",
    "LLAMA405-LAYER-H200": "DIAGNOSTIC-POST-FREEZE",
    "LLAMA405-LAYER-B300": "DIAGNOSTIC-POST-FREEZE",
    "QWEN-MIXED-H200": "EXCLUDED-FROM-PREDICTOR",
    "QWEN-MIXED-B300": "EXCLUDED-FROM-PREDICTOR",
    "QWEN-TP-SENS-H200": "EXCLUDED-FROM-PREDICTOR",
    "QWEN-TP-SENS-B300": "EXCLUDED-FROM-PREDICTOR",
    "SGLANG-ROBUST-H200": "EXCLUDED-FROM-PREDICTOR",
    "SGLANG-ROBUST-B300": "EXCLUDED-FROM-PREDICTOR",
    "PD-H200": "EXCLUDED-FROM-PREDICTOR",
    "PD-B300": "EXCLUDED-FROM-PREDICTOR",
    "REQUANT-B300-DECODE": "EXCLUDED-FROM-PREDICTOR",
    "REQUANT-B300-PREFILL": "EXCLUDED-FROM-PREDICTOR",
    "GATE1-B300-RUNBOOK": "GATE",
    "GATE1-H200-CONTROL": "GATE",
    "H4-SLO-H200": "EXCLUDED-FROM-PREDICTOR",
    "H4-SLO-B300": "EXCLUDED-FROM-PREDICTOR",
}

# Per-axis hold-out buckets from preregistration section 3.3. A cell inside a
# training-eligible experiment still becomes HELD-OUT when it lands in one of
# these buckets; the previous generator had no representation of this at all.
HELDOUT_ISL = {"8192"}
HELDOUT_LOAD = {
    "concurrency": {"3", "12", "48", "192"},
    "batch": {"3", "12", "24"},
    "active_tokens": {"3", "12", "48", "192", "768", "3072"},
}


def heldout_reason(exp_id, isl, load, shape=None):
    """Return a reason code when a cell falls in a per-axis hold-out bucket.

    A-9: the MICRO rows carry their load on the shape's M axis, not on the
    load field, so an M-only check was required or the active-token hold-out
    could never be tagged.
    """
    if DATA_ROLE.get(exp_id) != "TRAINING-ELIGIBLE":
        return None
    if str(isl) in HELDOUT_ISL:
        return "isl-8192-bucket"
    if isinstance(load, str) and "=" in load:
        axis, _, val = load.partition("=")
        if val in HELDOUT_LOAD.get(axis, ()):
            return f"off-grid-{axis}-{val}"
    if shape and shape.get("M") in OFFGRID_ACTIVE_TOKENS:
        return f"off-grid-active_tokens-{shape['M']}"
    return None
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

# On-grid powers of two plus the section 3.3 off-grid hold-out points
# {3,12,48,192,768,3072}. A-9: these were named in the protocol but never
# generated, so the load axis had no off-grid generalisation test at all.
ACTIVE_TOKENS = sorted([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                       + [3, 12, 48, 192, 768, 3072])
OFFGRID_ACTIVE_TOKENS = {3, 12, 48, 192, 768, 3072}

def split(v):
    """Split a pipe-separated axis value, propagating an 'axis=' prefix.

    A-9 (P0): the previous version kept the prefix only on the first element,
    so "batch=1|2|4" became ["batch=1", "2", "4"]. Every downstream consumer
    that parses 'axis=value' -- the hold-out tagger, the capacity envelope --
    therefore saw an axis for exactly one cell per row, which is why the
    section 3.3 off-grid concurrency and batch hold-outs were added to the
    matrix and still never got tagged.
    """
    raw = v.strip('"')
    parts = [s.strip() for s in raw.split("|")] if "|" in raw else [raw.strip()]
    if "=" in parts[0]:
        axis = parts[0].split("=", 1)[0]
        return [p if "=" in p else f"{axis}={p}" for p in parts]
    return parts

def expand(row):
    """Yield fully-specified run dicts for one matrix row."""
    axes = {
        "format": split(row["format"]),
        "isl": split(row["isl_tokens"]),
        "osl": split(row["osl_tokens"]),
        "load": split(row["load_or_concurrency"]),
    }
    # MICRO rows additionally sweep GEMM shapes x active tokens.
    #
    # P0 (held-out leakage): the previous gate was `tier == "mechanism"`, which
    # expanded BOTH Qwen and Llama-405B shapes for every MICRO-*-QWEN row and
    # expanded NEITHER for the MICRO-*-405B rows (whose tier is
    # diagnostic-post-freeze). The v3 split of MICRO into -QWEN and -405B was
    # therefore never realised in the generated specs: each
    # MICRO-*-QWEN.jsonl carried 1,820 llama405b_layer specs -- exactly the
    # pre-registered shape hold-out -- inside a row labelled
    # TRAINING-ELIGIBLE, while each MICRO-*-405B.jsonl carried no shapes at
    # all. Shape families are now selected from the row's declared model, so
    # the label and the contents cannot disagree.
    shapes = [None]
    if "gemm" in row["phase"] and row["experiment_id"].startswith("MICRO-"):
        model_field = row["model"]
        cfgs = []
        if "Qwen" in model_field:
            cfgs.append(("qwen3_32b", QWEN))
        if "405B" in model_field or "Llama" in model_field:
            cfgs.append(("llama405b_layer", LLAMA405B))
        if not cfgs:
            raise SystemExit(
                f"{row['experiment_id']}: cannot infer shape family from "
                f"model={model_field!r}; refusing to guess")
        shapes = [
            {"model": mname, "gemm": g[0], "N": g[1], "K": g[2], "M": m}
            for mname, mcfg in cfgs
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
                # P0: the pre-registration's data-role firewall (section 3.1)
                # existed only in prose; nothing in the generated specs marked
                # which cells a predictor may train on. Carry the role, and the
                # hold-out reason when applicable, on every spec.
                "data_role": DATA_ROLE.get(row["experiment_id"], "UNASSIGNED"),
            }
            ho = heldout_reason(row["experiment_id"], isl, load, shape)
            if ho:
                spec["data_role"] = "HELD-OUT"
                spec["heldout_reason"] = ho
            # A-9: Hopper has no native FP4 tensor-core pipe, so an "NVFP4" arm
            # on H200 is necessarily a dequantise-to-BF16 execution path -- a
            # different section 5.5 quadruple from B300's native NVFP4. The
            # NVFP4 boundary is therefore B300-only; these cells are kept as a
            # capability-boundary diagnostic and must not feed H2, H3 or the
            # NVFP4 boundary estimate.
            if spec["hardware"].startswith("H200") and "NVFP4" in fmt:
                spec["data_role"] = "DIAGNOSTIC-CAPABILITY"
                spec["diagnostic_reason"] = "h200-nvfp4-no-native-fp4-pipe"
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
