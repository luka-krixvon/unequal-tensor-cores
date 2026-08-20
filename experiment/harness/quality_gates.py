"""Quality gates for the H200/B300 study — implements preregistration §7 exactly.

Three gates, all against a BF16 reference from the SAME parent checkpoint:
  G1 PPL     : relative perplexity increase   (INT8/FP8 <=2%, NVFP4 <=5%)
  G2 AGREE   : greedy token agreement          (INT8/FP8 >=95%, NVFP4 >=90%)
  G3 NEEDLE  : long-context retrieval pass-rate drop (<=5 points)

Pinned per §7 (do not change without a dated amendment):
  dataset   Salesforce/wikitext, config wikitext-2-raw-v1, split test,
            revision b08601e04326c79dfdd32d625aee71d232d685c3
  PPL       whole split concatenated, window 4096, NON-OVERLAPPING stride,
            exp(mean token NLL)
  AGREE     concatenated split -> non-overlapping 320-token windows (GPT-2
            tokenizer), first 200 complete windows, first 64 tokens = prompt,
            greedy 256 new tokens; per-prompt score = matched tokens before
            first divergence / 256; gate on the mean over 200 prompts (A-3)
  NEEDLE    haystack = same concatenated split; needle sentence and question
            fixed below; depth {0,10,..,100}% x length {8k,16k,32k} = 33 cells,
            one deterministic greedy generation each; pass = answer contains
            "731942"; score = cells passed / 33

Usage
-----
  # reference pass (BF16), writes reference.json
  python3 quality_gates.py --model <bf16-path> --role reference --out ref.json
  # candidate pass, compares against reference.json
  python3 quality_gates.py --model <quant-path> --role candidate \
      --reference ref.json --arm INT8-W8A8 --out int8.json
  # optional subsets while developing: --gates ppl,agree  --limit-prompts 20

Runs inside the pinned vLLM image; needs `datasets` + `transformers` for the
tokenizer and corpus only (no HF model download for tokenization: GPT-2
tokenizer is used for windowing, matching the pinned spec).
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys

DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1", "test",
           "b08601e04326c79dfdd32d625aee71d232d685c3")
PPL_WINDOW = 4096
AGREE_WINDOW, AGREE_PROMPT, AGREE_GEN, AGREE_N = 320, 64, 256, 200
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # openai-community/gpt2 main @ pin
NEEDLE_SENT = "The secret checkpoint code is 731942."
NEEDLE_Q = "What is the secret checkpoint code?"
NEEDLE_ANS = "731942"
NEEDLE_DEPTHS = list(range(0, 101, 10))            # 11 depths
NEEDLE_LENGTHS = [8192, 16384, 32768]              # 3 lengths -> 33 cells
THRESH = {  # (ppl_max_rel_increase, agree_min_mean, needle_max_drop_points)
    "int8": (0.02, 0.95, 5.0),
    "fp8":  (0.02, 0.95, 5.0),
    "nvfp4": (0.05, 0.90, 5.0),
    "bf16": (0.02, 0.95, 5.0),
}


def arm_class(arm: str) -> str:
    a = arm.lower()
    for k in ("nvfp4", "int8", "fp8", "bf16"):
        if k in a:
            return k
    raise SystemExit(f"cannot classify arm '{arm}' (expect int8/fp8/nvfp4/bf16)")


# ---------------------------------------------------------------- corpus ----
def load_corpus():
    from datasets import load_dataset
    name, cfg, split, rev = DATASET
    ds = load_dataset(name, cfg, split=split, revision=rev)
    text = "".join(ds["text"])          # preserve original newlines verbatim
    return text


def gpt2_tokenizer():
    from transformers import AutoTokenizer
    # Revision pinned: the windowing tokenizer must not drift.
    return AutoTokenizer.from_pretrained("gpt2", revision=GPT2_REVISION)


def build_agree_prompts(text, tok, n=AGREE_N, limit=None):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    want = n if limit is None else min(n, limit)
    prompts, i = [], 0
    while len(prompts) < want and i + AGREE_WINDOW <= len(ids):
        prompts.append(tok.decode(ids[i:i + AGREE_PROMPT]))
        i += AGREE_WINDOW
    if len(prompts) < want:
        raise SystemExit(f"corpus yields only {len(prompts)} windows (<{want})")
    return prompts


def build_needle_cases(text, tok, limit=None):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    cases = []
    for L in NEEDLE_LENGTHS:
        if L > len(ids):
            raise SystemExit(f"corpus shorter than needle length {L}")
        body = ids[:L]
        for d in NEEDLE_DEPTHS:
            cut = int(len(body) * d / 100)
            hay = tok.decode(body[:cut]) + "\n" + NEEDLE_SENT + "\n" + tok.decode(body[cut:])
            cases.append({"length": L, "depth": d,
                          "prompt": hay + f"\n\n{NEEDLE_Q}\nAnswer:"})
    return cases[:limit] if limit else cases


# ------------------------------------------------------------------ gates ---
def gate_ppl(llm, text):
    """exp(mean token NLL) over non-overlapping 4096-token windows."""
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    windows = [ids[i:i + PPL_WINDOW] for i in range(0, len(ids), PPL_WINDOW)]
    windows = [w for w in windows if len(w) >= 2]
    # vLLM >=0.9 removed the prompt_token_ids kwarg; pass TokensPrompt objects.
    try:
        from vllm import TokensPrompt
        reqs = [TokensPrompt(prompt_token_ids=w) for w in windows]
    except ImportError:
        reqs = [{"prompt_token_ids": w} for w in windows]
    outs = llm.generate(reqs, sampling_params=sp)
    tot_nll, tot_tok = 0.0, 0
    for o in outs:
        pls = o.prompt_logprobs or []
        for pos, lp in enumerate(pls):
            if pos == 0 or lp is None:
                continue          # first token has no context
            tid = o.prompt_token_ids[pos]
            entry = lp.get(tid)
            if entry is None:
                continue
            tot_nll += -(entry.logprob if hasattr(entry, "logprob") else float(entry))
            tot_tok += 1
    if tot_tok == 0:
        raise SystemExit("PPL: no scored tokens (prompt_logprobs unsupported?)")
    return {"ppl": math.exp(tot_nll / tot_tok), "tokens": tot_tok,
            "windows": len(windows)}


def gate_agree(llm, prompts):
    """Greedy generations; per-prompt token ids for later comparison."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=AGREE_GEN, temperature=0.0, ignore_eos=True)
    outs = llm.generate(prompts, sp)
    return {"gen_token_ids": [list(o.outputs[0].token_ids) for o in outs]}


def agree_score(ref_ids, cand_ids):
    """Mean over prompts of (matched tokens before first divergence)/AGREE_GEN."""
    # P0: zip() silently truncated to the shorter list, so a reference with
    # fewer prompts than the candidate produced a score over a smaller,
    # unreported denominator.
    if len(ref_ids) != len(cand_ids):
        raise SystemExit(f"prompt-count mismatch: reference has {len(ref_ids)},"
                         f" candidate has {len(cand_ids)}")
    per = []
    for r, c in zip(ref_ids, cand_ids):
        m = 0
        for a, b in zip(r, c):
            if a != b:
                break
            m += 1
        per.append(m / AGREE_GEN)
    return sum(per) / len(per), per


def gate_needle(llm, cases):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    outs = llm.generate([c["prompt"] for c in cases], sp)
    cells = []
    for c, o in zip(cases, outs):
        txt = o.outputs[0].text
        cells.append({"length": c["length"], "depth": c["depth"],
                      "passed": NEEDLE_ANS in txt})
    rate = 100.0 * sum(1 for c in cells if c["passed"]) / len(cells)
    return {"pass_rate_pct": rate, "cells": cells}


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--role", choices=["reference", "candidate"], required=True)
    ap.add_argument("--arm", default="bf16")
    ap.add_argument("--reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gates", default="ppl,agree,needle")
    ap.add_argument("--limit-prompts", type=int)
    ap.add_argument("--limit-needle", type=int)
    ap.add_argument("--max-model-len", type=int, default=32768 + 512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]

    text = load_corpus()
    tok2 = gpt2_tokenizer()
    res = {"role": args.role, "arm": args.arm, "model": args.model,
           "dataset": {"name": DATASET[0], "config": DATASET[1],
                       "split": DATASET[2], "revision": DATASET[3],
                       "chars": len(text),
                       "sha256_16": hashlib.sha256(text.encode()).hexdigest()[:16]},
           "gates_run": gates}

    from vllm import LLM
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size,
              enforce_eager=True, disable_log_stats=True)

    if "ppl" in gates:
        res["ppl"] = gate_ppl(llm, text)
    if "agree" in gates:
        prompts = build_agree_prompts(text, tok2, limit=args.limit_prompts)
        # prompt-list digest: newline-separated, UTF-8, per preregistration
        # "prompt 清單 SHA 隨 artifacts 發布"
        res["agree_prompts_sha256_16"] = hashlib.sha256(
            "\n".join(prompts).encode("utf-8")).hexdigest()[:16]
        res["agree"] = gate_agree(llm, prompts)
        res["agree"]["n_prompts"] = len(prompts)
    if "needle" in gates:
        cases = build_needle_cases(text, tok2, limit=args.limit_needle)
        res["needle"] = gate_needle(llm, cases)

    # ---- compare against reference, apply §7 thresholds -------------------
    if args.role == "candidate":
        if not args.reference:
            raise SystemExit("--reference required for role=candidate")
        ref = json.load(open(args.reference))
        cls = arm_class(args.arm)
        ppl_max, agree_min, drop_max = THRESH[cls]
        verdict = {}
        if "ppl" in gates:
            rel = res["ppl"]["ppl"] / ref["ppl"]["ppl"] - 1.0
            verdict["ppl_rel_increase"] = rel
            verdict["ppl_pass"] = rel <= ppl_max
        if "agree" in gates:
            mean, per = agree_score(ref["agree"]["gen_token_ids"],
                                    res["agree"]["gen_token_ids"])
            verdict["agree_mean"] = mean
            verdict["agree_pass"] = mean >= agree_min
            res["agree"]["per_prompt"] = per
        if "needle" in gates:
            drop = ref["needle"]["pass_rate_pct"] - res["needle"]["pass_rate_pct"]
            verdict["needle_drop_points"] = drop
            verdict["needle_pass"] = drop <= drop_max
            verdict["needle_reference_pass_rate_pct"] = ref["needle"]["pass_rate_pct"]
        verdict["thresholds"] = {"arm_class": cls, "ppl_max_rel": ppl_max,
                                 "agree_min": agree_min,
                                 "needle_max_drop_pts": drop_max}
        passes = [v for k, v in verdict.items() if k.endswith("_pass")]
        # P0: ALL_GATES_PASS previously became true after running any subset of
        # gates on any subset of prompts. A partial run can never be a pass.
        full_run = (set(gates) == {"ppl", "agree", "needle"}
                    and not args.limit_prompts and not args.limit_needle)
        verdict["full_protocol_run"] = full_run
        verdict["ALL_GATES_PASS"] = bool(passes) and all(passes) and full_run
        if not full_run:
            verdict["note"] = ("PARTIAL RUN -- not a protocol verdict; "
                               "section 7 requires all three gates over the "
                               "full prompt and needle sets")
        # P0: the reference JSON was trusted blindly. A candidate scored
        # against the wrong reference (different dataset revision, prompt set,
        # parent model, or role) silently produces a meaningless verdict.
        for field, want in (("role", "reference"),):
            if ref.get(field) != want:
                raise SystemExit(f"reference JSON has {field}={ref.get(field)!r},"
                                 f" expected {want!r}")
        if ref.get("dataset", {}).get("sha256_16") != res["dataset"]["sha256_16"]:
            raise SystemExit("reference and candidate saw different corpora "
                             f"({ref.get('dataset',{}).get('sha256_16')} vs "
                             f"{res['dataset']['sha256_16']})")
        if ("agree" in gates and "agree_prompts_sha256_16" in ref
                and ref["agree_prompts_sha256_16"]
                != res.get("agree_prompts_sha256_16")):
            raise SystemExit("reference and candidate used different prompts")
        res["verdict"] = verdict

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res.get("verdict", {"role": "reference",
                                         "ppl": res.get("ppl", {}).get("ppl"),
                                         "needle": res.get("needle", {}).get("pass_rate_pct")}),
                     indent=2))
    print(f"[quality_gates] wrote {args.out}")
    # P0: a failed gate must fail the process so a runner can block on it.
    if args.role == "candidate" and not res["verdict"].get("ALL_GATES_PASS"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
