# equal-bytes

Reproducibility artifacts for a measurement study of quantization-format
crossovers across NVIDIA H200 and Blackwell Ultra B300.

Two outputs share this repository:

1. **Audit report** (submitted to arXiv 2026-08-12, `submit/7941203`) —
   *Spec Sheets Are Not Kernels: An ISA- and Source-Level Audit of INT8
   Availability on NVIDIA Blackwell Ultra*. A documentary audit: no
   performance measurements, every claim pinned to an official document,
   a source commit, or a released binary.
2. **Measurement study** (in progress) — attained INT8–FP8 and NVFP4–FP8
   crossovers on H200 and B300, with a mechanistic predictor validated on
   pre-registered held-out points.

## What is here

```
experiment/
  preregistration.md        pre-registered protocol, locked 2026-08-12
  config_matrix.csv         design matrix (26 experiments)
  harness/
    changepoint.py          crossover / knee estimators (locked artifact)
    validate_synthetic.py   estimator acceptance tests (19 criteria)
    quality_gates.py        PPL / token-agreement / needle gates
    gen_sweep.py            matrix -> executable run specs
    qwen3_32b_config.json   pinned architecture parameters
reproducibility/
  gate1_runbook/            scripted Gate-1 procedure for a rented B300
    runbook.md              step-by-step with GO/NO-GO decision tree
    imma_peak.cu            warp-level INT8 IMMA peak microbenchmark
    cublaslt_int8_probe.cu  cuBLASLt INT8 availability probe
    ncu_verdict.sh          IMMA-counter criterion (see below)
    build.sh, probe_kernel.py, capture_env.sh
  synthetic_validation_*.log  estimator validation output
  *_manifest_template.yaml    environment / run manifests
references/
  claim_source_ledger.csv   every factual claim with source, date, verdict
```

## Pre-registration

`experiment/preregistration.md` was locked before any H200 or B300 data
existed, and pins by SHA-256 the estimator, its acceptance tests, the sweep
expander, and the design matrix. It defines the held-out splits, the three
baselines, the quality-gate thresholds and datasets, and the metric
definitions including tie and censoring rules.

Amendments are two-class: **A** (pre-data, free-form) and **B** (post-data,
permitted only at decision points named in advance). Both are appended, never
rewritten. A-1 records a revision following an adversarial pre-data review;
A-2 records the estimator validation; A-3 corrects a prompt-source
specification found to be unsatisfiable before implementation.

## One methodological note worth surfacing

On `sm_103` (B300), checking for "native INT8" by grepping profiler output for
fifth-generation tensor-core kernel names produces a **false negative**: the
PTX ISA never exposes `tcgen05.mma` with `.kind::i8` on that target, so a
genuinely native INT8 execution cannot show one. The reliable criterion is a
nonzero `sm__inst_executed_pipe_tensor_op_imma.sum` in Nsight Compute. See the
audit report §VI and `reproducibility/gate1_runbook/ncu_verdict.sh`.

## Status

Estimators validated on synthetic data (19/19 acceptance criteria) and the
Gate-1 toolchain validated on known-answer hardware (`sm_89`). No
target-hardware measurements yet; this repository will carry them, together
with the predictor freeze record required by preregistration §1.2, as they are
produced.

## Citation

Citation metadata will be added once the audit report is announced on arXiv.

## License

Code: MIT. Documents (pre-registration, runbook, ledger): CC BY 4.0.
