#!/bin/bash
# Gate 1 decisive criterion: the IMMA instruction counter.
# Kernel-name matching is a FALSE-NEGATIVE trap on sm_103 (no tcgen05 integer
# kind exists there); the only valid signal is a nonzero
#   sm__inst_executed_pipe_tensor_op_imma.sum
# See audit report sec. VI(1) and claim ledger GATE1-PROFILER-001.
#
# Usage:
#   ./ncu_verdict.sh ./imma_peak 20000 2            # profile the microbench
#   ./ncu_verdict.sh python3 probe_forward.py ...   # profile one model forward
#
# Runs ncu from the CUDA devel container if not installed on the host.
set -uo pipefail
cd "$(dirname "$0")"

METRICS="sm__inst_executed_pipe_tensor_op_imma.sum,sm__inst_executed_pipe_tensor_op_hmma.sum"
IMAGE="${IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"
OUT="ncu_verdict_$(date -u +%Y%m%dT%H%M%SZ).log"

run_ncu() {
  if command -v ncu >/dev/null 2>&1; then
    ncu --metrics "$METRICS" --launch-count 32 --target-processes all "$@"
  else
    docker run --rm --gpus all --cap-add=SYS_ADMIN -v "$PWD":/w -w /w "$IMAGE" \
      ncu --metrics "$METRICS" --launch-count 32 --target-processes all "$@"
  fi
}

echo "[ncu_verdict] profiling: $*" | tee "$OUT"
run_ncu "$@" 2>&1 | tee -a "$OUT"

echo "----------------------------------------" | tee -a "$OUT"
IMMA=$(grep -oE 'tensor_op_imma\.sum[^0-9]*[0-9,\.]+' "$OUT" | grep -oE '[0-9][0-9,\.]*' | tr -d ',' | sort -rn | head -1)
IMMA=${IMMA:-0}
if [ "${IMMA%%.*}" -gt 0 ] 2>/dev/null; then
  echo "VERDICT: NATIVE-INT8-TENSOR-CORE (imma.sum=$IMMA > 0)" | tee -a "$OUT"
else
  echo "VERDICT: NO-IMMA-EXECUTED (imma.sum=0 -> dequant fallback or non-TC path; inspect $OUT)" | tee -a "$OUT"
fi
echo "[ncu_verdict] full log: $OUT"

# NOTE: if ncu reports ERR_NVGPUCTRPERM, the host restricts GPU performance
# counters. Fix requires host-level NVreg_RestrictProfilingToAdminUsers=0 or a
# provider that allows profiling; this is a Gate-0 acceptance criterion for the
# rental (GO_NO_GO report sec. 6, item 5). Without counters Gate 1 cannot pass.
