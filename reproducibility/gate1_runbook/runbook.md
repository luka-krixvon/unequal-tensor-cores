# Gate 1 Runbook：B300 原生 INT8 判定（單卡，4 小時內）

目標機：1–2×B300（Vast.ai 2× slice ~$15/h，或 DataCrunch 1× ~$7.5/h）。
前置：Docker + NVIDIA Container Toolkit + **profiler 權限**（見 Step 0）。
所有步驟的輸出都存檔——它們是論文 Gate 1 的稽核證據。

預算節奏（4h 租期）：Step 0–1 共 ~30 min；Step 2 ~30 min；Step 3 ~45 min；Step 4 ~30 min；Step 5 ~20 min；餘量給重跑與上傳結果。

---

## Step 0｜驗收與環境快照（不通過就立刻停止計費）

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
# 期待：B300 / ~275040 MiB 級 / compute_cap 10.3。不是 10.3 → 停，換機。
./capture_env.sh | tee env_manifest_gate1.yaml   # 從 dry_run_scripts/ 複製
docker run --rm --gpus all nvidia/cuda:13.0.0-devel-ubuntu24.04 nvidia-smi -L
```

**Profiler 權限檢查（決定性前置）**：
```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  bash -c "ncu --version && ncu --metrics sm__cycles_active.sum --launch-count 1 \
           --target-processes all nvidia-smi" 2>&1 | grep -i "ERR_NVGPUCTRPERM" \
  && echo "NO-GO: counters restricted — 換 host 或聯絡供應商" || echo "counters OK"
```
`ERR_NVGPUCTRPERM` 出現 → **這台機器無法完成 Gate 1**，立即停租換機（Gate 0 條款）。

## Step 1｜IMMA 峰值微基準（全文最重要的數字）

```bash
ARCH=native ./build.sh
docker run --rm --gpus '"device=0"' -v $PWD:/w -w /w nvidia/cuda:13.0.0-devel-ubuntu24.04 ./imma_peak | tee imma_peak_b300.log
```

讀數解釋（dense 規格錨點）：
- `int8_imma_attained_TOPS`：**第一份第三方 B300 INT8 實測**。對照 NVIDIA HGX B300 規格 150 TOPS dense。
- `int8_to_bf16_ratio`：規格上 B300 = 150/2200 ≈ **0.068**；sm_89/sm_90 正控組應 ≈ 2.0（已在 4090 驗證 harness 正確性）。
- 分支：
  - attained ≈ 150 TOPS 量級 → 規格證實，IMMA 路徑被限速但真實可用。
  - attained 遠低於 150（<50%）→ legacy IMMA 在 sm_103 上另有損耗，本身是可報告發現。
  - 編譯失敗於 `mma.m16n8k32.s8` → PTX 主張需要重新檢視（不預期發生；Programming Guide Table 33 說 cc 10.3 INT8=Yes）。

隨後立刻：
```bash
./ncu_verdict.sh ./imma_peak 20000 2
# 期待 VERDICT: NATIVE-INT8-TENSOR-CORE。這同時驗證判準腳本在 B300 上可用。
```

## Step 2｜vLLM 1B fixture 煙霧測試（預期：失敗，而且失敗訊息本身是證據）

用釘住的 image（不可用 :latest）：
```bash
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
docker pull $PIN
# 下載 fixture（~1.4 GB）
hf download RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8 --local-dir models/L32-1B-int8
# probe_kernel.py 取自 reproducibility/dry_run_scripts/
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD/models:/models:ro \
  -v $PWD/probe_kernel.py:/probe.py:ro --entrypoint python3 $PIN /probe.py /models/L32-1B-int8 \
  2>&1 | tee smoke_default.log
```
**期待**：載入成功、第一次 forward 丟出 `Int8 not supported on SM103...`。
- 如期失敗 → 記錄，進 Step 3。
- 意外成功 → 重大訊息：vLLM 已加入 Blackwell INT8 kernel。記錄 `Selected ...Kernel` 行，跳到 Step 4 直接 ncu 判定，並回頭更新 audit 報告的時效聲明。

## Step 3｜Triton 逃生路徑

```bash
docker run --rm --gpus '"device=0"' --shm-size=2g \
  -e VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel \
  -v $PWD/models:/models:ro -v $PWD/probe_kernel.py:/probe.py:ro \
  --entrypoint python3 $PIN /probe.py /models/L32-1B-int8 2>&1 | tee smoke_triton.log
grep "Selected .*Kernel" smoke_triton.log   # 期待 TritonInt8ScaledMMLinearKernel
```
- forward 成功 → 進 Step 4（判定它是不是真 IMMA）。
- Triton 編譯失敗於 sm_103 → INT8 在 vLLM 全路徑不可用；Gate 1 判 **NO-GO（serving 層）**，微基準結果（Step 1）仍成立，主矩陣 INT8 arm 降為 kernel 層。

## Step 4｜對 Triton 路徑做 IMMA 判定（Gate 1 的正式判決）

```bash
cat > probe_forward.py <<'PY'
import sys
def main():
    from vllm import LLM, SamplingParams
    llm=LLM(model=sys.argv[1],enforce_eager=True,max_model_len=256,gpu_memory_utilization=0.3)
    llm.generate(["hi"],SamplingParams(max_tokens=4))
if __name__=="__main__": main()
PY
# 在容器內以 ncu 包 vLLM 單次 forward（--launch-count 限制成本）
docker run --rm --gpus '"device=0"' --cap-add=SYS_ADMIN --shm-size=2g \
  -e VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel \
  -v $PWD:/w -w /w -v $PWD/models:/models:ro --entrypoint bash $PIN -c \
  "pip -q install --no-deps nvidia-nsight-compute 2>/dev/null; \
   ncu --metrics sm__inst_executed_pipe_tensor_op_imma.sum --launch-count 64 \
       --target-processes all python3 probe_forward.py /models/L32-1B-int8" \
  2>&1 | tee ncu_vllm_triton.log
```
（若 image 內無 ncu：改用 devel container 裝 vllm，或 host ncu attach——runbook 附錄有兩種備案指令。）

**判定**：
- `imma.sum > 0` 於 linear 層 kernel → **Gate 1 = GO（條件版）**：INT8 arm 以 Triton 路徑 + H200 校正的上界式報法進主矩陣。
- `imma.sum = 0` → Triton 把 s8 提升為整數以外路徑或 dequant → **Gate 1 = NO-GO（serving 層）**，INT8 只留 kernel 層證據（Step 1 的微基準）。

## Step 5｜cuBLASLt INT8 探針（NVIDIA 自家函式庫的行為，順手的量測點）

```bash
docker run --rm --gpus '"device=0"' -v $PWD:/w -w /w nvidia/cuda:13.0.0-devel-ubuntu24.04 bash -c \
  "nvcc -O3 -arch=native cublaslt_int8_probe.cu -lcublasLt -o lt_probe && ./lt_probe" | tee cublaslt_b300.log
```
記錄支援與否 + 若支援的 attained TOPS。無論結果如何都是論文素材（NVIDIA 自家 library 在 sm_103 的 INT8 立場）。

## Step 6｜FP8 / NVFP4 快速健全性（若還有時間）

同一 fixture 家族換 FP8 checkpoint 跑一次 probe（應成功且走 native 路徑），為 Phase 2 的 crossover 量測確認軟體堆疊就緒。

---

## 帶回來的檔案清單

`env_manifest_gate1.yaml`、`imma_peak_b300.log`、`ncu_verdict_*.log`、`smoke_default.log`、`smoke_triton.log`、`ncu_vllm_triton.log`、`cublaslt_b300.log`——全部進 `reproducibility/`，並更新 claim ledger 的 GATE1-* 列與 decision log。

---

## 附錄：Vast.ai 容器模式適配（2026-08-13）

Vast.ai instance 本身是容器（非 VM），原 runbook 的 docker-in-docker 步驟改為：

1. **Instance 映像**：`nvidia/cuda:13.0.0-devel-ubuntu24.04`（Step 1/5 的 imma_peak 與 cublaslt probe 直接在 instance 內 build，`ARCH=sm_103a`）。
2. **Step 0 profiler 檢查提前為第一動作**：`ncu --version` 確認存在（devel 映像含 toolkit；若無 → `apt install nsight-compute` 自 NVIDIA repo），然後對任意小 kernel 跑 `ncu --metrics sm__cycles_elapsed.sum`。出現 `ERR_NVGPUCTRPERM` = 房東主機未開放 counters 且容器內無法補救（需 host 端 `NVreg_RestrictProfilingToAdminUsers=0`）→ **立即銷毀 instance 換供應商（Verda VM）**，不做任何後續步驟。
3. **Step 2/3（vLLM fixture、Triton 逃生路徑）**：容器內無 docker，兩案：(a) 於同機另開第二個 instance、映像直接填釘住的 digest `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`；或 (b) 單 instance 內 `pip install vllm==0.27.1`（版本釘住但失去 image digest 等值性——manifest 須如實記錄為 pip 安裝，等值性主張降為版本級）。優先 (a)。
4. **Step 4（ncu 對 Triton kernel 判定）**：在 vLLM instance 內 apt 安裝 nsight-compute 後對 serving process 附掛，或以 (a) 案在 devel instance 內以 `torch` + Triton 復現該 kernel 再判定。若兩者皆受權限阻擋而 Step 0 曾通過，記錄差異（process-attach 權限與 counter 權限不同層）。
5. 其餘（環境 manifest、溫度基線、決策樹）不變。
