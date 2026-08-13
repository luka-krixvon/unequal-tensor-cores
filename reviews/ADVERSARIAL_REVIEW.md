# H200 × HGX B300 量化格式 Crossover：敵對技術審查

審查日期：2026-08-13（Asia/Taipei）

市場負面主張以 2026-08-12 為截止；GitHub pipeline 以 2026-08-13 快照核對。

## 摘要

硬體規格前提 C1 成立，但核心物理假說 C3 算錯；C6 的廣義 novelty 已被多篇先例擊穿；C12 也因「checkpoint 格式不等於執行格式」而不具任何 merge 都能存活的韌性。其餘多半是方向合理，但措辭、因果解釋或目前 preregistration 的實例化過強。

判決總表：

| Claim | Verdict |
|---|---|
| C1 | CONFIRM |
| C2 | PARTIAL |
| C3 | REFUTE |
| C4 | PARTIAL |
| C5 | PARTIAL |
| C6 | REFUTE |
| C7 | PARTIAL |
| C8 | PARTIAL |
| C9 | PARTIAL |
| C10 | PARTIAL（其中狹義 negative claim 為 CANNOT-VERIFY） |
| C11 | CANNOT-VERIFY |
| C12 | REFUTE |
| C13 | PARTIAL |
| C14 | PARTIAL |

## 規格口徑

以下均為 dense、per GPU；括號為 sparse：

| GPU | FP8 | INT8 | HBM BW | FP8:INT8 | FP8／INT8 ridge |
|---|---:|---:|---:|---:|---:|
| H200 SXM | 1.979 PF（3.958） | 1.979 POPS（3.958） | 4.8 TB/s | 1:1 | 412／412 op/B |
| HGX B200 GPU | 4.5 PF（9） | 4.5 POPS（9） | 7.7 TB/s | 1:1 | 584／584 op/B |
| HGX B300 GPU | 4.5 PF（9） | 0.150 POPS（0.300） | 7.7 TB/s | 30:1 | 584／19.5 op/B |

主要來源：

- [NVIDIA Blackwell Architecture Technical Brief，Table 3](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief)
- [NVIDIA Blackwell Ultra Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet)
- [NVIDIA H200 datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf)

B300 datasheet 給 sparse INT8 307 TOPS，即 dense 153.5 TOPS，故精確比值約 29.3×；Technical Brief 的 150 是四捨五入。H200 表中的 3,958 標為 with sparsity，dense 要除二。

不要混入 GB300 NVL72：其單 GPU為 dense FP8 5 PF、INT8 165 TOPS、8 TB/s。C3 原始的「4.5 PF + 8 TB/s」混用了 HGX B300 compute 與 GB300／整數化 bandwidth。

## C1 — CONFIRM

規格比值正確。H200 與 B200 的 advertised dense FP8、INT8 峰值皆約 1:1；HGX B300 才降為約 29.3–30:1，所以斷崖發生在 B200→B300，而不是 Hopper→一般 Blackwell。

限制：

- 這是 vendor peak operation-count ratio，不是 attained LLM throughput ratio。
- 未做 2:4 pruning 的 Qwen3 不能使用 sparse 欄。
- FLOPS 與 integer OPS 的比值只在 NVIDIA 相同的 FMA operation-count convention 下成立。

定案：規格主張已可由上述文件定案；實效需以相同大型 GEMM shapes 實測，並用 Nsight/SASS 確認真實 MMA 路徑。

## C2 — PARTIAL

「主要 W8A8 payload 都是 1 byte/value」成立；「實際搬動的 bytes 完全相同」不成立。較完整的 GEMM intensity 是：

\[
AI=\frac{2MKN}{KN+MK+b_DMN+\text{scales/padding/workspace}}.
\]

即使 A、W 各一 byte，輸出通常是 BF16/FP16；scale、padding、workspace、epilogue 也可能不同。只有 weight bytes 壓倒其他項時，才近似 AI≈2M。

具體反證：

- INT8 常用 per-channel weight、dynamic per-token activation；FP8 可能是 per-tensor、per-channel 或 128×128 block scaling，metadata 與量化 kernel 不同。[compressed-tensors quantization schemes](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py)
- LM head、ignored layers、scale granularity、resident layout 與 dynamic quantization fusion 皆可使 HBM traffic 不同。
- 即使 bytes 完全相同，也不必等 FP8 compute-bound 才出現差距；B300 INT8 約在 M≈10 就先跨入 compute-bound。

較安全的表述：

> 在兩臂都 bandwidth-bound，且 resident layout、scale granularity、ignored layers、KV dtype 與 quantization coverage 全部匹配時，主要 payload 接近，因此理想 memory lower bound 接近。

定案：逐 kernel 收集 DRAM bytes、L2 hit、quantization/epilogue 時間與 resident dtype/layout。

## C3 — REFUTE

核心錯誤是只算 FP8 ridge，卻用它判斷兩臂何時開始分岔。

對格式 f：

\[
t_f=\max(D/BW,F/P_f).
\]

HGX B300：

\[
I^*_{\rm FP8}=4500/7.7=584.4,\qquad
I^*_{\rm INT8}=150/7.7=19.5\ \text{op/B}.
\]

若 linear GEMM 的 AI≈2B：

- B<10：INT8、FP8 都可能 memory-bound；
- 10<B<292：INT8 已 compute-bound，FP8 仍 memory-bound；
- B>292：兩者才都 compute-bound，並漸近 30×。

所以差距大約從 B=10 開始，不是 B=280。max concurrency=256 足以觀察 INT8 分岔，只是不足以定位 FP8 自己的理想 compute knee。

對 [Qwen3-32B config](https://huggingface.co/Qwen/Qwen3-32B/blob/e3a3f59b0423be17f9582d1755fd3d5449b31a1f/config.json)，64 layers、64 query heads、8 KV heads、head dim 128：

- decoder linear weights 約 31.206B；
- 加 W8 LM head，每步 weight stream 約 31.984 GB；
- 每序列每步 KV read：

\[
D_{KV}=64\times2\times8\times128\times S\times q
      =131072Sq\ \text{bytes},
\]

其中 q=1 為 FP8 KV、q=2 為 BF16 KV。

- attention FLOPs：

\[
F_{att}\approx64\times4\times64\times128\times S
          =2{,}097{,}152S.
\]

GQA attention/KV intensity 只有 16/q：FP8 KV 為 16 FLOP/B、BF16 KV 為 8 FLOP/B。

在 B=256 時，只計 weights+KV：

| Context | FP8 KV：KV byte 占比／總 AI | BF16 KV：KV byte 占比／總 AI |
|---:|---:|---:|
| 2K | 68.2%／173.5 | 81.1%／103.1 |
| 8K | 89.6%／67.7 | 94.5%／35.7 |
| 32K | 97.2%／30.0 | 98.6%／15.2 |

KV 確實讓整體 decode 更 memory-heavy，尤其長 context；但它只會稀釋 INT8 linear penalty，不能消除 INT8 linear kernels 已在 B≈10 compute-bound 的事實。linear 與 paged-attention 是序列執行的不同 kernels，用全模型平均 AI 會掩蓋瓶頸切換。

簡化的 serialized-stage roofline 在 B=256 預測 FP8/native-INT8 speedup 約從 2K 的 5.7–8.8×，降到 32K 的 1.35–1.7×；不是「直到 >256 都幾乎相等」。

H200：

\[
I^*_{\rm FP8}=I^*_{\rm INT8}=1979/4.8=412.3,
\]

理想 linear knee B≈206，且沒有 silicon peak asymmetry。長 context 時兩者都更可能被 KV bandwidth 限制。

另一問題是 B=256×S=32K 不一定在所有 KV dtype、TP 與 graph/workspace 設定下可行；網格不能假設形成完整笛卡兒積。

定案：

- isolated QKV/O/MLP GEMM 掃 M=1…256；
- pure-decode 預先配置 KV，再依容量裁切網格；
- 分別記錄 linear、attention、quantization、all-reduce 時間及 HBM/tensor-pipe counters；
- 固定 TP、KV dtype、prefix reuse、CUDA graph padding、speculative decoding。

## C4 — PARTIAL

令 T=ΣSᵢ 為一次 prefill forward 的 active prompt tokens。weight-dominated W8A8 linear intensity 約為 2T。理想 B300 FP8 knee 約 T=292；加入 BF16 activation read/write 後，Qwen3-32B 的估算 knee 約移至 T≈339。

| T | 128 | 256 | 512 | 2048 |
|---:|---:|---:|---:|---:|
| Linear AI | 241 | 457 | 824 | 2077 |

因此 T≥512 的 standard dense prefill 很可能 compute-bound；短 prompt、小 batch 或 chunked-prefill chunk≤256 不一定。

反證：長 context 的 causal attention 是 O(S²) 的共同成本。在 Qwen3-32B，attention 約占 linear+attention FLOPs 的 3.3%（2K）、12.1%（8K）、35.5%（32K）。若 attention 路徑在兩臂相同，它會稀釋 INT8/FP8 linear 的峰值差。C3 也顯示中高 concurrency decode 可以出現顯著差距，所以「30×主要只在 prefill」過強。

定案：掃 active tokens 64–4096，分離 batch 與 sequence length，逐 kernel 報 linear/attention；關閉或獨立分析 prompt-logprobs 與 chunked prefill。

## C5 — PARTIAL

抽象上，若三類共享相同 action、constraints、quality/SLO 定義，且高階類允許退化成常數政策，則 Π₀⊂Π₁⊂Π₂ 成立，oracle objective 單調改善也是數學必然。

但目前 preregistration 有致命實例化錯誤：

- [experiment/preregistration.md:78](../experiment/preregistration.md#L78) 把 B1 定義成「每 hardware×phase 選格式」；
- 下一行 B2 又是「per-phase 選格式」。

B1 已經是 Π₁；B1/B2 實際不可區分，Π₀→Π₁ gap 無法估計。

其他問題：

- 若 Π₀ 只固定 format，但允許 config 隨 phase/load 改變，類別 nesting 必須重新形式化。
- oracle gap 是 clairvoyant upper bound，不等於可部署節省；尚未扣雙份模型 residency、idle pool、cold start、routing、切換與錯誤預測。
- regime 可能是內生的：format 改變服務率與排隊，會改變系統所在 regime。
- 只有四種格式時，argmax 本身不是研究貢獻。

「每點昂貴，所以 surrogate 有價值」可以成立，但必須以 sample efficiency 證明。直接先例包括 [OmniPilot，arXiv 2026](https://arxiv.org/abs/2607.01579) 與 [Vidur，MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)。

定案：修正 B1 後，畫「量測點數／GPU-hours → held-out regret、calibration、winner accuracy」曲線，與 interpolation、roofline、GP/BO、Vidur/OmniPilot-like baseline 比較。

## C6 — REFUTE

「DistServe、Splitwise、Mooncake、Dynamo 本身未自動選模型 quant format」大致成立；但推廣成「既有工作沒有 per-phase precision、沒有 format/load policy」是錯的。

| 工作 | 年份／venue | 直接相關內容 |
|---|---|---|
| [LLM-PQ](https://arxiv.org/abs/2403.01136) | PPoPP 2024 Poster | phase-aware partition、adaptive mixed precision、microbatch |
| [PMPD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5df4313ecd4875931fbdacc486cc1fcf-Abstract-Conference.html) | ICLR 2025 | phase-aware precision 與動態 precision-lowering scheduler |
| [SplitQuant](https://i2.cs.hku.hk/~cwu/papers/jtzhao-cluster25.pdf) | IEEE Cluster 2025 | 聯合 precision、phase partition、microbatch，並預測 latency |
| [HMA-Serve](https://arxiv.org/abs/2606.29986) | arXiv 2026 | PD 系統中的 phase-wise quantization；低精度 prefill、BF16 decode |
| [Mix-Quant](https://arxiv.org/abs/2605.20315) | arXiv 2026 | NVFP4 prefill、BF16 decode |
| [MorphServe](https://proceedings.mlsys.org/paper_files/paper/2026/hash/8144a9d62e506af0fcdeac0e456b2710-Abstract-Conference.html) | MLSys 2026 | 依 workload/SLO 動態交換量化 layers |

HMA-Serve、Mix-Quant 目前是預印本，不能冒充 peer-reviewed；但足以擊穿「沒有公開先例」。Sarathi-Serve 也不是典型 PD disaggregation，而是用 chunked prefill 與 decode 共排程。

「在同一 regime 內 exact decouple」不成立。format 會改變可行 batch、KV capacity、service-time distribution、最佳 parallelism、PD transfer representation 與 SLO feasible set。即使兩設定都叫 memory-bound，把 weights 減半而使最大 batch 從 16 提升至 32，就已使 inner scheduler 的可行域改變。

可保留的窄 novelty：

> 在真實 HGX B300 上，做經 execution-path 驗證的 matched INT8-vs-FP8 測量，量出 load-dependent crossover，並跨 H200/B300、held-out shape/model 驗證 calibrated predictor。

定案：在 held-out traces 上直接比較 joint optimizer 與 two-stage decoupled optimizer；逐步開放 batch、TP、KV、transfer、routing。「exact」需要明確假設與證明。

## C7 — PARTIAL

factory-first 作為量測順序合理：先取得 kernel/throughput calibration，再安排昂貴的 service-mode 實驗，有助於除錯和縮小設計空間。保留最後階段 dedicated SLO runs，使設計仍可辯護。

但「SLO/goodput 大部分可由 factory throughput 推導」不成立。即使平均服務率相同，M/G/1 有：

\[
E[W_q]=\frac{\lambda E[S^2]}{2(1-\rho)}.
\]

相同 E[S] 下，不同 E[S²]、重尾 service time 或 bursty arrivals 可產生完全不同的 tail latency。continuous batching 還涉及 prompt/output 長度聯合分布、HoL blocking、chunked prefill、prefix cache、preemption 與 KV pressure。

[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) 與 [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) 都直接用 workload traces 與 SLO/goodput 評估；[Vidur](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html) 能後推，是因為還建模 operator/service distributions，不是只靠飽和 throughput。

較安全表述：

> Factory sweep 是必要的上游 calibration 與診斷資料，不是 SLO/goodput 的充分母資料；SLO 層需獨立盲測。

定案：用 Poisson、MMPP/bursty、真實 trace，在多個 utilization 上盲測 TTFT/TPOT p50–p99 與 goodput。

## C8 — PARTIAL

公開文件中，未找到 vLLM、SGLang、TensorRT-LLM 或 Dynamo 自動依 phase×load 切換整個模型 quantization format：

- [vLLM online quantization](https://docs.vllm.ai/en/stable/features/quantization/online/) 仍以模型載入／forward recipe 為主；
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) 由 checkpoint/QuantConfig/engine recipe 決定；
- [Dynamo Global Router](https://docs.nvidia.com/dynamo/dev/reference/components/global-router-configuration) 能依 ISL、SLO、load 選 pool，但沒有公開的自動 precision selector。

所以 fixed deployment recipe 是合理 baseline。

但：

- 公開能力不能證明生產環境的 industry default；缺乏 survey/telemetry。
- 主流 checkpoint 可有 mixed precision、不同 KV dtype、BF16 LM head；「one global format」字面過強。
- 營運者可手工部署不同靜態 pools 再用外部 router。
- MorphServe 已是動態 precision 的研究反例。
- 目前 B1 更不是 global format，而是 per-phase。

Π₀→Π₂ 是 oracle upper bound，不宜直接稱 current practice 留在桌上的錢。還須扣 residency、idle capacity、routing、switching、prediction regret；GPU-seconds/token 也不自動等於 $/token。

定案：加入真正 single-recipe baseline、人工 heterogeneous pools、MorphServe-like adaptive baseline；以部署調查或匿名 telemetry 支持 industry-default claim。

## C9 — PARTIAL

以 PTX 9.3、CUTLASS dcf215af、vLLM 89c8401c、SGLang e8c7dddf 快照核對。

### C9(a) — CONFIRM，但只限原生第五代路徑

PTX 9.3 將 tcgen05.mma.kind::i8 支援列為 sm_100a、sm_101a/110a、sm_110a，沒有 sm_103a。[PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma)

但 legacy warp-level .s8/.u8 mma 對 sm_80+ 仍合法。因此正確表述是「SM103 沒有 tcgen05 INT8 UMMA」，不是「B300 完全沒有 Tensor Core INT8」。

### C9(b) — CONFIRM

CUTLASS generator 對包含 103a family 的 manifest 跳過 dense/sparse INT8 UMMA generation；SM103 專屬 generator 聚焦 FP4 ultra。[CUTLASS generator](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/python/cutlass_library/generator.py#L12035-L12091)

精確 sm_100a 仍有 INT8 kernels，不能泛化成所有 Blackwell。

### C9(c) — CONFIRM（source-level）

vLLM 將 SM100–119 送進 SM100 dispatcher，[SM100 wrapper](https://github.com/vllm-project/vllm/blob/89c8401c8aeb53ccee87060e913514068e0a1e82/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu#L11-L20) 把 INT8 function pointer 傳成 nullptr；實際呼叫時由 [helper](https://github.com/vllm-project/vllm/blob/89c8401c8aeb53ccee87060e913514068e0a1e82/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp#L22-L37) 報 Use FP8 quantization instead。

失敗發生在權重載入後第一次 forward；通常引擎初始化會先做 dummy/profile forward，所以不一定等第一個使用者請求。

### C9(d) — PARTIAL

vLLM 候選順序是 CUTLASS→Triton→Humming。設定 VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel 會跳過 CUTLASS，通常選到 Triton。[vLLM selector](https://github.com/vllm-project/vllm/blob/89c8401c8aeb53ccee87060e913514068e0a1e82/vllm/model_executor/kernels/linear/__init__.py#L549-L571)

但這只證明 reroute，不證明 B300 正確性、性能或實際 IMMA 指令。

SGLang dense per-channel W8A8 只 dispatch SM75、SM80–89、SM90；SM100/103 直接 NOT_IMPLEMENTED，[source](https://github.com/sgl-project/sglang/blob/e8c7dddfa0114479eec10514e89bea016652c43e/python/sglang/kernels/aot/csrc/gemm/int8_gemm_kernel.cu#L699-L744)，所以不是 B300 dense W8A8 的 practical Triton fallback。blockwise/MoE INT8 另有 Triton 路徑；不能說 SGLang 完全沒有 Triton INT8。公開 tuning configs 也沒有 B200/B300/GB300。

定案：在 B300 同時跑 default 與 forced Triton，記錄每層 selected kernel，用 Nsight/SASS 證實 instruction family、DRAM bytes、正確性與吞吐。

## C10 — PARTIAL

狹義「截至 2026-08-12 未有公開第三方 B300 INT8 execution 或 matched INT8-vs-FP8 測量」為 CANNOT-VERIFY。我未找到可信數據，但搜尋不到不能證明不存在。可寫成「依已記錄搜尋流程，未找到」。

廣義「公開 B300 多格式 benchmark 幾乎不存在」為 REFUTE。

直接反例：

- [Selectel，2026-05-14](https://selectel.ru/blog/llms-on-hgx-b300/) 在實體 8× HGX B300、vLLM 0.16 上比較 DeepSeek V3.2 NVFP4 與 FP8，掃 ISL 1K–16K、OSL 1K–4K、concurrency 4–256，並分 prefill/decode 報告。限制是 NVFP4 用 TP2×DP4、FP8 用 TP4×DP2，checkpoint 也不同，故不是 matched causal study。
- [TechInsights，2026-07-27](https://www.techinsights.com/blog/nvidia-hgx-b300-ai-infrastructure-benchmarking) 公開說明實體 HGX B300 測試涵蓋 FP4、FP8、BF16，以及 throughput、TTFT、TPOT、ITL、power/thermal；詳細數字付費，但足以否定「沒有人測 B300 多格式」。

新的 market positioning：

> 公開 B300 FP8/NVFP4/BF16 benchmark 已存在；缺口是經 SASS/NCU 驗證、控制 TP/DP/checkpoint/clock/power 的 B300 INT8-vs-FP8 matched measurement，以及跨 load/hardware 的 calibrated crossover predictor。

定案：公開搜尋 queries、日期、資料庫與排除表；真正跑最小 INT8 GEMM/LLM，證明走 INT8 arithmetic 而非 BF16/FP8 conversion。

## C11 — CANNOT-VERIFY

Unknown unknowns 無法封閉驗證，但至少有以下未明說、會改變研究結論的假設：

| 隱含假設 | 若不成立的後果 | 定案方式 |
|---|---|---|
| checkpoint、resident、MMA format 相同 | C1/C3 silicon framing 失效 | 記錄四層 dtype/layout、SASS |
| INT8/FP8 coverage、scale、ignored layers、品質等價 | 不是 matched comparison | layer manifest＋composed-policy quality gates |
| scheduler 產生相同 active-token/GEMM shapes | load 軸不可比 | 逐 iteration 記錄 effective batch、padding、graph bucket |
| format 不改變 KV capacity、admission、preemption | regime 變成內生 treatment | 固定資源與 natural-capacity 兩套 estimand |
| weight 每步只讀一次、KV 無額外 reuse | roofline bytes 錯 | NCU DRAM/L2、prefix-cache ablation |
| B×S 網格兩硬體都可行 | missing/censoring 有系統偏差 | 先做容量 envelope |
| TP/NVLink/all-reduce 在兩格式作用相同 | per-GPU roofline不能推 node | TP sensitivity、collective profiling |
| phase-isolated 結果代表 mixed/chunked serving | 政策無外部效度 | mixed workload、chunked/prefix/speculative ablations |
| switching/pool splitting 可免費實現 | oracle 不可部署 | 計入 duplicate weights、idle pool、cold start、KV transfer |
| crossover 單一且穩定 | first crossing 可能誤導 | 多交叉、tile/autotune thresholds、retest |
| clocks、power、thermal 相同 | peak/ridge workload-dependent | 鎖 power/clock，交錯 time blocks，收 telemetry |
| Qwen3-32B 可外推其他架構 | 結論只對單一 dense model | held-out dense、MoE、MLA、不同 TP |
| factory throughput 足以定 SLO | goodput story失效 | open-loop arrival blind tests |
| GPU-seconds/token 等於 $/token | 金錢主張失效 | 納入租價、利用率、保留容量與能源 |

## C12 — REFUTE

狹義「native INT8 arithmetic dense ceiling約 150–153.5 TOPS」成立；「任何 vLLM merge 都只能改善 kernel maturity」不成立。

### 路徑 (a)：INT8 checkpoint→BF16 execution

B300 dense BF16 約 2.2–2.25 PF，是 native INT8 ceiling 的約 14.7×。若 fused W8A16 on-the-fly dequantize，仍可能保留 W8 weight bandwidth；若 load-time 展開成 BF16，resident bytes 則改變。實效未必達 2.25 PF，但已不受 150 TOPS ceiling 限制。

現有 INT8-W8A16 control 只有以下皆匹配時才足夠：

- 同一 INT8 checkpoint、scale、ignored layers；
- 與未來 merge 相同的 load-time/on-the-fly conversion；
- profiler 證實 activation/MMA 為 BF16/FP16；
- resident layout、workspace、quality 相同。

另一個 W8A16 recipe 不能自動代理未來 default path。

### 路徑 (b)：INT8 checkpoint→FP8 execution

E4M3 在 [16,32) 間距為 2，17 等 INT8 值不可精確表示，因此必須重跑 quality gates；但 load-time 或 fused requantization 後可走 4.5 PF FP8 pipe。現有 W8A16 arm 未涵蓋此情境。

vLLM 已有 checkpoint/execution 分離先例：compressed-tensors 可在 checkpoint W8A8 不受支援時忽略 input quantization，改用 W8A16 execution。[vLLM compressed-tensors dispatch](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)

研究仍可存活，但 action 軸應改成：

serialized checkpoint × resident dtype/layout × execution/MMA format × backend/kernel

並新增 INT8-checkpoint→FP8-execution arm，與由同一 BF16 parent 直接量化的 FP8 checkpoint 比較品質與效能。不能再寫 survives any merge。

## C13 — PARTIAL

公開 pipeline 支持「目前沒有 performance-grade SM103/B300 INT8 PR」，但不能支持「沒有人會投資」的經濟學必然性，也無法排除 NVIDIA 私有 roadmap。

| 工作 | 狀態／日期 | 分類 | 判讀 |
|---|---|---|---|
| [vLLM #27182](https://github.com/vllm-project/vllm/issues/27182) | 2025-10-20 建立；2026-02-19 not_planned 關閉 | 相容性需求 | 無實作，負面訊號 |
| [vLLM #45126](https://github.com/vllm-project/vllm/pull/45126) | open，更新至 2026-08-10 | performance | 只調 SM89/90，非 Blackwell |
| [vLLM #44501](https://github.com/vllm-project/vllm/pull/44501) | draft，2026-06-04 | performance | CuTeDSL FP8/INT8，但明列 Hopper SM90 |
| [vLLM #34556](https://github.com/vllm-project/vllm/pull/34556) | 2026-04-24 merged | compatibility/JIT | Humming fallback/requant；非 SM103 UMMA |
| [SGLang #28137](https://github.com/sgl-project/sglang/pull/28137) | open，2026-06-13 | compatibility | SM120/121 復用 legacy SM80 path，明言不 tuning |
| [SGLang #31783](https://github.com/sgl-project/sglang/issues/31783) | open，2026-07-20 | roadmap | 重點 FP8/NVFP4/MXFP4/Humming，無 SM103 native INT8 |
| [SGLang #34072](https://github.com/sgl-project/sglang/pull/34072) | draft，2026-08-08 | FP8 performance | 標作 GB300 W8A8，實際是 per-token-group FP8 |
| [CUTLASS #2717](https://github.com/NVIDIA/cutlass/issues/2717) | open，2025-10-24 | compatibility request | SM120 INT8，無 PR，且非 SM103 |

未找到 SM103 native INT8、SM103-tuned Triton INT8，或專門的 B300 INT8-checkpoint→FP8/BF16 PR。不過 Humming 已有通用 origin quant→fp16/bf16→target quant 的 requantization 機制，[source](https://github.com/vllm-project/vllm/blob/89c8401c8aeb53ccee87060e913514068e0a1e82/vllm/model_executor/layers/quantization/humming.py#L287-L323)，所以 conversion loophole 並非純理論。

較可靠表述：

> 截至 2026-08-13，公開三專案沒有 performance-grade SM103/B300 INT8 PR；短期 merge hazard 較可能是 legacy/JIT fallback 或 requantization integration，而非新增 tcgen05 INT8。

定案：保存 GitHub search snapshot，在 predictor freeze 與首次 B300 run 前各重查一次；私有 roadmap 只能由 NVIDIA/CUTLASS 維護者確認。

## C14 — PARTIAL

同一新 digest 內開／關 backend，確實消除了「整個 release diff」這個主要 confound；但 treatment 是完整 backend path，不是單一 kernel body。

殘餘 confounds：

- selector 在 model construction 執行，兩臂需各自冷啟 process；
- CUTLASS/Triton/Humming 的 weight transpose、scale、activation quant、workspace、epilogue 不同；
- workspace/residency 可能改變 KV blocks、max concurrency、CUDA graph bucket 與 scheduler；
- 禁用 CUTLASS 後不保證落到 Triton，可能是 Humming；
- 同一層的不同 M/N/K、prefill/decode、dense/MoE 可能走不同 kernel；
- JIT、autotune、graph capture 與 cache warm-up 不同；
- 數值差異可能改變 EOS、MoE routing 與實際工作量。

建議三層設計：

1. 固定 (M,N,K)、相同 tensor/scales，做 CUDA-event＋NCU microkernel 比較。
2. 同一新 digest、乾淨重啟、明確 force backend、固定 graph/KV planning，量完整 backend total effect。
3. 要估計 PR 本身的 merge effect，用同一 parent commit 的 PR apply/revert 兩個 image，或加入只切新舊實作的狹義 runtime flag。

跨 digest 只能作描述性 corroboration；同 digest disable/enable 可以支持 stack-path effect，不能宣稱唯一改變的是 GEMM kernel。

## 最可能錯的三項

1. C3：用 FP8 ridge 判斷兩臂何時分岔；真正較弱的 B300 INT8 約在 B≈10 已跨 ridge，是直接的數學錯誤。
2. C6：已有多個 phase-aware、PD quantization、load-adaptive precision 先例；exact decoupling 也缺乏成立條件。
3. C12：把 checkpoint format 當成 execution arithmetic；BF16/FP8 conversion 可以繞過 150 TOPS native INT8 ceiling。
