# 預註冊 v3：H200×B300 量化格式 Crossover 量測研究

**v3 鎖定日期：2026-08-13（Asia/Taipei）。本文件在任何目標硬體（H200/B300）數據存在之前鎖定。**
v3 取代 v2（tag `prereg-v2-locked` 保存 v2 全文）：動因＝Codex 敵對審查（C1–C14，`notes/ADVERSARIAL_REVIEW.md`，SHA `bd6bba4722f28513`）及本專案對該審查的五線獨立覆核（數學重算、NVIDIA 一手文件、文獻、原始碼、GitHub pipeline）。全部修訂於任何目標硬體數據之前完成；變更清單見 A-4。
v2 取代同日 v1：v1 完成後立即接受三線敵對審查（洩漏／估計器／可 game 性），審查發現 4 FATAL、17 MAJOR、6 MINOR，全部在收取任何數據之前修入該版。審查全文存於 workflow 紀錄。

修改治理（取代 v1 的單一 Amendment 規則）：
- **A 類（預資料）**：首個目標硬體數據點之前的修改，自由記錄於文末。
- **B 類（資料後）**：僅限本文件**點名**的決策點（§8 NVFP4 recipe、§10 Gate 1 降級），各附決策準則與最晚時點。任何其他資料後修改＝協定偏離，須在論文中如實報告。

## 0. 鎖定的 artifacts（SHA-256 前 16 碼；v2 定稿時重算）

| 檔案 | SHA-256[:16] | 角色 |
|---|---|---|
| `experiment/harness/changepoint.py` | `a8b58ce3859a544f` | 估計器 v2（censored-inclusive CI、多交叉回報、失敗規則、paired block bootstrap、knee 存在性檢定、penalty 傳播） |
| `experiment/harness/validate_synthetic.py` | `6216d8c9e09088d0` | 驗收 v2（真實幾何 G13/G9/G9r5/G6、σ∈{.05,.10,.15}、S1–S10、K1–K2） |
| `experiment/harness/gen_sweep.py` | `a77697682c3029f4` | sweep 展開器（A-7：矩陣路徑解析明示化＋執行時印出所讀矩陣之路徑與 SHA；展開邏輯未變） |
| `experiment/config_matrix.csv` | `949d996a5c055e2e` | 實驗矩陣 v4（28 實驗、14,691 runs；v3 基礎上加入 REQUANT-B300-{DECODE,PREFILL} 診斷臂）。名目網格，實際執行受 §3.5 容量包絡裁切 |
| `experiment/harness/capacity_envelope.py` | `580105ebb64e3018` | §3.5 容量包絡規則之實作（B_max、crop_grid、infeasible 記帳；含 self-test） |
| `experiment/harness/qwen3_32b_config.json` | `97e295b632839357` | 主模型架構參數 |
| `experiment/harness/quality_gates.py` | `41b922dd7d2ce1d5` | §7 三個 quality gates 之實作（PPL／一致率／needle；含 A-3 之窗口切分規則） |

驗證設定聲明：crossover 覆蓋率以鎖定之 B=1000 驗證；knee-CI 與 penalty 情境以 B=300、60–100 trials 驗證（覆蓋率標準誤 4–6%，驗收帶已吸收），此為預先聲明的驗證設定。

## 1. 假說與判定準則

### 1.0 機制框架：逐格式 ridge（v3 新增；修正 v2 時期只算 FP8 ridge 的框架錯誤）

Roofline ridge 依（硬體, 格式）分開計算：ridge_f = dense peak_f ÷ HBM BW。以一手文件釘定的規格（§4 B3 表）：

| 硬體 | FP8 ridge | INT8-native ridge | BF16 ridge |
|---|---:|---:|---:|
| HGX B300（7.7 TB/s） | 584 op/B | **19.5–19.9 op/B** | 292 op/B |
| H200（4.8 TB/s） | 412 op/B | 412 op/B | 206 op/B |

主模型（Qwen3-32B，鎖定 config）之 weight-dominated 線性層 AI ≈ 2·B（B＝共享一次權重讀取的 active tokens）。由此登記以下**預資料機制預期**（非判定準則，供結果解讀與 B3 對照）：

- **B300 INT8-native 的線性層於 B≈10 即進入 compute-bound**（150–153.5 TOPS 天花板）；FP8 線性層在 B≤256 全程 memory-bound（weights-only knee ≈292，含 activation IO 後 ≈330–341）。兩臂差距自 B≈10 開始張開，**非** v2 時期誤登記的「到 B>256 都幾乎相等」。
- 因此 B300 上 INT8-native 對 FP8 的預期結構為 **FP8 ⪰ INT8 於全負載域**（memory-bound 區由 equal-bytes 下限貼近、compute-bound 區由 30:1 拉開）：winner-flip 型 crossover 預期**不存在**，估計器將以 `below`/`above`/`no-crossing-indicated` 狀態回報，divergence-onset（差距張開點）由 knee 估計器定位。此預期**不適用**於 kernel 效應主導的情形（如 fallback 路徑），該情形正是 H2 校正分析的對象。
- **Decode 的 KV 稀釋**：attention/KV 讀取之 AI＝16/q（q＝KV bytes/value），永遠 memory-bound 且兩臂相同（KV dtype 匹配時）。端到端 decode 差距隨 context 增長而被稀釋（B=256 時 KV byte 占比：2K/8K/32K ≈ 68/90/97%，FP8 KV）；線性層與 attention 為序列執行之不同 kernels，分析不得以全模型平均 AI 掩蓋瓶頸切換，逐 kernel 計時為準（§3.5、§5.5）。
- H200 為同矽對照：兩格式 peak 相同（1,979），serialized 模型下比值恆為 1；**H200 上任何實測 INT8–FP8 差距均為 kernel/軟體效應**，此為 backend penalty 校正（§5）的識別基礎。

v2 時期於 decision log 登記的「B300 decode crossover 右設限（>256）」假說**於此撤回並取代**（預資料撤回；原紀錄保留於 decision log 不刪改）。

- **H1**：`hardware × format × regime` 交互作用。判定：§9 模型的交互項以 time-block cluster-bootstrap Wald 聯合檢定，α=0.05；regime 編碼＝log₂(load) 連續軸 + phase 類別。
- **H2**：兩組 crossover（INT8–FP8、NVFP4–FP8）位置在兩硬體間位移；INT8–FP8 於 B300 依 §1.0 預期呈非交叉結構時，H2 之該側改以 divergence-onset（knee）位置與 below/above 狀態的硬體間對比表述。**B300 INT8 側的 H2 主分析使用校正曲線**（`corrected_crossover_ci`，P 不確定性在每個 bootstrap draw 內重抽傳播）；raw 曲線並列報告。`penalty_extrapolated` 的格點不進入 H2 確證。
- **H3**：predictor 在 §3 held-out 上優於三 baseline。**適用矩陣**：winner accuracy 與 regret 對 B1/B2/B3；boundary error 與 calibration 僅對 B3（B1/B2 不產生位置與區間，該欄不適用——此為 v1「全部指標優於全部 baseline」的修正）。**H3 的 B300 INT8 標籤一律使用 RAW 量測**（predictor 亦以 raw 預測），校正僅屬 H2 的硬體歸因分析——預測與標籤不得同時通過同一條校正曲線。
- **H4**：predictor 引導配置於**下列兩個 SLO 至少其一**優於最佳固定格式配置。
  - **SLO-A（互動）**：TTFT p99 ≤ 500 ms 且 TPOT p50 ≤ 50 ms。
  - **SLO-B（批次）**：TTFT p99 ≤ 2000 ms 且 TPOT p99 ≤ 100 ms。
  - **Goodput** ≝ 每 GPU-秒內完成且全程滿足該 SLO 兩項條件的請求數。**GPU-hour/token** ≝ 掛鐘時間×GPU 數 ÷ 產出 tokens。判定：配對 bootstrap（B=1000）之 goodput 差 90% CI > 0，逐 SLO 報告，兩個都報，不得只報有利者。
  - **推導限制（v3 新增）**：吞吐 sweep（closed-loop 飽和量測）僅作為 H4 的**校準輸入**（縮小候選配置空間、供 predictor 擬合），**不得**用以推導任何 SLO/goodput 結論——尾延遲由服務時間二階矩與到達過程決定（M/G/1：E[W_q]=λE[S²]/2(1−ρ)），穩態吞吐不含此資訊。H4 判定僅得出自專用 open-loop 量測：到達過程至少含 Poisson 與一種 bursty（MMPP 或實 trace），各於 ≥2 個 utilization 水準；此等 runs 於 §3.1 屬 EXCLUDED-FROM-PREDICTOR。

### 1.1 Predictor 規格（現在鎖定，杜絕凍結漏洞）

Predictor ＝ **確定性機制模型，無可調超參數**：對每（hardware, format, phase），由 TRAINING-ELIGIBLE 量測擬合 attained-roofline 參數（attained slope、attained peak；B300 INT8 另乘 raw 路徑之 P 曲線——僅供其 raw 預測），對任一 query 點輸出預測吞吐曲線，crossover 由同一 `changepoint.py` 估計器在預測曲線上求得；預測區間＝roofline 參數之 bootstrap 經預測函數傳播。**沒有模型選擇、沒有超參數搜尋、沒有學習組件**；因此凍結內容＝擬合出的 roofline 參數與選點清單。攀 roofline 之 attained 參數依 §1.0 逐格式擬合（各臂各自的 attained peak 與 slope），不共用 FP8 ridge。

### 1.2 Predictor Freeze Amendment（必要程序）

首個 held-out 量測（含 405B、ISL=8192 開封、off-grid 值、MICRO-*-405B）之前，必須存在一則 Freeze Amendment，內含：predictor 程式與擬合參數之 SHA-256、405B 選點完整清單、以及**可外部驗證的時間戳**（推送至公開 git remote 的 signed tag，或 OSF registration）。缺少先行凍結紀錄的 held-out 量測一律作廢。

## 2. 估計器（v2 語義；程式為準）

- Crossover：中位數曲線、log₂ 內插；**回報全部交叉**（`n_crossings`、清單），主分析取第一個；`n_crossings>1` 的 pair 排除於單交叉 calibration 指標並全數列報。
- 無交叉：端點 |d| 相差 <20% → `no-crossing-indicated`（不猜方向）；否則 `below`/`above`。**不外插。**
- CI：paired **block** bootstrap（repeat index＝time block；A/B 同格點同 block 綁定重抽），B=1000；censored draws 以 ±∞ 進入百分位，落在 censored 質量上的界以開界（`<x₁`/`>xₙ`）回報；另報 `censor_frac` 與 `multimodal_frac`。
- 失敗規則：y≤0 或缺 repeats → 該格點自**兩臂**同時剔除並記錄；存活格點 <5 → `unestimable`。**容量不可行格點（§3.5）非失敗亦非 censored**：自兩臂剔除、獨立記帳、不進入任何狀態判定。
- Knee：僅適用 n≥6 之網格（適用列：MICRO-*、QWEN-PREFILL-*、QWEN-DECODE-*）；存在性檢定 SSE₁段/SSE₂段 ≥ 3.0（v2 驗證中 1.5 對純冪律假 knee 率 22%，故提高；見 A-2），否則 `no-knee`；QWEN-MIXED（3 點）與 BACKEND-PENALTY（4 點）**不做 knee 估計**。
- B300 INT8 校正：`corrected_crossover_ci`，P 於每 draw 重抽；P 支撐域外→nearest-P 且標記 `penalty_extrapolated`。

## 3. 資料角色與 held-out split

### 3.1 全實驗角色表（每個 experiment_id 恰一角色）

| 角色 | experiment_id |
|---|---|
| TRAINING-ELIGIBLE | MICRO-H200-QWEN、MICRO-B300-QWEN、QWEN-PREFILL-{H200,B300}、QWEN-DECODE-{H200,B300}（扣除 §3.3 保留桶） |
| CALIBRATION-ONLY（防火牆：不入 predictor 訓練，僅供 §5 校正與品質分析） | BACKEND-PENALTY-H200、BACKEND-PENALTY-DECODE-H200 |
| HELD-OUT | LLAMA405-{H200,B300}、MICRO-{H200,B300}-405B、各 grid 之 ISL=8192 桶、off-grid load 值（§3.3）、SESSION-RETEST-* |
| DIAGNOSTIC-POST-FREEZE | LLAMA405-LAYER-{H200,B300} |
| EXCLUDED-FROM-PREDICTOR（robustness/應用，不入訓練亦不入 H3 評分） | QWEN-MIXED-*、QWEN-TP-SENS-*、SGLANG-ROBUST-*、PD-*、H4 open-loop SLO runs、INT8→FP8-requant 診斷臂（§5.5） |

「selected-boundary-points」（TP-SENS/SGLANG 列）＝以**量測到的** causal-grid crossover 兩側相鄰格點為準的確定性規則，非 predictor 選擇。

### 3.2 Shape hold-out
405B 全部 layer shapes（含 MICRO-*-405B 與 LLAMA405-LAYER-*）。Roofline 擬合、特徵縮放、任何超參選擇不得接觸（v1 FATAL 修正：MICRO 已拆分為 -QWEN 與 -405B 兩列）。

### 3.3 Load 與 context hold-out（逐軸、僅限網格範圍內）
- MICRO active-token 軸（1..4096）：hold-out {3, 12, 48, 192, 768, 3072}。
- Decode concurrency 軸（1..256）：hold-out {3, 12, 48, 192}。
- Prefill batch 軸（1..32）：hold-out {3, 12, 24}。
- Context：ISL=8192 桶整層（**修正 v1 不一致**：prefill 訓練 ISL={128,512,2048,32768}；decode 訓練 ISL={128,2048,32768}——decode 網格本無 512）。
- **隔離協定**：harness 將全部 held-out 格點寫入獨立檔案，收集當下即計 SHA-256 並記錄；Freeze Amendment 之前不得開啟；每次開封記錄於 Amendment。
- Held-out 格點若依 §3.5 落入容量不可行域：記錄為 infeasible、自評分分母剔除、計數列報（不以任何方式替換或移點——替換即洩漏選點自由度）。

### 3.4 405B 端到端選點（規則現鎖）
每（hardware, crossover pair, phase）：以凍結後 predictor 的 x̂* 為中心，取 **7 個 log-uniform 點於 [x̂*/4, x̂*×4]**（裁剪至可行域），**加 4 個固定錨點**（decode: concurrency {8, 64, 224}；prefill: batch {2, 24}——與任何 hold-out 值不重合），使嚴重誤預測可被觀測。全部點列入 Freeze Amendment。

### 3.5 容量可行性包絡（v3 新增）

名目網格（decode concurrency ≤256 × ISL ≤32768）含物理上不可實現的格點。規則：

- 每（hardware, model, 量化臂, KV dtype, TP）於權重載入後量測**實際可用 KV 記憶體** M_KV（engine 回報值為準，非規格 VRAM），計 **B_max(S) = ⌊M_KV ÷ (c_KV·S·q)⌋**；Qwen3-32B 之 c_KV=131,072 bytes/token（q=1）。B > B_max(S) 之格點標記 **INFEASIBLE-BY-CAPACITY**，自兩臂同步剔除、獨立記帳、不計入 censored/failed、不進入估計。包絡表隨資料發布。
- **預資料預期**（規格試算，270 GB HGX B300 / 141 GB H200、W8 權重、FP8 KV）：B300 之 B_max ≈ 867/216/54（ISL 2K/8K/32K）；H200 ≈ 394/98/24。即 **B=256 於 ISL≥8192 在兩硬體皆預期不可行**；BF16 KV 或 BF16 權重臂再減半。實際以量測 M_KV 為準。
- 兩臂之可行域可能**因格式而異**（BF16 權重臂 vs W8 臂）：配對分析僅於兩臂交集可行域內進行；差集格點列報但不進入 crossover 估計。

## 4. Baselines（可唯一重實作）

- **B1 全域單一格式（Π₀）**：每 **hardware**，對該硬體 QWEN-{PREFILL,DECODE} **兩個 phase 聯集**之全部 TRAINING-ELIGIBLE 格點，取**中位吞吐之幾何平均**最高的**單一格式**（跨 phase 共用同一格式）。〔v3 修正：v2 誤寫為「每 hardware×phase」，使 B1 與 B2 不可區分、Π₀→Π₁ gap 不可估——Codex 審查 C5 指出。〕
- **B2 phase-only（Π₁）**：同 B1 聚合，惟每 **hardware×phase** 各自選格式。
- **B3 spec-peak roofline**：crossover 預測 = 兩格式規格 dense 峰值與公布頻寬構成之理想 roofline 交點，**逐格式各用其自身 ridge**（§1.0）。**規格數字表（v3 以一手文件填定）**：
  | (硬體, 格式) | dense 峰值 | 來源 |
  |---|---|---|
  | H200 FP8 / INT8 | 1,979 TFLOPS / 1,979 TOPS | H200 datasheet（3,958 sparse ÷2；SXM 欄） |
  | H200 BF16 | 989.5 TFLOPS | 同上（1,979 sparse ÷2） |
  | B300 FP8 / INT8 | 4.5 PFLOPS / 150 TOPS | Blackwell Architecture Technical Brief Table 3（Dense/Sparse 標註列）。Blackwell Ultra Datasheet 給 INT8 sparse 307→dense 153.5；**B3 主用 150，153.5 作敏感度**（NVIDIA 文件間 ±2.3% 不一致如實記錄） |
  | B300 BF16 | 2.25 PFLOPS | Blackwell Ultra Datasheet（4.5 sparse ÷2；Brief 捨入為 2.2） |
  | B300 NVFP4 | 14 PFLOPS | Blackwell Ultra Datasheet per-GPU 欄（18 sparse｜14 dense）；系統總值 108÷8=13.5 之內部不一致如實記錄，**B3 主用 14，13.5 作敏感度** |
  | HBM 頻寬 H200 / HGX B300 | 4.8 TB/s / **7.7 TB/s** | H200 datasheet；Blackwell Ultra Datasheet「270 GB HBM3E｜7.7 TB/s」。**8.0 TB/s 屬 GB300 NVL72（279 GB），非本研究 SKU，不得混用** |
  B3 公式：x*_B3 = 兩格式 peak 相等點於理想 roofline min(bw·x·bytes⁻¹, peak_f) 下之解析交點；無交點→censored 預測。

## 5. B300 INT8 報法與 P(regime)

- P 參數化：per-phase，對 log₂(load) 之分段線性內插；prefill 軸由 BACKEND-PENALTY-H200（batch 1|4|16|64；ISL 128|2048|32768，**8192 由內插**）、decode 軸由 BACKEND-PENALTY-DECODE-H200（concurrency 1|4|16|64|256）。支撐域外→nearest-P 並標記，`penalty_extrapolated` 格點不入 H2 確證。
- **路徑同一性條件（v3 新增）**：P 為 **fallback-path penalty**（非泛稱 Triton penalty）。P 在 H200 上以「CUTLASS 路徑 vs 實選 fallback 路徑」量測；其對 B300 之適用**以 §5.5 verification gate 確認 B300 實選 fallback 與 H200 量測之 fallback 為同一 kernel 家族為前提**（vLLM 之 INT8 候選序為 CUTLASS→Triton→Humming，禁用 CUTLASS 後不保證落於 Triton）。若 B300 實選路徑不同（如 Humming），P 不得套用，須以 A/B 類 Amendment（依時點）記錄並改以該路徑重量 P 或放棄校正。
- H3 一律 raw；H2 主分析 corrected（§1）。校正臂與參照臂**不得**共用同一條校正（參照臂 FP8 永為 native、不校正）。

### 5.5 格式軸分解與執行驗證 gate（v3 新增）

**「格式」非單一變數。**每個 run 記錄四個觀測欄位，臂由四元組定義，任何主張須指名四元組而非僅稱「INT8」：

1. **checkpoint 格式**（序列化權重：INT8-W8A8 / FP8-W8A8 / NVFP4 / BF16）；
2. **resident dtype/layout**（載入後駐留形式，含 load-time 轉換）；
3. **execution/MMA dtype**（實際張量核心算術，profiler 驗證）；
4. **backend/kernel identity**（實選 kernel，逐層記錄於 run manifest）。

命名臂：**INT8-native**（exec=INT8，UMMA 或 legacy IMMA，逐層記何者）、**INT8-fallback**（exec=INT8 經非 CUTLASS 路徑）、**INT8-W8A16**（INT8 checkpoint→BF16 execution 對照）、**INT8→FP8-requant**（INT8 checkpoint 經 requantization→FP8 execution；**v3 新增之 P2 診斷臂**，動因＝C12 模擬漏洞：vLLM 對 FP8-W8A8 已有 silent W8A16 降級先例、Humming 已有 origin→bf16→target requant 機制，故 conversion 路徑非理論；此臂須先過 §7 quality gates 方可量吞吐，且屬 EXCLUDED-FROM-PREDICTOR）、FP8-native、NVFP4、BF16。矩陣 v4 加入 requant 臂後以 A 類 Amendment 重釘 SHA。

**執行驗證 gate**：每（hardware×臂×backend）之數據**進入任何分析前**，須有該組合之 NCU 證據：INT8 execution 以 `sm__inst_executed_pipe_tensor_op_imma.sum > 0` 且無其他 tensor-pipe 主導為準（kernel 名稱比對在 sm_103 上為假陰性，禁用作判準）；FP8/BF16/NVFP4 臂以對應 pipe counters 為準。無驗證證據之數據隔離不用。逐層實選 kernel log 隨 run manifest 保存。

**Merge 韌性表述（v3 收窄，取代「survives any merge」）**：150–153.5 TOPS 之矽天花板僅約束 **execution=INT8** 之臂；任何以 conversion（W8A16、requant→FP8）服務 INT8 checkpoint 的未來 merge 不受此天花板約束，屬**執行路徑改變**，由本節四元組觀測與驗證 gate 捕捉為新臂，而非推翻既有臂之結果。同 digest 內 kernel 開關之 ablation 估計的是 **backend-path total effect**（含 epilogue、workspace、graph、記憶體規劃之全部差異），非單一 kernel body effect，全部主張依此措辭；跨 digest 對比僅作描述性佐證；PR 級歸因需同 parent commit 之 apply/revert 建置（選做）。

## 6. 評估指標（含平手與混合情形）

- **Winner accuracy**：量測勝者＝中位吞吐較高者；若配對 block-bootstrap 90% CI（中位 log 差）含 0 → **statistical tie**，自分母剔除並報 tie 數（predictor 與全部 baseline 同規則）。
- **Boundary error**：雙方皆數值 → |log₂x̂*−log₂x*|，報中位數；**censored 對 censored 僅在方向一致時記正確**；數值對 censored（任一向）→ accuracy 記錯、不入 |log₂| 聚合，計數列報。
- **Calibration**：predictor 90% 區間之覆蓋率；比較量 |coverage−0.90|，僅對 B3。
- **Regret**：選錯格式相對 oracle 的中位吞吐損失。
- **OOD degradation**：Qwen-trained → 405B 之上述指標變化。
- **Oracle gap 之表述限制（v3 新增）**：Π₀→Π₁→Π₂ 之 oracle 差距為 **clairvoyant upper bound**，不得表述為可部署節省或「現行做法留在桌上的錢」。主 estimand **明確排除**下列成本：pool 拆分之雙份權重駐留、閒置容量、cold start、路由、KV 轉換/搬移、切換延遲、誤預測——此等成本僅於 PD-* 與 H4 之**已實現政策**量測中內生地計入；可部署價值主張只得出自 H4。GPU-seconds/token 至 $/token 之換算需另設租價與利用率假設，僅得於論文以獨立敏感度小節呈現，不進入任何假說判定。

## 7. Quality gates（全部釘死）

資料集：`Salesforce/wikitext`，config `wikitext-2-raw-v1`，**test split**，dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- **PPL**：串接全 test split，context window 4096、**非重疊 stride**，token-level NLL 之 exp；相對增幅 INT8/FP8 ≤2%、NVFP4 ≤5%。
- **一致率**：prompts＝該 test split 全文串接後（保留原始換行）以 GPT-2 tokenizer 切成不重疊的 320-token 窗口，**依序取前 200 個完整窗口**，每個窗口取**前 64 tokens** 為 prompt（確定性；prompt 清單 SHA 隨 artifacts 發布）。〔A-3 修正：v2 原寫「前 200 篇 ≥256 tokens 之文件」，但該 split 僅含 63 篇頂層文件，物理上不可能取 200 篇；改為窗口切分，維持「確定性、可重算、涵蓋整個 split」的原意。〕greedy 256 tokens；**每 prompt 分數＝首次分歧前吻合 tokens 數 ÷256，gate 於 200 篇平均**；INT8/FP8 ≥95%、NVFP4 ≥90%。**BF16 參照**＝H200、釘住之 vLLM image、enforce_eager、greedy；各（arm×hardware×backend）對同一參照計分；另報 BF16-on-B300 對參照的一致率作為跨硬體數值基線。
- **Needle-in-haystack**：haystack＝同資料集 test split 串接；needle＝固定句 "The secret checkpoint code is 731942."（於 depth 處插入）；depth {0,10,…,100}%×length {8k,16k,32k}＝33 cells、每 cell 1 次確定性 greedy；問句固定 "What is the secret checkpoint code?"；通過＝回答含子字串 "731942"；分數＝33 cells 通過率；降幅 ≤5 pts（相對同硬體 BF16 參照，參照分數一併列報）。評測腳本＝`experiment/harness/quality_gates.py`（SHA-256[:16] `41b922dd7d2ce1d5`，見 §0；A-6 釘定，先於任何目標硬體 quality-gate 執行）。
- Gate 範圍：逐（arm×hardware×實際 benchmark 之 backend）評定；某硬體上未過 → 僅該硬體之等品質比較排除該臂，全部 gate 數字照報。INT8→FP8-requant 臂（§5.5）之 gates 於其任何吞吐量測之前完成。

## 8. Matched-quantization recipes

- Parent：`Qwen/Qwen3-32B` BF16（sha `9216db5781bf21249d130ec9da846c4624c16137`）；校準集 512 樣本、固定 IDs 與 seed（隨 artifacts 發布）。
- **INT8-W8A8（鎖定）**：llm-compressor，channel-wise symmetric weights + dynamic per-token activations。
- **FP8-W8A8（現在鎖定，無 Amendment 路徑）**：llm-compressor，**per-channel weights + dynamic per-token activations**——與 INT8 同 granularity 家族，消除參照臂自由度。
- **NVFP4（B 類決策點）**：候選＝llm-compressor 原生 與 Four-Over-Six。決策準則：**僅依 §7 quality gates**（對吞吐與 crossover 全盲）；兩者皆過 → 取 llm-compressor 原生；最晚時點＝任一硬體首次 NVFP4 吞吐量測之前；先於 Amendment 的任何 NVFP4 吞吐量測作廢。
- KV dtype、scheduler、CUDA graph 兩硬體一致；不一致處記錄並做敏感度分析。

## 9. 統計模型

主分析＝各條件內配對 block bootstrap（B=1000）中位差；交互作用＝log-throughput 對 hardware×format×regime 線性模型、time-block cluster bootstrap SE、Wald 聯合檢定 α=0.05；每點 ≥7 重複（P2 允許 5，其 CI 依 §0 驗證之較寬驗收帶解讀），跨 time-block 隨機化順序、固定輸出長度。

## 10. 停止規則與既定決策點

- 熱節流／`:latest` 漂移／單 host 限制：同 v1（Gate 0/4；digest 釘死；主張限縮）。
- **Gate 1 失敗（B 類決策點）**：Triton 路徑無 IMMA → INT8 降為 kernel 層證據，§5 作廢、H2 之 INT8 部分改由 IMMA 微基準回答；Amendment 須引 `ncu_verdict` log。Gate 1 同時記錄 B300 實選 fallback 路徑之 kernel 家族（§5 路徑同一性條件之輸入）。

---

## Amendments

### A-1（A 類，2026-08-12，v2 定稿同時）
v1→v2 全面修訂，動因＝預資料敵對審查（4 FATAL/17 MAJOR/6 MINOR）。要點：MICRO 拆分修 shape 洩漏；penalty 移出 ISL=8192 並新增 decode 軸列；H3 raw-label／H2 corrected 分離；H4 SLO 具體化；NVFP4/FP8 決策收緊；quality gates 全釘；估計器 v2（censored-CI、多交叉、失敗規則、block bootstrap、knee 存在性、P 傳播）；驗證擴至真實幾何與 S1–S10/K1–K2。v1 全文保留於版本歷史。

### A-2（A 類，2026-08-12，v2 驗證完成）
§0 SHA 已填。驗證於 `gpu-worker-A` 全數通過（19/19）：S1 六種幾何（G13/G9/G6、r∈{5,7}、σ∈{.05,.10,.15}）誤差 0.023–0.089、覆蓋 0.905–0.95；S2 自信假交叉率 0.015（≤0.10；總假交叉 28% 屬解析度極限內、98.5% 帶不確定性旗標——邊緣 gap <2.5×中位數標準誤時本方法不宣稱能分辨無交叉與交叉，見 §2）；S3 離群穩健 0.028；S4 近切線情境 99.5% 帶旗標（文件化解析度極限）；S5 平滑曲線 0.119／knee 偏差 0.144；S6 雙交叉偵測率 1.00、首交叉誤差 0.024；S7 block 效應下覆蓋 0.94；S8 penalty 傳播覆蓋 0.94；S9 純冪律 no-knee 率 0.98（existence 門檻 1.5→3.0，門檻 1.5 時假 knee 率 22% 之教訓記於程式註解）；S10 近邊緣覆蓋 0.935；K1 knee 誤差達標、K2 knee CI 覆蓋達標。完整 log：`logs/synthetic_v2b.log`（VM）。

### A-3（A 類，2026-08-12）
§7 一致率的 prompt 來源由「前 200 篇文件」改為「串接後不重疊 320-token 窗口取前 200 個、各取前 64 tokens」。動因：實作前查證發現 `wikitext-2-raw-v1` test split 僅有 63 篇頂層文件（4,358 行、1,294,336 字元），原規格無法滿足 200 篇。資料集 revision、PPL 協定、needle 協定均不變。此為預資料修訂（尚無任何目標硬體數據）。

### A-4（A 類，2026-08-13，v2→v3：Codex 敵對審查 C1–C14 及五線獨立覆核後之修訂）
所有修訂於任何目標硬體數據之前完成。審查檔：`notes/ADVERSARIAL_REVIEW.md`（SHA `bd6bba4722f28513`）；覆核＝五個獨立 agent（數學重算、NVIDIA 一手文件、文獻、原始碼逐行、GitHub pipeline），結論：Codex 數學與來源實質全數成立。變更：
1. **§1.0 新增**：逐格式 ridge 框架。v2 時期框架僅算 FP8 ridge（584 op/B）而漏算 INT8-native ridge（19.5 op/B），誤預期「decode 兩臂到 B>256 幾乎相等」；正確結構為差距自 B≈10 張開、B300 上 FP8⪰INT8-native 全域。decision log 中的右設限假說預資料撤回。
2. **§3.5 新增**：容量可行性包絡。規格試算顯示 B=256×ISL 32K 需 1.1 TB KV（q=1）、於任何候選 GPU 皆不可行；B=256×8K 於 B300(270GB)/H200 亦不可行（B_max≈216/98）。名目網格依量測 M_KV 裁切，infeasible ≠ censored ≠ failed。
3. **§4 B1 修正（FATAL）**：v2 的 B1 誤寫「每 hardware×phase」與 B2 不可區分；改為每 hardware 之跨 phase 單一格式（Π₀）。B3 規格表以一手文件填定（HGX B300：7.7 TB/s、270 GB、INT8 150/153.5 雙源、NVFP4 14/13.5、BF16 2.25；8.0 TB/s 屬 GB300 NVL72 明文禁混）。
4. **§5.5 新增**：格式四元組（checkpoint/resident/execution/backend）、命名臂（含 INT8→FP8-requant 新診斷臂，動因＝C12 conversion 漏洞：compressed-tensors 對 FP8-W8A8 有 silent W8A16 降級先例〔INT8-W8A8 無能力檢查、直接 crash——覆核修正 Codex 此處過廣之表述〕、Humming 有通用 requant 機制）、NCU 執行驗證 gate（IMMA counter 判準）、merge 韌性表述收窄（「survives any merge」撤回）、同 digest ablation＝backend-path total effect 之措辭約束。
5. **§5 路徑同一性條件**：P 改稱 fallback-path penalty；適用前提＝B300 實選路徑與 H200 量測路徑同 kernel 家族（候選序 CUTLASS→Triton→Humming，禁用後不保證 Triton）。
6. **§1 H4 推導限制**：吞吐 sweep 僅為校準輸入，SLO 結論僅得出自 open-loop 專用量測（M/G/1 二階矩論證）；到達過程與 utilization 水準規格化。
7. **§6 oracle gap 表述限制**：clairvoyant upper bound ≠ 可部署節省；排除成本清單明文化；$/token 換算獨立於假說判定。
8. **定位聲明收窄（供論文，非假說）**：公開 B300 多格式 benchmark 已存在（Selectel 2026-05：8×HGX B300、vLLM 0.16、DeepSeek V3.2 NVFP4 vs FP8、ISL 1K–16K、concurrency 4–256，NVFP4/FP8 不同 TP×DP 非 matched；TechInsights 2026-07：FP4/FP8/BF16——兩者皆無 INT8）；本研究之缺口主張收窄為「經 execution-path 驗證、checkpoint/TP/shape/power 受控之 matched INT8-vs-FP8 量測，與跨 H200/B300、held-out 驗證之 calibrated crossover predictor」。相近先行文獻（LLM-PQ PPoPP'24 poster、PMPD ICLR'25、SplitQuant Cluster'25、HMA-Serve arXiv'26、Mix-Quant arXiv'26、MorphServe MLSys'26 oral、OmniPilot arXiv'26、Vidur MLSys'24、NestedFP NeurIPS'25）已錄入 annotated bibliography，論文 related work 據此撰寫；其中無一者量測或預測 load-regime 依賴之格式 crossover、無一者於 B300 矽上做 matched INT8 量測。
9. 矩陣 v4（加 requant 臂）與 harness 容量包絡實作完成後，以 A-5 重釘 §0 SHA。

### A-5（A 類，2026-08-13，A-4 後續：artifacts 重釘）
矩陣 v4（28 實驗、14,691 runs）：新增 REQUANT-B300-DECODE 與 REQUANT-B300-PREFILL（P2 診斷、§7 gates 先行、EXCLUDED-FROM-PREDICTOR、NCU 驗證 execution=FP8）。新增 `capacity_envelope.py`（§3.5 之 B_max/crop_grid 實作，self-test 重現預資料試算）。§0 SHA 已更新：matrix `949d996a5c055e2e`、capacity_envelope `580105ebb64e3018`；estimator 與驗收套件未動（SHA 不變）。gen_sweep.py 未修改（新列由既有 schema 展開）。

### A-6（A 類，2026-08-14）
釘定 §7 quality gates 之評測腳本：`experiment/harness/quality_gates.py`，SHA-256[:16] `41b922dd7d2ce1d5`，並列入 §0。此為 v2 §7 所要求之「首次使用前補入」義務之履行，非新增自由度：三個 gate 的資料集 revision、window/stride、prompt 來源規則（A-3）、needle 句與問句、門檻值均已於 v2/v3 鎖定，本腳本僅為其實作。管線已於 sm_89 代理硬體驗證可執行（三 gate exit 0；該次比較使用兩個不同來源的 1B fixture，非同一 BF16 母模型，故其數值非有效 gate 結果，僅驗證管線）。尚無任何目標硬體（H200/B300）quality-gate 數據。

### A-7（A 類，2026-08-14，缺陷修復）
`gen_sweep.py` 重釘為 `a77697682c3029f4`（原 `6198eb61a6cf3f71`）。**動因＝實際發現的靜默錯誤輸入缺陷**：原程式寫死 `MATRIX = HERE/../config_matrix.csv`（配合 repo 之 `experiment/harness/` 佈局）；在扁平部署（代理 VM 的工作目錄）中 `..` 解析至無關目錄，讀到一份 2026-08-12 遺留的 v3 舊矩陣（26 列、`a99071715d5e5992`），產生 14,451 runs 而非 v4 的 14,691——**且所有 SHA 檢查皆通過，因為被檢查的檔案不是被讀取的那一份**。修法：路徑解析改為「先同目錄、再上層」，且每次執行印出實際所讀矩陣之絕對路徑與 SHA-256[:16]。展開邏輯一字未動，同一矩陣之輸出不變（本機重跑仍為 14,691，矩陣 SHA 印為 `949d996a5c055e2e`）。舊矩陣殘留已自 VM 移除。教訓：SHA 釘定只在「被釘的檔案就是被讀的檔案」時有效；凡讀取外部輸入之腳本，執行時必須自報所讀路徑與雜湊。
