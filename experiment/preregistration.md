# 預註冊 v2：H200×B300 量化格式 Crossover 量測研究

**v2 鎖定日期：2026-08-12（Asia/Taipei）。本文件在任何目標硬體（H200/B300）數據存在之前鎖定。**
v2 取代同日 v1：v1 完成後立即接受三線敵對審查（洩漏／估計器／可 game 性），審查發現 4 FATAL、17 MAJOR、6 MINOR，全部在收取任何數據之前修入本版。審查全文存於 workflow 紀錄；此為預資料修訂，非資料後改動。

修改治理（取代 v1 的單一 Amendment 規則）：
- **A 類（預資料）**：首個目標硬體數據點之前的修改，自由記錄於文末。
- **B 類（資料後）**：僅限本文件**點名**的決策點（§8 NVFP4 recipe、§10 Gate 1 降級），各附決策準則與最晚時點。任何其他資料後修改＝協定偏離，須在論文中如實報告。

## 0. 鎖定的 artifacts（SHA-256 前 16 碼；v2 定稿時重算）

| 檔案 | SHA-256[:16] | 角色 |
|---|---|---|
| `experiment/harness/changepoint.py` | `a8b58ce3859a544f` | 估計器 v2（censored-inclusive CI、多交叉回報、失敗規則、paired block bootstrap、knee 存在性檢定、penalty 傳播） |
| `experiment/harness/validate_synthetic.py` | `6216d8c9e09088d0` | 驗收 v2（真實幾何 G13/G9/G9r5/G6、σ∈{.05,.10,.15}、S1–S10、K1–K2） |
| `experiment/harness/gen_sweep.py` | `6198eb61a6cf3f71` | sweep 展開器 |
| `experiment/config_matrix.csv` | `a99071715d5e5992` | 實驗矩陣 v3（26 實驗、14,451 runs；MICRO 已拆分、penalty 已移出 held-out ISL） |
| `experiment/harness/qwen3_32b_config.json` | `97e295b632839357` | 主模型架構參數 |

驗證設定聲明：crossover 覆蓋率以鎖定之 B=1000 驗證；knee-CI 與 penalty 情境以 B=300、60–100 trials 驗證（覆蓋率標準誤 4–6%，驗收帶已吸收），此為預先聲明的驗證設定。

## 1. 假說與判定準則

- **H1**：`hardware × format × regime` 交互作用。判定：§9 模型的交互項以 time-block cluster-bootstrap Wald 聯合檢定，α=0.05；regime 編碼＝log₂(load) 連續軸 + phase 類別。
- **H2**：兩組 crossover（INT8–FP8、NVFP4–FP8）位置在兩硬體間位移。**B300 INT8 側的 H2 主分析使用校正曲線**（`corrected_crossover_ci`，P 不確定性在每個 bootstrap draw 內重抽傳播）；raw 曲線並列報告。`penalty_extrapolated` 的格點不進入 H2 確證。
- **H3**：predictor 在 §3 held-out 上優於三 baseline。**適用矩陣**：winner accuracy 與 regret 對 B1/B2/B3；boundary error 與 calibration 僅對 B3（B1/B2 不產生位置與區間，該欄不適用——此為 v1「全部指標優於全部 baseline」的修正）。**H3 的 B300 INT8 標籤一律使用 RAW 量測**（predictor 亦以 raw 預測），校正僅屬 H2 的硬體歸因分析——預測與標籤不得同時通過同一條校正曲線。
- **H4**：predictor 引導配置於**下列兩個 SLO 至少其一**優於最佳固定格式配置。
  - **SLO-A（互動）**：TTFT p99 ≤ 500 ms 且 TPOT p50 ≤ 50 ms。
  - **SLO-B（批次）**：TTFT p99 ≤ 2000 ms 且 TPOT p99 ≤ 100 ms。
  - **Goodput** ≝ 每 GPU-秒內完成且全程滿足該 SLO 兩項條件的請求數。**GPU-hour/token** ≝ 掛鐘時間×GPU 數 ÷ 產出 tokens。判定：配對 bootstrap（B=1000）之 goodput 差 90% CI > 0，逐 SLO 報告，兩個都報，不得只報有利者。

### 1.1 Predictor 規格（現在鎖定，杜絕凍結漏洞）

Predictor ＝ **確定性機制模型，無可調超參數**：對每（hardware, format, phase），由 TRAINING-ELIGIBLE 量測擬合 attained-roofline 參數（attained slope、attained peak；B300 INT8 另乘 raw 路徑之 P 曲線——僅供其 raw 預測），對任一 query 點輸出預測吞吐曲線，crossover 由同一 `changepoint.py` 估計器在預測曲線上求得；預測區間＝roofline 參數之 bootstrap 經預測函數傳播。**沒有模型選擇、沒有超參數搜尋、沒有學習組件**；因此凍結內容＝擬合出的 roofline 參數與選點清單。

### 1.2 Predictor Freeze Amendment（必要程序）

首個 held-out 量測（含 405B、ISL=8192 開封、off-grid 值、MICRO-*-405B）之前，必須存在一則 Freeze Amendment，內含：predictor 程式與擬合參數之 SHA-256、405B 選點完整清單、以及**可外部驗證的時間戳**（推送至公開 git remote 的 signed tag，或 OSF registration）。缺少先行凍結紀錄的 held-out 量測一律作廢。

## 2. 估計器（v2 語義；程式為準）

- Crossover：中位數曲線、log₂ 內插；**回報全部交叉**（`n_crossings`、清單），主分析取第一個；`n_crossings>1` 的 pair 排除於單交叉 calibration 指標並全數列報。
- 無交叉：端點 |d| 相差 <20% → `no-crossing-indicated`（不猜方向）；否則 `below`/`above`。**不外插。**
- CI：paired **block** bootstrap（repeat index＝time block；A/B 同格點同 block 綁定重抽），B=1000；censored draws 以 ±∞ 進入百分位，落在 censored 質量上的界以開界（`<x₁`/`>xₙ`）回報；另報 `censor_frac` 與 `multimodal_frac`。
- 失敗規則：y≤0 或缺 repeats → 該格點自**兩臂**同時剔除並記錄；存活格點 <5 → `unestimable`。
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
| EXCLUDED-FROM-PREDICTOR（robustness/應用，不入訓練亦不入 H3 評分） | QWEN-MIXED-*、QWEN-TP-SENS-*、SGLANG-ROBUST-*、PD-* |

「selected-boundary-points」（TP-SENS/SGLANG 列）＝以**量測到的** causal-grid crossover 兩側相鄰格點為準的確定性規則，非 predictor 選擇。

### 3.2 Shape hold-out
405B 全部 layer shapes（含 MICRO-*-405B 與 LLAMA405-LAYER-*）。Roofline 擬合、特徵縮放、任何超參選擇不得接觸（v1 FATAL 修正：MICRO 已拆分為 -QWEN 與 -405B 兩列）。

### 3.3 Load 與 context hold-out（逐軸、僅限網格範圍內）
- MICRO active-token 軸（1..4096）：hold-out {3, 12, 48, 192, 768, 3072}。
- Decode concurrency 軸（1..256）：hold-out {3, 12, 48, 192}。
- Prefill batch 軸（1..32）：hold-out {3, 12, 24}。
- Context：ISL=8192 桶整層（**修正 v1 不一致**：prefill 訓練 ISL={128,512,2048,32768}；decode 訓練 ISL={128,2048,32768}——decode 網格本無 512）。
- **隔離協定**：harness 將全部 held-out 格點寫入獨立檔案，收集當下即計 SHA-256 並記錄；Freeze Amendment 之前不得開啟；每次開封記錄於 Amendment。

### 3.4 405B 端到端選點（規則現鎖）
每（hardware, crossover pair, phase）：以凍結後 predictor 的 x̂* 為中心，取 **7 個 log-uniform 點於 [x̂*/4, x̂*×4]**（裁剪至可行域），**加 4 個固定錨點**（decode: concurrency {8, 64, 224}；prefill: batch {2, 24}——與任何 hold-out 值不重合），使嚴重誤預測可被觀測。全部點列入 Freeze Amendment。

## 4. Baselines（可唯一重實作）

- **B1 固定格式**：每（hardware×phase），對 QWEN-{PREFILL,DECODE}-{HW} 之全部 TRAINING-ELIGIBLE 格點取**中位吞吐之幾何平均**最高的格式。
- **B2 phase-only**：同 B1 聚合，per-phase 選格式。
- **B3 spec-peak roofline**：crossover 預測 = 兩格式規格 dense 峰值與公布頻寬構成之理想 roofline 交點。**規格數字表（現鎖）**：
  | (硬體, 格式) | dense 峰值 | 來源 |
  |---|---|---|
  | H200 FP8 / INT8 | 1,979 TFLOPS / 1,979 TOPS | NVIDIA H200 頁（sparse 3,958 ÷2；INT8 行單位標示 "TFLOPS" sic） |
  | H200 BF16 | 989.5 TFLOPS | 同上（1,979 sparse ÷2） |
  | B300 FP8 / INT8 | 4.5 PFLOPS / 150 TOPS | Blackwell Architecture Technical Brief Table 3（Dense/Sparse 標註列） |
  | B300 NVFP4 | 《A 類 Amendment 填入，限引 Technical Brief/Datasheet 逐字》 | 同上 |
  | HBM 頻寬 H200 / B300 | 《同上規則填入》 | H200 頁 / Technical Brief |
  B3 公式：x*_B3 = 兩格式 peak 相等點於理想 roofline min(bw·x·bytes⁻¹, peak) 下之解析交點；無交點→censored 預測。

## 5. B300 INT8 報法與 P(regime)

- P 參數化：per-phase，對 log₂(load) 之分段線性內插；prefill 軸由 BACKEND-PENALTY-H200（batch 1|4|16|64；ISL 128|2048|32768，**8192 由內插**）、decode 軸由 BACKEND-PENALTY-DECODE-H200（concurrency 1|4|16|64|256）。支撐域外→nearest-P 並標記，`penalty_extrapolated` 格點不入 H2 確證。
- H3 一律 raw；H2 主分析 corrected（§1）。校正臂與參照臂**不得**共用同一條校正（參照臂 FP8 永為 native、不校正）。

## 6. 評估指標（含平手與混合情形）

- **Winner accuracy**：量測勝者＝中位吞吐較高者；若配對 block-bootstrap 90% CI（中位 log 差）含 0 → **statistical tie**，自分母剔除並報 tie 數（predictor 與全部 baseline 同規則）。
- **Boundary error**：雙方皆數值 → |log₂x̂*−log₂x*|，報中位數；**censored 對 censored 僅在方向一致時記正確**；數值對 censored（任一向）→ accuracy 記錯、不入 |log₂| 聚合，計數列報。
- **Calibration**：predictor 90% 區間之覆蓋率；比較量 |coverage−0.90|，僅對 B3。
- **Regret**：選錯格式相對 oracle 的中位吞吐損失。
- **OOD degradation**：Qwen-trained → 405B 之上述指標變化。

## 7. Quality gates（全部釘死）

資料集：`Salesforce/wikitext`，config `wikitext-2-raw-v1`，**test split**，dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- **PPL**：串接全 test split，context window 4096、**非重疊 stride**，token-level NLL 之 exp；相對增幅 INT8/FP8 ≤2%、NVFP4 ≤5%。
- **一致率**：prompts＝該 test split 全文串接後（保留原始換行）以 GPT-2 tokenizer 切成不重疊的 320-token 窗口，**依序取前 200 個完整窗口**，每個窗口取**前 64 tokens** 為 prompt（確定性；prompt 清單 SHA 隨 artifacts 發布）。〔A-3 修正：v2 原寫「前 200 篇 ≥256 tokens 之文件」，但該 split 僅含 63 篇頂層文件，物理上不可能取 200 篇；改為窗口切分，維持「確定性、可重算、涵蓋整個 split」的原意。〕greedy 256 tokens；**每 prompt 分數＝首次分歧前吻合 tokens 數 ÷256，gate 於 200 篇平均**；INT8/FP8 ≥95%、NVFP4 ≥90%。**BF16 參照**＝H200、釘住之 vLLM image、enforce_eager、greedy；各（arm×hardware×backend）對同一參照計分；另報 BF16-on-B300 對參照的一致率作為跨硬體數值基線。
- **Needle-in-haystack**：haystack＝同資料集 test split 串接；needle＝固定句 "The secret checkpoint code is 731942."（於 depth 處插入）；depth {0,10,…,100}%×length {8k,16k,32k}＝33 cells、每 cell 1 次確定性 greedy；問句固定 "What is the secret checkpoint code?"；通過＝回答含子字串 "731942"；分數＝33 cells 通過率；降幅 ≤5 pts（相對同硬體 BF16 參照，參照分數一併列報）。評測腳本 SHA 於首次使用前以 A 類 Amendment 補入。
- Gate 範圍：逐（arm×hardware×實際 benchmark 之 backend）評定；某硬體上未過 → 僅該硬體之等品質比較排除該臂，全部 gate 數字照報。

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
- **Gate 1 失敗（B 類決策點）**：Triton 路徑無 IMMA → INT8 降為 kernel 層證據，§5 作廢、H2 之 INT8 部分改由 IMMA 微基準回答；Amendment 須引 `ncu_verdict` log。

---

## Amendments

### A-1（A 類，2026-08-12，v2 定稿同時）
v1→v2 全面修訂，動因＝預資料敵對審查（4 FATAL/17 MAJOR/6 MINOR）。要點：MICRO 拆分修 shape 洩漏；penalty 移出 ISL=8192 並新增 decode 軸列；H3 raw-label／H2 corrected 分離；H4 SLO 具體化；NVFP4/FP8 決策收緊；quality gates 全釘；估計器 v2（censored-CI、多交叉、失敗規則、block bootstrap、knee 存在性、P 傳播）；驗證擴至真實幾何與 S1–S10/K1–K2。v1 全文保留於版本歷史。

### A-2（A 類，2026-08-12，v2 驗證完成）
§0 SHA 已填。驗證於 `gpu-worker-A` 全數通過（19/19）：S1 六種幾何（G13/G9/G6、r∈{5,7}、σ∈{.05,.10,.15}）誤差 0.023–0.089、覆蓋 0.905–0.95；S2 自信假交叉率 0.015（≤0.10；總假交叉 28% 屬解析度極限內、98.5% 帶不確定性旗標——邊緣 gap <2.5×中位數標準誤時本方法不宣稱能分辨無交叉與交叉，見 §2）；S3 離群穩健 0.028；S4 近切線情境 99.5% 帶旗標（文件化解析度極限）；S5 平滑曲線 0.119／knee 偏差 0.144；S6 雙交叉偵測率 1.00、首交叉誤差 0.024；S7 block 效應下覆蓋 0.94；S8 penalty 傳播覆蓋 0.94；S9 純冪律 no-knee 率 0.98（existence 門檻 1.5→3.0，門檻 1.5 時假 knee 率 22% 之教訓記於程式註解）；S10 近邊緣覆蓋 0.935；K1 knee 誤差達標、K2 knee CI 覆蓋達標。完整 log：`logs/synthetic_v2b.log`（VM）。

### A-3（A 類，2026-08-12）
§7 一致率的 prompt 來源由「前 200 篇文件」改為「串接後不重疊 320-token 窗口取前 200 個、各取前 64 tokens」。動因：實作前查證發現 `wikitext-2-raw-v1` test split 僅有 63 篇頂層文件（4,358 行、1,294,336 字元），原規格無法滿足 200 篇。資料集 revision、PPL 協定、needle 協定均不變。此為預資料修訂（尚無任何目標硬體數據）。
